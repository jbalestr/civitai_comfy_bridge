"""reclassify_cli.py — the middle stage of the pipeline, run between
cli.py's --download-only and --embed-only:

    uv run python -m civitai_comfy_bridge.reclassify_cli \
        --input ./json_pulls/ \
        --data-root ./data

Recomputes every record's filter bucket against the *current* filter
code (downloader.py's _filter_bucket/_has_prompt/_has_character —
these evolve as filters get tuned) and compares it to what's actually
on disk right now, per --data-root's manifest.json.

--input accepts either one civitai_output.json or a DIRECTORY of them
(every *.json inside, merged and deduped by imageId). Use the
directory form if data_root has accumulated images across multiple
pulls over time — reclassify needs each image's ORIGINAL record to
recompute its bucket.

Two different ways metadata can go missing, both routed to a flat
data_root/orphaned/ folder so they're easy to find and deal with by
hand, but handled separately since they mean different things:

    orphaned  A manifest entry exists (modelId/createdAt/etc. are
              known) but its original civitai record isn't in whatever
              --input covers this run — e.g. an older pull's JSON
              wasn't included. Recoverable: point --input at a
              directory covering more of your pulls and re-run.

    unlinked  A raw file sits on disk but the manifest has NO entry
              for it at all — manifest.json was overwritten/corrupted,
              a file got moved by hand outside this tool, etc. Nothing
              left to recover from; found by scanning data_root
              directly rather than via any summary row.

This scan for unlinked files always runs and is reported every time
(even with no --input/--from-summary at all — data-root alone is
enough), moved only under --apply, same as everything else.

Always writes two files next to --input (or inside it, if --input is
a directory):

    <name>_summary.csv        one row per record — imageId, prompt
                               actually used for filtering, and every
                               bucket signal (see downloader.py's
                               build_filter_summary), plus one row per
                               orphaned entry
    <name>_summary_diff.html  thumbnail + current-vs-proposed bucket,
                               for every record whose proposed bucket
                               differs from what's on disk.
                               Self-contained (base64 thumbnails) —
                               just open it in a browser.

Bare filenames/imageIds mean nothing to a human — the diff HTML exists
so you can actually SEE which images would move and where before
touching anything on disk.

By DEFAULT THIS IS A DRY RUN — nothing is moved. Review the diff HTML
(and hand-edit the summary CSV if a few rows need manual correction),
then either:

    re-run this same command with --apply, or
    re-run with --from-summary <name>_summary.csv --apply
    (skips recomputing from --input, uses your hand-edited CSV as-is)

--apply moves ordinary bucket changes and unlinked files, but NEVER
moves "orphaned" rows — that needs a separate --apply-orphaned, since
a partial --input (e.g. one day's pull, not your full history) makes
most of your manifest look orphaned, and that must never get swept
into orphaned/ just because --apply was passed for something else.
Only add --apply-orphaned once you've confirmed --input genuinely
covers your full pull history and the remaining orphaned rows are real.

Moving files this way is deliberately decoupled from downloading —
filters get tuned after looking at a batch, and re-running this against
already-downloaded raw files costs nothing, unlike re-pulling several
GB from civitai.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import downloader


def _load_records(input_path: Path) -> list[dict]:
    """Load civitai records from --input: one JSON file, or a
    directory of them merged and deduped by imageId (last file wins on
    a clash, sorted by filename for determinism — shouldn't matter
    since a given imageId's record shouldn't change between pulls).

    Reclassify needs every image currently on disk's ORIGINAL record
    to recompute its bucket, which usually means the full history of
    pulls, not just the latest one — that's the point of the directory
    form.
    """
    if input_path.is_dir():
        json_files = sorted(input_path.glob("*.json"))
        if not json_files:
            raise SystemExit(f"no .json files found in {input_path}")
        by_id: dict[str, dict] = {}
        for jf in json_files:
            for rec in json.loads(jf.read_text(encoding="utf-8")):
                by_id[str(rec["imageId"])] = rec
        print(f"loaded {len(by_id)} unique record(s) from {len(json_files)} file(s) in {input_path}", flush=True)
        return list(by_id.values())

    records = json.loads(input_path.read_text(encoding="utf-8"))
    print(f"loaded {len(records)} record(s) from {input_path}", flush=True)
    return records


def _default_summary_path(input_path: Path) -> Path:
    """<input>_summary.csv next to a file, or reclassify_summary.csv
    inside a directory."""
    if input_path.is_dir():
        return input_path / "reclassify_summary.csv"
    return input_path.with_name(f"{input_path.stem}_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, type=Path,
                         help="root folder for dated download runs + manifest.json")
    parser.add_argument("--input", type=Path,
                         help="path to civitai_fetcher's civitai_output.json — recomputes filter decisions fresh "
                              "from the current filter code. Omit if using --from-summary, or if you only want "
                              "the unlinked-file scan below")
    parser.add_argument("--from-summary", type=Path, metavar="PATH",
                         help="use this existing summary CSV as-is (e.g. one you hand-edited) instead of "
                              "recomputing from --input")
    parser.add_argument("--apply", action="store_true",
                         help="actually move files whose recomputed bucket differs from the manifest, and unlinked "
                              "files (see below). Default is a dry run: reports/writes only, moves nothing. Does "
                              "NOT move orphaned entries (see --apply-orphaned)")
    parser.add_argument("--apply-orphaned", action="store_true",
                         help="ALSO move manifest entries with no matching record in --input into orphaned/ "
                              "(see 'orphaned' in the docstring above). Separate from --apply on purpose: a "
                              "partial --input (e.g. one day's pull, not your full history) makes most of your "
                              "manifest look orphaned, and that should never be swept into orphaned/ by accident. "
                              "Only pass this once you've confirmed --input genuinely covers everything")
    parser.add_argument("--restore-embeds", action="store_true",
                         help="one-off fix for a bug in earlier versions of this tool where already-embedded PNGs "
                              "got wrongly swept into orphaned/ by the unlinked-file scan. Moves them back to "
                              "their correct <date>/images/<bucket>/ location and exits; ignores --input/--apply")
    parser.add_argument("--check-drift", action="store_true",
                         help="report manifest entries whose raw_path doesn't exist on disk (what shows up as "
                              "'raw file missing' elsewhere), and whether a same-named file was found elsewhere "
                              "in data_root. Dry run only — see --repair-drift to fix. Ignores --input/--apply")
    parser.add_argument("--repair-drift", action="store_true",
                         help="like --check-drift, but also updates raw_path for any UNAMBIGUOUS match (exactly "
                              "one same-named file found elsewhere). Doesn't move files, only corrects the "
                              "manifest's record of where they are. Ignores --input/--apply")
    args = parser.parse_args()

    if args.restore_embeds:
        result = downloader.restore_misplaced_embeds(args.data_root)
        print(f"restored {result['restored']}, left alone {result['left_alone']}, conflicts {result['conflicts']}", flush=True)
        return

    if args.check_drift or args.repair_drift:
        if args.repair_drift:
            result = downloader.repair_manifest_raw_paths(args.data_root)
            print(f"repaired {result['repaired']}, not found {result['not_found']}, ambiguous {result['ambiguous']}", flush=True)
        else:
            drifted = downloader.find_manifest_drift(args.data_root)
            print(f"{len(drifted)} manifest entries drifted from disk", flush=True)
            for d in drifted:
                found = d["found_at"]
                if not found:
                    status = "NOT FOUND anywhere else"
                elif len(found) == 1:
                    status = f"found at {found[0].relative_to(args.data_root)} (would auto-repair)"
                else:
                    status = f"{len(found)} ambiguous candidates"
                print(f"  {d['imageId']}: expected {d['expected_path'].relative_to(args.data_root)} — {status}", flush=True)
        return

    if args.input and args.from_summary:
        parser.error("--input and --from-summary are mutually exclusive")

    rows = []
    orphan_rows = []
    summary_path = None

    if args.from_summary:
        summary_path = args.from_summary
        all_rows = downloader.read_summary_csv(summary_path)
        rows = [r for r in all_rows if r.get("final_bucket") != "orphaned"]
        orphan_rows = [r for r in all_rows if r.get("final_bucket") == "orphaned"]
        print(f"using existing summary {summary_path} ({len(rows)} bucket row(s), {len(orphan_rows)} orphaned row(s))", flush=True)
    elif args.input:
        civitai_records = _load_records(args.input)

        manifest = downloader.load_manifest(args.data_root)
        orphan_rows = downloader.orphaned_summary_rows(manifest, civitai_records)
        if orphan_rows:
            pct = 100 * len(orphan_rows) / len(manifest)
            print(f"note: {len(orphan_rows)} of {len(manifest)} ({pct:.0f}%) manifest entries have no matching "
                  f"record in --input — flagged as final_bucket=orphaned below. NOT moved by --apply; only "
                  f"--apply-orphaned moves these. If this is most/all of your manifest, --input almost certainly "
                  f"doesn't cover your full pull history yet — point it at a directory with everything instead "
                  f"of assuming these are really orphaned", flush=True)

        rows = downloader.build_filter_summary(civitai_records)
        summary_path = _default_summary_path(args.input)
        downloader.write_filter_summary_csv_rows(rows + orphan_rows, summary_path)
        print(f"wrote summary to {summary_path}", flush=True)

    if summary_path:
        diff_path = summary_path.with_name(f"{summary_path.stem}_diff.html")
        changed = downloader.write_diff_report_html(rows + orphan_rows, args.data_root, diff_path)
        print(f"wrote {changed} changed row(s) to {diff_path}", flush=True)
    else:
        changed = 0

    # --- unlinked-file scan: always runs, regardless of --input/--from-summary ---
    unlinked = downloader.find_unlinked_raw_files(args.data_root)
    if unlinked:
        print(f"found {len(unlinked)} unlinked file(s) on disk with no manifest entry at all "
              f"(lost metadata, not just missing from --input):", flush=True)
        for path in unlinked:
            print(f"    {path.relative_to(args.data_root)}", flush=True)

    if args.apply:
        if rows:
            result = downloader.apply_summary(rows, args.data_root)
            print(f"moved {result['moved']}, unchanged {result['unchanged']}, skipped {result['skipped']}", flush=True)
        if unlinked:
            moved = downloader.move_unlinked_raw_files(args.data_root)
            print(f"moved {moved} unlinked file(s) to {args.data_root / 'orphaned'}", flush=True)
    else:
        if changed:
            print(f"dry run only — review {diff_path}, hand-edit {summary_path} if needed, "
                  f"then re-run with --from-summary {summary_path} --apply", flush=True)
        elif summary_path:
            print("no bucket changes — every record already matches the manifest", flush=True)
        if unlinked:
            print("re-run with --apply to move the unlinked file(s) above into orphaned/", flush=True)

    if args.apply_orphaned and orphan_rows:
        result = downloader.apply_summary(orphan_rows, args.data_root)
        print(f"[orphaned] moved {result['moved']}, unchanged {result['unchanged']}, skipped {result['skipped']}", flush=True)
    elif orphan_rows and not args.apply_orphaned:
        print(f"{len(orphan_rows)} orphaned row(s) NOT moved — re-run with --apply-orphaned once you've confirmed "
              f"--input truly covers everything", flush=True)


if __name__ == "__main__":
    main()