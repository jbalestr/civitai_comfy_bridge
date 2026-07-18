"""downloader.py — downloads images referenced in civitai_fetcher's output
JSON into a dated folder under --data-root, skipping anything already
present in the shared manifest.json so repeat daily/weekly runs never
duplicate a file across folders.

Dedup key: civitai's imageId (globally unique, per civitai_fetcher's
README), not createdAt — createdAt is kept in the manifest as metadata
for later filtering ("only images from last week"), not as the identity
key. See README.md "Open questions" for the content-hash fallback we're
deliberately not building yet.

manifest.json lives at the data-root, not inside a dated folder — it's
the cross-run source of truth. Shape:

{
  "<imageId>": {
    "raw_path": "2026-07-13/raw/123_456.jpeg",   relative to data-root
    "modelId": 123,
    "createdAt": "...",                          from civitai, as-is
    "downloaded_at": "2026-07-13",                our own run date
    "embedded": false                             set true by cli.py once
                                                    png_writer.py has produced
                                                    the final PNG (see below)
  },
  ...
}

Filename convention: "{modelId}_{imageId}.<ext>" — imageId alone is a
sufficient dedup key, but modelId is included in the filename for
human-readability and to keep the on-disk name self-describing without
needing to open the manifest. Extension is taken from the source URL
(civitai serves jpeg/png/webp — not necessarily PNG) rather than
assumed, since this is the raw, unmodified download, not the final
embedded PNG.

Raw vs final split: this module ONLY ever writes into "<run>/raw/" and
never touches those files again once written — they're a permanent
untouched cache of exactly what civitai served. The separate, final
PNG-with-embedded-metadata that portfolio-explorer actually indexes
lives in "<run>/images/", produced by png_writer.py from these raw
bytes, orchestrated by cli.py. Keeping raw and final in separate
folders means png_writer.py / graph_builder.py can change or be re-run
later without ever needing to re-download anything.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import httpx
import re as _re

MANIFEST_FILE = "manifest.json"

# civitai posts can be video (mp4/webm/mov), not just images — same
# imageUrl field either way, no separate type field to check. Videos
# aren't useful for image-similarity/region-embedding and are often
# 10-60x the size of a jpeg, so they're filtered out before download
# rather than downloaded and discarded later.
_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov"}

# Fallback when a civitai imageUrl has no recognisable suffix (seen
# rarely — most URLs end in .jpeg/.png/.webp) — .bin rather than
# guessing wrong, since png_writer.py decodes via PIL regardless of
# what's on disk, not by trusting this extension.
_DEFAULT_EXTENSION = ".bin"

# Keywords checked against the POSITIVE prompt only.
_FURRY_KEYWORDS = {"furry", "tail","anthro", "feral", "kemono", "scales", "scaled", "fursona", "animal"}

# Bondage/restraint context — excludes clothing alone (bodysuits, latex)
# since those appear in superhero/villain content.
_BONDAGE_KEYWORDS = _re.compile(
    r'\b(bondage|posture\.collar|posturecollar|padded\.room|padded\.cell|ballgag|ball\.gag|shackle)\b'
)

# Horror — specific terms only, avoids false positives on action/fantasy.
# Model blocklist handles cases where the prompt is descriptive prose.
_HORROR_KEYWORDS = _re.compile(
    r'\b(sharp\.teeth|pointed\.teeth|monster\.mouth|jagged\.teeth|eldritch|'
    r'gore|guro|body\.horror|cosmic\.horror|sharp\.fangs|creepy)\b'
)

# Model name blocklist for horror — these models produce horror by default
# regardless of prompt keywords.
_HORROR_MODEL_BLOCKLIST = {"weallstarve"}

# Solo male — must have BOTH a male tag AND explicit masculine descriptors.
# Androgynous/femboy characters (Venti, Astolfo etc.) are tagged 1boy but
# lack masculine descriptors so they pass through correctly.
_SOLO_MALE = _re.compile(r'\b(1boy|male focus|gay)\b')
_MASCULINE = _re.compile(r'\b(masculine|gay|muscular|beard|chest hair|bulge|pecs|abs|stubble|facial hair)\b')
_FEMALE_CHAR = _re.compile(r'\b(\d*girl|\d*girls|woman|women|female|lady|ladies|waifu)\b')

# Buckets set aside for manual review rather than nsfw-level routing —
# these live flat under data_root (no dated raw/ folder), and are
# excluded from portfolio-explorer's index (see apply_summary).
# "orphaned" is special: it's not a content filter decision at all —
# it means the manifest entry's original civitai record couldn't be
# found (e.g. an older pull's JSON wasn't included in --input), so
# there's nothing to recompute a bucket from. See orphaned_summary_rows.
_REVIEW_BUCKETS = {"furry", "bondage", "horror", "solomale", "nocharacter", "noprompt", "orphaned"}


def _bucket_from_raw_path(raw_path_str: str) -> str:
    """Derive which bucket a file is ACTUALLY sitting in from its
    raw_path, rather than trusting the manifest's separately-tracked
    "filter" label. The two can drift apart — e.g. repair_manifest_raw_paths
    fixes raw_path after files get relocated (by hand, by a bug, by
    anything outside apply_summary's own bookkeeping) but has no way
    to know what "filter" should now say. Comparing against physical
    reality here means that kind of drift can't cause apply_summary to
    wrongly call something "unchanged" just because a stale label
    happens to match the target.

    Recognizes the two path shapes this tool ever writes:
        "<bucket>/filename"                flat review bucket
        "<date>/raw/<bucket>/filename"      dated download
    Returns "" for anything else (unrecognized shape — caller should
    not treat that as a meaningful bucket match).
    """
    parts = Path(raw_path_str).parts
    if len(parts) == 2:
        return parts[0]
    if len(parts) == 4 and parts[1] == "raw":
        return parts[2]
    return ""


def _resolve_comfy_text(nodes: dict, value) -> str:
    """Resolve a ComfyUI widget input to its actual string value.

    `value` is either a literal (already the string we want) or a link
    `[node_id, output_index]` pointing at the node that produces it.
    Recurses through StringConcatenate (joining string_a/string_b with
    its delimiter) and generic single-"text"-input nodes (CR Text and
    similar custom nodes all shape their prompt as a plain "text"
    input), since civitai's raw embedded workflow builds prompts this
    way rather than always giving one literal string.
    """
    if isinstance(value, str):
        return value
    if not (isinstance(value, list) and len(value) == 2):
        return ""

    node = nodes.get(str(value[0]))
    if node is None:
        return ""
    inputs = node.get("inputs") or {}

    if node.get("class_type") == "StringConcatenate":
        a = _resolve_comfy_text(nodes, inputs.get("string_a", ""))
        b = _resolve_comfy_text(nodes, inputs.get("string_b", ""))
        delim = _resolve_comfy_text(nodes, inputs.get("delimiter", ""))
        return f"{a}{delim}{b}"

    if "text" in inputs:
        return _resolve_comfy_text(nodes, inputs["text"])

    return ""


def _extract_comfy_prompt_text(record: dict) -> str:
    """Best-effort positive-prompt text pulled from civitai's embedded
    raw ComfyUI workflow (meta.comfy), for records where civitai gave
    no flat meta.prompt — seen with custom/complex workflows (e.g.
    Anima-based ones using CR Text + StringConcatenate chains instead
    of a single CLIPTextEncode). Without this, such records look
    prompt-less to _has_prompt/_has_character/_filter_bucket even
    though a real prompt exists in the graph.

    Walks every KSampler node's "positive" input and resolves whatever
    text feeds it; doesn't try to pick a single "final" sampler (hires
    fix / disabled passes make that ambiguous) since for filtering
    purposes collecting all candidate positive text is sufficient and
    more robust than guessing wrong.
    """
    meta = record.get("meta") or {}
    raw_comfy = meta.get("comfy")
    if not raw_comfy:
        return ""

    try:
        workflow = json.loads(raw_comfy)
        nodes = workflow.get("prompt") or {}
    except (json.JSONDecodeError, TypeError, AttributeError):
        return ""

    found = []
    for node in nodes.values():
        if node.get("class_type") != "KSampler":
            continue
        positive = (node.get("inputs") or {}).get("positive")
        text = _resolve_comfy_text(nodes, positive)
        if text:
            found.append(text)

    # Dedup while preserving order — hires-fix passes usually share the
    # same positive prompt, no need to repeat it.
    seen = set()
    unique = [t for t in found if not (t in seen or seen.add(t))]
    return ", ".join(unique)


def _effective_prompt_text(record: dict) -> str:
    """The positive prompt text to use for all filtering decisions:
    civitai's flat meta.prompt when present, otherwise whatever can be
    recovered from the embedded raw ComfyUI workflow. Centralised here
    so _filter_bucket/_has_prompt/_has_character can't drift out of
    sync on which source of truth they check.
    """
    meta = record.get("meta") or {}
    prompt = str(meta.get("prompt") or "").strip()
    if prompt:
        return prompt
    return _extract_comfy_prompt_text(record)


def _filter_bucket(record: dict) -> str | None:
    """Return a review bucket name if the record should be filtered,
    or None if it should proceed to normal nsfw routing.

    Filters are checked in priority order — furry first, then content filters.
    All filtered images go to flat review buckets under data_root/.
    """
    prompt = _effective_prompt_text(record).lower()
    model_name = (record.get("modelName") or "").lower()
    tokens = set(_re.split(r"[,\s:|()\.\[\]]+", prompt))

    if tokens & _FURRY_KEYWORDS:
        return "furry"
    if _BONDAGE_KEYWORDS.search(prompt):
        return "bondage"
    if _HORROR_KEYWORDS.search(prompt) or any(b in model_name for b in _HORROR_MODEL_BLOCKLIST):
        return "horror"
    if _SOLO_MALE.search(prompt) and _MASCULINE.search(prompt) and not _FEMALE_CHAR.search(prompt):
        return "solomale"
    return None


def _is_furry(record: dict) -> bool:
    """Return True if the positive prompt contains a known furry keyword."""
    meta = record.get("meta") or {}
    prompt = str(meta.get("prompt") or "").lower()
    # split on common delimiters so we match whole tokens only
    import re
    tokens = set(re.split(r"[,\s:|()\[\]]+", prompt))
    return bool(tokens & _FURRY_KEYWORDS)


# Positive prompt tokens that confirm a character is present.
# A match here means the image is kept regardless of landscape keywords.
_CHARACTER_TOKENS = _re.compile(
    # numeric prefix variants: 1girl, 2boys, 3women etc.
    r'\b\d*(?:girl|boy|girls|boys|woman|women|man|men)\b' +
    r'|\b(solo|' +
    # gender/age
    r'male|female|lady|gentleman|guy|gal|lad|lass|' +
    # anime/game archetypes
    r'waifu|husbando|mecha|character|portrait|cyborg|human|people|crowd|person|face|' +
    # titles and roles (female + male)
    r'princess|prince|queen|king|goddess|god|warrior|knight|' +
    r'vampire|witch|wizard|mage|ninja|samurai|' +
    r'sister|brother|mother|father|daughter|son|' +
    # Japanese character archetypes
    r'yuki.onna|onmyoji|geisha|kunoichi|shrine.maiden|miko)\b' +
    # Chinese woman/man characters (no word boundary needed)
    r'|[女男]'
)

# Tokens that indicate a characterless environment/scene.
# Only acted on when _CHARACTER_TOKENS is absent.
_LANDSCAPE_TOKENS = _re.compile(
    r'\b(landscape|scenery|architecture|building|interior|cityscape|room|' +
    r'still.life|scenic|nature|background|view|hills|station|environment|object|props)\b'
)


def _has_prompt(record: dict) -> bool:
    """Return True if the record has a non-empty positive prompt,
    checking civitai's flat meta.prompt and falling back to text
    recovered from the embedded raw ComfyUI workflow (see
    _effective_prompt_text)."""
    return bool(_effective_prompt_text(record).strip())


def _has_character(record: dict) -> bool:
    """Return True if the positive prompt positively confirms a character is
    present. Returns False if prompt is missing or contains no character tokens.
    """
    prompt = _effective_prompt_text(record)

    if not prompt.strip():
        return False

    if _CHARACTER_TOKENS.search(prompt):
        return True

    return False


def _extension_from_url(url: str) -> str:
    """Best-effort file extension from a civitai imageUrl's path
    component (ignoring query string). Falls back to _DEFAULT_EXTENSION
    rather than raising — an unrecognised suffix shouldn't abort a
    download, since it's only used for the raw cache's filename, not
    for decoding.
    """
    suffix = Path(urlparse(url).path).suffix
    return suffix if suffix else _DEFAULT_EXTENSION


def load_manifest(data_root: Path) -> dict:
    """Load the shared manifest, or an empty dict if this is the first
    run against this data_root. Not raising on a missing file is
    deliberate — first-run-ever is an expected, not exceptional, case.
    """
    path = data_root / MANIFEST_FILE
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            # Corrupt/truncated manifest (e.g. killed mid-write, though
            # save_manifest's atomic replace should make that rare) —
            # treat as first-run rather than crash. Worst case this
            # re-downloads images already on disk; it never loses them.
            return {}
    return {}


def save_manifest(data_root: Path, manifest: dict) -> None:
    """Write the manifest atomically (write to .tmp, then replace) so a
    kill mid-write can't corrupt it — same pattern as build_index.py's
    own progress file in portfolio-explorer.
    """
    data_root.mkdir(parents=True, exist_ok=True)
    tmp_path = data_root / (MANIFEST_FILE + ".tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2))
    tmp_path.replace(data_root / MANIFEST_FILE)


def _creation_date(rec: dict) -> str:
    """Extract YYYY-MM-DD from civitai's createdAt ISO timestamp.
    Falls back to today if the field is missing or malformed.
    """
    created_at = rec.get("createdAt") or ""
    try:
        d = created_at[:10]
        if len(d) == 10 and d[4] == "-" and d[7] == "-":
            return d
    except Exception:
        pass
    return date.today().isoformat()


def run_download(civitai_records: list[dict], data_root: Path, source_json: str = "") -> dict:
    """Download every record's imageUrl not already in the manifest,
    into data_root/<creation-date>/raw/<nsfw-subdir>/, updating +
    periodically flushing the manifest as it goes.

    Images are filed under their civitai createdAt date (not today's
    download date) so that the same image fetched in two different runs
    — e.g. a Week run followed by a Month run — lands at the same path
    and is naturally deduped: if the file already exists on disk, the
    download is skipped and the manifest is updated from the existing
    file without hitting the network at all.

    civitai_records: the flat per-image dicts civitai_fetcher.py writes
    to civitai_output.json (imageId, modelId, imageUrl, meta, ...).

    Returns the final manifest dict (also persisted to disk). New
    entries are written with "embedded": False — cli.py flips that to
    True once png_writer.py has produced the corresponding final PNG,
    so a later run knows which raw files still need embedding without
    re-downloading anything.

    A single failed download (bad URL, timeout, 404) must not abort the
    whole run — log and continue. Do not add failed images to the
    manifest so the next run retries them.
    """
    from civitai_comfy_bridge.png_writer import nsfw_subdir
    manifest = load_manifest(data_root)

    already = len(manifest)
    is_video = lambda r: _extension_from_url(r["imageUrl"]) in _VIDEO_EXTENSIONS
    videos_skipped = [r for r in civitai_records if is_video(r)]
    seen_ids: set[str] = set()
    pending = []
    for r in civitai_records:
        iid = str(r["imageId"])
        if iid not in manifest and not is_video(r) and iid not in seen_ids:
            pending.append(r)
            seen_ids.add(iid)
    print(f"{already} already downloaded, {len(videos_skipped)} videos skipped, {len(pending)} pending", flush=True)

    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for i, rec in enumerate(pending):
            image_id = str(rec["imageId"])
            model_id = rec["modelId"]
            ext = _extension_from_url(rec["imageUrl"])
            created_date = _creation_date(rec)

            # --- pre-download content filters ---
            review = _filter_bucket(rec)
            if review:
                subdir = review
                run_dir = data_root / subdir
            else:
                subdir = nsfw_subdir(rec.get("nsfwLevel"))
                run_dir = data_root / created_date / "raw" / subdir

            run_dir.mkdir(parents=True, exist_ok=True)
            dest = run_dir / f"{model_id}_{image_id}{ext}"

            # --- file existence check: same image from a prior run ---
            if dest.exists():
                total_bytes = dest.stat().st_size
                print(f"  [{i + 1}/{len(pending)}] {dest.name} (exists, {total_bytes:,} bytes) [{subdir}]", flush=True)
            else:
                try:
                    with client.stream("GET", rec["imageUrl"]) as resp:
                        resp.raise_for_status()
                        total_bytes = 0
                        with open(dest, "wb") as f:
                            for chunk in resp.iter_bytes():
                                f.write(chunk)
                                total_bytes += len(chunk)
                except (httpx.HTTPError, OSError) as e:
                    dest.unlink(missing_ok=True)
                    print(f"  [{i + 1}/{len(pending)}] FAILED {image_id}: {e}", flush=True)
                    continue

                # --- prompt + character filters: only applied to soft bucket ---
                # Mature and explicit kept as-is — explicit almost always
                # has a subject; poorly-tagged ones aren't worth reviewing.
                if subdir == "soft":
                    if not _has_prompt(rec):
                        post_subdir = "noprompt"
                    elif not _has_character(rec):
                        post_subdir = "nocharacter"
                    else:
                        post_subdir = None

                    if post_subdir:
                        new_dest = data_root / post_subdir / dest.name
                        new_dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.replace(new_dest)
                        dest = new_dest
                        subdir = post_subdir

                print(f"  [{i + 1}/{len(pending)}] {dest.name} ({total_bytes:,} bytes) [{subdir}]", flush=True)

            manifest[image_id] = {
                "raw_path": str(dest.relative_to(data_root)),
                "modelId": model_id,
                "nsfwLevel": rec.get("nsfwLevel"),
                "filter": subdir,
                "createdAt": rec.get("createdAt"),
                "downloaded_at": date.today().isoformat(),
                "source_json": source_json,
                "embedded": False,
            }

            if (i + 1) % 50 == 0 or (i + 1) == len(pending):
                # Flush periodically, not only at the end — a killed run
                # partway through a large batch shouldn't lose already-
                # completed downloads to a manifest that was never saved.
                save_manifest(data_root, manifest)

    save_manifest(data_root, manifest)
    return manifest


def build_filter_summary(civitai_records: list[dict]) -> list[dict]:
    """One row per record, showing exactly what run_download's filters
    would decide for it — imageId, the prompt text actually used for
    filtering (post meta.comfy fallback, see _effective_prompt_text),
    and every filter signal that fed into the bucket decision.

    Doesn't touch the network or the manifest — pure read of
    civitai_records, so it's safe to run over the same input JSON
    before or after downloading, purely to review/audit filtering.

    review_bucket mirrors _filter_bucket (furry/bondage/horror/
    solomale, checked regardless of nsfw level). nsfw_bucket is what
    nsfw_subdir would file it under absent a review-bucket hit.
    has_prompt/has_character are only meaningful for the "soft" nsfw
    bucket — that's the only bucket run_download applies them to — but
    are reported for every row so you can see why a "mature"/"explicit"
    row was exempted from that check.
    """
    from civitai_comfy_bridge.png_writer import nsfw_subdir

    rows = []
    for rec in civitai_records:
        prompt = _effective_prompt_text(rec)
        nsfw_bucket = nsfw_subdir(rec.get("nsfwLevel"))
        review_bucket = _filter_bucket(rec)
        has_prompt = _has_prompt(rec)
        has_character = _has_character(rec)

        if review_bucket:
            final_bucket = review_bucket
        elif nsfw_bucket == "soft" and not has_prompt:
            final_bucket = "noprompt"
        elif nsfw_bucket == "soft" and not has_character:
            final_bucket = "nocharacter"
        else:
            final_bucket = nsfw_bucket

        rows.append({
            "imageId": rec.get("imageId"),
            "modelId": rec.get("modelId"),
            "modelName": rec.get("modelName"),
            "nsfwLevel": rec.get("nsfwLevel"),
            "nsfw_bucket": nsfw_bucket,
            "review_bucket": review_bucket or "",
            "final_bucket": final_bucket,
            "has_prompt": has_prompt,
            "has_character": has_character,
            "prompt_used": prompt,
        })
    return rows


def orphaned_summary_rows(manifest: dict, civitai_records: list[dict]) -> list[dict]:
    """Synthetic summary rows — same shape as build_filter_summary's,
    so they can be concatenated onto its output and flow through the
    same write_diff_report_html/apply_summary machinery — for every
    manifest entry whose original civitai record isn't present in
    civitai_records (e.g. an older pull's JSON wasn't included in
    --input this run). final_bucket is always "orphaned": there's no
    prompt/meta left to recompute a real bucket from, so rather than
    silently leaving these stuck wherever they happen to be, they're
    routed to a dedicated review bucket where they're easy to find
    and deal with by hand.

    Entries already sitting in "orphaned" are naturally left alone by
    apply_summary (current bucket already matches target), so re-runs
    don't keep re-flagging the same images.
    """
    record_ids = {str(r["imageId"]) for r in civitai_records}
    rows = []
    for image_id, entry in manifest.items():
        if image_id in record_ids:
            continue
        rows.append({
            "imageId": image_id,
            "modelId": entry.get("modelId"),
            "modelName": "(orphaned — no matching record in --input)",
            "nsfwLevel": entry.get("nsfwLevel"),
            "nsfw_bucket": "",
            "review_bucket": "orphaned",
            "final_bucket": "orphaned",
            "has_prompt": False,
            "has_character": False,
            "prompt_used": "",
        })
    return rows


def write_filter_summary_csv_rows(rows: list[dict], out_path: Path) -> int:
    """Write already-built summary rows (e.g. build_filter_summary's
    output, possibly concatenated with orphaned_summary_rows) to a CSV
    at out_path (parent dirs created as needed). Returns the row count
    written. Shared by write_filter_summary_csv and reclassify_cli.py,
    which needs to write rows from more than one source into one file.
    """
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["imageId", "modelId", "modelName", "nsfwLevel", "nsfw_bucket",
                      "review_bucket", "final_bucket", "has_prompt", "has_character", "prompt_used"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_filter_summary_csv(civitai_records: list[dict], out_path: Path) -> int:
    """Write build_filter_summary()'s rows to a CSV at out_path (parent
    dirs created as needed). Returns the row count written."""
    return write_filter_summary_csv_rows(build_filter_summary(civitai_records), out_path)


def read_summary_csv(path: Path) -> list[dict]:
    """Read a filter-summary CSV (as written by write_filter_summary_csv,
    hand-edited or not) back into row dicts for apply_summary."""
    import csv

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _locate_embed(data_root: Path, model_id, image_id: str, expected: Path, exclude: Path | None = None) -> Path | None:
    """Return the embedded PNG's actual on-disk location, or None if
    it can't be found anywhere.

    Checks `expected` first (cheap, correct in the common case: the
    PNG is exactly where build_dest_path says it should be given the
    manifest's CURRENT raw_path). Falls back to a full search by exact
    filename across ALL of data_root — not just images/ folders,
    because a PNG can end up sitting flat in a review bucket like
    orphaned/ with no images/ subfolder at all (e.g. never actually
    restored there, just relocated wholesale along with its raw file
    during a repair). Reclassify has moved things around enough times
    across different tool versions that "the PNG lives under images/"
    isn't a safe assumption to gate the search on; matching by exact
    filename is.

    `exclude` should be the raw file's own current path, so a raw file
    that happens to share the embed's exact filename (rare, but
    possible if civitai served the original as .png) can never be
    mistaken for the embed and moved as if it were one.
    """
    if expected.exists():
        return expected
    filename = f"{model_id}_{image_id}.png"
    exclude_resolved = exclude.resolve() if exclude is not None else None
    for match in data_root.rglob(filename):
        if exclude_resolved is not None and match.resolve() == exclude_resolved:
            continue
        return match
    return None


def apply_summary(summary_rows: list[dict], data_root: Path) -> dict:
    """Move already-downloaded raw files to match each row's
    final_bucket, without re-downloading anything — the point being to
    iterate on filter logic (tweak code, regenerate the summary; or
    hand-edit a few rows) and try it against files already on disk,
    instead of re-pulling a couple of GB every time.

    Only touches images this data_root's manifest already knows about
    (matched by imageId) and whose raw file still exists; anything else
    is skipped with a message rather than erroring — the summary may
    have been generated against a different data_root, or cover images
    not downloaded yet.

    Rows whose final_bucket already matches the manifest's recorded
    "filter" are left alone (no-op).

    If the image was already embedded (entry["embedded"] is True and
    its PNG is actually found at the expected old location — see
    png_writer.build_dest_path), the embedded PNG is moved right along
    with the raw file, into the equivalent images/ location under the
    new bucket. This SHADOWS the raw move rather than leaving the PNG
    stranded at its old path (which would otherwise need a separate
    unlinked-file cleanup later) or discarding it by resetting
    "embedded" to False and losing track of it. Review-bucket moves
    (furry/bondage/horror/solomale/nocharacter/noprompt/orphaned) stay
    excluded from portfolio-explorer's index the same way the raw file
    is: by living under a flat, non-dated folder rather than a dated
    run, not by discarding the embed.

    If "embedded" is True but no PNG is found where expected, nothing
    is moved and a note is printed — could mean it was moved/deleted
    outside this tool; "embedded" is left as-is since forcing a
    re-embed on ambiguous state isn't this function's call to make.

    Returns {"moved": n, "unchanged": n, "skipped": n}.
    """
    from civitai_comfy_bridge.png_writer import build_dest_path

    manifest = load_manifest(data_root)
    moved = unchanged = skipped = 0

    for row in summary_rows:
        image_id = str(row.get("imageId") or "").strip()
        target_bucket = str(row.get("final_bucket") or "").strip()
        if not image_id or not target_bucket:
            skipped += 1
            continue

        entry = manifest.get(image_id)
        if entry is None:
            print(f"SKIP {image_id}: not in this data_root's manifest", flush=True)
            skipped += 1
            continue

        current_path = data_root / entry["raw_path"]
        if not current_path.exists():
            print(f"SKIP {image_id}: raw file missing at {current_path}", flush=True)
            skipped += 1
            continue

        current_bucket = _bucket_from_raw_path(entry.get("raw_path", ""))
        if current_bucket == target_bucket:
            if entry.get("filter") != target_bucket:
                entry["filter"] = target_bucket  # was stale, now in sync with reality
            unchanged += 1
            continue

        old_run_dir_name = Path(entry["raw_path"]).parts[0]

        if target_bucket in _REVIEW_BUCKETS:
            new_dest = data_root / target_bucket / current_path.name
            new_run_dir_name = target_bucket
        else:
            created_date = _creation_date(entry)
            new_dest = data_root / created_date / "raw" / target_bucket / current_path.name
            new_run_dir_name = created_date

        new_dest.parent.mkdir(parents=True, exist_ok=True)
        current_path.rename(new_dest)

        entry["raw_path"] = str(new_dest.relative_to(data_root))
        entry["filter"] = target_bucket

        if entry.get("embedded") and new_run_dir_name != old_run_dir_name:
            nsfw_level = entry.get("nsfwLevel")
            expected_old = build_dest_path(data_root, old_run_dir_name, entry["modelId"], image_id, nsfw_level=nsfw_level)
            old_embed_path = _locate_embed(data_root, entry["modelId"], image_id, expected_old, exclude=new_dest)
            if old_embed_path is not None:
                new_embed_path = build_dest_path(data_root, new_run_dir_name, entry["modelId"], image_id, nsfw_level=nsfw_level)
                new_embed_path.parent.mkdir(parents=True, exist_ok=True)
                old_embed_path.rename(new_embed_path)
                print(f"  shadowed embed: {old_embed_path.relative_to(data_root)} -> {new_embed_path.relative_to(data_root)}", flush=True)
            else:
                print(f"  note: {image_id} marked embedded but no PNG found anywhere under images/ "
                      f"(checked {expected_old.relative_to(data_root)} plus a full search by filename), "
                      f"nothing to shadow", flush=True)

        print(f"MOVED {image_id}: {current_bucket} -> {target_bucket}", flush=True)
        moved += 1

    save_manifest(data_root, manifest)
    return {"moved": moved, "unchanged": unchanged, "skipped": skipped}


def find_unlinked_raw_files(data_root: Path) -> list[Path]:
    """Walk data_root for files that exist on disk but aren't
    referenced by ANY manifest entry's raw_path — true orphans, as
    opposed to orphaned_summary_rows' manifest entries (which still
    have modelId/createdAt, just no record for THIS run's --input).
    These have lost their tracking entirely: manifest.json was
    overwritten/corrupted, a file got moved by hand outside this
    tool, etc. — there's no imageId left to even look one up by.

    Only scans raw_path candidates: MUST skip every data_root/*/images/
    folder (png_writer.build_dest_path's embedded-PNG output — see
    restore_misplaced_embeds for the fallout when this didn't skip
    them). The manifest never records embedded-PNG paths, only
    raw_path, so without this exclusion every already-embedded PNG
    looks "unlinked" and isn't one.

    Also skips manifest.json itself and anything already sitting in
    data_root/orphaned/ (so re-scans don't keep re-finding what a
    previous move_unlinked_raw_files call already relocated there).
    """
    manifest = load_manifest(data_root)
    known_paths = {(data_root / entry["raw_path"]).resolve() for entry in manifest.values()}
    orphaned_dir = (data_root / "orphaned").resolve()

    unlinked = []
    for path in data_root.rglob("*"):
        if path.is_dir():
            continue
        if path.name == MANIFEST_FILE and path.parent == data_root:
            continue
        resolved = path.resolve()
        if resolved == orphaned_dir or orphaned_dir in resolved.parents:
            continue
        if "images" in path.relative_to(data_root).parts:
            continue
        if resolved not in known_paths:
            unlinked.append(path)
    return unlinked


def restore_misplaced_embeds(data_root: Path) -> dict:
    """One-off remediation for the find_unlinked_raw_files bug that
    shipped before the "images" exclusion above: it treated every
    already-embedded PNG as unlinked (manifest never tracked their
    path) and moved them into data_root/orphaned/ alongside genuinely
    unlinked files.

    For every file directly in data_root/orphaned/ whose name matches
    png_writer's "{model_id}_{image_id}.ext" pattern AND whose
    image_id IS in the manifest: that file wasn't actually unlinked —
    it was a legitimate embedded PNG swept up by the bug. Recomputes
    where png_writer.build_dest_path would have put it (using the
    manifest entry's raw_path to recover which dated run it belongs to,
    and its nsfwLevel for the bucket) and moves it back there.

    Files in orphaned/ whose image_id ISN'T in the manifest are left
    alone — those are genuinely unlinked and belong there. Also left
    alone: a file that IS the manifest's current raw_path (i.e. the
    real raw download legitimately living in orphaned/ or another
    review bucket after a bucket move) — without this check, that raw
    file would get wrongly renamed into a "*.png"-named destination
    under images/, corrupting the raw/embed distinction and leaving
    raw_path pointing at a now-empty location.

    Won't overwrite an existing file at the restored destination
    (logged as a conflict instead of silently clobbering — shouldn't
    normally happen since embedded stays True for these entries and
    embed_pending skips anything already embedded).

    Returns {"restored": n, "left_alone": n, "conflicts": n}.
    """
    from civitai_comfy_bridge.png_writer import build_dest_path

    manifest = load_manifest(data_root)
    orphaned_dir = data_root / "orphaned"
    restored = left_alone = conflicts = 0

    if not orphaned_dir.is_dir():
        return {"restored": 0, "left_alone": 0, "conflicts": 0}

    for path in sorted(orphaned_dir.iterdir()):
        if path.is_dir():
            continue
        if "_" not in path.stem:
            left_alone += 1
            continue

        model_id_str, image_id = path.stem.split("_", 1)
        entry = manifest.get(image_id)
        if entry is None:
            left_alone += 1
            continue

        raw_path_str = entry.get("raw_path", "")
        if raw_path_str and (data_root / raw_path_str).resolve() == path.resolve():
            # this file IS the currently-tracked raw download, not a
            # stray embed — leave it exactly where it is
            left_alone += 1
            continue

        raw_path = entry.get("raw_path", "")
        if not raw_path:
            left_alone += 1
            continue
        run_dir_name = Path(raw_path).parts[0]
        model_id = entry.get("modelId") or model_id_str

        dest = build_dest_path(data_root, run_dir_name, model_id, image_id, nsfw_level=entry.get("nsfwLevel"))
        if dest.exists():
            print(f"CONFLICT: {path.relative_to(data_root)} -> {dest.relative_to(data_root)} already exists, left in place", flush=True)
            conflicts += 1
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        path.rename(dest)
        print(f"RESTORED {path.relative_to(data_root)} -> {dest.relative_to(data_root)}", flush=True)
        restored += 1

    return {"restored": restored, "left_alone": left_alone, "conflicts": conflicts}


def find_manifest_drift(data_root: Path) -> list[dict]:
    """Diagnostic for manifest entries whose raw_path doesn't exist on
    disk — i.e. exactly the entries apply_summary/embed_pending would
    print as "raw file missing". For each one, searches data_root for
    any file with the same basename outside images/ folders, so you
    can tell drift (file's still there, just at a different path —
    fixable, see repair_manifest_raw_paths) apart from genuine loss
    (no matching file anywhere — actually gone).

    Returns one dict per drifted entry:
        {"imageId", "expected_path", "found_at": [...] or []}
    found_at is a list since more than one match means the drift can't
    be auto-repaired unambiguously (repair_manifest_raw_paths skips
    those, logging why).
    """
    manifest = load_manifest(data_root)
    drifted = []

    for image_id, entry in manifest.items():
        raw_path_str = entry.get("raw_path", "")
        if not raw_path_str:
            continue
        expected = data_root / raw_path_str
        if expected.exists():
            continue

        found = []
        for match in data_root.rglob(expected.name):
            if "images" in match.relative_to(data_root).parts:
                continue
            found.append(match)

        drifted.append({"imageId": image_id, "expected_path": expected, "found_at": found})

    return drifted


def repair_manifest_raw_paths(data_root: Path) -> dict:
    """Auto-fix find_manifest_drift's unambiguous cases: a manifest
    entry whose raw_path doesn't exist, but exactly ONE file with that
    same basename was found elsewhere in data_root — updates raw_path
    to point at wherever it actually is. Doesn't move any files, only
    corrects the manifest's record of where they are.

    Entries with zero matches (genuinely gone) or more than one match
    (ambiguous — can't tell which is the real one) are left alone and
    printed for manual investigation.

    Returns {"repaired": n, "not_found": n, "ambiguous": n}.
    """
    manifest = load_manifest(data_root)
    drifted = find_manifest_drift(data_root)
    repaired = not_found = ambiguous = 0

    for d in drifted:
        image_id = d["imageId"]
        found = d["found_at"]
        if len(found) == 0:
            print(f"NOT FOUND {image_id}: expected {d['expected_path'].relative_to(data_root)}, "
                  f"no file with that name anywhere else in data_root", flush=True)
            not_found += 1
        elif len(found) > 1:
            candidates = ", ".join(str(p.relative_to(data_root)) for p in found)
            print(f"AMBIGUOUS {image_id}: expected {d['expected_path'].relative_to(data_root)}, "
                  f"multiple candidates found ({candidates}), left alone", flush=True)
            ambiguous += 1
        else:
            new_raw_path = str(found[0].relative_to(data_root))
            print(f"REPAIRED {image_id}: raw_path {manifest[image_id]['raw_path']} -> {new_raw_path}", flush=True)
            manifest[image_id]["raw_path"] = new_raw_path
            repaired += 1

    save_manifest(data_root, manifest)
    return {"repaired": repaired, "not_found": not_found, "ambiguous": ambiguous}


def move_unlinked_raw_files(data_root: Path) -> int:
    """Move every file found by find_unlinked_raw_files into a flat
    data_root/orphaned/ folder. There's no manifest entry to update —
    these files were never tracked by one — so this just physically
    relocates them so they're easy to find and deal with by hand,
    rather than left scattered wherever they turned up.

    On a filename clash (two different dated folders producing the
    same "{modelId}_{imageId}.ext", or an untracked file that just
    happens to share a name), disambiguates with a numeric suffix
    rather than silently overwriting one file with another.

    Returns the number of files moved.
    """
    unlinked = find_unlinked_raw_files(data_root)
    dest_dir = data_root / "orphaned"
    dest_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for path in unlinked:
        dest = dest_dir / path.name
        n = 1
        while dest.exists():
            dest = dest_dir / f"{path.stem}_{n}{path.suffix}"
            n += 1
        path.rename(dest)
        print(f"MOVED (unlinked) {path.relative_to(data_root)} -> {dest.relative_to(data_root)}", flush=True)
        moved += 1
    return moved


def write_diff_report_html(summary_rows: list[dict], data_root: Path, out_path: Path, thumbnail_size: int = 256) -> int:
    """Build a self-contained HTML report (thumbnails embedded as base64
    data URIs, no external files/server needed — just open it in a
    browser) showing only the rows whose final_bucket differs from
    what's currently in the manifest.

    This exists because a diff of raw imageId filenames is useless to
    a human — you can't tell what "soft -> nocharacter" means for
    136746092.jpeg without opening it. Showing the actual thumbnail
    next to current vs proposed bucket makes it something you can
    actually eyeball and hand-correct before running --apply-summary.

    Silently skips rows whose raw file is missing or unreadable as an
    image (thumbnail failure), noting the count at the bottom of the
    report rather than failing the whole build over one bad file.

    Returns the number of changed rows included in the report.
    """
    import base64
    import io
    import html as _html
    from PIL import Image, UnidentifiedImageError

    manifest = load_manifest(data_root)
    changed = []
    thumb_failures = 0

    for row in summary_rows:
        image_id = str(row.get("imageId") or "").strip()
        target_bucket = str(row.get("final_bucket") or "").strip()
        entry = manifest.get(image_id)
        if entry is None or not target_bucket:
            continue

        current_bucket = _bucket_from_raw_path(entry.get("raw_path", ""))
        if current_bucket == target_bucket:
            continue

        raw_path = data_root / entry["raw_path"]
        data_uri = ""
        try:
            with Image.open(raw_path) as im:
                im.thumbnail((thumbnail_size, thumbnail_size))
                buf = io.BytesIO()
                im.convert("RGB").save(buf, format="JPEG", quality=80)
                data_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        except (OSError, UnidentifiedImageError):
            thumb_failures += 1

        changed.append({
            "image_id": image_id,
            "model_name": row.get("modelName") or entry.get("modelId") or "",
            "current_bucket": current_bucket or "(none)",
            "target_bucket": target_bucket,
            "prompt": (row.get("prompt_used") or "")[:300],
            "data_uri": data_uri,
        })

    cards = []
    for c in changed:
        img_html = (f'<img src="{c["data_uri"]}" alt="{c["image_id"]}">'
                    if c["data_uri"] else '<div class="broken">no preview</div>')
        cards.append(f'''
        <div class="card">
          {img_html}
          <div class="meta">
            <div class="model">{_html.escape(str(c["model_name"]))}</div>
            <div class="id">id {_html.escape(c["image_id"])}</div>
            <div class="bucket-change">
              <span class="from">{_html.escape(c["current_bucket"])}</span>
              &rarr;
              <span class="to">{_html.escape(c["target_bucket"])}</span>
            </div>
            <div class="prompt">{_html.escape(c["prompt"])}</div>
          </div>
        </div>''')

    html_doc = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Filter diff report</title>
<style>
  body {{ font-family: -apple-system, sans-serif; background: #1a1a1a; color: #eee; margin: 2rem; }}
  h1 {{ font-size: 1.2rem; color: #aaa; font-weight: normal; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 1rem; }}
  .card {{ background: #262626; border-radius: 8px; overflow: hidden; }}
  .card img {{ width: 100%; display: block; }}
  .broken {{ width: 100%; height: {thumbnail_size}px; display: flex; align-items: center;
             justify-content: center; color: #666; background: #333; }}
  .meta {{ padding: 0.6rem 0.8rem; }}
  .model {{ font-weight: 600; font-size: 0.9rem; }}
  .id {{ color: #888; font-size: 0.75rem; margin-bottom: 0.3rem; }}
  .bucket-change {{ font-size: 0.85rem; margin-bottom: 0.4rem; }}
  .from {{ color: #e08; }}
  .to {{ color: #4c8; font-weight: 600; }}
  .prompt {{ font-size: 0.75rem; color: #999; line-height: 1.3; max-height: 4.5em; overflow: hidden; }}
  .empty {{ color: #888; }}
</style></head>
<body>
  <h1>{len(changed)} image(s) changing bucket{f" &middot; {thumb_failures} preview(s) failed to load" if thumb_failures else ""}</h1>
  <div class="grid">
    {"".join(cards) if cards else '<p class="empty">No differences between this summary and the current manifest.</p>'}
  </div>
</body></html>'''

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    return len(changed)