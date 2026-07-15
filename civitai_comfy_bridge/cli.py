"""cli.py — entrypoint wiring downloader.py -> graph_builder.py ->
png_writer.py into one command:

    uv run python -m civitai_comfy_bridge.cli \
        --input civitai_output.json \
        --data-root ./data

Reads civitai_fetcher's flat output JSON (one record per image — see
graph_builder.py's docstring for the fields each stage consumes),
downloads whatever isn't already in --data-root's manifest.json, builds
a fake ComfyUI node graph per image, and writes it out as a PNG with
that graph embedded — ready for portfolio-explorer's build_index.py to
index, unmodified, by just pointing it at the resulting dated folder.

Intended to be run daily/weekly against a fresh civitai_output.json
pull. Safe to re-run against the same --data-root: already-downloaded
images (by imageId, per downloader.py's manifest) are skipped, not
re-fetched or duplicated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import downloader, graph_builder, png_writer


def embed_pending(civitai_records: list[dict], data_root: Path) -> dict:
    """Second stage: for every manifest entry with "embedded": False,
    look up its original civitai record (by imageId), build the fake
    ComfyUI graph + extra metadata, and write the final embedded PNG
    into <run>/images/ via png_writer.py. Raw files are only ever read,
    never modified — see downloader.py's module docstring for why.

    civitai_records is matched against the manifest by str(imageId),
    same key downloader.py uses. If a manifest entry's record isn't in
    the current civitai_records list (e.g. this JSON is a later pull
    that dropped an older image), or its raw file is missing on disk,
    that entry is skipped with a message rather than crashing the run
    — same per-item resilience principle as downloader.py.

    Returns the updated manifest (also persisted to disk).
    """
    manifest = downloader.load_manifest(data_root)
    records_by_id = {str(r["imageId"]): r for r in civitai_records}

    pending_ids = [
        image_id for image_id, entry in manifest.items()
        if not entry.get("embedded", False)
    ]
    print(f"{len(pending_ids)} pending embed", flush=True)

    for i, image_id in enumerate(pending_ids):
        entry = manifest[image_id]

        record = records_by_id.get(image_id)
        if record is None:
            print(f"  [{i + 1}/{len(pending_ids)}] SKIP {image_id}: "
                  f"not present in this input JSON", flush=True)
            continue

        raw_path = data_root / entry["raw_path"]
        if not raw_path.exists():
            print(f"  [{i + 1}/{len(pending_ids)}] SKIP {image_id}: "
                  f"raw file missing at {raw_path}", flush=True)
            continue

        # images/ lives alongside raw/ under the same dated run folder
        # the raw file was downloaded into, e.g. "2026-07-13/raw/..."
        # -> "2026-07-13/images/...".
        run_dir_name = Path(entry["raw_path"]).parts[0]
        nsfw_level = entry.get("nsfwLevel") or record.get("nsfwLevel")
        dest = png_writer.build_dest_path(data_root, run_dir_name, entry["modelId"], image_id, nsfw_level=nsfw_level)
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            graph = graph_builder.build_fake_comfy_graph(record)
            extra = graph_builder.build_extra_metadata(record)
            used_real_metadata = png_writer.write_png_with_prompt_chunk(
                raw_path.read_bytes(), graph, dest, extra_metadata=extra,
            )
        except (ValueError, OSError) as e:
            # Corrupt/unreadable raw bytes, or a disk write failure —
            # log and move on, don't lose the rest of the batch to one
            # bad file. entry["embedded"] stays False so a later run
            # retries it (useful if e.g. the disk was full this time).
            print(f"  [{i + 1}/{len(pending_ids)}] FAILED embed {image_id}: {e}", flush=True)
            continue

        entry["embedded"] = True
        tag = "real metadata" if used_real_metadata else "fabricated"
        print(f"  [{i + 1}/{len(pending_ids)}] embedded {dest.name} ({tag})", flush=True)

        if (i + 1) % 50 == 0 or (i + 1) == len(pending_ids):
            downloader.save_manifest(data_root, manifest)

    downloader.save_manifest(data_root, manifest)
    return manifest


def run(input_path: Path, data_root: Path) -> None:
    """Load civitai_output.json, download pending images, embed a fake
    ComfyUI graph into each, save into data_root/<today>/images/.

    Each image should be handled independently (download failure or a
    graph-building/write failure on one record must not abort the
    whole batch) — same per-item resilience principle used throughout
    portfolio-explorer's own pipeline (scanner.py, build_index.py).
    """
    civitai_records = json.loads(input_path.read_text())
    print(f"loaded {len(civitai_records)} records from {input_path}", flush=True)

    downloader.run_download(civitai_records, data_root)
    embed_pending(civitai_records, data_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                         help="path to civitai_fetcher's civitai_output.json")
    parser.add_argument("--data-root", required=True, type=Path,
                         help="root folder for dated download runs + manifest.json")
    parser.add_argument("--embed-only", action="store_true",
                         help="skip download stage, only embed already-downloaded raw files into PNGs")
    args = parser.parse_args()

    civitai_records = json.loads(args.input.read_text())
    print(f"loaded {len(civitai_records)} records from {args.input}", flush=True)

    if not args.embed_only:
        downloader.run_download(civitai_records, args.data_root)
    embed_pending(civitai_records, args.data_root)


if __name__ == "__main__":
    main()