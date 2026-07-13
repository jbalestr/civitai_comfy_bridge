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


def run(input_path: Path, data_root: Path) -> None:
    """Load civitai_output.json, download pending images, embed a fake
    ComfyUI graph into each, save into data_root/<today>/images/.

    Each image should be handled independently (download failure or a
    graph-building/write failure on one record must not abort the
    whole batch) — same per-item resilience principle used throughout
    portfolio-explorer's own pipeline (scanner.py, build_index.py).
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                         help="path to civitai_fetcher's civitai_output.json")
    parser.add_argument("--data-root", required=True, type=Path,
                         help="root folder for dated download runs + manifest.json")
    args = parser.parse_args()

    run(input_path=args.input, data_root=args.data_root)


if __name__ == "__main__":
    main()
