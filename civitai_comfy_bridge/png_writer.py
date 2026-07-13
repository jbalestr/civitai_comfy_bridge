"""png_writer.py — embeds a fake ComfyUI node graph (from graph_builder.py)
into a downloaded image as a PNG "prompt" tEXt chunk, and saves it to
disk under the naming convention downloader.py expects
("{modelId}_{imageId}.png").

portfolio-explorer's metadata.py reads this chunk via
extract_metadata_from_png_text(), which looks specifically for a tEXt
(or iTXt) chunk with keyword "prompt" — not "workflow", and not any
other keyword — containing the JSON-encoded node graph as its value.
That's the one contract this module has to honour exactly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

NodeGraph = dict[str, dict]


def write_png_with_prompt_chunk(
    image_bytes: bytes,
    graph: NodeGraph,
    dest_path: Path,
) -> None:
    """Load image_bytes (as downloaded from civitai's imageUrl — may be
    JPEG or WEBP source-side, not necessarily PNG already), re-encode
    as PNG if needed, embed `graph` as a "prompt" tEXt chunk via
    PIL.PngImagePlugin.PngInfo, and save to dest_path.

    dest_path's parent directory is assumed to already exist
    (downloader.py creates the dated run folder) — this function only
    writes the file, it doesn't create directories.

    Should raise on genuine corruption (unreadable image bytes) rather
    than silently skipping — the caller (cli.py) is responsible for
    catching that and logging + continuing, same per-item resilience
    pattern as the rest of this pipeline.
    """
    raise NotImplementedError


def build_dest_path(data_root: Path, run_dir_name: str, model_id: int, image_id: str) -> Path:
    """Return data_root/run_dir_name/images/{model_id}_{image_id}.png —
    centralised here (rather than left to each caller) so downloader.py
    and any future re-run/repair tooling construct the exact same path
    for the exact same record.
    """
    raise NotImplementedError
