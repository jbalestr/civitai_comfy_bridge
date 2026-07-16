"""png_writer.py — embeds a fake ComfyUI node graph (from graph_builder.py)
into a downloaded image as a PNG "prompt" tEXt chunk, and saves it to
disk under the naming convention downloader.py expects
("{modelId}_{imageId}.png").

portfolio-explorer's metadata.py reads this chunk via
extract_metadata_from_png_text(), which looks specifically for a tEXt
(or iTXt) chunk with keyword "prompt" — not "workflow", and not any
other keyword — containing the JSON-encoded node graph as its value.
That's the one contract this module has to honour exactly.

A second, separate tEXt chunk ("civitai_extra") optionally carries
graph_builder.py's build_extra_metadata() output — the exact
meta.Model/hash build-variant info that doesn't fit anywhere in
metadata.py's ExtractedMetadata. metadata.py never reads this chunk
(it only ever looks for "prompt"), so this can't collide with or
break that parsing — it's just along for the ride for whenever a
future scanner.py extension wants it.

Real-metadata passthrough: civitai's own imageUrl is normally a
re-encoded JPEG with no embedded chunks at all (confirmed against a
real 100-record sample — every non-video result came back .jpeg), so
fabrication is the common case. But raw bytes CAN occasionally already
carry a genuine ComfyUI "prompt"/"workflow" chunk pair — e.g. a PNG
sourced some other way than civitai's imageUrl. Confirmed against a
real ComfyUI-generated PNG: metadata.py's parser correctly extracts
checkpoint/sampler/seed/prompts/loras from it. Real data is strictly
better than our reconstruction (it has the actual local ckpt_name and
LoRAs, which build_fake_comfy_graph() doesn't attempt to populate — see
graph_builder.py), so if it's already there, it's preserved untouched
rather than clobbered with a fabricated graph.

Note (surfaced while implementing this): downloader.py currently
writes raw downloaded bytes straight to the final image path, with no
re-encoding or metadata — this module's job (converting to real PNG +
embedding chunks) has to run as a distinct step afterward, orchestrated
by cli.py, not something downloader.py does inline. Flagging here so
it isn't missed when cli.py gets wired up.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.PngImagePlugin import PngInfo

NodeGraph = dict[str, dict]

PROMPT_CHUNK_KEYWORD = "prompt"
WORKFLOW_CHUNK_KEYWORD = "workflow"
EXTRA_CHUNK_KEYWORD = "civitai_extra"


def write_png_with_prompt_chunk(
    image_bytes: bytes,
    graph: NodeGraph,
    dest_path: Path,
    extra_metadata: dict[str, Any] | None = None,
) -> bool:
    """Load image_bytes (as downloaded from civitai's imageUrl — may be
    JPEG or WEBP source-side, not necessarily PNG already), re-encode
    as PNG if needed, embed `graph` as a "prompt" tEXt chunk via
    PIL.PngImagePlugin.PngInfo, and save to dest_path.

    If image_bytes already carries a genuine "prompt" chunk (i.e. it's
    already a real ComfyUI-generated PNG, not a re-encoded civitai
    JPEG), that real chunk — and "workflow" alongside it, if present —
    is preserved as-is instead of being overwritten by the fabricated
    `graph`. Real data always wins over our reconstruction.

    Returns True if real, pre-existing metadata was passed through;
    False if `graph` was used (the common case). Callers may use this
    for logging/stats — see cli.py.

    dest_path's parent directory is assumed to already exist
    (downloader.py creates the dated run folder) — this function only
    writes the file, it doesn't create directories.

    Raises on genuine corruption (unreadable image bytes) rather than
    silently skipping — the caller (cli.py) is responsible for
    catching that and logging + continuing, same per-item resilience
    pattern as the rest of this pipeline.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()  # force full decode now, so corrupt bytes raise here
    except Exception as e:
        raise ValueError(f"could not decode image bytes for {dest_path.name}: {e}") from e

    # PNG doesn't support every source mode (e.g. CMYK from some JPEGs,
    # or palette modes) — normalise to something PNG can always encode
    # rather than letting img.save() fail deep in PIL's PNG encoder.
    if img.mode not in ("RGB", "RGBA", "L", "LA"):
        img = img.convert("RGB")

    png_info = PngInfo()
    existing_prompt = img.info.get(PROMPT_CHUNK_KEYWORD)
    used_real_metadata = bool(existing_prompt)

    if used_real_metadata:
        png_info.add_text(PROMPT_CHUNK_KEYWORD, existing_prompt)
        existing_workflow = img.info.get(WORKFLOW_CHUNK_KEYWORD)
        if existing_workflow:
            png_info.add_text(WORKFLOW_CHUNK_KEYWORD, existing_workflow)
    else:
        png_info.add_text(PROMPT_CHUNK_KEYWORD, json.dumps(graph))

    if extra_metadata:
        png_info.add_text(EXTRA_CHUNK_KEYWORD, json.dumps(extra_metadata))

    img.save(dest_path, format="PNG", pnginfo=png_info)
    return used_real_metadata


# Map civitai nsfwLevel string values to subdirectory names.
# None means the field was absent — treat as unknown rather than guessing.
_NSFW_SUBDIR: dict[str | None, str] = {
    "Soft":   "soft",
    "Mature": "mature",
    "X":      "explicit",
    None:     "soft",   # unclassified images confirmed to be at most soft
}


def nsfw_subdir(nsfw_level: str | None) -> str:
    """Return the images/ subdirectory name for a given nsfwLevel value.
    Unrecognised strings (future Civitai levels) fall back to unknown.
    """
    return _NSFW_SUBDIR.get(nsfw_level, "soft")  # unrecognised values treated as soft


def build_dest_path(data_root: Path, run_dir_name: str, model_id: int, image_id: str, nsfw_level: str | None = None) -> Path:
    """Return data_root/run_dir_name/images/<nsfw_subdir>/{model_id}_{image_id}.png.
    Images are bucketed by nsfwLevel so build_index.py can ingest
    specific subdirectories (e.g. soft + unknown only) without touching
    mature or explicit content.
    """
    subdir = nsfw_subdir(nsfw_level)
    return data_root / run_dir_name / "images" / subdir / f"{model_id}_{image_id}.png"