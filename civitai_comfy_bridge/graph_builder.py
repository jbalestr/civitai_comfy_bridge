"""graph_builder.py — maps one civitai_fetcher record's flat metadata
into a synthetic ComfyUI "prompt" node graph. Two things it's built to
satisfy at once:

1. portfolio-explorer's metadata.py (registry-based handlers keyed on
   class_type — see that module's docstring) needs to extract the same
   checkpoint/sampler/seed/prompt/lora fields it would from a real
   ComfyUI PNG. This is the graph's primary job, feeding the
   Qdrant-indexing pipeline.

2. The graph's links (model/clip/latent_image/positive/negative) are
   real and resolvable — not just structurally-walkable for
   metadata.py's parser, but the actual connections a person would see
   if they opened this in ComfyUI, so it's a reasonable starting point
   to manually fix up (swap ckpt_name/lora_name for real local files)
   and re-run. Deliberately NOT a fully turnkey execution graph though
   — see "What this graph is not" below.

    {
      "1": {"class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "<modelName>"},
            "_meta": {"title": "civitai model <modelId> (version <modelVersionId>)"}},
      "lora_0": {"class_type": "LoraLoader",              # 0 or more, only
            "inputs": {"lora_name": ..., "strength_model": ...,   # if resources
                        "strength_clip": ..., "model": [...], "clip": [...]},
            "_meta": {"title": "civitai lora: <name> (<modelVersionId or hash>)"}},
      "2": {"class_type": "CLIPTextEncode",
            "inputs": {"text": "<meta.prompt>", "clip": [...]}},
      "3": {"class_type": "CLIPTextEncode",              # always present, "" if
            "inputs": {"text": "<meta.negativePrompt>",   # civitai gave none —
                        "clip": [...]}},                    # matches civitai's UI
      "4": {"class_type": "KSampler",
            "inputs": {
                "seed": <meta.seed>, "steps": <meta.steps>, "cfg": <meta.cfgScale>,
                "sampler_name": <meta.sampler>, "scheduler": "normal", "denoise": 1.0,
                "model": [...],                            # checkpoint or last lora
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["5", 0],
            }},
      "5": {"class_type": "EmptyLatentImage",
            "inputs": {"width": <width>, "height": <height>, "batch_size": 1}},
      "6": {"class_type": "VAEDecode",
            "inputs": {"samples": ["4", 0], "vae": ["1", 2]}},
      "7": {"class_type": "SaveImage",
            "inputs": {"filename_prefix": "civitai_bridge", "images": ["6", 0]}},
    }

civitai field -> node mapping (see portfolio-explorer/metadata.py's
registered handlers for exactly what each class_type contributes):

    modelName            -> CheckpointLoaderSimple.ckpt_name
    meta.seed             -> KSampler.seed
    meta.sampler           -> KSampler.sampler_name
    meta.cfgScale            -> KSampler.cfg
    meta.prompt                -> CLIPTextEncode (positive) .text
    meta.negativePrompt          -> CLIPTextEncode (negative) .text, "" if absent
    width / height                 -> EmptyLatentImage
    meta.civitaiResources /          -> LoraLoader nodes, one per lora
      meta.resources (type=="lora")     entry found (see _extract_lora_resources)

What this graph is NOT: a turnkey, guaranteed-runnable workflow.
ckpt_name is civitai's modelName, lora_name is civitai's resource
name — neither matches a real local filename, so both need manually
pointing at real files in ComfyUI before this queues successfully.
Each CheckpointLoaderSimple/LoraLoader node's `_meta.title` carries the
civitai reference (model id + version id, or file hash when no version
id is available) needed to go find the right file. That's the whole
point of doing this at all: this project already writes every image
out as a PNG with an embedded graph for metadata.py's benefit, so
making that same graph's links real costs little extra and turns it
into a genuinely useful starting point for "found this in Qdrant, want
to re-run/modify it" — see README "Roadmap" for the fuller AIR-based
alternative that was considered and deferred instead of this.

Negative prompt is always populated (defaulting to "" when civitai's
meta had none), matching civitai's own UI, which now always shows a
negative prompt field. This is a deliberate change from an earlier
version of this function, which omitted the node entirely when
negativePrompt was absent, so metadata.py's extracted result stayed
None rather than "". That distinction is gone now — anything querying
Qdrant for "no negative prompt" needs to check for "" going forward,
not None, once build_index.py has been re-run against PNGs produced by
this version.
"""

from __future__ import annotations

from typing import Any

NodeGraph = dict[str, dict]

CLASS_CHECKPOINT = "CheckpointLoaderSimple"
CLASS_LORA = "LoraLoader"
CLASS_CLIP_TEXT = "CLIPTextEncode"
CLASS_SAMPLER = "KSampler"
CLASS_LATENT = "EmptyLatentImage"
CLASS_VAE_DECODE = "VAEDecode"
CLASS_SAVE_IMAGE = "SaveImage"


def build_extra_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Build the civitai-specific facts that don't fit anywhere in
    metadata.py's ExtractedMetadata (checkpoint/sampler/seed/prompts/
    dimensions only — no generic "extra" slot) but are worth keeping:
    the exact build/quantisation variant actually used (meta.Model,
    e.g. "Kreamania_v3a_bf16"), as distinct from the stable modelName
    ("Kreamania") used as the checkpoint field in the fake node graph.

    Real-sample finding (100 records, 10 distinct modelIds): meta.Model
    is inconsistent enough — 26% missing, and the *same* modelName
    spans several meta.Model values that are build variants, not
    different models (e.g. Kreamania -> kreamania_v1 / kreamania_v2 /
    Kreamania_v3a_bf16; FASCIUM KREA2 -> five fp8/nvfp4 variants) —
    that it would fragment cross-model comparison if used as the
    primary checkpoint identity. That's why it's kept here as
    supplementary data, not promoted into build_fake_comfy_graph()'s
    CheckpointLoaderSimple node.

    This does NOT get embedded via the "prompt" tEXt chunk — png_writer.py
    embeds it as a second, separate chunk (metadata.py only ever reads
    "prompt", so this can't collide with or break that parsing).

    Also carries civitai's own baseModel (added to civitai_fetcher's
    output after this project's node-graph work surfaced the need for
    it — e.g. "Anima", "Illustrious", "Pony", "Flux.1 D", not always a
    familiar legacy name, sometimes an emerging architecture in its own
    right). Confirmed against a real record that this can differ
    meaningfully from what the classic CheckpointLoaderSimple-shaped
    graph in build_fake_comfy_graph() assumes — e.g. "Anima" turned out
    to be a Qwen-based split-loader architecture, not SDXL-family, going
    by that record's own meta.Module1/Module2/ecosystem fields. Kept as
    data only for now — nothing in this project's graph-building logic
    branches on it yet.

    hashes.model and the top-level "Model hash" meta key were confirmed
    identical whenever both are present in the same real sample, so
    hashes.model is preferred with "Model hash" as fallback rather than
    picking one arbitrarily.
    """
    meta = record.get("meta") or {}
    hashes = meta.get("hashes") or {}

    return {
        "model_name": record.get("modelName"),
        "model_id": record.get("modelId"),
        "model_version_id": record.get("modelVersionId"),
        "base_model": record.get("baseModel"),
        "meta_model": meta.get("Model"),
        "meta_model_hash": hashes.get("model") or meta.get("Model hash"),
    }


def _extract_lora_resources(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return [{"name": ..., "weight": ..., "ref": ...}, ...] for every
    LoRA used, from whichever of civitai's two resource-list formats
    this record has (they were never observed co-occurring on the same
    record in a real 100-record sample):

    - meta.civitaiResources: richer format, has type/weight/name/
      modelVersionId (present on ~15% of a real sample; 18 lora
      entries across those). `ref` becomes "modelVersionId:<id>".
    - meta.resources: plainer A1111-style format, has type/name/hash,
      no weight or version id (present on ~67% of a real sample; 22 of
      those entries are type "lora", rest "model"). `ref` becomes
      "hash:<hash>" — still enough to look the file up manually via
      civitai's hash-lookup, just not an AIR-resolvable id.

    Prefer civitaiResources when both are somehow present, since it
    carries an explicit weight rather than an assumed default. Records
    with neither field (~18% of a real sample) yield an empty list —
    not every generation used a LoRA.
    """
    meta = record.get("meta") or {}

    civitai_resources = meta.get("civitaiResources")
    if civitai_resources:
        return [
            {
                "name": r["name"],
                "weight": r.get("weight", 1),
                "ref": f"modelVersionId:{r['modelVersionId']}" if r.get("modelVersionId") else "unknown",
            }
            for r in civitai_resources
            if r.get("type") == "lora" and r.get("name")
        ]

    resources = meta.get("resources")
    if resources:
        return [
            {
                "name": r["name"],
                "weight": 1,
                "ref": f"hash:{r['hash']}" if r.get("hash") else "unknown",
            }
            for r in resources
            if r.get("type") == "lora" and r.get("name")
        ]

    return []


def build_fake_comfy_graph(record: dict[str, Any]) -> NodeGraph:
    """Build a synthetic ComfyUI "prompt" node graph from one
    civitai_fetcher record (one entry from civitai_output.json), with
    real, resolvable links throughout — see module docstring for what
    that does and doesn't get you.

    record is expected to have the shape documented in
    civitai_fetcher's README: modelName, meta.{prompt, sampler, seed,
    cfgScale, ...}, width, height, etc.

    Raises nothing on missing optional fields — a field civitai didn't
    give us should just be absent from the relevant node's inputs
    (metadata.py's handlers already tolerate missing keys via
    dict.get() with fallback), not synthesized as a fake default.
    """
    meta = record.get("meta") or {}

    graph: NodeGraph = {
        "1": {
            "class_type": CLASS_CHECKPOINT,
            "inputs": {"ckpt_name": record.get("modelName")},
            "_meta": {
                "title": f"civitai model {record.get('modelId')} "
                         f"(version {record.get('modelVersionId')}) — "
                         f"pick your local equivalent checkpoint",
            },
        },
    }

    # Chain LoRAs sequentially: each one's model/clip inputs come from
    # the previous node's model/clip outputs (or the checkpoint's, for
    # the first one). LoraLoader nodes don't strictly need to be wired
    # in for metadata.py's _handle_lora_loader() to find them (it reads
    # any node with class_type "LoraLoader" regardless of links) — but
    # real links are the whole point now, so CLIPTextEncode/KSampler
    # downstream correctly pull from the end of this chain, not
    # straight from the checkpoint.
    current_model_link = ["1", 0]
    current_clip_link = ["1", 1]

    for i, lora in enumerate(_extract_lora_resources(record)):
        node_id = f"lora_{i}"
        graph[node_id] = {
            "class_type": CLASS_LORA,
            "inputs": {
                "lora_name": lora["name"],
                "strength_model": lora["weight"],
                "strength_clip": lora["weight"],
                "model": current_model_link,
                "clip": current_clip_link,
            },
            "_meta": {
                "title": f"civitai lora: {lora['name']} ({lora['ref']}) "
                         f"— pick or download the matching file",
            },
        }
        current_model_link = [node_id, 0]
        current_clip_link = [node_id, 1]

    # meta.prompt is absent on a small minority of real records (seen
    # in a live sample: ~3%) — a save node that didn't embed full meta,
    # or a post with generation data stripped. Still emit the node so
    # KSampler always has something to link "positive" to; metadata.py's
    # CLIPTextEncode handler tolerates a None .text just fine, and
    # downstream (payload storage) treating "no prompt" as None rather
    # than a missing node is more consistent to query against.
    graph["2"] = {
        "class_type": CLASS_CLIP_TEXT,
        "inputs": {"text": meta.get("prompt"), "clip": current_clip_link},
    }

    ksampler_inputs: dict[str, Any] = {
        "seed": meta.get("seed"),
        "steps": meta.get("steps", 20),
        "sampler_name": meta.get("sampler"),
        "scheduler": "normal",
        "cfg": meta.get("cfgScale"),
        "denoise": 1.0,
        "model": current_model_link,
        "positive": ["2", 0],
    }

    # civitai's own UI now always shows a negative prompt field (even
    # when empty), so this graph does the same — always emit the node,
    # defaulting to "" rather than omitting it when civitai's meta had
    # no negativePrompt. (Earlier versions of this function omitted the
    # node entirely to keep metadata.py's extracted result as None
    # rather than "" for records with no negative prompt — deliberately
    # dropped now in favour of matching civitai's own standard.)
    graph["3"] = {
        "class_type": CLASS_CLIP_TEXT,
        "inputs": {"text": meta.get("negativePrompt", ""), "clip": current_clip_link},
    }
    ksampler_inputs["negative"] = ["3", 0]

    graph["5"] = {
        "class_type": CLASS_LATENT,
        "inputs": {
            "width": record.get("width"),
            "height": record.get("height"),
            "batch_size": 1,
        },
    }
    ksampler_inputs["latent_image"] = ["5", 0]

    graph["4"] = {"class_type": CLASS_SAMPLER, "inputs": ksampler_inputs}

    graph["6"] = {
        "class_type": CLASS_VAE_DECODE,
        "inputs": {"samples": ["4", 0], "vae": ["1", 2]},
    }

    graph["7"] = {
        "class_type": CLASS_SAVE_IMAGE,
        "inputs": {"filename_prefix": "civitai_bridge", "images": ["6", 0]},
    }

    return graph