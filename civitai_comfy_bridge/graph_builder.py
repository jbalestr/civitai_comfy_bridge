"""graph_builder.py — maps one civitai_fetcher record's flat metadata
into a minimal, synthetic ComfyUI "prompt" node graph: just enough for
portfolio-explorer's metadata.py (registry-based handlers keyed on
class_type — see that module's docstring) to extract the same
checkpoint/sampler/seed/prompt fields it would from a real ComfyUI PNG.

This is NOT a real execution graph — node ids are made up, and there's
no actual sampling/checkpoint-loading happening. It only needs to be
structurally valid enough for metadata.py's handlers to walk it:

    {
      "1": {"class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "<modelName>"}},
      "2": {"class_type": "CLIPTextEncode",
            "inputs": {"text": "<meta.prompt>"}},
      "3": {"class_type": "CLIPTextEncode",              # only if present
            "inputs": {"text": "<meta.negativePrompt>"}},
      "4": {"class_type": "KSampler",
            "inputs": {
                "seed": <meta.seed>,
                "sampler_name": <meta.sampler>,
                "cfg": <meta.cfgScale>,
                "positive": ["2", 0],
                "negative": ["3", 0],                     # only if node 3 exists
            }},
      "5": {"class_type": "EmptyLatentImage",
            "inputs": {"width": <width>, "height": <height>}},
      "lora_0": {"class_type": "LoraLoader",              # 0 or more, only
            "inputs": {"lora_name": ..., "strength_model": ...,  # if resources
                        "strength_clip": ...}},                   # list them
    }

civitai field -> node mapping (see portfolio-explorer/metadata.py's
registered handlers for exactly what each class_type contributes):

    modelName            -> CheckpointLoaderSimple.ckpt_name
    meta.seed             -> KSampler.seed
    meta.sampler           -> KSampler.sampler_name
    meta.cfgScale            -> KSampler.cfg
    meta.prompt                -> CLIPTextEncode (positive) .text
    meta.negativePrompt          -> CLIPTextEncode (negative) .text, IF PRESENT
    width / height                 -> EmptyLatentImage
    meta.civitaiResources /          -> LoraLoader nodes, one per lora
      meta.resources (type=="lora")     entry found (see _extract_lora_resources)

Known gap (see README "Open questions"): civitai_fetcher's documented
output shape doesn't show a negativePrompt key in its example payload.
Don't assume it's always there — check meta.get("negativePrompt") and
omit node "3" / the KSampler "negative" link entirely when absent,
rather than emitting an empty-string prompt node.
"""

from __future__ import annotations

from typing import Any

NodeGraph = dict[str, dict]


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
        "meta_model": meta.get("Model"),
        "meta_model_hash": hashes.get("model") or meta.get("Model hash"),
    }


def _extract_lora_resources(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return [{"name": ..., "weight": ...}, ...] for every LoRA used,
    from whichever of civitai's two resource-list formats this record
    has (they were never observed co-occurring on the same record in a
    real 100-record sample):

    - meta.civitaiResources: richer format, has type/weight/name/
      modelVersionId (present on ~15% of a real sample; 18 lora
      entries across those).
    - meta.resources: plainer A1111-style format, has type/name/hash,
      no weight field, so weight defaults to 1 (present on ~67% of a
      real sample; 22 of those entries are type "lora", rest "model").

    Prefer civitaiResources when both are somehow present, since it
    carries an explicit weight rather than an assumed default. Records
    with neither field (~18% of a real sample) yield an empty list —
    not every generation used a LoRA.
    """
    meta = record.get("meta") or {}

    civitai_resources = meta.get("civitaiResources")
    if civitai_resources:
        return [
            {"name": r["name"], "weight": r.get("weight", 1)}
            for r in civitai_resources
            if r.get("type") == "lora" and r.get("name")
        ]

    resources = meta.get("resources")
    if resources:
        return [
            {"name": r["name"], "weight": 1}
            for r in resources
            if r.get("type") == "lora" and r.get("name")
        ]

    return []


def build_fake_comfy_graph(record: dict[str, Any]) -> NodeGraph:
    """Build a synthetic ComfyUI "prompt" node graph from one
    civitai_fetcher record (one entry from civitai_output.json).

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
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": record.get("modelName")},
        },
    }

    # meta.prompt is absent on a small minority of real records (seen
    # in a live sample: ~3%) — a save node that didn't embed full meta,
    # or a post with generation data stripped. Still emit the node so
    # KSampler always has something to link "positive" to; metadata.py's
    # CLIPTextEncode handler tolerates a None .text just fine, and
    # downstream (payload storage) treating "no prompt" as None rather
    # than a missing node is more consistent to query against.
    graph["2"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": meta.get("prompt")},
    }

    ksampler_inputs: dict[str, Any] = {
        "seed": meta.get("seed"),
        "sampler_name": meta.get("sampler"),
        "cfg": meta.get("cfgScale"),
        "positive": ["2", 0],
    }

    # negativePrompt is genuinely optional (present on roughly half of
    # a real sample) — per this module's docstring, omit the node and
    # the link entirely rather than emit an empty-string prompt, so
    # "no negative prompt" and "empty negative prompt" aren't conflated.
    negative_prompt = meta.get("negativePrompt")
    if negative_prompt:
        graph["3"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt},
        }
        ksampler_inputs["negative"] = ["3", 0]

    graph["4"] = {"class_type": "KSampler", "inputs": ksampler_inputs}

    graph["5"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": record.get("width"),
            "height": record.get("height"),
        },
    }

    # LoraLoader nodes don't need to be wired into the sampler chain —
    # metadata.py's _handle_lora_loader() reads any node with
    # class_type "LoraLoader" regardless of graph links (confirmed
    # against metadata.py's extract_metadata(), which iterates every
    # node in the dict, not just ones reachable from KSampler). They
    # just need to exist, each with its own node id.
    for i, lora in enumerate(_extract_lora_resources(record)):
        graph[f"lora_{i}"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": lora["name"],
                "strength_model": lora["weight"],
                "strength_clip": lora["weight"],
            },
        }

    return graph