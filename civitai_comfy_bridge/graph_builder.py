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

Known gap (see README "Open questions"): civitai_fetcher's documented
output shape doesn't show a negativePrompt key in its example payload.
Don't assume it's always there — check meta.get("negativePrompt") and
omit node "3" / the KSampler "negative" link entirely when absent,
rather than emitting an empty-string prompt node.
"""

from __future__ import annotations

from typing import Any

NodeGraph = dict[str, dict]


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
    raise NotImplementedError
