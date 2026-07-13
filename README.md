# civitai-comfy-bridge

Takes civitai_fetcher's output JSON, downloads the images, and writes them
back out as PNGs carrying a synthetic ComfyUI "prompt" metadata chunk —
so portfolio-explorer's existing `scanner.py` / `metadata.py` / `build_index.py`
can index them completely unmodified, as if they were local ComfyUI outputs.

## Why

`metadata.py` in portfolio-explorer parses a ComfyUI PNG's embedded "prompt"
tEXt chunk (a resolved node graph: CheckpointLoaderSimple, KSampler,
CLIPTextEncode, ...). civitai's API gives us the same underlying facts
(checkpoint/model name, sampler, seed, prompt text, LoRAs) but as flat
JSON, not a node graph. This project's only job is translating flat
civitai metadata into a minimal, fake — but structurally valid — node
graph, embedding it, and getting the file safely onto disk without
duplicating images across repeat runs.

Nothing downstream needs to know these PNGs didn't come out of ComfyUI —
and if a raw file already carries genuine ComfyUI metadata (see
"Real-metadata passthrough" below), that real data is used instead of
being overwritten.

## Pipeline

```
civitai_output.json (from jbalestr/civitai_fetcher)
        |
        v
  downloader.py    -- downloads imageUrl, dedups via manifest.json,
   |                  skips video posts, writes raw bytes into
   |                  data/<run-date>/raw/ (untouched, never rewritten)
   v
graph_builder.py   -- civitai meta dict -> fake ComfyUI node graph dict
   |                  (checkpoint, prompts, sampler/seed/cfg, LoRAs)
   v
 png_writer.py     -- embeds graph as "prompt" tEXt chunk on a real PNG,
   |                  passes through real metadata untouched if the raw
   |                  bytes already had it, writes into
   |                  data/<run-date>/images/
   v
data/<run-date>/images/*.png   -- ready for portfolio-explorer's
                                   build_index.py, unmodified
```

`cli.py` orchestrates both stages and is resumable at each: `downloader.py`
tracks completed downloads in `manifest.json` by imageId, and each
manifest entry's `"embedded": false/true` flag tracks the second stage
independently — so raw files are only ever downloaded once, and can be
re-embedded later (e.g. after a `graph_builder.py` change) without
re-fetching anything.

## Status

Fully implemented and tested against a real 100-record
`civitai_output.json` pull (10 distinct models, mixed image/video posts,
~1/3 with LoRAs, ~1/2 with negative prompts) — round-tripped through
portfolio-explorer's actual `metadata.py` parser, not just structurally
validated. See each module's docstring for the specific contract it
honours.

## Usage

```
uv sync
uv run python -m civitai_comfy_bridge.cli \
    --input civitai_output.json \
    --data-root ./data
```

Designed to be run daily/weekly against fresh `civitai_output.json` pulls —
repeat runs against the same `--data-root` will not re-download or
duplicate images already present in `manifest.json`.

## Project files

```
civitai-comfy-bridge/
├── README.md
├── pyproject.toml
└── civitai_comfy_bridge/
    ├── __init__.py
    ├── cli.py             entrypoint: wires download + embed stages, resumable
    ├── downloader.py      civitai_output.json -> raw/ files, deduped, video-filtered
    ├── graph_builder.py   civitai meta dict -> fake ComfyUI node graph + LoRAs
    └── png_writer.py      embeds the graph as a PNG "prompt" tEXt chunk,
                            or passes through real metadata if already present
```

## Design decisions worth knowing

- **Checkpoint identity uses `modelName`, not `meta.Model`.** The latter is
  a build/quantisation variant (e.g. `Kreamania_v3a_bf16` vs `kreamania_v1`
  under the same model) — inconsistent enough (26% missing, same model
  spanning several variant strings in a real sample) that using it as the
  primary comparison key would fragment cross-model clustering. It's kept
  as supplementary data (`build_extra_metadata()`, embedded as a separate
  `"civitai_extra"` PNG chunk) instead.
- **Raw downloads are permanent and untouched.** `downloader.py` writes to
  `raw/`, `png_writer.py` writes the final embedded PNG to a separate
  `images/` folder — so the embedding logic can change and be re-run later
  without ever re-downloading.
- **Videos are filtered out before download**, not after — civitai serves
  video posts (mp4/webm/mov) through the same `imageUrl` field, often
  10-60x the size of an image, and they're not useful for image-similarity
  clustering.
- **Real-metadata passthrough.** If raw bytes already carry a genuine
  ComfyUI `"prompt"` chunk (confirmed against a real example — extracts
  correctly including LoRAs), it's preserved as-is rather than overwritten
  by the fabricated graph. Real data is always better than our
  reconstruction. In the normal civitai_fetcher flow this rarely triggers,
  since civitai's `imageUrl` is typically a re-encoded JPEG with no
  metadata at all — but it matters for raw files sourced another way.

## Open questions / not yet decided

- Content-hash dedup (sha256 of image bytes) alongside imageId, in case
  civitai ever re-serves one image under two ids. Not needed for v1.
- NSFW split (civitai.com vs civitai.red) noted in civitai_fetcher's
  README — shouldn't affect this project (it only follows `imageUrl` as
  given) but worth a real-run sanity check.

## Roadmap / considered and deferred

- **Runnable ComfyUI workflows (not just parseable metadata).** Rather
  than a graph that only needs to satisfy `metadata.py`'s parser, build a
  fully-linked graph (real `model`/`clip`/`latent_image` links,
  `VAEDecode`, `SaveImage`, and a `"workflow"` UI chunk) that could
  actually be opened and re-run in ComfyUI.

  The blocker: a `ckpt_name` string won't match any file in someone's
  local `models/checkpoints/`. The fix would be using
  [`civitai/civitai_comfy_nodes`](https://github.com/civitai/civitai_comfy_nodes)'
  AIR-based loaders (reference by Civitai model/version id, fetched on
  demand) instead of the standard `CheckpointLoaderSimple`/`LoraLoader`
  nodes — but that only works for someone who has that node pack
  installed, and coverage is uneven: the checkpoint's `modelVersionId` is
  present on 100% of a real sample, but the *LoRA-specific* version id
  (needed for the same AIR trick on LoRAs) was only present on 6/100
  records — the rest only had a plain hash+name with no resolvable
  version id.

  Also a genuinely separate tool from this one — different node types,
  a real (not just-enough) execution graph, a `"workflow"` chunk on top
  of `"prompt"`, and a hard dependency on the person's ComfyUI setup —
  not an extension of `civitai_comfy_bridge`'s job of feeding
  portfolio-explorer's indexer. Deferred rather than built, given the
  LoRA coverage gap; worth revisiting if that data improves or if
  checkpoint-only reproduction turns out to be enough on its own.