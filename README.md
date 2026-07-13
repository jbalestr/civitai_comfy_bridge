# civitai-comfy-bridge

Takes civitai_fetcher's output JSON, downloads the images, and writes them
back out as PNGs carrying a synthetic ComfyUI "prompt" metadata chunk —
so portfolio-explorer's existing `scanner.py` / `metadata.py` / `build_index.py`
can index them completely unmodified, as if they were local ComfyUI outputs.

## Why

`metadata.py` in portfolio-explorer parses a ComfyUI PNG's embedded "prompt"
tEXt chunk (a resolved node graph: CheckpointLoaderSimple, KSampler,
CLIPTextEncode, ...). civitai's API gives us the same underlying facts
(checkpoint/model name, sampler, seed, prompt text) but as flat JSON, not a
node graph. This project's only job is translating flat civitai metadata
into a minimal, fake — but structurally valid — node graph, embedding it,
and getting the file safely onto disk without duplicating images across
repeat runs.

Nothing downstream needs to know these PNGs didn't come out of ComfyUI.

## Pipeline

```
civitai_output.json (from jbalestr/civitai_fetcher)
        |
        v
  downloader.py    -- downloads imageUrl, dedups via manifest.json,
   |                  writes into data/<run-date>/images/
   v
graph_builder.py   -- civitai meta dict -> fake ComfyUI node graph dict
   |
   v
 png_writer.py     -- embeds graph as "prompt" tEXt chunk on the PNG
   |
   v
data/<run-date>/images/*.png   -- ready for portfolio-explorer's
                                   build_index.py, unmodified
```

## Status

Scaffold only — stubs to be filled in. See docstrings in each module for
the intended contract. Nothing has been run against the live API yet;
treat all HTTP/IO paths as unverified until tested against real
`civitai_output.json` output.

## Usage (once implemented)

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
    ├── cli.py             entrypoint: wires the three stages together
    ├── downloader.py      civitai_output.json -> downloaded PNGs, deduped
    ├── graph_builder.py   civitai meta dict -> fake ComfyUI node graph
    └── png_writer.py      embeds the graph as a PNG "prompt" tEXt chunk
```

## Open questions / not yet decided

- Content-hash dedup (sha256 of image bytes) alongside imageId, in case
  civitai ever re-serves one image under two ids. Not needed for v1.
- Whether `negativePrompt` is reliably present in civitai's `meta` object
  across all models, or needs a fallback (graph_builder.py should not
  assume it's there — see its docstring).
- NSFW split (civitai.com vs civitai.red) noted in civitai_fetcher's
  README — shouldn't affect this project (it only follows `imageUrl` as
  given) but worth a real-run sanity check.
