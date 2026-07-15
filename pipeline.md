# Civitai → Qdrant → Portfolio Explorer: Full Pipeline Runbook

End-to-end guide for fetching Civitai images, converting them to indexed PNGs, and browsing them in the 2D similarity map.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Windows + WSL2 (Ubuntu 24.04) | Bridge runs on Windows, everything else in WSL |
| NVIDIA GPU with CUDA 13+ | RTX 4050 confirmed working |
| Docker (in WSL) | For Qdrant |
| `uv` | Package manager for all three projects |

### One-time WSL setup

Ensure CUDA libraries are on the dynamic linker path. Add to `~/.bashrc`:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

Then install cuDNN (needed for insightface GPU inference):

```bash
sudo apt-get install -y libcudnn9-cuda-13
source ~/.bashrc
```

---

## Step 1 — Fetch images from Civitai

**Runs on: Windows**, in `C:\Users\jbale\projects\civitai_fetcher`

```powershell
$env:CIVITAI_API_TOKEN = "your_token_here"   # optional, raises rate limits

uv run python -m civitai_fetcher.images_cli `
    --period Week `
    --top-models 150 `
    --top-reactions-per-model 50
```

Output: `civitai_output_week_<ddMMMyy>_<HHMM>.json` in the project root.

Key flags:
- `--period` — `Day`, `Week` (recommended), `Month`. Week gives the best signal-to-noise.
- `--top-models` — how many activity-ranked models to fetch images for.
- `--top-reactions-per-model` — keeps the top N images per model (prevents one popular model crowding everything else out). Use `--top-reactions 0` to keep all fetched images.

---

## Step 2 — Bridge: download + embed as ComfyUI PNGs

**Runs on: Windows**, in `C:\Users\jbale\projects\civitai_comfy_bridge`

Copy the fetcher output into the bridge's `json/` folder, then run:

```powershell
# Full run (download + embed):
uv run python -m civitai_comfy_bridge.cli `
    --input json\civitai_output_week_14jul26_1028.json `
    --data-root data

# Embed only (skip download, use already-downloaded raw files):
uv run python -m civitai_comfy_bridge.cli `
    --input json\civitai_output_week_14jul26_1028.json `
    --data-root data `
    --embed-only
```

Output: `data\<run-date>\images\*.png` — real PNGs with ComfyUI metadata embedded, ready for indexing.

Notes:
- Safe to interrupt and re-run — `manifest.json` tracks completed downloads by imageId, `embedded` flag tracks the second stage independently.
- Video posts (mp4/webm/mov) are filtered out automatically.
- If a raw file already carried genuine ComfyUI metadata, it's preserved untouched rather than overwritten.

---

## Step 3 — Start Qdrant

**Runs on: WSL**

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Leave this running in its own terminal. If restarting a fresh run, stop and remove the old container first:

```bash
docker stop <container_name> && docker rm <container_name>
# find container name with: docker ps
```

> **Note:** Qdrant's default Docker setup stores data inside the container — wiped on removal. This is intentional for a re-index workflow. If you want persistence across container restarts, mount a volume:
> `docker run -p 6333:6333 -p 6334:6334 -v ~/qdrant_storage:/qdrant/storage qdrant/qdrant`

---

## Step 4 — Index into Qdrant

**Runs on: WSL**, in `~/projects/portfolioExplorer`

Delete any stale progress files if starting fresh:

```bash
rm -f .build_index_progress.json .index_writer_progress.json
```

Test with 10 images first:

```bash
uv run python3 build_index.py \
    /mnt/c/Users/jbale/projects/civitai_comfy_bridge/data/2026-07-14/images \
    --pose-model pose_landmarker.task \
    --limit 10
```

If that looks good (GPU active, points written to Qdrant), run the full batch:

```bash
uv run python3 build_index.py \
    /mnt/c/Users/jbale/projects/civitai_comfy_bridge/data/2026-07-14/images \
    --pose-model pose_landmarker.task
```

Expect ~2 images/sec on GPU (RTX 4050). Safe to interrupt and re-run — progress is saved per image.

Confirm GPU is active by checking the log output includes:
```
Applied providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```
If you only see `CPUExecutionProvider`, check `LD_LIBRARY_PATH` includes both CUDA and cuDNN paths.

---

## Step 5 — Build 2D projections

**Runs on: WSL**, in `~/projects/portfolioExplorer`

Run after `build_index.py` completes — needs vectors in Qdrant first.

```bash
uv run python3 projection_builder.py --collection faces
uv run python3 projection_builder.py --collection hair
uv run python3 projection_builder.py --collection outfits
uv run python3 projection_builder.py --collection backgrounds
```

This writes `map_x`/`map_y` coordinates back into each point's Qdrant payload via UMAP. Re-run any time you add more data — the map updates on browser reload.

---

## Step 6 — Serve and browse

**Runs on: WSL**, in `~/projects/portfolioExplorer`

```bash
uv run python3 server.py \
    --port 8765 \
    --backend qdrant \
    --qdrant-url http://localhost:6333 \
    --image-root /mnt/c/Users/jbale/projects/civitai_comfy_bridge/data/2026-07-14/images
```

Open `http://127.0.0.1:8765` in a browser.

`--image-root` must match exactly what you passed to `build_index.py` — image paths in Qdrant are stored relative to this folder. Without it, the map works but thumbnails 503.

---

## Repeat runs (adding more data)

1. Run `images_cli` again with the same or updated flags → new JSON.
2. Copy new JSON to bridge's `json/` folder.
3. Run bridge `cli` — skips already-downloaded imageIds, only fetches new ones.
4. Run `build_index.py` without deleting progress files — resumes from where it left off.
5. Re-run all four `projection_builder.py` commands to rebuild the 2D map with the new points included.
6. Restart `server.py` (or just reload the browser — the map fetches projections live from Qdrant).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `libcudart.so.13: cannot open shared object file` | Add `/usr/local/cuda/targets/x86_64-linux/lib` to `LD_LIBRARY_PATH` in `~/.bashrc` |
| `libcudnn.so.9: cannot open shared object file` | `sudo apt-get install -y libcudnn9-cuda-13` and add `/usr/lib/x86_64-linux-gnu` to `LD_LIBRARY_PATH` |
| `Applied providers: ['CPUExecutionProvider']` only | GPU libs not found — check both `LD_LIBRARY_PATH` entries are set and `source ~/.bashrc` was run |
| `0 pending / nothing to do` on fresh Qdrant | Stale progress files — `rm .build_index_progress.json .index_writer_progress.json` |
| Thumbnails show "click to seed" placeholder | `--image-root` not set, or pointing at wrong folder |
| Stale points from old run mixed into map | Wipe Qdrant (stop/rm container, restart), delete progress files, re-run from Step 4 |
| `--embed-only: unrecognized argument` | Update `cli.py` `main()` — see civitai_comfy_bridge README |