"""downloader.py — downloads images referenced in civitai_fetcher's output
JSON into a dated folder under --data-root, skipping anything already
present in the shared manifest.json so repeat daily/weekly runs never
duplicate a file across folders.

Dedup key: civitai's imageId (globally unique, per civitai_fetcher's
README), not createdAt — createdAt is kept in the manifest as metadata
for later filtering ("only images from last week"), not as the identity
key. See README.md "Open questions" for the content-hash fallback we're
deliberately not building yet.

manifest.json lives at the data-root, not inside a dated folder — it's
the cross-run source of truth. Shape:

{
  "<imageId>": {
    "raw_path": "2026-07-13/raw/123_456.jpeg",   relative to data-root
    "modelId": 123,
    "createdAt": "...",                          from civitai, as-is
    "downloaded_at": "2026-07-13",                our own run date
    "embedded": false                             set true by cli.py once
                                                    png_writer.py has produced
                                                    the final PNG (see below)
  },
  ...
}

Filename convention: "{modelId}_{imageId}.<ext>" — imageId alone is a
sufficient dedup key, but modelId is included in the filename for
human-readability and to keep the on-disk name self-describing without
needing to open the manifest. Extension is taken from the source URL
(civitai serves jpeg/png/webp — not necessarily PNG) rather than
assumed, since this is the raw, unmodified download, not the final
embedded PNG.

Raw vs final split: this module ONLY ever writes into "<run>/raw/" and
never touches those files again once written — they're a permanent
untouched cache of exactly what civitai served. The separate, final
PNG-with-embedded-metadata that portfolio-explorer actually indexes
lives in "<run>/images/", produced by png_writer.py from these raw
bytes, orchestrated by cli.py. Keeping raw and final in separate
folders means png_writer.py / graph_builder.py can change or be re-run
later without ever needing to re-download anything.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import httpx

MANIFEST_FILE = "manifest.json"

# civitai posts can be video (mp4/webm/mov), not just images — same
# imageUrl field either way, no separate type field to check. Videos
# aren't useful for image-similarity/region-embedding and are often
# 10-60x the size of a jpeg, so they're filtered out before download
# rather than downloaded and discarded later.
_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov"}

# Fallback when a civitai imageUrl has no recognisable suffix (seen
# rarely — most URLs end in .jpeg/.png/.webp) — .bin rather than
# guessing wrong, since png_writer.py decodes via PIL regardless of
# what's on disk, not by trusting this extension.
_DEFAULT_EXTENSION = ".bin"


def _extension_from_url(url: str) -> str:
    """Best-effort file extension from a civitai imageUrl's path
    component (ignoring query string). Falls back to _DEFAULT_EXTENSION
    rather than raising — an unrecognised suffix shouldn't abort a
    download, since it's only used for the raw cache's filename, not
    for decoding.
    """
    suffix = Path(urlparse(url).path).suffix
    return suffix if suffix else _DEFAULT_EXTENSION


def load_manifest(data_root: Path) -> dict:
    """Load the shared manifest, or an empty dict if this is the first
    run against this data_root. Not raising on a missing file is
    deliberate — first-run-ever is an expected, not exceptional, case.
    """
    path = data_root / MANIFEST_FILE
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            # Corrupt/truncated manifest (e.g. killed mid-write, though
            # save_manifest's atomic replace should make that rare) —
            # treat as first-run rather than crash. Worst case this
            # re-downloads images already on disk; it never loses them.
            return {}
    return {}


def save_manifest(data_root: Path, manifest: dict) -> None:
    """Write the manifest atomically (write to .tmp, then replace) so a
    kill mid-write can't corrupt it — same pattern as build_index.py's
    own progress file in portfolio-explorer.
    """
    data_root.mkdir(parents=True, exist_ok=True)
    tmp_path = data_root / (MANIFEST_FILE + ".tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2))
    tmp_path.replace(data_root / MANIFEST_FILE)


def run_download(civitai_records: list[dict], data_root: Path) -> dict:
    """Download every record's imageUrl not already in the manifest,
    into data_root/<today>/raw/, updating + periodically flushing the
    manifest as it goes (same "don't lose completed work to a late
    crash" reasoning as build_index.py's batch flushing).

    civitai_records: the flat per-image dicts civitai_fetcher.py writes
    to civitai_output.json (imageId, modelId, imageUrl, meta, ...).

    Returns the final manifest dict (also persisted to disk). New
    entries are written with "embedded": False — cli.py flips that to
    True once png_writer.py has produced the corresponding final PNG,
    so a later run knows which raw files still need embedding without
    re-downloading anything.

    A single failed download (bad URL, timeout, 404) must not abort the
    whole run — log and continue, same resilience principle as
    scanner.py / build_index.py's per-item try/except elsewhere in this
    pipeline. Do not add the failed image to the manifest, so the next
    run retries it.
    """
    manifest = load_manifest(data_root)
    run_dir = data_root / date.today().isoformat() / "raw"
    run_dir.mkdir(parents=True, exist_ok=True)

    already = len(manifest)
    is_video = lambda r: _extension_from_url(r["imageUrl"]) in _VIDEO_EXTENSIONS
    videos_skipped = [r for r in civitai_records if is_video(r)]
    pending = [
        r for r in civitai_records
        if str(r["imageId"]) not in manifest and not is_video(r)
    ]
    print(f"{already} already downloaded, {len(videos_skipped)} videos skipped, {len(pending)} pending", flush=True)

    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for i, rec in enumerate(pending):
            image_id = str(rec["imageId"])
            model_id = rec["modelId"]
            ext = _extension_from_url(rec["imageUrl"])
            dest = run_dir / f"{model_id}_{image_id}{ext}"

            try:
                # Stream to disk rather than buffering the whole response
                # in memory via resp.content — civitai's "original=true"
                # URLs can be several MB each, and buffering every one of
                # them in RAM adds up fast across a batch of 100+.
                with client.stream("GET", rec["imageUrl"]) as resp:
                    resp.raise_for_status()
                    total_bytes = 0
                    with open(dest, "wb") as f:
                        for chunk in resp.iter_bytes():
                            f.write(chunk)
                            total_bytes += len(chunk)
            except (httpx.HTTPError, OSError) as e:
                # OSError catches ssl.SSLError and raw socket resets that
                # surface from deeper in httpcore without being wrapped
                # as an httpx.HTTPError — seen in practice as a mid-
                # download connection reset. A network blip on one image
                # must not take down the whole batch.
                dest.unlink(missing_ok=True)  # don't leave a partial file behind
                print(f"  [{i + 1}/{len(pending)}] FAILED {image_id}: {e}", flush=True)
                continue

            print(f"  [{i + 1}/{len(pending)}] {dest.name} ({total_bytes:,} bytes)", flush=True)

            manifest[image_id] = {
                "raw_path": str(dest.relative_to(data_root)),
                "modelId": model_id,
                "createdAt": rec.get("createdAt"),
                "downloaded_at": date.today().isoformat(),
                "embedded": False,
            }

            if (i + 1) % 50 == 0 or (i + 1) == len(pending):
                # Flush periodically, not only at the end — a killed run
                # partway through a large batch shouldn't lose already-
                # completed downloads to a manifest that was never saved.
                save_manifest(data_root, manifest)

    save_manifest(data_root, manifest)
    return manifest