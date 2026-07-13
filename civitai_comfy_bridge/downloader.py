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
    "path": "2026-07-13/images/123_456.png",   relative to data-root
    "modelId": 123,
    "createdAt": "...",                          from civitai, as-is
    "downloaded_at": "2026-07-13"                 our own run date
  },
  ...
}

Filename convention: "{modelId}_{imageId}.png" — imageId alone is a
sufficient dedup key, but modelId is included in the filename for
human-readability and to keep the on-disk name self-describing without
needing to open the manifest.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx

MANIFEST_FILE = "manifest.json"


def load_manifest(data_root: Path) -> dict:
    """Load the shared manifest, or an empty dict if this is the first
    run against this data_root. Not raising on a missing file is
    deliberate — first-run-ever is an expected, not exceptional, case.
    """
    path = data_root / MANIFEST_FILE
    raise NotImplementedError


def save_manifest(data_root: Path, manifest: dict) -> None:
    """Write the manifest atomically (write to .tmp, then replace) so a
    kill mid-write can't corrupt it — same pattern as build_index.py's
    own progress file in portfolio-explorer.
    """
    raise NotImplementedError


def run_download(civitai_records: list[dict], data_root: Path) -> dict:
    """Download every record's imageUrl not already in the manifest,
    into data_root/<today>/images/, updating + periodically flushing
    the manifest as it goes (same "don't lose completed work to a late
    crash" reasoning as build_index.py's batch flushing).

    civitai_records: the flat per-image dicts civitai_fetcher.py writes
    to civitai_output.json (imageId, modelId, imageUrl, meta, ...).

    Returns the final manifest dict (also persisted to disk).

    A single failed download (bad URL, timeout, 404) must not abort the
    whole run — log and continue, same resilience principle as
    scanner.py / build_index.py's per-item try/except elsewhere in this
    pipeline. Do not add the failed image to the manifest, so the next
    run retries it.
    """
    raise NotImplementedError
