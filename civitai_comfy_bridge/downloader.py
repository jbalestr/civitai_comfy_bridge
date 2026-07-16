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
import re as _re

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

# Keywords checked against the POSITIVE prompt only.
_FURRY_KEYWORDS = {"furry", "fur", "tail","anthro", "feral", "kemono", "scale", "scales", "scaled", "fursona", "animal"}

# Bondage/restraint context — excludes clothing alone (bodysuits, latex)
# since those appear in superhero/villain content.
_BONDAGE_KEYWORDS = _re.compile(
    r'\b(bondage|posture\.collar|posturecollar|padded\.room|padded\.cell|ballgag|ball\.gag|shackle)\b'
)

# Horror — specific terms only, avoids false positives on action/fantasy.
# Model blocklist handles cases where the prompt is descriptive prose.
_HORROR_KEYWORDS = _re.compile(
    r'\b(sharp\.teeth|pointed\.teeth|monster\.mouth|jagged\.teeth|eldritch|'
    r'gore|guro|body\.horror|cosmic\.horror|sharp\.fangs|creepy)\b'
)

# Model name blocklist for horror — these models produce horror by default
# regardless of prompt keywords.
_HORROR_MODEL_BLOCKLIST = {"weallstarve"}

# Solo male — must have BOTH a male tag AND explicit masculine descriptors.
# Androgynous/femboy characters (Venti, Astolfo etc.) are tagged 1boy but
# lack masculine descriptors so they pass through correctly.
_SOLO_MALE = _re.compile(r'\b(1boy|male focus|gay)\b')
_MASCULINE = _re.compile(r'\b(masculine|gay|muscular|beard|chest hair|bulge|pecs|abs|stubble|facial hair)\b')
_FEMALE_CHAR = _re.compile(r'\b(\d*girl|\d*girls|woman|women|female|lady|ladies|waifu)\b')


def _filter_bucket(record: dict) -> str | None:
    """Return a review bucket name if the record should be filtered,
    or None if it should proceed to normal nsfw routing.

    Filters are checked in priority order — furry first, then content filters.
    All filtered images go to flat review buckets under data_root/.
    """
    prompt = str((record.get("meta") or {}).get("prompt") or "").lower()
    model_name = (record.get("modelName") or "").lower()
    tokens = set(_re.split(r"[,\s:|()\.\[\]]+", prompt))

    if tokens & _FURRY_KEYWORDS:
        return "furry"
    if _BONDAGE_KEYWORDS.search(prompt):
        return "bondage"
    if _HORROR_KEYWORDS.search(prompt) or any(b in model_name for b in _HORROR_MODEL_BLOCKLIST):
        return "horror"
    if _SOLO_MALE.search(prompt) and _MASCULINE.search(prompt) and not _FEMALE_CHAR.search(prompt):
        return "solomale"
    return None


def _is_furry(record: dict) -> bool:
    """Return True if the positive prompt contains a known furry keyword."""
    meta = record.get("meta") or {}
    prompt = str(meta.get("prompt") or "").lower()
    # split on common delimiters so we match whole tokens only
    import re
    tokens = set(re.split(r"[,\s:|()\[\]]+", prompt))
    return bool(tokens & _FURRY_KEYWORDS)


# Positive prompt tokens that confirm a character is present.
# A match here means the image is kept regardless of landscape keywords.
_CHARACTER_TOKENS = _re.compile(
    # numeric prefix variants: 1girl, 2boys, 3women etc.
    r'\b\d*(?:girl|boy|girls|boys|woman|women|man|men)\b' +
    r'|\b(solo|' +
    # gender/age
    r'male|female|lady|gentleman|guy|gal|lad|lass|' +
    # anime/game archetypes
    r'waifu|husbando|mecha|character|portrait|cyborg|human|people|crowd|person|face|' +
    # titles and roles (female + male)
    r'princess|prince|queen|king|goddess|god|warrior|knight|' +
    r'vampire|witch|wizard|mage|ninja|samurai|' +
    r'sister|brother|mother|father|daughter|son|' +
    # Japanese character archetypes
    r'yuki.onna|onmyoji|geisha|kunoichi|shrine.maiden|miko)\b' +
    # Chinese woman/man characters (no word boundary needed)
    r'|[女男]'
)

# Tokens that indicate a characterless environment/scene.
# Only acted on when _CHARACTER_TOKENS is absent.
_LANDSCAPE_TOKENS = _re.compile(
    r'\b(landscape|scenery|architecture|building|interior|cityscape|room|' +
    r'still.life|scenic|nature|background|view|hills|station|environment|object|props)\b'
)


def _has_prompt(record: dict) -> bool:
    """Return True if the record has a non-empty positive prompt."""
    meta = record.get("meta") or {}
    return bool(str(meta.get("prompt") or "").strip())


def _has_character(record: dict) -> bool:
    """Return True if the positive prompt positively confirms a character is
    present. Returns False if prompt is missing or contains no character tokens.
    """
    meta = record.get("meta") or {}
    prompt = str(meta.get("prompt") or "")

    if not prompt.strip():
        return False

    if _CHARACTER_TOKENS.search(prompt):
        return True

    return False


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


def _creation_date(rec: dict) -> str:
    """Extract YYYY-MM-DD from civitai's createdAt ISO timestamp.
    Falls back to today if the field is missing or malformed.
    """
    created_at = rec.get("createdAt") or ""
    try:
        d = created_at[:10]
        if len(d) == 10 and d[4] == "-" and d[7] == "-":
            return d
    except Exception:
        pass
    return date.today().isoformat()


def run_download(civitai_records: list[dict], data_root: Path, source_json: str = "") -> dict:
    """Download every record's imageUrl not already in the manifest,
    into data_root/<creation-date>/raw/<nsfw-subdir>/, updating +
    periodically flushing the manifest as it goes.

    Images are filed under their civitai createdAt date (not today's
    download date) so that the same image fetched in two different runs
    — e.g. a Week run followed by a Month run — lands at the same path
    and is naturally deduped: if the file already exists on disk, the
    download is skipped and the manifest is updated from the existing
    file without hitting the network at all.

    civitai_records: the flat per-image dicts civitai_fetcher.py writes
    to civitai_output.json (imageId, modelId, imageUrl, meta, ...).

    Returns the final manifest dict (also persisted to disk). New
    entries are written with "embedded": False — cli.py flips that to
    True once png_writer.py has produced the corresponding final PNG,
    so a later run knows which raw files still need embedding without
    re-downloading anything.

    A single failed download (bad URL, timeout, 404) must not abort the
    whole run — log and continue. Do not add failed images to the
    manifest so the next run retries them.
    """
    from civitai_comfy_bridge.png_writer import nsfw_subdir
    manifest = load_manifest(data_root)

    already = len(manifest)
    is_video = lambda r: _extension_from_url(r["imageUrl"]) in _VIDEO_EXTENSIONS
    videos_skipped = [r for r in civitai_records if is_video(r)]
    seen_ids: set[str] = set()
    pending = []
    for r in civitai_records:
        iid = str(r["imageId"])
        if iid not in manifest and not is_video(r) and iid not in seen_ids:
            pending.append(r)
            seen_ids.add(iid)
    print(f"{already} already downloaded, {len(videos_skipped)} videos skipped, {len(pending)} pending", flush=True)

    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for i, rec in enumerate(pending):
            image_id = str(rec["imageId"])
            model_id = rec["modelId"]
            ext = _extension_from_url(rec["imageUrl"])
            created_date = _creation_date(rec)

            # Review buckets go flat under data_root — no date folder needed.
            _REVIEW_BUCKETS = {"furry", "bondage", "horror", "solomale", "nocharacter", "noprompt"}

            # --- pre-download content filters ---
            review = _filter_bucket(rec)
            if review:
                subdir = review
                run_dir = data_root / subdir
            else:
                subdir = nsfw_subdir(rec.get("nsfwLevel"))
                run_dir = data_root / created_date / "raw" / subdir

            run_dir.mkdir(parents=True, exist_ok=True)
            dest = run_dir / f"{model_id}_{image_id}{ext}"

            # --- file existence check: same image from a prior run ---
            if dest.exists():
                total_bytes = dest.stat().st_size
                print(f"  [{i + 1}/{len(pending)}] {dest.name} (exists, {total_bytes:,} bytes) [{subdir}]", flush=True)
            else:
                try:
                    with client.stream("GET", rec["imageUrl"]) as resp:
                        resp.raise_for_status()
                        total_bytes = 0
                        with open(dest, "wb") as f:
                            for chunk in resp.iter_bytes():
                                f.write(chunk)
                                total_bytes += len(chunk)
                except (httpx.HTTPError, OSError) as e:
                    dest.unlink(missing_ok=True)
                    print(f"  [{i + 1}/{len(pending)}] FAILED {image_id}: {e}", flush=True)
                    continue

                # --- prompt + character filters: only applied to soft bucket ---
                # Mature and explicit kept as-is — explicit almost always
                # has a subject; poorly-tagged ones aren't worth reviewing.
                if subdir == "soft":
                    if not _has_prompt(rec):
                        post_subdir = "noprompt"
                    elif not _has_character(rec):
                        post_subdir = "nocharacter"
                    else:
                        post_subdir = None

                    if post_subdir:
                        new_dest = data_root / post_subdir / dest.name
                        new_dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.replace(new_dest)
                        dest = new_dest
                        subdir = post_subdir

                print(f"  [{i + 1}/{len(pending)}] {dest.name} ({total_bytes:,} bytes) [{subdir}]", flush=True)

            manifest[image_id] = {
                "raw_path": str(dest.relative_to(data_root)),
                "modelId": model_id,
                "nsfwLevel": rec.get("nsfwLevel"),
                "filter": subdir,
                "createdAt": rec.get("createdAt"),
                "downloaded_at": date.today().isoformat(),
                "source_json": source_json,
                "embedded": False,
            }

            if (i + 1) % 50 == 0 or (i + 1) == len(pending):
                # Flush periodically, not only at the end — a killed run
                # partway through a large batch shouldn't lose already-
                # completed downloads to a manifest that was never saved.
                save_manifest(data_root, manifest)

    save_manifest(data_root, manifest)
    return manifest