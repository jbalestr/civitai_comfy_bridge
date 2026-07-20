"""
Minimal sketch: image-focused, tag-based include/exclude filter session.
No prompt text, no NLP — just set overlap on structured JSON fields.

Run: uv run uvicorn filter_session:app --reload
"""

import html
import re
import yaml
from collections import Counter, defaultdict
from pathlib import Path
import json

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

# --- config ---
DATA_ROOT = Path("./data")               # manifest.json lives here (written by downloader.py)
JSON_METADATA_DIR = Path("./json")       # batch files, arrays of civitai_fetcher records
THUMB_CACHE_DIR = Path("./.thumb_cache")  # generated once per image, reused after
THUMB_SIZE = 200

# Persisted extracted-prompt summary — imageId -> prompt text, written
# after every load_catalogue() pass so a server RESTART doesn't have
# to re-walk every meta.comfy ComfyUI graph from scratch. Only
# imageIds missing from this file get real extraction done on load;
# everything else is a straight read. Safe to delete any time — it's
# purely a derived cache, never the source of truth (source JSON in
# JSON_METADATA_DIR is never modified, so this file can always be
# fully regenerated from scratch if deleted or corrupted).
PROMPT_SUMMARY_PATH = Path("./prompt_summary_cache.json")

# Resources that show up in most images regardless of subject — aesthetic
# boosters, quality/highres loras, etc. Not useful as category signal.
# Add to this as you spot more in the ranked output below.
RESOURCE_STOPLIST = {
    "high_detailed",
}
STOPLIST_PCT_THRESHOLD = 0.35  # auto-flag (not auto-drop) anything above this

def get_field(data: dict, dot_path: str):
    for part in dot_path.split("."):
        if not isinstance(data, dict) or part not in data:
            return []
        data = data[part]
    return data if isinstance(data, list) else [data] if data else []


_resources_cache: dict[str, list[dict]] = {}


def get_resources(record: dict) -> list[dict]:
    """meta.resources[] as (name, type) pairs — type is one of
    'model' (checkpoint), 'lora', 'used_embeddings', 'vae', etc.
    Lumping all types together mixes checkpoint selection with subject
    loras, which are different signals — let callers filter by type.

    Cached by imageId, same reasoning as effective_prompt_text — this
    gets called repeatedly per record (dominant_lora, any 'resources'-
    field filter condition during predicate matching, resource_frequency,
    resource_cooccurrence), rebuilding the same list from scratch each
    time otherwise. Cleared by /reload-catalogue.
    """
    image_id = record.get("imageId")
    image_id = str(image_id) if image_id is not None else None  # normalize to match records dict keys (str(imageId))
    if image_id is not None and image_id in _resources_cache:
        return _resources_cache[image_id]

    resources = get_field(record, "meta.resources")
    result = [
        {"name": r["name"], "type": r.get("type", "")}
        for r in resources if isinstance(r, dict) and r.get("name")
    ]

    if image_id is not None:
        _resources_cache[image_id] = result
    return result


def load_manifest_jpgs() -> dict[str, Path]:
    """imageId -> absolute jpg path, straight from downloader.py's own
    manifest.json. Skips any entry whose raw_path isn't a .jpg (per
    'only jpg in data subdirectories') rather than guessing at the
    filename convention ourselves.
    """
    manifest_path = DATA_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return {
        image_id: (DATA_ROOT / entry["raw_path"])
        for image_id, entry in manifest.items()
        if entry.get("raw_path", "").lower().endswith((".jpg", ".jpeg"))
    }


_record_source_file: dict[str, Path] = {}


def load_metadata_records() -> dict[str, dict]:
    """imageId -> metadata record, flattened across every batch file
    in JSON_METADATA_DIR (each file is an array of civitai_fetcher
    records, per the civitai_comfy_bridge format).

    Also records which file each imageId came from (_record_source_file)
    — needed so a record's full raw JSON (including its meta.comfy
    workflow graph) can be re-read on demand later, after load_catalogue
    has stripped that field from the in-memory copy to save RAM.
    """
    records: dict[str, dict] = {}
    _record_source_file.clear()
    for path in JSON_METADATA_DIR.glob("*.json"):
        for rec in json.loads(path.read_text()):
            image_id = str(rec["imageId"])
            records[image_id] = rec
            _record_source_file[image_id] = path
    return records


def reload_record_from_disk(image_id: str) -> dict | None:
    """Re-read ONE record's full raw JSON straight from its source
    batch file, bypassing the in-memory catalogue entirely — so it
    still has the original meta.comfy workflow graph even after
    load_catalogue has cleared that field from the cached copy. Only
    the single batch file containing this record is re-parsed, not
    the whole JSON_METADATA_DIR. Returns None if the imageId isn't
    known or its source file can't be found (e.g. deleted since load).
    """
    path = _record_source_file.get(image_id)
    if path is None or not path.exists():
        return None
    for rec in json.loads(path.read_text()):
        if str(rec.get("imageId")) == image_id:
            return rec
    return None


# --- in-memory session state (swap for redis/sqlite if it needs to survive restarts) ---
class SessionState:
    def __init__(self):
        self.include: set[str] = set()
        self.exclude: set[str] = set()

    def unlabeled(self, all_ids: set[str]) -> set[str]:
        return all_ids - self.include - self.exclude


state = SessionState()


_catalogue_cache: dict = {}


def load_prompt_summary() -> dict[str, str]:
    """Read the persisted imageId -> prompt-text summary from disk, if
    it exists. Corrupt/missing file just means starting from an empty
    summary — never fatal, since it's purely a derived cache."""
    if not PROMPT_SUMMARY_PATH.exists():
        return {}
    try:
        return json.loads(PROMPT_SUMMARY_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_prompt_summary(summary: dict[str, str]) -> None:
    """Write the current full imageId -> prompt-text summary to disk.
    Called once per load_catalogue() pass — cheap even at tens of
    thousands of entries, since it's just strings, no node graphs."""
    try:
        PROMPT_SUMMARY_PATH.write_text(json.dumps(summary))
    except OSError:
        pass  # best-effort — a failed save just means next restart redoes extraction


def load_catalogue(force_reload: bool = False) -> tuple[dict[str, Path], dict[str, dict]]:
    """(imageId -> jpg path, imageId -> metadata record), joined only
    on imageIds present in BOTH — an image with no metadata record (or
    vice versa) can't be filtered or displayed meaningfully anyway.

    Cached at module level — this was being re-read (manifest.json +
    every JSON batch file) on EVERY /thumb request, which is why grid
    pages with 40+ thumbnails were slow beyond just thumbnail
    generation. Call with force_reload=True after re-running the
    downloader if you want fresh data without restarting the server.

    Each record's prompt text is extracted and cached HERE, eagerly,
    at load time — then meta.comfy (the raw embedded ComfyUI workflow
    graph) is dropped from the in-memory copy. For catalogues with
    many custom workflows, that embedded graph JSON is easily the
    single largest field per record, and once the prompt text has
    been derived from it, the raw graph is never needed again for
    filtering or display. The one thing that DOES still need the
    original graph — /noprompt-diagnostic's forced re-extraction, to
    correctly log which node types caused a failure — re-reads that
    specific record fresh from disk via reload_record_from_disk
    instead of relying on this (now-trimmed) in-memory copy.

    The extracted prompt text is also persisted to PROMPT_SUMMARY_PATH
    (see load/save_prompt_summary) — this is what "we could build a
    proper summary file" was asking for: a SERVER RESTART pre-loads
    this file first, so extraction only actually runs for imageIds
    that aren't in it yet (i.e. new images from a freshly-added JSON
    batch), instead of re-walking every ComfyUI graph in the whole
    catalogue from scratch every time the process starts.
    """
    if _catalogue_cache and not force_reload:
        return _catalogue_cache["jpgs"], _catalogue_cache["records"]

    jpgs = load_manifest_jpgs()
    records = load_metadata_records()
    common = jpgs.keys() & records.keys()

    persisted_summary = load_prompt_summary()
    for iid, prompt in persisted_summary.items():
        _prompt_text_cache.setdefault(iid, prompt)

    trimmed: dict[str, dict] = {}
    for iid in common:
        rec = records[iid]
        effective_prompt_text(rec)  # cache hit if iid was in persisted_summary, otherwise real extraction
        meta = rec.get("meta")
        if isinstance(meta, dict) and "comfy" in meta:
            meta["comfy"] = None
        trimmed[iid] = rec

    save_prompt_summary({iid: _prompt_text_cache[iid] for iid in common if iid in _prompt_text_cache})

    result = ({i: jpgs[i] for i in common}, trimmed)
    _catalogue_cache["jpgs"], _catalogue_cache["records"] = result
    return result


def get_thumb_path(image_id: str, source_path: Path) -> Path:
    """Return a cached thumbnail path for image_id, generating it first
    if it doesn't exist yet (or the source is newer than the cached
    thumb — handles a re-download replacing the original)."""
    THUMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    thumb_path = THUMB_CACHE_DIR / f"{image_id}.jpg"

    if thumb_path.exists() and thumb_path.stat().st_mtime >= source_path.stat().st_mtime:
        return thumb_path

    from PIL import Image
    with Image.open(source_path) as im:
        im.thumbnail((THUMB_SIZE, THUMB_SIZE))
        im.convert("RGB").save(thumb_path, format="JPEG", quality=80)
    return thumb_path


@app.get("/resource-cooccurrence")
def resource_cooccurrence(resource: str, min_count: int = 5):
    """Given a seed resource (lora) name, find which OTHER resources
    tend to appear in the same images, ranked by lift — not raw count.

    Lift = P(B present | seed present) / P(B present overall).
    Raw co-occurrence counts are misleading (a generic aesthetic-booster
    lora shows up next to EVERYTHING just because it's everywhere) —
    lift asks "does B show up more than you'd expect by chance, given
    the seed", which is what actually indicates a real pairing (e.g.
    two loras habitually bundled for the same character/style).

    min_count filters out low-sample noise (a pair seen only once or
    twice can show enormous lift purely by chance).
    """
    _, records = load_catalogue()
    total = len(records) or 1

    seed_images = []
    overall_counts = Counter()
    for rec in records.values():
        names = {r["name"] for r in get_resources(rec)}
        overall_counts.update(names)
        if resource in names:
            seed_images.append(names)

    seed_total = len(seed_images) or 1
    with_seed_counts = Counter()
    for names in seed_images:
        with_seed_counts.update(names - {resource})

    ranked = []
    for name, count in with_seed_counts.items():
        if count < min_count:
            continue
        p_given_seed = count / seed_total
        p_overall = overall_counts[name] / total
        lift = round(p_given_seed / p_overall, 2) if p_overall else 0
        ranked.append({
            "resource": name,
            "count_with_seed": count,
            "count_overall": overall_counts[name],
            "lift": lift,
        })

    ranked.sort(key=lambda r: r["lift"], reverse=True)
    return {"seed": resource, "seed_image_count": seed_total, "co_occurring": ranked}


@app.get("/debug-prompt/{image_id}")
def debug_prompt(image_id: str):
    """Spot-check what effective_prompt_text actually extracts for one
    image — flat meta.prompt if present, else best-effort recovery from
    meta.comfy. Use this to find more comfy-graph node shapes that
    aren't handled yet (returns "" for those) instead of pasting whole
    metadata records for review."""
    _, records = load_catalogue()
    rec = records.get(image_id)
    if rec is None:
        return {"error": "imageId not found in catalogue"}
    return {
        "image_id": image_id,
        "has_flat_prompt": bool((rec.get("meta") or {}).get("prompt")),
        "extracted_prompt": effective_prompt_text(rec),
    }


@app.get("/noprompt-diagnostic")
def noprompt_diagnostic():
    """Runs prompt extraction across every currently-noprompt record and
    reports which comfy-graph node class_types most often caused a dead
    end (empty string returned) — i.e. the ones actually worth adding a
    handler for next, ranked by how many records they'd unblock, rather
    than guessing from one pasted example at a time."""
    _comfy_miss_log.clear()
    _, records = load_catalogue()

    still_empty = 0
    checked = 0
    for iid, rec in records.items():
        if not has_no_prompt(rec):
            continue
        checked += 1
        # The in-memory record's meta.comfy has already been cleared
        # by load_catalogue (RAM saving) — re-read the original raw
        # record from disk so _resolve_comfy_text has the real graph
        # to walk, and use_cache=False forces it to actually run
        # rather than returning the (already-empty) cached result.
        raw_rec = reload_record_from_disk(iid) or rec
        if not effective_prompt_text(raw_rec, use_cache=False).strip():
            still_empty += 1

    return {
        "checked": checked,
        "still_empty_after_extraction": still_empty,
        "unhandled_node_types": Counter(_comfy_miss_log).most_common(20),
    }


class LabelRequest(BaseModel):
    image_ids: list[str]
    label: str  # "include" | "exclude" | "unlabel"


@app.post("/label")
def label_images(req: LabelRequest):
    for iid in req.image_ids:
        state.include.discard(iid)
        state.exclude.discard(iid)
        if req.label == "include":
            state.include.add(iid)
        elif req.label == "exclude":
            state.exclude.add(iid)
    return {"include": len(state.include), "exclude": len(state.exclude)}


@app.get("/resource-frequency")
def resource_frequency(resource_type: str | None = "lora"):
    """Rank resource names by frequency across the WHOLE catalogue
    (not just include) — this is category discovery, not filter
    refinement. resource_type filters to one type (default 'lora',
    since that's the subject-signal one — pass 'model' to see
    checkpoint distribution instead, or None/'' for everything mixed).

    pct_of_catalogue above STOPLIST_PCT_THRESHOLD is flagged as likely
    generic tooling (aesthetic/quality loras) rather than a real
    subject signal, same idea as downloader.py's model blocklist — but
    not auto-dropped, since the threshold is a heuristic, not ground
    truth. Add confirmed noise to RESOURCE_STOPLIST once you've
    eyeballed it.
    """
    _, records = load_catalogue()
    total = len(records) or 1
    counter = Counter()
    for rec in records.values():
        for res in get_resources(rec):
            if resource_type and res["type"] != resource_type:
                continue
            counter[res["name"]] += 1

    ranked = [
        {
            "resource": name,
            "count": count,
            "pct_of_catalogue": round(count / total, 3),
            "in_stoplist": name in RESOURCE_STOPLIST,
            "likely_generic": (count / total) >= STOPLIST_PCT_THRESHOLD,
        }
        for name, count in counter.most_common()
    ]
    return {"total_images": total, "resource_type_filter": resource_type, "ranked_resources": ranked}


@app.get("/tag-frequency")
def tag_frequency():
    """Rank tags by how often they appear across the current include set."""
    _, records = load_catalogue()
    counter = Counter()
    for iid in state.include:
        if iid in records:
            tags = get_field(records[iid], TAG_FIELD)
            counter.update(str(t).lower() for t in tags)

    total_included = len(state.include) or 1
    ranked = [
        {"tag": tag, "count": count, "pct_of_include": round(count / total_included, 2)}
        for tag, count in counter.most_common()
    ]
    return {"ranked_tags": ranked}


class MatchRequest(BaseModel):
    keywords: list[str]
    match: str = "any"  # "any" | "all" | "none"


def check(values, keywords, match) -> bool:
    values = set(str(v).lower() for v in values)
    keywords = set(k.lower() for k in keywords)
    if match == "any":
        return bool(values & keywords)
    if match == "all":
        return keywords.issubset(values)
    if match == "none":
        return values.isdisjoint(keywords)
    raise ValueError(f"unknown match mode: {match}")


@app.post("/run-match")
def run_match(req: MatchRequest):
    _, records = load_catalogue()
    results = {"matched_include": [], "matched_new": [], "matched_excluded": []}

    for iid, data in records.items():
        tags = get_field(data, TAG_FIELD)
        if not check(tags, req.keywords, req.match):
            continue
        if iid in state.include:
            results["matched_include"].append(iid)
        elif iid in state.exclude:
            results["matched_excluded"].append(iid)  # matched but you already rejected it — flag for review
        else:
            results["matched_new"].append(iid)

    return results


# Different custom-node packages implement "join two strings" and
# "literal text box" the same way but under different input key names.
# Extend these dicts as more workflow variants turn up — this can't
# cover every custom node that exists, only the patterns seen so far.
_JOIN_NODE_KEYS = {
    "StringConcatenate": ("string_a", "string_b", "delimiter"),
    "JoinStrings": ("string1", "string2", "delimiter"),
}
_LITERAL_KEY_BY_CLASS = {
    "easy positive": "positive",
    "easy negative": "negative",
}


_comfy_miss_log: list[str] = []  # dev diagnostic only — see /noprompt-diagnostic


def _resolve_comfy_text(nodes: dict, value) -> str:
    """Resolve a ComfyUI widget input to its actual string value — same
    logic as downloader.py's _resolve_comfy_text, extended to cover a
    couple more common custom-node shapes seen in real workflows:
    JoinStrings (KJNodes' StringConcatenate equivalent), literal text
    boxes like 'easy positive'/'easy negative' (text sits under a
    'positive'/'negative' key, not 'text'), and ImpactWildcardProcessor
    (resolved wildcard text sits under 'populated_text', not an
    upstream link at all). `value` is either a literal string or a
    link [node_id, output_index]."""
    if isinstance(value, str):
        return value
    if not (isinstance(value, list) and len(value) == 2):
        return ""

    node = nodes.get(str(value[0]))
    if node is None:
        return ""
    inputs = node.get("inputs") or {}
    class_type = node.get("class_type")

    if class_type in _JOIN_NODE_KEYS:
        key_a, key_b, key_delim = _JOIN_NODE_KEYS[class_type]
        a = _resolve_comfy_text(nodes, inputs.get(key_a, ""))
        b = _resolve_comfy_text(nodes, inputs.get(key_b, ""))
        delim = inputs.get(key_delim, "")
        delim = delim if isinstance(delim, str) else _resolve_comfy_text(nodes, delim)
        return f"{a}{delim}{b}"

    if class_type == "JoinStringMulti":
        delim = inputs.get("delimiter", "")
        delim = delim if isinstance(delim, str) else _resolve_comfy_text(nodes, delim)
        string_keys = sorted(
            (k for k in inputs if k.startswith("string_")),
            key=lambda k: int(k.rsplit("_", 1)[-1]) if k.rsplit("_", 1)[-1].isdigit() else 0,
        )
        parts = [_resolve_comfy_text(nodes, inputs[k]) for k in string_keys]
        return delim.join(p for p in parts if p)

    if class_type == "PreviewAny":
        # Pure pass-through node — just forwards whatever feeds its
        # "source" input, doesn't transform it.
        return _resolve_comfy_text(nodes, inputs.get("source", ""))

    if class_type == "ImpactWildcardProcessor":
        return str(inputs.get("populated_text") or "")

    if class_type in _LITERAL_KEY_BY_CLASS:
        val = inputs.get(_LITERAL_KEY_BY_CLASS[class_type], "")
        return val if isinstance(val, str) else _resolve_comfy_text(nodes, val)

    if class_type == "TriggerWord Toggle (LoraManager)":
        toggles = (inputs.get("toggle_trigger_words") or {}).get("__value__") or []
        active = [str(item.get("text", "")) for item in toggles if isinstance(item, dict) and item.get("active")]
        return ", ".join(t for t in active if t)

    if "text" in inputs:
        return _resolve_comfy_text(nodes, inputs["text"])

    # Generic fallback for unknown literal-text node types (PrimitiveStringMultiline,
    # TextBox1, TextGenerate, etc.) — if there's exactly one plain string among this
    # node's inputs, it's almost certainly the text, whatever the key is called.
    # We only ever reach here via a real text-resolution chain rooted at a
    # KSampler's positive input, so any node landed on is part of that chain,
    # not an unrelated infra node — safe to guess in this specific context.
    string_inputs = [v for v in inputs.values() if isinstance(v, str)]
    if len(string_inputs) == 1:
        return string_inputs[0]

    _comfy_miss_log.append(class_type or "(unknown)")
    return ""


def _extract_comfy_prompt_text(record: dict) -> str:
    """Best-effort positive-prompt text from the embedded raw ComfyUI
    workflow (meta.comfy), for records with no flat meta.prompt — seen
    with custom workflows using CR Text + StringConcatenate chains
    instead of a single CLIPTextEncode. Without this, such records
    look prompt-less even though a real prompt exists in the graph."""
    meta = record.get("meta") or {}
    raw_comfy = meta.get("comfy")
    if not raw_comfy:
        return ""

    try:
        workflow = json.loads(raw_comfy)
        nodes = workflow.get("prompt") or {}
    except (json.JSONDecodeError, TypeError, AttributeError):
        return ""

    found = []
    for node in nodes.values():
        if node.get("class_type") != "KSampler":
            continue
        positive = (node.get("inputs") or {}).get("positive")
        text = _resolve_comfy_text(nodes, positive)
        if text:
            found.append(text)

    seen = set()
    unique = [t for t in found if not (t in seen or seen.add(t))]
    return ", ".join(unique)


_prompt_text_cache: dict[str, str] = {}


def effective_prompt_text(record: dict, use_cache: bool = True) -> str:
    """The positive prompt to use everywhere in this tool: flat
    meta.prompt when present, otherwise recovered from the embedded
    raw ComfyUI graph. Centralised so has_no_prompt/other callers
    can't drift out of sync on which source of truth they check.

    Cached by imageId — this used to be recomputed from scratch every
    time it was called, and /grid calls it 3+ times per matched record
    per page load (once during predicate matching, once for keyword-
    hit tracing, once for phrase tokenizing) — including a fresh
    json.loads + node-graph walk of meta.comfy for records with no
    flat prompt. Fine at a few thousand images, painfully slow once
    the catalogue grew past that. Cleared by /reload-catalogue.

    use_cache=False forces a fresh extraction, bypassing (but still
    refreshing) the cache — needed by /noprompt-diagnostic, since
    _resolve_comfy_text's failure logging to _comfy_miss_log only
    happens during actual extraction. A cached return would silently
    skip re-extraction for any record already resolved by an earlier
    /grid call, under-reporting which node types are actually causing
    failures.
    """
    image_id = record.get("imageId")
    image_id = str(image_id) if image_id is not None else None  # normalize to match records dict keys (str(imageId))
    if use_cache and image_id is not None and image_id in _prompt_text_cache:
        return _prompt_text_cache[image_id]

    meta = record.get("meta") or {}
    prompt = str(meta.get("prompt") or "").strip()
    if not prompt:
        prompt = _extract_comfy_prompt_text(record)

    if image_id is not None:
        _prompt_text_cache[image_id] = prompt
    return prompt


def has_no_prompt(record: dict) -> bool:
    return not effective_prompt_text(record).strip()


def has_no_lora(record: dict) -> bool:
    return not any(r["type"] == "lora" for r in get_resources(record))


FILTERS_CONFIG_PATH = Path("./filters.yaml")


def _tokenize(text: str) -> set[str]:
    return set(t for t in re.split(r"[,\s:|()\.\[\]]+", text.lower()) if t)


# Comma-separated phrase splitting for co-occurrence discovery — unlike
# _tokenize (word-level, for keyword matching), this keeps multi-word
# phrases like "cat ears" intact as one unit, matching how booru-style
# prompts are actually structured. Parens/brackets/braces count as
# implicit comma delimiters too — booru-style prompts use them for
# emphasis grouping, including nested, e.g. "((masterpiece, best
# quality))" or "(fur collar:1.2)" — and any ":<weight>" suffix is
# stripped first so it doesn't get glued onto the phrase to its left
# (otherwise "score_8)" or "fur collar:1.2)" leak through as their own
# bogus phrases instead of "score_8" / "fur collar").
#
# Embedded resource tags like "<lora:pearly_zootopia>" or
# "<lora:name:0.8>" (some workflows write these straight into the flat
# prompt string, resolved by a lora-loader node reading the text) are
# stripped entirely, not just delimited — they're a resource reference
# already captured properly via get_resources()/meta.resources, not a
# descriptive phrase, and left whole they either vanish (unescaped "<"
# read as a stray HTML tag) or get mangled by the weight-strip regex
# into junk like "pearlyzost".
#
# Civitai ARTICLE-sourced prompts sometimes bundle several images
# under one shared caption using a "__ 1st image:"/"__ 2nd image:"
# marker (optionally preceded by the article's own URL) instead of a
# comma — e.g. "...tag1, tag2 __ 3rd image: masterpiece, tag3...".
# With no comma on either side of the marker, the trailing tag before
# it and the leading tag after it glue into one garbage phrase (and
# critically, that means "masterpiece" never appears as its own exact
# token, so it slips past _PHRASE_STOPLIST entirely). Treating the URL
# and the marker itself as delimiters splits both neighbouring tags
# out cleanly and lets the stoplist catch "masterpiece" as intended.
def _phrase_tokenize(text: str) -> set[str]:
    text = re.sub(r"https?://\S+", ",", text)               # strip stray URLs (article-caption artifact)
    text = re.sub(r"__+\s*\d+(?:st|nd|rd|th)\s+image\s*:", ",", text)  # strip "__ Nth image:" markers
    text = re.sub(r"<[^<>]+>", ",", text)         # strip embedded <lora:...>/<embedding:...> tags
    text = re.sub(r":\d+(\.\d+)?", "", text)      # strip emphasis weights like :1.2
    text = re.sub(r"[()\[\]{}]+", ",", text)      # parens/brackets -> delimiter
    phrases = set()
    for p in re.split(r"[,\n]+", text.lower()):
        # Trim stray leading/trailing punctuation (periods, stray quotes,
        # etc.) left over from prose-style prompts or a missing comma —
        # otherwise "female." and "female" count as different phrases.
        p = re.sub(r"^[^\w]+|[^\w]+$", "", p.strip())
        if len(p) > 1:
            phrases.add(p)
    return phrases


# Quality/aesthetic boilerplate that shows up in nearly every prompt
# regardless of subject — not useful as a co-occurrence signal. Add to
# this as you spot more generic phrases in the association output.
_PHRASE_STOPLIST = {
    "masterpiece", "best quality", "very aesthetic", "absurdres", "highres",
    "newest", "sensitive", "explicit", "high contrast", "ultra detailed",
    "professional quality", "high resolution", "sharp focus", "rich contrast",
    "hyperrealistic", "amazing quality",
}

# Structural/count tags that scale indefinitely (score_9, score_8_up,
# etc.) — noise for subject discovery same as _PHRASE_STOPLIST, but
# enumerating every count variant by hand doesn't scale. Regex
# patterns instead — extend this list as new scaling patterns turn up
# (don't add individual tags here, that's what _PHRASE_STOPLIST is for).
#
# Headcount tags (1girl/2girls/1boy/...) were stoplisted here too at
# one point, but are back in the main tables for now — no concrete use
# case yet for hiding them, revisit if/when one comes up.
_PHRASE_STOPLIST_PATTERNS = [
    re.compile(r"^score_\d+(_up)?$"),
]


def _is_stoplisted_phrase(phrase: str) -> bool:
    return phrase in _PHRASE_STOPLIST or any(p.match(phrase) for p in _PHRASE_STOPLIST_PATTERNS)


_prompt_lower_cache: dict[str, str] = {}
_resources_lower_cache: dict[str, frozenset] = {}


def _condition_values(record: dict, field: str):
    """The set of values to match a condition's keywords against.
    'prompt' is free text, tokenized into whole words (so 'tail'
    doesn't match inside 'detail'). 'resources' is the record's lora/
    model names, matched as whole discrete strings, not tokenized (a
    lora name shouldn't be split on its own words). Anything else is
    treated as a dot-path into the record via get_field.

    'prompt'/'resources' results are cached by imageId — this used to
    re-run .lower() (and rebuild the resources set) on every single
    call, and this function is called once per condition per record
    per /grid request. Cleared by /reload-catalogue.
    """
    image_id = record.get("imageId")
    image_id = str(image_id) if image_id is not None else None  # normalize to match records dict keys (str(imageId))

    if field == "prompt":
        if image_id is not None and image_id in _prompt_lower_cache:
            return _prompt_lower_cache[image_id]
        value = effective_prompt_text(record).lower()
        if image_id is not None:
            _prompt_lower_cache[image_id] = value
        return value

    if field == "resources":
        if image_id is not None and image_id in _resources_lower_cache:
            return _resources_lower_cache[image_id]
        value = frozenset(r["name"].lower() for r in get_resources(record))
        if image_id is not None:
            _resources_lower_cache[image_id] = value
        return value

    raw = get_field(record, field)
    return {str(v).lower() for v in raw if v}


def _apply_match(values, keywords: set[str], mode: str) -> bool:
    if isinstance(values, str):
        if mode == "any":
            return any(bool(re.search(r"\b" + re.escape(kw) + r"\b", values)) for kw in keywords)
        if mode == "all":
            return all(bool(re.search(r"\b" + re.escape(kw) + r"\b", values)) for kw in keywords)
        if mode == "none":
            return not any(bool(re.search(r"\b" + re.escape(kw) + r"\b", values)) for kw in keywords)
        raise ValueError(f"unknown match mode: {mode!r}")

    # Fallback to precise set operations for lists/tags/resources
    if mode == "any":
        return bool(values & keywords)
    if mode == "all":
        return keywords.issubset(values)
    if mode == "none":
        return values.isdisjoint(keywords)
    raise ValueError(f"unknown match mode: {mode!r} (use any/all/none)")


def _combine(results: list[bool], mode: str) -> bool:
    if mode == "any":
        return any(results)
    if mode == "all":
        return all(results)
    if mode == "none":
        return not any(results)
    raise ValueError(f"unknown match mode: {mode!r} (use any/all/none)")


_grid_filters_cache: dict = {}


def load_grid_filters() -> dict:
    """Filters selectable from /grid via ?view=<n> — 'all'/'noprompt'/
    'nolora' are always available (structural checks, not subject
    filters), everything else comes from filters.yaml. No restart
    needed to add/edit a filter: save the YAML file and reload the
    page.

    Cached by filters.yaml's mtime — this was being read and
    yaml.safe_load'd from disk, with every filter's closures rebuilt
    from scratch, on EVERY /grid request (including just clicking
    between filter tabs). The mtime check is one cheap stat() call;
    the full re-parse only happens again when the file actually
    changed, so editing filters.yaml still takes effect immediately
    with no restart.

    Each entry is {"predicate": fn, "conditions": list or None} —
    conditions is None for the built-in structural filters (no
    keyword list to trace), and the raw condition list for YAML-
    defined filters, so the grid can report which specific keywords
    drove each match, not just true/false.
    """
    if not FILTERS_CONFIG_PATH.exists():
        return {
            "all": {"predicate": lambda rec: True, "conditions": None},
            "noprompt": {"predicate": has_no_prompt, "conditions": None},
            "nolora": {"predicate": has_no_lora, "conditions": None},
        }

    mtime = FILTERS_CONFIG_PATH.stat().st_mtime
    if _grid_filters_cache.get("mtime") == mtime:
        return _grid_filters_cache["filters"]

    filters: dict = {
        "all": {"predicate": lambda rec: True, "conditions": None},
        "noprompt": {"predicate": has_no_prompt, "conditions": None},
        "nolora": {"predicate": has_no_lora, "conditions": None},
    }

    config = yaml.safe_load(FILTERS_CONFIG_PATH.read_text()) or {}
    for entry in config.get("filters", []):
        name = entry["name"]
        filter_match_mode = entry.get("match", "any")
        conditions = entry.get("conditions", [])

        def make_predicate(conditions=conditions, filter_match_mode=filter_match_mode):
            def predicate(record: dict) -> bool:
                results = [
                    _apply_match(
                        _condition_values(record, cond["field"]),
                        {k.lower() for k in cond["keywords"]},
                        cond.get("match", "any"),
                    )
                    for cond in conditions
                ]
                return _combine(results, filter_match_mode)
            return predicate

        filters[name] = {
            "predicate": make_predicate(),
            "conditions": conditions,
            "match": filter_match_mode,
        }

    _grid_filters_cache["mtime"] = mtime
    _grid_filters_cache["filters"] = filters
    return filters


def matched_keywords_for(record: dict, conditions: list) -> set[str]:
    """Which specific keywords this record actually hit, across all of
    a filter's non-'none' conditions — 'none' conditions confirm an
    ABSENCE, so they never contribute a "why this matched" keyword."""
    matched: set[str] = set()
    for cond in conditions:
        if cond.get("match", "any") == "none":
            continue
        values = _condition_values(record, cond["field"])
        keywords = {k.lower() for k in cond["keywords"]}

        if isinstance(values, str):
            matched |= {kw for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", values)}
        else:
            matched |= (values & keywords)
    return matched


def dominant_lora(record: dict) -> str:
    """Pick a grouping key for this image: the first non-stoplisted
    lora if one exists, otherwise fall back to the checkpoint's
    modelName — most images (2267/2956 in the initial check) use no
    subject lora at all, just a base checkpoint, so lora alone only
    covers a minority. modelName (top-level, the civitai checkpoint
    page name — NOT meta.Model, which is workflow-specific and often
    missing) gave a well-distributed ~20+ way split on that leftover
    set, so it's the fallback tier rather than one giant catch-all
    bucket."""
    for res in get_resources(record):
        if res["type"] == "lora" and res["name"] not in RESOURCE_STOPLIST:
            return res["name"]
    checkpoint = record.get("modelName")
    return f"checkpoint: {checkpoint}" if checkpoint else "(no lora or checkpoint)"


@app.post("/reload-catalogue")
def reload_catalogue():
    _prompt_text_cache.clear()
    _phrase_index_cache.clear()
    _resources_cache.clear()
    _prompt_lower_cache.clear()
    _resources_lower_cache.clear()
    _word_index_cache.clear()
    jpgs, records = load_catalogue(force_reload=True)
    return {"jpgs": len(jpgs), "records": len(records)}


_phrase_index_cache: dict = {}


def get_phrase_index(force_reload: bool = False) -> tuple[dict[str, frozenset], Counter]:
    """(imageId -> phrase set, GLOBAL phrase -> count across the whole
    catalogue), built once and cached.

    /grid's association tables used to compute global counts by doing,
    for EVERY candidate phrase, a fresh pass over every record calling
    _phrase_tokenize(effective_prompt_text(rec)) again — O(records x
    phrases). With ~160 candidate phrases (60 + the 100-row compound
    table) and a catalogue that's grown past a few thousand images,
    that's the dominant cost of loading /grid at all.

    This does the tokenizing pass ONCE (O(records)), and builds the
    global Counter in the same pass — so a global count for any phrase
    afterwards is an O(1) dict lookup, and per-filter phrase_counts in
    grid() can reuse each record's already-tokenized set instead of
    re-tokenizing it again on every /grid call. Cleared by
    /reload-catalogue.
    """
    if _phrase_index_cache and not force_reload:
        return _phrase_index_cache["by_id"], _phrase_index_cache["global_counts"]

    _, records = load_catalogue()
    by_id: dict[str, frozenset] = {}
    global_counts: Counter = Counter()
    for iid, rec in records.items():
        phrases = frozenset(_phrase_tokenize(effective_prompt_text(rec)))
        by_id[iid] = phrases
        global_counts.update(phrases)

    _phrase_index_cache["by_id"] = by_id
    _phrase_index_cache["global_counts"] = global_counts
    return by_id, global_counts


_word_index_cache: dict = {}


def get_word_index(force_reload: bool = False) -> tuple[dict[str, frozenset], dict[str, frozenset]]:
    """(imageId -> word-token set, word -> set of imageIds containing
    it as a whole word), built once over the whole catalogue's prompt
    text using _tokenize (word-level splitting — distinct from
    _phrase_tokenize's comma-delimited phrases above).

    This is what lets a filter condition on the 'prompt' field with
    ALL-single-word keywords skip the per-record regex scan entirely:
    look up each keyword's id set directly (O(1)) and union/intersect
    them, instead of an O(records) sweep doing a \\b<kw>\\b regex search
    per record per keyword. See _fast_condition_ids, used in grid().

    Multi-word keywords (e.g. "elf ears") aren't single tokens, so
    they can't be resolved here — grid()'s fast path bails to the
    normal predicate() scan for any condition containing one, which
    stays fully correct, just not accelerated. Cleared by
    /reload-catalogue.
    """
    if _word_index_cache and not force_reload:
        return _word_index_cache["by_id"], _word_index_cache["word_to_ids"]

    _, records = load_catalogue()
    by_id: dict[str, frozenset] = {}
    word_to_ids: dict[str, set] = defaultdict(set)
    for iid, rec in records.items():
        tokens = frozenset(_tokenize(effective_prompt_text(rec)))
        by_id[iid] = tokens
        for tok in tokens:
            word_to_ids[tok].add(iid)

    frozen_word_to_ids = {tok: frozenset(ids) for tok, ids in word_to_ids.items()}
    _word_index_cache["by_id"] = by_id
    _word_index_cache["word_to_ids"] = frozen_word_to_ids
    return by_id, frozen_word_to_ids


def _fast_condition_ids(cond: dict, all_ids: frozenset, word_to_ids: dict[str, frozenset]):
    """Resolve ONE filter condition to a set of matching imageIds using
    the word inverted index — O(1) lookups + set ops instead of an
    O(records) regex scan.

    Returns None if this condition can't be safely fast-pathed:
    - field isn't 'prompt' (resources/dot-path conditions still use
      the regular per-record path — no index built for those)
    - ANY keyword in this condition is multi-word (e.g. "elf ears")
      — the word index only has single tokens, and mixing a fast
      single-word lookup with one keyword that genuinely needs a
      regex check would either miss matches or require a per-record
      check anyway, so the whole condition bails rather than risk an
      incorrect partial result.

    A filter mixing single- and multi-word keywords in ONE condition
    gets no speedup for that condition (falls back, still correct) —
    splitting multi-word keywords into their own condition (combined
    at the filter level via match: any/all) unlocks acceleration for
    the single-word condition alongside it.
    """
    if cond["field"] != "prompt":
        return None
    keywords = [k.lower() for k in cond["keywords"]]
    if any(re.search(r"\s", kw) for kw in keywords):
        return None

    mode = cond.get("match", "any")
    sets = [word_to_ids.get(kw, frozenset()) for kw in keywords]

    if mode == "any":
        result: set = set()
        for s in sets:
            result |= s
        return result
    if mode == "all":
        if not sets:
            return set(all_ids)
        result = set(sets[0])
        for s in sets[1:]:
            result &= s
        return result
    if mode == "none":
        present: set = set()
        for s in sets:
            present |= s
        return set(all_ids) - present
    return None


@app.get("/grid", response_class=HTMLResponse)
def grid(view: str = "all"):
    """Thumbnails grouped by dominant lora — a visual check on whether
    lora-based grouping actually clusters similar subjects, before
    trusting it as a category source. Groups sorted by size, largest
    first; each group capped at 40 thumbnails so one huge group
    doesn't dominate the page.

    view selects a GRID_FILTERS predicate to restrict the catalogue to
    a named subset first (e.g. 'noprompt', 'nolora') — buttons for
    each are rendered at the top of the page.
    """
    jpgs, records = load_catalogue()
    grid_filters = load_grid_filters()
    filter_entry = grid_filters.get(view, grid_filters["all"])
    predicate = filter_entry["predicate"]
    conditions = filter_entry["conditions"]
    filter_match_mode = filter_entry.get("match", "any")

    groups: dict[str, list[str]] = defaultdict(list)
    matched_keywords_by_id: dict[str, set[str]] = {}
    keyword_counts: Counter = Counter()
    own_keywords: set[str] = set()
    if conditions:
        for cond in conditions:
            if cond.get("match", "any") != "none":
                for kw in cond["keywords"]:
                    keyword_counts[kw.lower()] += 0
                    own_keywords.add(kw.lower())

    # Fast path: resolve matched_ids via the word inverted index
    # (get_word_index) instead of scanning every record and calling
    # predicate() on each — O(1) lookups + set ops per condition,
    # instead of an O(records) regex sweep. Only attempted when EVERY
    # condition qualifies (see _fast_condition_ids); any one condition
    # that can't be safely fast-pathed (non-prompt field, or a multi-
    # word keyword) drops the whole filter back to the exact original
    # predicate() scan, so results are always correct either way.
    matched_ids: list[str] = []
    if conditions:
        all_ids = frozenset(records.keys())
        _, word_to_ids = get_word_index()
        per_condition_ids = []
        fast_path_ok = True
        for cond in conditions:
            ids = _fast_condition_ids(cond, all_ids, word_to_ids)
            if ids is None:
                fast_path_ok = False
                break
            per_condition_ids.append(ids)

        if fast_path_ok:
            if filter_match_mode == "any":
                matched_set = set()
                for s in per_condition_ids:
                    matched_set |= s
            elif filter_match_mode == "all":
                matched_set = set(per_condition_ids[0]) if per_condition_ids else set(all_ids)
                for s in per_condition_ids[1:]:
                    matched_set &= s
            elif filter_match_mode == "none":
                present: set = set()
                for s in per_condition_ids:
                    present |= s
                matched_set = set(all_ids) - present
            else:
                fast_path_ok = False  # unrecognised mode — stay safe, fall back

            if fast_path_ok:
                matched_ids = [iid for iid in matched_set if iid in jpgs]

        if not fast_path_ok:
            matched_ids = [iid for iid, rec in records.items() if iid in jpgs and predicate(rec)]
    else:
        matched_ids = [iid for iid, rec in records.items() if iid in jpgs and predicate(rec)]

    for iid in matched_ids:
        rec = records[iid]
        if conditions:
            hits = matched_keywords_for(rec, conditions)
            matched_keywords_by_id[iid] = hits
            keyword_counts.update(hits)
        groups[dominant_lora(rec)].append(iid)

    # Phrase co-occurrence: which phrases (comma/paren-delimited chunks
    # of the prompt) show up alongside this filter's own keywords, and
    # how "specific" each one is to this filter vs the whole catalogue.
    # Uses the cached phrase index (see get_phrase_index) — per-record
    # tokenizing happens once for the whole catalogue's lifetime, not
    # once per /grid call.
    phrase_by_id, global_phrase_counts = get_phrase_index()
    phrase_counts: Counter = Counter()
    if conditions:
        for iid in matched_ids:
            phrases = phrase_by_id.get(iid, frozenset())
            phrase_counts.update(
                p for p in (phrases - own_keywords) if not _is_stoplisted_phrase(p)
            )

    def tile(iid):
        border = "green" if iid in state.include else "red" if iid in state.exclude else "#ccc"
        hits = matched_keywords_by_id.get(iid)
        hits_html = (
            f'<div style="font-size:0.75rem; color:#888;">{html.escape(", ".join(sorted(hits)))}</div>'
            if hits else ""
        )
        # width AND height both set (not just width) — without an explicit
        # height, the browser doesn't know each tile's size until the lazy
        # image actually loads, forcing a layout recalculation per image as
        # they trickle in instead of laying the whole page out once upfront.
        return f"""
        <div class="tile" style="border: 3px solid {border}" data-id="{html.escape(iid)}">
            <img src="/thumb/{html.escape(iid)}" loading="lazy" width="150" height="150" style="object-fit:cover;">
            <div>{html.escape(iid)}</div>
            {hits_html}
        </div>"""

    # Total tiles actually rendered into the DOM, across ALL groups — a
    # catalogue with many dominant_lora groups (lots of distinct
    # checkpoints as the fallback grouping key) can add up to thousands
    # of tiles even with the existing 40-per-group cap, and building/
    # laying out that much DOM is exactly what was burning browser CPU.
    # loading="lazy" only defers the image FETCH, not the DOM/layout
    # work — this cap bounds that work directly instead.
    MAX_TOTAL_TILES = 1000
    sections = []
    tiles_rendered = 0
    groups_sorted = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    for name, ids in groups_sorted:
        if tiles_rendered >= MAX_TOTAL_TILES:
            break
        remaining_budget = MAX_TOTAL_TILES - tiles_rendered
        shown = ids[:min(40, remaining_budget)]
        tiles_rendered += len(shown)
        # content-visibility:auto tells the browser to skip layout/paint
        # entirely for groups currently offscreen (it estimates their
        # size from contain-intrinsic-size instead) — this is the main
        # fix for CPU cost from many groups, since the browser stops
        # doing real layout work for the ~9 out of 10 groups you aren't
        # currently looking at.
        sections.append(f"""
        <h3 style="font-family:sans-serif;">{html.escape(name)} ({len(ids)} image{'s' if len(ids) != 1 else ''})</h3>
        <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:2rem;
                     content-visibility:auto; contain-intrinsic-size: 1000px 200px;">
            {"".join(tile(i) for i in shown)}
        </div>""")

    omitted_notice = (
        f'<p style="font-family:sans-serif; color:#888;">'
        f'Showing {tiles_rendered} of {sum(len(ids) for _, ids in groups_sorted)} matched images '
        f'(capped at {MAX_TOTAL_TILES} per page load — narrow the filter/view to see the rest).</p>'
        if tiles_rendered < sum(len(ids) for _, ids in groups_sorted) else ""
    )

    filter_buttons = "".join(
        f'<a href="/grid?view={html.escape(key)}" style="margin-right:1rem; '
        f'{"font-weight:bold;" if key == view else ""}">{html.escape(key)}</a>'
        for key in grid_filters
    )

    keyword_summary = ""
    if keyword_counts:
        rows = "".join(
            f"<tr><td>{html.escape(kw)}</td><td>{count}</td></tr>"
            for kw, count in keyword_counts.most_common()
        )
        keyword_summary = f"""
        <table style="font-family:sans-serif; border-collapse:collapse; margin-bottom:1.5rem;">
            <tr><th style="text-align:left; padding-right:1rem;">keyword</th><th style="text-align:left;">image count</th></tr>
            {rows}
        </table>"""

    # Association tables: two views over the same candidate phrase pool.
    # "Most common" surfaces high-volume phrases (often generic/global
    # boilerplate). "Most specific" ranks by % of that phrase's GLOBAL
    # catalogue occurrences that fall inside this filter — a
    # concentration/lift signal, since a phrase appearing 39 times and
    # ALL 39 being inside this filter (100%) is a much stronger signal
    # than a phrase appearing 97 times with only 8% inside this filter.
    #
    # Pool is widened to 60 candidates (vs the 30 actually displayed per
    # table) before computing global counts — otherwise the specificity
    # table only ever re-sorts the same top-30-by-count phrases, and a
    # low-count-but-100%-specific phrase would never surface at all.
    association_summary = ""
    if phrase_counts:
        total_matched = len(matched_ids) or 1

        # Compound variants of the filter's OWN keywords — e.g. filtering
        # on "ear" should surface "pointy ears", "elf ears", "ear wax" as
        # their own dedicated list, since these are the phrases most
        # directly relevant to why you built this filter in the first
        # place. Pulled from the FULL phrase_counts (no top-N cap), since
        # a compound variant can be low-count/high-relevance and would
        # otherwise never make it into the general top-60 candidate pool
        # used by the two tables below.
        compound_variant_phrases = [
            (p, c) for p, c in phrase_counts.items()
            if c >= 2 and any(re.search(r"\b" + re.escape(kw) + r"\b", p) for kw in own_keywords)
        ]

        candidate_phrases = [(p, c) for p, c in phrase_counts.most_common(60) if c >= 2]

        # Global counts: O(1) lookups into the catalogue-wide index
        # built once by get_phrase_index — replaces what used to be a
        # fresh full-catalogue rescan per candidate phrase.
        global_counts = global_phrase_counts

        def pct_global(phrase, count):
            gc = global_counts[phrase]
            return round(count / gc * 100) if gc else 0

        def make_table(ordered, heading, limit=30):
            rows = "".join(
                f"<tr><td>{html.escape(phrase)}</td><td>{count}</td>"
                f"<td>{round(count / total_matched * 100)}%</td>"
                f"<td>{global_counts[phrase]}</td><td>{pct_global(phrase, count)}%</td></tr>"
                for phrase, count in ordered[:limit]
            )
            return f"""
            <h4 style="font-family:sans-serif;">{heading}</h4>
            <table style="font-family:sans-serif; border-collapse:collapse; margin-bottom:1.5rem;">
                <tr><th style="text-align:left; padding-right:1rem;">co-occurring phrase</th>
                    <th style="text-align:left; padding-right:1rem;">count</th>
                    <th style="text-align:left; padding-right:1rem;">% of matches</th>
                    <th style="text-align:left; padding-right:1rem;">global count</th>
                    <th style="text-align:left;">% of global</th></tr>
                {rows}
            </table>""" if rows else ""

        by_count = sorted(candidate_phrases, key=lambda pc: -pc[1])
        by_specificity = sorted(candidate_phrases, key=lambda pc: -pct_global(*pc))
        by_compound_count = sorted(compound_variant_phrases, key=lambda pc: -pc[1])

        association_summary = (
            make_table(by_compound_count, "Compound phrases containing a filter keyword", limit=100)
            + make_table(by_count, "Most common in this filter")
            + make_table(by_specificity, "Most specific to this filter (% of global)")
        )

    return f"""
    <html><body>
    <div style="font-family:sans-serif; margin-bottom:1rem;">{filter_buttons}</div>
    {keyword_summary}
    {association_summary}
    {omitted_notice}
    {"".join(sections)}
    <script>
      // click a tile to toggle include/exclude — wire this up to POST /label
    </script>
    </body></html>
    """


@app.get("/thumb/{image_id}")
def get_thumb(image_id: str):
    from fastapi.responses import FileResponse
    jpgs, _ = load_catalogue()
    thumb_path = get_thumb_path(image_id, jpgs[image_id])
    return FileResponse(thumb_path)


@app.get("/image/{image_id}")
def get_image(image_id: str):
    from fastapi.responses import FileResponse
    jpgs, _ = load_catalogue()
    return FileResponse(jpgs[image_id])