"""
Minimal sketch: image-focused, tag-based include/exclude filter session.
No prompt text, no NLP — just set overlap on structured JSON fields.

Run: uv run uvicorn filter_session:app --reload
"""

import csv
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

# Resources that show up in most images regardless of subject — aesthetic
# boosters, quality/highres loras, etc. Not useful as category signal.
# Add to this as you spot more in the ranked output below.
RESOURCE_STOPLIST = {
    "high_detailed",
}
STOPLIST_PCT_THRESHOLD = 0.35  # auto-flag (not auto-drop) anything above this

# Danbooru seed graph — implications fetched once via fetch_danbooru_implications.py,
# tags/aliases from the deepghs/site_tags HuggingFace bulk dump.
DANBOORU_IMPLICATIONS_PATH = Path("./danbooru_implications.json")
DANBOORU_TAGS_CSV = Path("./tags.csv")
DANBOORU_ALIASES_CSV = Path("./tag_aliases.csv")
DANBOORU_CATEGORY_GENERAL = "0"  # only general tags are useful subject candidates,
                                  # not artist(1)/copyright(3)/character(4)/meta(5)


def get_field(data: dict, dot_path: str):
    for part in dot_path.split("."):
        if not isinstance(data, dict) or part not in data:
            return []
        data = data[part]
    return data if isinstance(data, list) else [data] if data else []


def get_resources(record: dict) -> list[dict]:
    """meta.resources[] as (name, type) pairs — type is one of
    'model' (checkpoint), 'lora', 'used_embeddings', 'vae', etc.
    Lumping all types together mixes checkpoint selection with subject
    loras, which are different signals — let callers filter by type."""
    resources = get_field(record, "meta.resources")
    return [
        {"name": r["name"], "type": r.get("type", "")}
        for r in resources if isinstance(r, dict) and r.get("name")
    ]


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


def load_metadata_records() -> dict[str, dict]:
    """imageId -> metadata record, flattened across every batch file
    in JSON_METADATA_DIR (each file is an array of civitai_fetcher
    records, per the civitai_comfy_bridge format)."""
    records: dict[str, dict] = {}
    for path in JSON_METADATA_DIR.glob("*.json"):
        for rec in json.loads(path.read_text()):
            records[str(rec["imageId"])] = rec
    return records


# --- in-memory session state (swap for redis/sqlite if it needs to survive restarts) ---
class SessionState:
    def __init__(self):
        self.include: set[str] = set()
        self.exclude: set[str] = set()

    def unlabeled(self, all_ids: set[str]) -> set[str]:
        return all_ids - self.include - self.exclude


state = SessionState()


_catalogue_cache: dict = {}


def load_catalogue(force_reload: bool = False) -> tuple[dict[str, Path], dict[str, dict]]:
    """(imageId -> jpg path, imageId -> metadata record), joined only
    on imageIds present in BOTH — an image with no metadata record (or
    vice versa) can't be filtered or displayed meaningfully anyway.

    Cached at module level — this was being re-read (manifest.json +
    every JSON batch file) on EVERY /thumb request, which is why grid
    pages with 40+ thumbnails were slow beyond just thumbnail
    generation. Call with force_reload=True after re-running the
    downloader if you want fresh data without restarting the server.
    """
    if _catalogue_cache and not force_reload:
        return _catalogue_cache["jpgs"], _catalogue_cache["records"]

    jpgs = load_manifest_jpgs()
    records = load_metadata_records()
    common = jpgs.keys() & records.keys()
    result = ({i: jpgs[i] for i in common}, {i: records[i] for i in common})
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


_danbooru_cache: dict = {}


def load_danbooru_graph() -> dict:
    """Build (and cache in memory) the Danbooru seed graph:
    - alias_map: alias name -> canonical name
    - parent_to_children: canonical parent tag -> set of canonical child tags
      (child implies parent, e.g. children_of["animal"] includes "cat", "dog")
    - child_to_parents: canonical child tag -> set of canonical parent tags
    - deprecated: set of deprecated tag names, excluded from the graph

    Only category "general" (0) tags are kept — artist/copyright/character/
    meta tags aren't useful subject-category candidates for this tool.

    Cached at module level since tags.csv is ~195MB — re-parsing it per
    request would make every /suggest-siblings call slow for no reason.
    """
    if _danbooru_cache:
        return _danbooru_cache

    alias_map: dict[str, str] = {}
    with open(DANBOORU_ALIASES_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            alias_map[row["alias"]] = row["tag"]

    def resolve(name: str) -> str:
        seen = set()
        while name in alias_map and name not in seen:
            seen.add(name)
            name = alias_map[name]
        return name

    deprecated: set[str] = set()
    general_tags: set[str] = set()
    with open(DANBOORU_TAGS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("is_deprecated") == "True":
                deprecated.add(row["name"])
            if row.get("category") == DANBOORU_CATEGORY_GENERAL:
                general_tags.add(row["name"])

    implications = json.loads(DANBOORU_IMPLICATIONS_PATH.read_text())
    parent_to_children: dict[str, set[str]] = defaultdict(set)
    child_to_parents: dict[str, set[str]] = defaultdict(set)
    for imp in implications:
        child = resolve(imp["antecedent"])
        parent = resolve(imp["consequent"])
        if child in deprecated or parent in deprecated:
            continue
        if child not in general_tags or parent not in general_tags:
            continue  # keep only general<->general relations
        parent_to_children[parent].add(child)
        child_to_parents[child].add(parent)

    _danbooru_cache.update({
        "resolve": resolve,
        "parent_to_children": dict(parent_to_children),
        "child_to_parents": dict(child_to_parents),
    })
    return _danbooru_cache


def prompt_text_for(record: dict) -> str:
    return effective_prompt_text(record).lower()


@app.get("/suggest-siblings")
def suggest_siblings(tag: str):
    """Given a seed tag (e.g. 'cat'), find its Danbooru parent(s) (e.g.
    'animal'), collect all sibling tags under those parents (e.g. 'dog',
    'bird', 'fox', ...), then check which siblings actually show up in
    THIS catalogue's prompts — the "real magic" being that Danbooru
    tells you what's possible, your data tells you what's present.

    Only siblings with at least one hit are returned, ranked by count.
    A sibling never appearing in your data is just noise for this tool.
    """
    graph = load_danbooru_graph()
    seed = graph["resolve"](tag.strip().lower().replace(" ", "_"))

    parents = graph["child_to_parents"].get(seed, set())
    siblings: set[str] = set()
    for parent in parents:
        siblings |= graph["parent_to_children"].get(parent, set())
    siblings.discard(seed)

    if not siblings:
        return {"seed": seed, "parents": sorted(parents), "siblings_found_in_data": []}

    _, records = load_catalogue()
    total = len(records) or 1
    counter = Counter()
    for rec in records.values():
        prompt = prompt_text_for(rec)
        for sib in siblings:
            needle = sib.replace("_", " ")
            if needle in prompt:
                counter[sib] += 1

    ranked = [
        {"tag": name, "count": count, "pct_of_catalogue": round(count / total, 3)}
        for name, count in counter.most_common()
    ]
    return {"seed": seed, "parents": sorted(parents), "siblings_found_in_data": ranked}


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
    for rec in records.values():
        if not has_no_prompt(rec):
            continue
        checked += 1
        if not effective_prompt_text(rec).strip():
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


def effective_prompt_text(record: dict) -> str:
    """The positive prompt to use everywhere in this tool: flat
    meta.prompt when present, otherwise recovered from the embedded
    raw ComfyUI graph. Centralised so has_no_prompt/prompt_text_for
    can't drift out of sync on which source of truth they check."""
    meta = record.get("meta") or {}
    prompt = str(meta.get("prompt") or "").strip()
    if prompt:
        return prompt
    return _extract_comfy_prompt_text(record)


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
# prompts are actually structured.
def _phrase_tokenize(text: str) -> set[str]:
    return {p.strip() for p in re.split(r"[,\n]+", text.lower()) if len(p.strip()) > 1}


# Quality/aesthetic boilerplate that shows up in nearly every prompt
# regardless of subject — not useful as a co-occurrence signal. Add to
# this as you spot more generic phrases in the association output.
_PHRASE_STOPLIST = {
    "masterpiece", "best quality", "very aesthetic", "absurdres", "highres",
    "newest", "sensitive", "explicit", "high contrast", "ultra detailed",
    "professional quality", "high resolution", "sharp focus", "rich contrast",
    "hyperrealistic", "amazing quality",
}


def _condition_values(record: dict, field: str) -> set[str]:
    """The set of values to match a condition's keywords against.
    'prompt' is free text, tokenized into whole words (so 'tail'
    doesn't match inside 'detail'). 'resources' is the record's lora/
    model names, matched as whole discrete strings, not tokenized (a
    lora name shouldn't be split on its own words). Anything else is
    treated as a dot-path into the record via get_field."""
    if field == "prompt":
        return effective_prompt_text(record).lower()  # Keep raw text intact
    if field == "resources":
        return {r["name"].lower() for r in get_resources(record)}
    raw = get_field(record, field)
    return {str(v).lower() for v in raw if v}

def _apply_match(values: set[str], keywords: set[str], mode: str) -> bool:
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


def load_grid_filters() -> dict:
    """Filters selectable from /grid via ?view=<name> — 'all'/'noprompt'/
    'nolora' are always available (structural checks, not subject
    filters), everything else comes fresh from filters.yaml on every
    call. No restart needed to add/edit a filter: save the YAML file
    and reload the page.

    Each entry is {"predicate": fn, "conditions": list or None} —
    conditions is None for the built-in structural filters (no
    keyword list to trace), and the raw condition list for YAML-
    defined filters, so the grid can report which specific keywords
    drove each match, not just true/false.
    """
    filters: dict = {
        "all": {"predicate": lambda rec: True, "conditions": None},
        "noprompt": {"predicate": has_no_prompt, "conditions": None},
        "nolora": {"predicate": has_no_lora, "conditions": None},
    }

    if not FILTERS_CONFIG_PATH.exists():
        return filters

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

        filters[name] = {"predicate": make_predicate(), "conditions": conditions}

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
    jpgs, records = load_catalogue(force_reload=True)
    return {"jpgs": len(jpgs), "records": len(records)}


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

    groups: dict[str, list[str]] = defaultdict(list)
    matched_keywords_by_id: dict[str, set[str]] = {}
    keyword_counts: Counter = Counter()
    matched_ids: list[str] = []
    own_keywords: set[str] = set()
    if conditions:
        for cond in conditions:
            if cond.get("match", "any") != "none":
                for kw in cond["keywords"]:
                    keyword_counts[kw.lower()] += 0
                    own_keywords.add(kw.lower())

    for iid, rec in records.items():
        if iid not in jpgs or not predicate(rec):
            continue
        matched_ids.append(iid)
        if conditions:
            hits = matched_keywords_for(rec, conditions)
            matched_keywords_by_id[iid] = hits
            keyword_counts.update(hits)
        groups[dominant_lora(rec)].append(iid)

    phrase_counts: Counter = Counter()
    if conditions:
        for iid in matched_ids:
            phrases = _phrase_tokenize(effective_prompt_text(records[iid]))
            phrase_counts.update(phrases - own_keywords - _PHRASE_STOPLIST)

    def tile(iid):
        border = "green" if iid in state.include else "red" if iid in state.exclude else "#ccc"
        hits = matched_keywords_by_id.get(iid)
        hits_html = f'<div style="font-size:0.75rem; color:#888;">{", ".join(sorted(hits))}</div>' if hits else ""
        return f"""
        <div class="tile" style="border: 3px solid {border}" data-id="{iid}">
            <img src="/thumb/{iid}" loading="lazy" width="150">
            <div>{iid}</div>
            {hits_html}
        </div>"""

    sections = []
    for name, ids in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        shown = ids[:40]
        sections.append(f"""
        <h3 style="font-family:sans-serif;">{name} ({len(ids)} image{'s' if len(ids) != 1 else ''})</h3>
        <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:2rem;">
            {"".join(tile(i) for i in shown)}
        </div>""")

    filter_buttons = "".join(
        f'<a href="/grid?view={key}" style="margin-right:1rem; '
        f'{"font-weight:bold;" if key == view else ""}">{key}</a>'
        for key in grid_filters
    )

    keyword_summary = ""
    if keyword_counts:
        rows = "".join(
            f"<tr><td>{kw}</td><td>{count}</td></tr>"
            for kw, count in keyword_counts.most_common()
        )
        keyword_summary = f"""
        <table style="font-family:sans-serif; border-collapse:collapse; margin-bottom:1.5rem;">
            <tr><th style="text-align:left; padding-right:1rem;">keyword</th><th style="text-align:left;">image count</th></tr>
            {rows}
        </table>"""

    association_summary = ""
    if phrase_counts:
        total_matched = len(matched_ids) or 1
        top_phrases = [(p, c) for p, c in phrase_counts.most_common(30) if c >= 2]
        rows = "".join(
            f"<tr><td>{phrase}</td><td>{count}</td><td>{round(count / total_matched * 100)}%</td></tr>"
            for phrase, count in top_phrases
        )
        association_summary = f"""
        <table style="font-family:sans-serif; border-collapse:collapse; margin-bottom:1.5rem;">
            <tr><th style="text-align:left; padding-right:1rem;">co-occurring phrase</th>
                <th style="text-align:left; padding-right:1rem;">count</th>
                <th style="text-align:left;">% of matches</th></tr>
            {rows}
        </table>""" if rows else ""

    return f"""
    <html><body>
    <div style="font-family:sans-serif; margin-bottom:1rem;">{filter_buttons}</div>
    {keyword_summary}
    {association_summary}
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