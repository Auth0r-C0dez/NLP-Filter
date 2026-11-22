"""The three text sources.

1. Plain templates : hand-written frames + generated room lists   (source="template")
2. Method A        : LLM-written frames + the same room lists      (source="llm_frame")
3. Method B        : LLM rewrites whole sentences, [[ ]] markers   (source="llm_full")

Labels never depend on trusting the LLM: in Method A the LLM never sees the room
list; in Method B every rewrite is re-parsed and rejected unless it still adds up.
"""
import json
import math
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import config as cfg
from common import (BED_KEYWORD_RE, extract_spans, mark_spans, normalize,
                    validate_row)

NUM_BY_INT = {v: k for k, v in cfg.NUMBER_WORDS.items()}
PLACEHOLDER = "{ROOMS}"


# ============================================================================
# Room-list builder (shared by all sources)
# ============================================================================
def weighted_sample(rng, items, weights, k):
    items, weights, out = list(items), list(weights), []
    for _ in range(min(k, len(items))):
        i = rng.choices(range(len(items)), weights=weights)[0]
        out.append(items.pop(i))
        weights.pop(i)
    return out


def _plural(name, n):
    return name + "s" if n != 1 else name


def _article(name):
    return "an" if name[0] in "aeiou" else "a"


def make_part(name, n, rng):
    """'2 kitchens' / 'two kitchens' / 'a kitchen' / 'kitchen'"""
    style = rng.choices(["digit", "word", "article", "bare"], weights=[50, 32, 13, 5])[0]
    if style in ("article", "bare") and n != 1:
        style = rng.choice(["digit", "word"])
    if style == "digit":
        return f"{n} {_plural(name, n)}"
    if style == "word":
        return f"{NUM_BY_INT[n]} {_plural(name, n)}"
    if style == "article":
        return f"{_article(name)} {name}"
    return name


JOIN_STYLES = [("oxford", 30), ("plain", 35), ("amp", 5), ("semi", 8),
               ("plus", 4), ("comma", 8), ("bullets", 10)]


def join_parts(parts, rng):
    """Join phrases into one string and return the (start, end) of each phrase."""
    style = rng.choices([s for s, _ in JOIN_STYLES], weights=[w for _, w in JOIN_STYLES])[0]
    text, offs, n = "", [], len(parts)
    for i, p in enumerate(parts):
        if style == "bullets":
            text += "\n- "
        elif i > 0:
            last = i == n - 1
            if style == "oxford":
                text += " and " if (last and n == 2) else (", and " if last else ", ")
            elif style == "plain":
                text += " and " if last else ", "
            elif style == "amp":
                text += " & " if last else ", "
            elif style == "semi":
                text += "; "
            elif style == "plus":
                text += " + "
            else:
                text += ", "
        offs.append((len(text), len(text) + len(p)))
        text += p
    return text, offs


def build_rooms(rng, target_bedrooms=None):
    """Returns {"text": room list, "spans": [[s, e], ...] (relative), "bedrooms": int}."""
    mode = (rng.choices(["standard", "bhk", "colon", "zero"], weights=[78, 10, 7, 5])[0]
            if target_bedrooms is None else rng.choices(["standard", "bhk", "colon"], weights=[78, 12, 10])[0])
    parts = []  # (text, quantity, is_bedroom)

    others = weighted_sample(rng, cfg.OTHER_ROOMS, [1] * len(cfg.OTHER_ROOMS), rng.randint(1, 5))
    for name in others:
        n = rng.choices([1, 2, 3], weights=[75, 18, 7])[0]
        text = f"{name}s: {n}" if mode == "colon" else make_part(name, n, rng)
        parts.append((text, n, False))

    if mode == "zero":
        if rng.random() < 0.4:
            parts.append(("no bedrooms", 0, False))
    elif mode == "bhk":
        n = (rng.choices([1, 2, 3, 4, 5], weights=[10, 35, 35, 15, 5])[0]
             if target_bedrooms is None else target_bedrooms)
        text = rng.choice([f"{n}BHK", f"{n} BHK", f"{n}bhk", f"{n}-BHK", f"{n} BR", f"{n}BR"])
        parts.append((text, n, True))
    else:
        k = (rng.choices([1, 2, 3, 4], weights=[55, 32, 11, 2])[0]
             if target_bedrooms is None else rng.randint(1, min(4, target_bedrooms)))
        names = [t for t, _ in cfg.BEDROOM_TYPES]
        weights = [w for _, w in cfg.BEDROOM_TYPES]
        selected = weighted_sample(rng, names, weights, k)
        quantities = []
        if target_bedrooms is not None:
            remaining = target_bedrooms
            for j in range(k):
                slots = k - j - 1
                q = remaining if slots == 0 else rng.randint(1, remaining - slots)
                quantities.append(q)
                remaining -= q
        else:
            quantities = [rng.choices([1, 2, 3, 4], weights=[78, 18, 3, 1])[0] for _ in range(k)]
        for name, n in zip(selected, quantities):
            text = f"{name}s: {n}" if mode == "colon" else make_part(name, n, rng)
            parts.append((text, n, True))

    rng.shuffle(parts)
    text, offs = join_parts([p[0] for p in parts], rng)
    spans = [list(o) for o, p in zip(offs, parts) if p[2]]
    bedrooms = sum(p[1] for p in parts if p[2])
    return {"text": text, "spans": spans, "bedrooms": bedrooms}


def fill_frame(frame, rooms):
    assert frame.count(PLACEHOLDER) == 1, "frame needs exactly one {ROOMS}"
    i = frame.index(PLACEHOLDER)
    text = frame.replace(PLACEHOLDER, rooms["text"])
    return text, [[s + i, e + i] for s, e in rooms["spans"]]


def sample_row(frame, rng):
    rooms = build_rooms(rng)
    text, spans = fill_frame(frame, rooms)
    if rng.random() < cfg.DISTRACTOR_PROB:
        if not text.rstrip().endswith((".", "!", "?")):
            text += "."
        text += " " + rng.choice(cfg.DISTRACTORS)
    return {"text": text, "spans": spans, "bedrooms": rooms["bedrooms"]}


def sample_row_for_class(frame, label, rng):
    if label == 0:
        return sample_zero_row(rng)
    # Class 5 examples are generated as 5; the configured fold policy also
    # accepts naturally generated totals above 5 in the unbalanced generator.
    rooms = build_rooms(rng, target_bedrooms=label)
    text, spans = fill_frame(frame, rooms)
    if rng.random() < cfg.DISTRACTOR_PROB:
        if not text.rstrip().endswith((".", "!", "?")):
            text += "."
        text += " " + rng.choice(cfg.DISTRACTORS)
    return {"text": text, "spans": spans, "bedrooms": rooms["bedrooms"]}


def sample_overflow_row(frame, rng):
    """Create a six-bedroom row so the configured fold policy is exercised."""
    rooms = build_rooms(rng, target_bedrooms=6)
    text, spans = fill_frame(frame, rooms)
    if rng.random() < cfg.DISTRACTOR_PROB:
        if not text.rstrip().endswith((".", "!", "?")):
            text += "."
        text += " " + rng.choice(cfg.DISTRACTORS)
    return {"text": text, "spans": spans, "bedrooms": rooms["bedrooms"]}


def sample_zero_row(rng):
    """Create an explicit floor-plan request with no bedrooms."""
    frame = rng.choice(cfg.HAND_FRAMES)
    options = ["a studio apartment", "a home office and a kitchen",
               "a small shop with a counter", "a hall, a kitchen and a bathroom",
               "a laundry room and a storage area", "a workshop with a utility area",
               "a living room, kitchen and bathroom", "a reception area and a washroom",
               "no bedrooms, one kitchen and one bathroom", "a guest lounge and a pantry"]
    text, spans = fill_frame(frame, {"text": rng.choice(options), "spans": [], "bedrooms": 0})
    return {"text": text, "spans": spans, "bedrooms": 0}


# ============================================================================
# Utilities
# ============================================================================
def run_parallel(tasks, fn, workers, desc=""):
    def safe(t):
        try:
            return fn(t)
        except Exception as e:  # network/API errors must not kill the whole run
            print(f"[warn] task failed: {type(e).__name__}: {e}", file=sys.stderr)
            return None

    if workers <= 1:
        return [safe(t) for t in tasks]
    with ThreadPoolExecutor(workers) as ex:
        try:
            from tqdm import tqdm
            return list(tqdm(ex.map(safe, tasks), total=len(tasks), desc=desc))
        except ImportError:
            return list(ex.map(safe, tasks))


def extract_json_list(raw):
    i, j = raw.find("["), raw.rfind("]")
    if i == -1 or j <= i:
        return []
    try:
        data = json.loads(raw[i:j + 1])
    except json.JSONDecodeError:
        return []
    return [x for x in data if isinstance(x, str)] if isinstance(data, list) else []


def clean_llm_text(raw):
    t = raw.strip()
    t = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", t).strip()
    if len(t) > 1 and t[0] == t[-1] and t[0] in "\"'":
        t = t[1:-1].strip()
    return t


# ============================================================================
# Method A: LLM-written frames
# ============================================================================
def frame_prompt(style, n):
    return (
        "You are helping build a dataset of how people ask an AI to generate a "
        "residential floor plan image.\n"
        f"Write {n} different user requests. Style: {style}.\n"
        "Rules:\n"
        f"- Every request must contain the placeholder {PLACEHOLDER} exactly once, "
        "where a list of rooms will be inserted later.\n"
        "- Do NOT mention bedrooms, bed counts, any numbers or digits, and do not "
        "name specific rooms yourself.\n"
        "- Vary sentence structure, length and word choice. Do not reuse openings.\n"
        f"- Return ONLY a JSON array of {n} strings."
    )


def frame_problem(frame):
    if frame.count(PLACEHOLDER) != 1:
        return "placeholder_count"
    if not 15 <= len(frame) <= 300:
        return "bad_length"
    if re.search(r"\d", frame):
        return "contains_digit"
    if BED_KEYWORD_RE.search(frame.replace(PLACEHOLDER, "")) or "[[" in frame:
        return "contains_bedroom_word"
    return None


def generate_frames(pool, cache, target, workers, rej):
    calls = math.ceil(target / cfg.FRAMES_PER_CALL * 1.5)
    keys = [f"frames-{i}" for i in range(calls)]

    def work(key):
        rec = cache.get(key)
        if rec is None:
            style = cfg.STYLES[int(key.split("-")[1]) % len(cfg.STYLES)]
            llm, raw = pool.call(frame_prompt(style, cfg.FRAMES_PER_CALL),
                                 temperature=random.uniform(0.7, 1.1))
            rec = {"key": key, "llm": llm, "style": style, "raw": raw}
            cache.put(key, rec)
        return rec

    recs = run_parallel(keys, work, workers, "frames")
    frames, seen = [], set()
    for rec in recs:
        if not rec:
            continue
        for f in extract_json_list(rec["raw"]):
            f = f.strip()
            reason = frame_problem(f)
            if reason is None and normalize(f) in seen:
                reason = "duplicate_frame"
            if reason:
                rej[("frame", rec["llm"], reason)] += 1
                continue
            seen.add(normalize(f))
            frames.append({"id": len(frames), "frame": f, "llm": rec["llm"], "style": rec["style"]})
    return frames


def method_a_rows(frames, n, rng):
    rows = []
    for _ in range(n):
        f = rng.choice(frames)
        row = sample_row(f["frame"], rng)
        row.update(source="llm_frame", llm=f["llm"], group=f"frame_{f['id']}")
        rows.append(row)
    return rows


def method_a_balanced_rows(frames, counts, rng):
    rows = []
    for label, count in counts.items():
        for _ in range(count):
            f = rng.choice(frames)
            row = sample_row_for_class(f["frame"], label, rng)
            row.update(source="llm_frame", llm=f["llm"], group=f"frame_{f['id']}")
            rows.append(row)
    return rows


# ============================================================================
# Method B: LLM rewrites whole sentences, bedroom phrases wrapped in [[ ]]
# ============================================================================
def rewrite_prompt(style, marked):
    return (
        "Rewrite the floor-plan request below in a different way. "
        f"Style: {style}.\n"
        "Rules:\n"
        "- Some phrases are wrapped in double square brackets, like [[2 master bedrooms]]. "
        "Keep EVERY bracketed phrase, with its double brackets, exactly once. Do not "
        "add, remove, split or merge brackets.\n"
        "- Inside the brackets you may change a digit to a number word (or the reverse) "
        "or change the article, but the quantity and the room type must stay the same.\n"
        "- Do not add new rooms, new mentions of bedrooms, or new numbers outside the "
        "brackets. Keep any other numbers (areas, floors, sizes) unchanged.\n"
        "- Return ONLY the rewritten text, with no quotes and no explanation.\n\n"
        "TEXT TO REWRITE:\n" + marked
    )


def base_row(i, seed):
    """Deterministic base sentence #i (always has at least one bedroom span)."""
    rng = random.Random(f"{seed}-base-{i}")
    row = sample_row(rng.choice(cfg.HAND_FRAMES), rng)
    while not row["spans"]:
        row = sample_row(rng.choice(cfg.HAND_FRAMES), rng)
    return row


def base_row_for_class(i, seed, label):
    rng = random.Random(f"{seed}-base-{i}-{label}")
    return sample_row_for_class(rng.choice(cfg.HAND_FRAMES), label, rng)


def check_rewrite(base, raw):
    """Returns (row, None) if the rewrite is safe, else (None, reason)."""
    text, spans = extract_spans(clean_llm_text(raw))
    if "[" in text or "]" in text:
        return None, "bad_markers"
    if len(spans) != len(base["spans"]):
        return None, "marker_count_changed"
    if normalize(text) == normalize(base["text"]):
        return None, "unchanged"
    if not 10 <= len(text) <= 400:
        return None, "bad_length"
    row = {"text": text, "spans": spans, "bedrooms": base["bedrooms"]}
    reason = validate_row(row)
    return (None, reason) if reason else (row, None)


def generate_rewrites(pool, cache, target, workers, seed, rej, target_class=None):
    accepted, next_base = [], 0
    batch_bases = max(workers * 4, 32)
    max_bases = max(target * 10, 200)

    def work(task):
        b, k = task
        key = f"{b}-{k}" if target_class is None else f"{b}-{k}-c{target_class}"
        rec = cache.get(key)
        if rec is None:
            base = (base_row(b, seed) if target_class is None
                    else base_row_for_class(b, seed, target_class))
            marked = mark_spans(base["text"], base["spans"])
            style = cfg.STYLES[(b + k) % len(cfg.STYLES)]
            llm, raw = pool.call(rewrite_prompt(style, marked),
                                 temperature=random.uniform(0.7, 1.1))
            rec = {"key": key, "llm": llm, "style": style, "raw": raw}
            cache.put(key, rec)
        return rec

    while len(accepted) < target and next_base < max_bases:
        bases = range(next_base, next_base + batch_bases)
        next_base += batch_bases
        tasks = [(b, k) for b in bases for k in range(cfg.REWRITES_PER_BASE)]
        recs = run_parallel(tasks, work, workers, "rewrites")
        if not any(recs):
            raise RuntimeError("Every LLM call in this batch failed. Check API keys / model ids.")
        for (b, k), rec in zip(tasks, recs):
            if not rec:
                continue
            base = (base_row(b, seed) if target_class is None
                    else base_row_for_class(b, seed, target_class))
            row, reason = check_rewrite(base, rec["raw"])
            if reason:
                rej[("llm_full", rec["llm"], reason)] += 1
                continue
            group_id = f"base_{target_class}_{b}" if target_class is not None else f"base_{b}"
            row.update(source="llm_full", llm=rec["llm"], group=group_id)
            accepted.append(row)
    return accepted[:target]
