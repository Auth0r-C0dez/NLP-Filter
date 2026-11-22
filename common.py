"""Shared helpers: quantity parsing, span validation, [[ ]] markers, noise, dedupe, split."""
import random
import re
from collections import Counter, defaultdict

import config as cfg
from config import BED_KEYWORD_RE, NUMBER_WORDS


def bedroom_label(bedrooms):
    if cfg.OVERFLOW_POLICY not in {"fold", "drop"}:
        raise ValueError("OVERFLOW_POLICY must be 'fold' or 'drop'")
    if bedrooms > cfg.MAX_BEDROOM_CLASS and cfg.OVERFLOW_POLICY == "drop":
        return None
    return min(bedrooms, cfg.MAX_BEDROOM_CLASS)

# ----------------------------------------------------------------------------
# Quantity parsing: "2 kitchens" -> 2, "one master bedroom" -> 1, "2BHK" -> 2,
# "a guest room" -> 1, bare "master bedroom" -> 1.
# ----------------------------------------------------------------------------
_DIGITS = re.compile(r"\d+")
_WORDS = re.compile(r"[a-z]+")


def parse_quantity(phrase):
    p = phrase.lower()
    m = _DIGITS.search(p)
    if m:
        return int(m.group())
    for w in _WORDS.findall(p):
        if w in NUMBER_WORDS:
            return NUMBER_WORDS[w]
    return 1  # "a guest room", "master bedroom" (no quantity) -> 1


# ----------------------------------------------------------------------------
# Row validation. Returns None if OK, otherwise a short reason string.
# A row is: {"text", "spans": [[start, end], ...], "bedrooms": int, ...}
# ----------------------------------------------------------------------------
def validate_row(row, check_keywords=True):
    text, spans = row["text"], row["spans"]
    label = row.get("label")
    if label == cfg.NOT_RELATED_ID:
        if spans != [] or row.get("bedrooms") is not None:
            return "invalid_negative_metadata"
        if row.get("label_name", cfg.LABEL_NAMES[label]) != cfg.LABEL_NAMES[label]:
            return "label_name_mismatch"
        return None
    if label is not None:
        if row.get("bedrooms") is None:
            return "missing_bedroom_count"
        expected = bedroom_label(row["bedrooms"])
        if expected is None or label != expected:
            return "label_mismatch"
        if row.get("label_name", cfg.LABEL_NAMES[label]) != cfg.LABEL_NAMES[label]:
            return "label_name_mismatch"
    if row.get("bedrooms") == 0 and spans:
        return "zero_bedrooms_has_spans"
    prev = 0
    outside = []
    for s, e in spans:
        if not (0 <= prev <= s < e <= len(text)):
            return "bad_span_bounds"
        outside.append(text[prev:s])
        prev = e
        if check_keywords and not BED_KEYWORD_RE.search(text[s:e]):
            return "span_without_bedroom_word"
    outside.append(text[prev:])
    if sum(parse_quantity(text[s:e]) for s, e in spans) != row["bedrooms"]:
        return "count_mismatch"
    if check_keywords and spans and BED_KEYWORD_RE.search("".join(outside)):
        return "bedroom_word_outside_spans"
    return None


# ----------------------------------------------------------------------------
# [[ ]] markers (Method B)
# ----------------------------------------------------------------------------
_MARK = re.compile(r"\[\[(.+?)\]\]", re.S)


def mark_spans(text, spans):
    out, prev = [], 0
    for s, e in spans:
        out += [text[prev:s], "[[", text[s:e], "]]"]
        prev = e
    out.append(text[prev:])
    return "".join(out)


def extract_spans(marked):
    """Remove [[ ]] markers and return (clean_text, spans)."""
    parts, spans, pos, length = [], [], 0, 0
    for m in _MARK.finditer(marked):
        pre = marked[pos:m.start()]
        parts.append(pre)
        length += len(pre)
        inner = m.group(1)
        spans.append([length, length + len(inner)])
        parts.append(inner)
        length += len(inner)
        pos = m.end()
    parts.append(marked[pos:])
    return "".join(parts), spans


# ----------------------------------------------------------------------------
# Noise (applied AFTER LLM generation so the LLM cannot "fix" it).
# Quantity tokens (digits, number words) and short words are never touched,
# so labels stay correct; spans are recomputed segment by segment.
# ----------------------------------------------------------------------------
_WORD = re.compile(r"[A-Za-z']+")


def _typo(word, rng):
    i = rng.randrange(1, len(word) - 1)
    op = rng.choice(["swap", "drop", "double"])
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "drop":
        return word[:i] + word[i + 1:]
    return word[:i] + word[i] + word[i:]


def _mutate(segment, rng, p):
    def sub(m):
        w = m.group()
        if len(w) >= 5 and w.lower() not in NUMBER_WORDS and rng.random() < p:
            return _typo(w, rng)
        return w
    return _WORD.sub(sub, segment)


def add_noise(row, rng, typo_p=0.15):
    text, spans = row["text"], row["spans"]
    out, new_spans, cursor, prev = [], [], 0, 0
    for s, e in spans:
        pre = _mutate(text[prev:s], rng, typo_p)
        out.append(pre)
        cursor += len(pre)
        inner = _mutate(text[s:e], rng, typo_p)
        new_spans.append([cursor, cursor + len(inner)])
        out.append(inner)
        cursor += len(inner)
        prev = e
    out.append(_mutate(text[prev:], rng, typo_p))
    new_text = "".join(out)

    # casing noise (skip if it would change the length, e.g. non-ASCII letters)
    r = rng.random()
    cased = new_text.lower() if r < 0.15 else new_text.upper() if r < 0.20 else new_text
    if len(cased) == len(new_text):
        new_text = cased

    # sometimes drop the final period (only if no span touches it)
    if (new_text.endswith(".") and rng.random() < 0.25
            and (not new_spans or new_spans[-1][1] < len(new_text))):
        new_text = new_text[:-1]

    return {**row, "text": new_text, "spans": new_spans}


# ----------------------------------------------------------------------------
# Dedupe and split
# ----------------------------------------------------------------------------
def normalize(text):
    return " ".join(text.lower().split())


def dedupe(rows):
    seen, out = set(), []
    for r in rows:
        k = normalize(r["text"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def _split_one_source(rows, val_frac, test_frac, rng):
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    ids = sorted(groups)
    rng.shuffle(ids)
    n = len(rows)
    t_target, v_target = int(n * test_frac), int(n * val_frac)
    train, val, test = [], [], []
    for gid in ids:
        g = groups[gid]
        if len(test) < t_target:
            test.extend(g)
        elif len(val) < v_target:
            val.extend(g)
        else:
            train.extend(g)
    return train, val, test


def group_split(rows, val_frac, test_frac, seed):
    """Greedily stratify labels while keeping every group wholly in one split."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    names = ("train", "val", "test")
    fracs = (max(0.0, 1.0 - val_frac - test_frac), val_frac, test_frac)
    features = sorted({("label", r.get("label", 0)) for r in rows} |
                      {("source", r.get("source", "")) for r in rows})
    totals = {feature: sum((feature[0] == "label" and r.get("label", 0) == feature[1]) or
                           (feature[0] == "source" and r.get("source", "") == feature[1])
                           for r in rows) for feature in features}
    target = {name: {feature: totals[feature] * frac for feature in features}
              for name, frac in zip(names, fracs)}
    assigned = {name: {feature: 0 for feature in features} for name in names}
    splits = {name: [] for name in names}
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: (-len(item[1]), -max(
        (1 / max(1, totals.get(r.get("label", 0), 1)) for r in item[1]), default=0)))
    for _, group in items:
        counts = Counter(("label", r.get("label", 0)) for r in group)
        counts.update(Counter(("source", r.get("source", "")) for r in group))
        choices = []
        for name in names:
            # Minimize the change in normalized squared error, so an oversized
            # group is discouraged from overshooting a small validation target.
            cost = 0.0
            for feature, count in counts.items():
                goal = max(1.0, target[name][feature])
                before = (assigned[name][feature] - goal) / goal
                after = (assigned[name][feature] + count - goal) / goal
                cost += after * after - before * before
            choices.append((cost, rng.random(), name))
        name = min(choices)[2]
        splits[name].extend(group)
        for label, count in counts.items():
            assigned[name][label] += count
    train, val, test = (splits[n] for n in names)
    for split in (train, val, test):
        rng.shuffle(split)
    return train, val, test
