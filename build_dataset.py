"""Build a balanced seven-class bedroom-count classification dataset."""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import config as cfg
from common import add_noise, bedroom_label, dedupe, group_split, validate_row
from generators import (generate_frames, generate_rewrites, sample_overflow_row,
                        sample_row_for_class)
from llm import JsonlCache, LLMPool, MockPool
from negatives import generate_negatives


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-total", type=int, default=20000, help="final target row count before split")
    p.add_argument("--out-dir", default="out")
    p.add_argument("--ratios", default=None, help="template,llm_frame,llm_full proportions")
    p.add_argument("--no-llm", action="store_true", help="use hand-written templates only")
    p.add_argument("--mock-llm", action="store_true", help="offline stand-in for LLM calls")
    p.add_argument("--negative-llm", action="store_true", help="also request cached LLM negatives")
    p.add_argument("--offtopic-file", default=None, help="local TXT (one prompt per line) or JSONL source")
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--workers", type=int, default=cfg.WORKERS)
    p.add_argument("--noise-prob", type=float, default=cfg.NOISE_PROB)
    p.add_argument("--val-frac", type=float, default=cfg.VAL_FRAC)
    p.add_argument("--test-frac", type=float, default=cfg.TEST_FRAC)
    p.add_argument("--holdout-llms", default=",".join(cfg.HOLDOUT_LLMS))
    return p.parse_args()


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _target_counts(n):
    weights = cfg.TARGET_CLASS_DISTRIBUTION
    total = sum(weights.values())
    raw = {k: n * v / total for k, v in weights.items()}
    counts = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda x: raw[x] - counts[x], reverse=True)[:n - sum(counts.values())]:
        counts[k] += 1
    return counts


def _decorate_floor(row):
    label = bedroom_label(row["bedrooms"])
    if label is None:
        return None
    row.update(label=label, label_name=cfg.LABEL_NAMES[label])
    return row


def _keep_floor(candidates, row, rej):
    decorated = _decorate_floor(row)
    if decorated is None:
        rej[(row.get("source", ""), row.get("llm", ""), "overflow_dropped")] += 1
    else:
        candidates.append(decorated)


def _class_hist(rows):
    c = Counter(r["label"] for r in rows)
    return {str(i): c.get(i, 0) for i in range(len(cfg.LABEL_NAMES))}


def _balanced_select(candidates, targets, rng, rej):
    by_label = defaultdict(list)
    for row in dedupe(candidates):
        by_label[row["label"]].append(row)
    selected = []
    for label, target in targets.items():
        rows = by_label[label]
        if len(rows) < target:
            rej[("balance", "", f"class_{label}_shortfall")] += target - len(rows)
            print(f"[warn] class {label}: requested {target}, only {len(rows)} unique rows available")
            target = len(rows)
        rng.shuffle(rows)
        selected.extend(rows[:target])
    return selected


def main():
    a = parse_args()
    if a.n_total <= 0:
        raise ValueError("--n-total must be positive")
    if not 0 <= a.val_frac < 1 or not 0 <= a.test_frac < 1 or a.val_frac + a.test_frac >= 1:
        raise ValueError("--val-frac and --test-frac must be non-negative and sum to less than 1")
    if not 0 <= a.noise_prob <= 1:
        raise ValueError("--noise-prob must be between 0 and 1")
    if a.workers < 1:
        raise ValueError("--workers must be at least 1")
    if cfg.OVERFLOW_POLICY not in {"fold", "drop"}:
        raise ValueError("OVERFLOW_POLICY must be 'fold' or 'drop'")
    out = Path(a.out_dir)
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    targets = _target_counts(a.n_total)
    ratios = ([float(x) for x in a.ratios.split(",")] if a.ratios else
              [cfg.RATIOS["template"], cfg.RATIOS["llm_frame"], cfg.RATIOS["llm_full"]])
    if len(ratios) != 3 or any(x < 0 for x in ratios) or sum(ratios) <= 0:
        raise ValueError("--ratios must contain three non-negative values with positive sum")
    ratio_total = sum(ratios)
    ratios = [x / ratio_total for x in ratios]
    if a.no_llm:
        ratios = [1.0, 0.0, 0.0]

    needs_llm = bool(ratios[1] or ratios[2] or a.negative_llm or cfg.NEGATIVE_LLM)
    pool = (MockPool(a.seed) if a.mock_llm else LLMPool(cfg.MODELS)) if needs_llm else None
    cache_dir = out / ("cache_mock" if a.mock_llm else "cache")
    frame_cache = JsonlCache(cache_dir / "frames.jsonl")
    rewrite_cache = JsonlCache(cache_dir / "rewrites.jsonl")
    negative_cache = JsonlCache(cache_dir / "negatives.jsonl")
    rej = Counter()
    candidates = []

    # Create extra examples, then select exactly the configured class targets.
    for label in range(cfg.MAX_BEDROOM_CLASS + 1):
        target = targets.get(label, 0)
        candidate_count = max(target, target * cfg.OVERSAMPLE_FACTOR)
        if label == 0 or ratios[1] == 0 and ratios[2] == 0:
            source_counts = [candidate_count, 0, 0]
        else:
            source_counts = [int(candidate_count * r) for r in ratios]
            source_counts[2] = candidate_count - source_counts[0] - source_counts[1]
        for i in range(source_counts[0]):
            frame_id = rng.randrange(len(cfg.HAND_FRAMES))
            frame = cfg.HAND_FRAMES[frame_id]
            row = (sample_overflow_row(frame, rng)
                   if label == 5 and cfg.OVERFLOW_POLICY == "fold" and i % 10 == 0
                   else sample_row_for_class(frame, label, rng))
            row.update(source="template", llm="", group=f"tpl_{frame_id}")
            _keep_floor(candidates, row, rej)
        if source_counts[1]:
            frames = generate_frames(pool, frame_cache,
                                     max(40, source_counts[1] // cfg.ROWS_PER_FRAME), a.workers, rej)
            if not frames:
                raise RuntimeError("No valid LLM frames were produced; check the provider/model configuration.")
            write_jsonl(out / "frames.jsonl", frames)
            for _ in range(source_counts[1]):
                fr = rng.choice(frames)
                row = (sample_overflow_row(fr["frame"], rng)
                       if label == 5 and cfg.OVERFLOW_POLICY == "fold" and _ % 10 == 0
                       else sample_row_for_class(fr["frame"], label, rng))
                row.update(source="llm_frame", llm=fr["llm"], group=f"frame_{fr['id']}")
                _keep_floor(candidates, row, rej)
        if source_counts[2]:
            rewrites = generate_rewrites(pool, rewrite_cache, source_counts[2], a.workers,
                                         a.seed + label, rej, target_class=label)
            for row in rewrites:
                _keep_floor(candidates, row, rej)

    neg_count = targets.get(cfg.NOT_RELATED_ID, 0)
    source_file = a.offtopic_file
    if source_file and not Path(source_file).is_absolute():
        source_file = str(Path(__file__).resolve().parent / source_file)
    neg_candidates = generate_negatives(neg_count * cfg.OVERSAMPLE_FACTOR,
                                        neg_count * cfg.OVERSAMPLE_FACTOR, rng,
                                        source_file, pool,
                                        negative_cache if (a.negative_llm or cfg.NEGATIVE_LLM) else None,
                                        rej)
    candidates.extend(neg_candidates)

    # Validate before noise. Fold/drop policy is applied before balancing.
    valid = []
    for row in candidates:
        reason = validate_row(row, check_keywords=row["label"] != cfg.NOT_RELATED_ID)
        if reason:
            rej[(row["source"], row["llm"], reason)] += 1
        else:
            valid.append(row)
    noisy = []
    for row in valid:
        if rng.random() < a.noise_prob:
            changed = add_noise(row, rng, cfg.TYPO_WORD_PROB)
            if validate_row(changed, check_keywords=changed["label"] != cfg.NOT_RELATED_ID) is None:
                row = changed
            else:
                rej[(row["source"], row["llm"], "noise_reverted")] += 1
        noisy.append(row)
    rows = _balanced_select(noisy, targets, rng, rej)
    for row in rows:
        reason = validate_row(row, check_keywords=row["label"] != cfg.NOT_RELATED_ID)
        assert reason is None, f"final validation failed: {reason}: {row}"

    # Optional unseen-model holdout retains complete groups.
    holdout = {x for x in a.holdout_llms.split(",") if x}
    unseen, keep = [], rows
    if holdout:
        by_group = defaultdict(list)
        for row in rows:
            by_group[row["group"]].append(row)
        keep, unseen = [], []
        for group in by_group.values():
            if all(row["llm"] in holdout for row in group):
                unseen.extend(group)
            elif any(row["llm"] in holdout for row in group):
                rej[("holdout", "", "mixed_group_dropped")] += len(group)
            else:
                keep.extend(group)
        rows = keep

    train, val, test = group_split(rows, a.val_frac, a.test_frac, a.seed)
    splits = {"train": train, "val": val, "test": test, "test_unseen_llm": unseen}
    for name, split in splits.items():
        if name != "test_unseen_llm" and split and set(r["label"] for r in split) != set(range(7)):
            missing = sorted(set(range(7)) - {r["label"] for r in split})
            print(f"[warn] {name} split is missing labels {missing}; adjust target size/groups")
    for name in ("train", "val", "test"):
        write_jsonl(out / f"{name}.jsonl", splits[name])
    if unseen:
        write_jsonl(out / "test_unseen_llm.jsonl", unseen)
    stats = {
        "args": vars(a), "label_names": {str(i): n for i, n in enumerate(cfg.LABEL_NAMES)},
        "overflow_policy": cfg.OVERFLOW_POLICY, "target_class_distribution": cfg.TARGET_CLASS_DISTRIBUTION,
        "target_class_counts": {str(k): v for k, v in targets.items()},
        "rows": {k: len(v) for k, v in splits.items()},
        "label_histogram": {k: _class_hist(v) for k, v in splits.items()},
        "rows_by_split_and_source": {k: dict(Counter(r["source"] for r in v)) for k, v in splits.items()},
        "rows_by_llm": dict(Counter(r["llm"] or "none" for s in splits.values() for r in s)),
        "bedroom_histogram": {k: dict(sorted(Counter(str(r["bedrooms"]) for r in v
                                                     if r["bedrooms"] is not None).items()))
                              for k, v in splits.items()},
        "rejections": {"|".join(map(str, k)): v for k, v in sorted(rej.items())},
        "llm_pool": {k: dict(v) for k, v in (pool.stats.items() if pool else [])},
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nWrote dataset to {out}")
    for name, split in splits.items():
        if split:
            print(f"  {name:<16}{len(split):>7} rows   labels={_class_hist(split)}")
    print(f"  rejected/reverted: {sum(rej.values())} (details in stats.json)")


if __name__ == "__main__":
    main()
