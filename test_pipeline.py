"""Offline self-checks. Run with: python test_pipeline.py"""
import random
from collections import defaultdict

import config as cfg
from common import (add_noise, bedroom_label, extract_spans, group_split,
                    mark_spans, parse_quantity, validate_row)
from generators import (check_rewrite, frame_problem, rewrite_prompt,
                        sample_overflow_row, sample_row, sample_row_for_class)
from negatives import generate_negatives


def _floor_row(text, bedrooms, spans):
    label = bedroom_label(bedrooms)
    return {"text": text, "spans": spans, "bedrooms": bedrooms,
            "label": label, "label_name": cfg.LABEL_NAMES[label]}


def test_parse_quantity():
    cases = {"2 kitchens": 2, "one master bedroom": 1, "a guest room": 1,
             "an ensuite bedroom": 1, "master bedroom": 1, "2BHK": 2,
             "3 BR": 3, "4-BHK": 4, "bedrooms: 3": 3,
             "Two Guest Rooms": 2, "TWO BEDROOMS": 2, "five bedrooms": 5}
    for phrase, expected in cases.items():
        assert parse_quantity(phrase) == expected


def test_hand_frames_are_clean():
    for frame in cfg.HAND_FRAMES:
        assert frame_problem(frame) is None


def test_generated_rows_are_valid():
    rng = random.Random(0)
    for _ in range(5000):
        row = sample_row(rng.choice(cfg.HAND_FRAMES), rng)
        bedrooms = row["bedrooms"]
        label = bedroom_label(bedrooms)
        if label is None:
            continue
        row.update(label=label, label_name=cfg.LABEL_NAMES[label])
        assert validate_row(row) is None, (validate_row(row), row)


def test_the_example_from_the_brief():
    text = ("Create an image of a floor plan which consists of 1 master bedroom, "
            "1 guest room, 1 kitchen, 1 bathroom, 1 living room and 1 dining room.")
    a, b = text.index("1 master bedroom"), text.index("1 guest room")
    row = _floor_row(text, 2, [[a, a + len("1 master bedroom")],
                               [b, b + len("1 guest room")]])
    assert validate_row(row) is None
    assert row["label"] == 2


def test_markers_round_trip():
    rng = random.Random(1)
    for _ in range(500):
        row = sample_row(rng.choice(cfg.HAND_FRAMES), rng)
        text, spans = extract_spans(mark_spans(row["text"], row["spans"]))
        assert text == row["text"] and spans == row["spans"]


def test_noise_keeps_labels_valid():
    rng = random.Random(2)
    changed = 0
    for _ in range(3000):
        row = sample_row(rng.choice(cfg.HAND_FRAMES), rng)
        label = bedroom_label(row["bedrooms"])
        row.update(label=label, label_name=cfg.LABEL_NAMES[label])
        noisy = add_noise(row, rng, typo_p=0.4)
        assert validate_row(noisy, check_keywords=False) is None
        changed += noisy["text"] != row["text"]
    assert changed > 1000


def test_rewrite_validator_rejects_bad_outputs():
    text = "Plan with 2 master bedrooms and a kitchen."
    start = text.index("2 master bedrooms")
    base = _floor_row(text, 2, [[start, start + len("2 master bedrooms")]])
    marked = mark_spans(base["text"], base["spans"])
    good = "pls make me a plan: " + marked.split("Plan with ", 1)[1]
    assert check_rewrite(base, good)[1] is None
    assert check_rewrite(base, good.replace("]]", ""))[1] == "bad_markers"
    assert check_rewrite(base, good.replace("2 master", "3 master"))[1] == "count_mismatch"
    assert check_rewrite(base, "no markers at all in this long enough text")[1] == "marker_count_changed"
    assert check_rewrite(base, "I want [[2 master bedrooms]] and 2 bedrooms too.")[1] == "bedroom_word_outside_spans"
    assert "TEXT TO REWRITE:" in rewrite_prompt("casual", marked)


def test_group_split_has_no_leakage():
    rng = random.Random(3)
    rows = []
    for source, n_groups in [("template", 30), ("llm_frame", 200), ("llm_full", 300)]:
        for group in range(n_groups):
            for _ in range(rng.randint(1, 15)):
                label = rng.randrange(7)
                rows.append({"text": f"{source} {group} {len(rows)}", "spans": [],
                             "bedrooms": None, "label": label, "label_name": cfg.LABEL_NAMES[label],
                             "source": source, "llm": "", "group": f"{source}_{group}"})
    train, val, test = group_split(rows, 0.1, 0.1, seed=0)
    seen = defaultdict(set)
    for name, split in (("train", train), ("val", val), ("test", test)):
        for row in split:
            seen[row["group"]].add(name)
    assert all(len(splits) == 1 for splits in seen.values())
    assert len(train) + len(val) + len(test) == len(rows)
    for split in (val, test):
        assert {row["source"] for row in split} == {"template", "llm_frame", "llm_full"}
        assert {row["label"] for row in split} == set(range(7))


def test_label_mapping_and_overflow_policy():
    assert [bedroom_label(n) for n in range(6)] == list(range(6))
    previous = cfg.OVERFLOW_POLICY
    try:
        cfg.OVERFLOW_POLICY = "fold"
        assert bedroom_label(6) == 5 and bedroom_label(12) == 5
        overflow = sample_overflow_row(cfg.HAND_FRAMES[0], random.Random(44))
        overflow.update(label=bedroom_label(overflow["bedrooms"]),
                        label_name=cfg.LABEL_NAMES[5])
        assert overflow["bedrooms"] == 6 and validate_row(overflow) is None
        cfg.OVERFLOW_POLICY = "drop"
        assert bedroom_label(6) is None
    finally:
        cfg.OVERFLOW_POLICY = previous


def test_negative_rows_validate_and_keep_class_six():
    rows = generate_negatives(40, 40, random.Random(7))
    assert rows
    for row in rows:
        assert row["label"] == 6 and row["label_name"] == cfg.LABEL_NAMES[6]
        assert row["spans"] == [] and row["bedrooms"] is None
        assert validate_row(row) is None
    hard = next(row for row in rows if row["source"] == "neg_bedroom_mention")
    assert "bed" in hard["text"].lower()
    assert validate_row(hard) is None


def test_class_zero_examples_validate():
    rng = random.Random(9)
    for _ in range(100):
        row = sample_row_for_class(rng.choice(cfg.HAND_FRAMES), 0, rng)
        row.update(label=0, label_name=cfg.LABEL_NAMES[0])
        assert row["bedrooms"] == 0 and row["spans"] == []
        assert validate_row(row) is None


if __name__ == "__main__":
    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\nall {len(tests)} checks passed")
