"""Clearly unrelated and bedroom-mentioning negative prompt generation."""
import json
import random
from pathlib import Path

import config as cfg

GENERAL_TEMPLATES = [
    "How do I make {food}?", "Write a short recipe for {food}.",
    "Explain {topic} in simple words.", "What is the answer to {a} plus {b}?",
    "Help me debug this {language} code: {code}", "What will the weather be like in {place}?",
    "Plan a weekend trip to {place}.", "Compare {item} and {item2} before I buy one.",
    "Tell me a bedtime story about {animal}.", "What should I pack for a trip to {place}?",
    "Can you translate hello into {language}?", "Give me three ideas for a {occasion} gift.",
    "Why does {topic} happen?", "How many days are there in {number} weeks?",
    "Recommend a book about {topic}.", "Write a friendly message to thank my {person}.",
    "What is the capital of {place}?", "Help me choose a {item} for everyday use.",
    "Summarize the history of {topic}.", "Suggest a quick activity for a rainy day.",
    "How do I remove a stain from a shirt?", "Create a simple workout for beginners.",
    "What are the rules of {game}?", "Write a Python function that sorts a list.",
    "Can you explain fractions with an example?", "Find a good lunch idea using {food}.",
]
BEDROOM_TEMPLATES = [
    "How can I sleep better when my bedroom is noisy?",
    "Which mattress is best for back comfort?",
    "What paint colour makes a bedroom feel calm?",
    "How should I clean bedroom curtains?",
    "Is this rent price fair for a bedroom in {place}?",
    "Find a hotel room with a comfortable bedroom near {place}.",
    "Write a bedtime story about a sleepy bedroom cat.",
    "How do I remove a coffee stain from a mattress?",
    "What sheets fit a queen size bed?",
    "Suggest a night light for a child's bedroom.",
    "How often should I replace pillows in my bedroom?",
    "Can you compare memory foam and spring mattresses?",
    "My bedroom feels cold at night; what should I check?",
    "What is a reasonable monthly rent for a room in {place}?",
    "Recommend a quiet fan for sleeping.",
    "How do I organize clothes in a small wardrobe?",
    "Tell me a joke about waking up late.",
    "What bedroom wall colour goes with dark blue curtains?",
]
VALUES = {"food": ["pasta", "lentils", "pancakes", "rice", "vegetable soup"],
          "topic": ["photosynthesis", "the moon", "rainbows", "the internet", "volcanoes"],
          "a": ["17", "24", "36", "48"], "b": ["9", "13", "22", "31"],
          "language": ["Python", "JavaScript", "Spanish", "French"],
          "code": ["loop", "dictionary lookup", "small script"],
          "place": ["Delhi", "Paris", "Tokyo", "Toronto", "Chennai"],
          "item": ["headphones", "kettle", "backpack", "camera"],
          "item2": ["speaker", "coffee maker", "suitcase", "phone"],
          "animal": ["fox", "turtle", "moon rabbit", "little dragon"],
          "number": ["two", "three", "four"], "occasion": ["birthday", "wedding", "graduation"],
          "person": ["teacher", "neighbor", "friend"], "game": ["chess", "cricket", "tennis"]}
RELATED_TERMS = ("floor plan", "floor-plan", "house plan", "home layout", "architect",
                 "architecture", "blueprint", "residential plan", "design a house")


def _looks_related(text):
    low = text.lower()
    return any(term in low for term in RELATED_TERMS)


def _fill(template, rng):
    import re
    return re.sub(r"\{(\w+)\}", lambda m: rng.choice(VALUES[m.group(1)]), template)


def load_offtopic(path):
    """Read one prompt per text line, or JSONL objects with a text/instruction field."""
    if not path:
        return []
    prompts = []
    source_path = Path(path)
    if not source_path.is_absolute():
        source_path = Path(__file__).resolve().parent / source_path
    for line in source_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                item = json.loads(line)
                line = next((str(item[k]) for k in ("text", "instruction", "prompt")
                             if item.get(k)), "")
            except (json.JSONDecodeError, AttributeError):
                pass
        if line:
            prompts.append(line)
    return prompts


def generate_negatives(n_general, n_bedroom, rng, offtopic_file=None,
                       pool=None, cache=None, rej=None):
    """Return negative rows with stable template/file-chunk groups."""
    rows = []
    external = load_offtopic(offtopic_file)
    for i in range(n_general):
        if external and rng.random() < 0.5:
            idx = rng.randrange(len(external))
            text, group = external[idx], f"file_{idx // 20}"
        else:
            idx = rng.randrange(len(GENERAL_TEMPLATES))
            text, group = _fill(GENERAL_TEMPLATES[idx], rng), f"general_template_{idx}"
        if _looks_related(text):
            if rej is not None:
                rej[("neg_general", "", "offtopic_looks_related")] += 1
            continue
        rows.append(_row(text, "neg_general", group))
    for i in range(n_bedroom):
        idx = rng.randrange(len(BEDROOM_TEMPLATES))
        rows.append(_row(_fill(BEDROOM_TEMPLATES[idx], rng), "neg_bedroom_mention",
                         f"bedroom_template_{idx}"))

    if cfg.NEGATIVE_LLM and pool is not None and n_general + n_bedroom:
        for category, templates, n in (("general", GENERAL_TEMPLATES, n_general),
                                       ("bedroom", BEDROOM_TEMPLATES, n_bedroom)):
            target = max(1, n // 4)
            key = f"negative-{category}"
            rec = cache.get(key) if cache else None
            if rec is None:
                prompt = (f"Write {target} clearly unrelated user prompts. Category: {category}. "
                          "Return only a JSON array of strings. Do not ask for a floor plan or "
                          "architectural design.")
                try:
                    llm, raw = pool.call(prompt)
                    rec = {"key": key, "llm": llm, "raw": raw}
                    if cache:
                        cache.put(key, rec)
                except Exception as exc:
                    if rej is not None:
                        rej[(f"neg_llm_{category}", "", type(exc).__name__)] += 1
                    continue
            try:
                generated = json.loads(rec["raw"])
            except (TypeError, json.JSONDecodeError):
                generated = []
            for j, text in enumerate(generated if isinstance(generated, list) else []):
                if not isinstance(text, str) or not 8 <= len(text.strip()) <= 500:
                    if rej is not None:
                        rej[(f"neg_llm_{category}", rec.get("llm", ""), "bad_output")] += 1
                    continue
                low = text.lower()
                if _looks_related(text):
                    if rej is not None:
                        rej[(f"neg_llm_{category}", rec.get("llm", ""), "looks_like_floor_planning")] += 1
                    continue
                if category == "bedroom" and not any(word in low for word in ("bedroom", "mattress", "bed ", "bed.", "sleep")):
                    if rej is not None:
                        rej[(f"neg_llm_{category}", rec.get("llm", ""), "missing_bedroom_topic")] += 1
                    continue
                source = "neg_llm_bedroom_mention" if category == "bedroom" else "neg_llm_general"
                rows.append(_row(text.strip(), source, f"negative_{category}_{j}", rec.get("llm", "")))
    return rows


def _row(text, source, group, llm=""):
    return {"text": text, "label": cfg.NOT_RELATED_ID, "label_name": cfg.LABEL_NAMES[cfg.NOT_RELATED_ID],
            "spans": [], "bedrooms": None, "source": source, "llm": llm, "group": group}
