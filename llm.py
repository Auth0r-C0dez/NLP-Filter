"""LLM access for offline data generation.

LLMPool    - rotates between models on OpenRouter and NVIDIA NIM (OpenAI-compatible APIs)
MockPool   - fake LLM for testing the pipeline without network or API keys
JsonlCache - append-only cache so a crashed run can resume without repeating calls
"""
import json
import os
import random
import re
import threading
import time
from collections import defaultdict
from pathlib import Path

PROVIDERS = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "nim": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
}


class JsonlCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {}
        self.lock = threading.Lock()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec

    def get(self, key):
        return self.data.get(key)

    def put(self, key, rec):
        with self.lock:
            self.data[key] = rec
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class LLMPool:
    """Picks a random healthy model per call; retries with backoff; drops models
    that fail many times in a row. Returns (model_id, text)."""

    def __init__(self, models, retries=3, dead_after=8):
        if not models:
            raise ValueError(
                "config.MODELS is empty. Add model ids (see config.py) or run with "
                "--mock-llm / --no-llm."
            )
        from openai import OpenAI  # imported lazily: not needed for --mock-llm

        self.models = [{**m, "id": f"{m['provider']}:{m['model']}"} for m in models]
        self.retries, self.dead_after = retries, dead_after
        self.clients, self.dead = {}, set()
        self.stats = defaultdict(lambda: {"calls": 0, "errors": 0, "streak": 0})
        self.lock = threading.Lock()
        for m in self.models:
            p = m["provider"]
            if p in self.clients:
                continue
            if p not in PROVIDERS:
                raise ValueError(f"Unknown provider '{p}'. Use one of {list(PROVIDERS)}")
            base_url, env = PROVIDERS[p]
            key = os.environ.get(env)
            if not key:
                raise RuntimeError(f"Set the {env} environment variable.")
            self.clients[p] = OpenAI(base_url=base_url, api_key=key, timeout=90)

    def call(self, prompt, temperature=0.9):
        tried = set()
        while True:
            cands = [m for m in self.models
                     if m["id"] not in self.dead and m["id"] not in tried]
            if not cands:
                raise RuntimeError("All models failed or were disabled.")
            m = random.choice(cands)
            tried.add(m["id"])
            for attempt in range(self.retries):
                with self.lock:
                    self.stats[m["id"]]["calls"] += 1
                try:
                    r = self.clients[m["provider"]].chat.completions.create(
                        model=m["model"],
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=1500,
                    )
                    text = r.choices[0].message.content or ""
                    with self.lock:
                        self.stats[m["id"]]["streak"] = 0
                    return m["id"], text
                except Exception:
                    with self.lock:
                        st = self.stats[m["id"]]
                        st["errors"] += 1
                        st["streak"] += 1
                        if st["streak"] >= self.dead_after:
                            self.dead.add(m["id"])
                    time.sleep(2 ** attempt + random.random())


class MockPool:
    """Offline stand-in for LLMPool. Produces plausible frames and rewrites, and
    deliberately breaks ~14% of rewrites so the validators get exercised."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.models = ["mock:model-a", "mock:model-b", "mock:model-c"]
        self.stats = defaultdict(lambda: {"calls": 0, "errors": 0, "streak": 0})
        self.lock = threading.Lock()

    def call(self, prompt, temperature=0.9):
        with self.lock:
            llm = self.rng.choice(self.models)
            self.stats[llm]["calls"] += 1
            if "clearly unrelated user prompts" in prompt:
                return llm, json.dumps(["How do I make vegetable soup?", "Explain the moon phases.",
                                        "Which mattress is best for back comfort?",
                                        "How can I sleep better in a noisy bedroom?"])
            if "TEXT TO REWRITE:" in prompt:
                return llm, self._rewrite(prompt)
            return llm, self._frames()

    def _frames(self):
        openers = [
            "Make a ADJ floor plan that has {ROOMS}",
            "I'd like a ADJ home layout with {ROOMS}",
            "Draw a ADJ plan including {ROOMS}",
            "Please generate a ADJ residential floor plan: {ROOMS}",
            "ADJ house plan needed, with {ROOMS}",
            "Can you sketch a ADJ layout featuring {ROOMS}?",
        ]
        tails = ["", ".", " Thanks!", " Keep it simple.", " Make it clean."]
        adjs = ["modern", "compact", "spacious", "cozy", "traditional", "minimal"]
        frames = [
            self.rng.choice(openers).replace("ADJ", self.rng.choice(adjs))
            + self.rng.choice(tails)
            for _ in range(12)
        ]
        return json.dumps(frames)

    def _rewrite(self, prompt):
        marked = prompt.split("TEXT TO REWRITE:\n", 1)[1].strip()
        r = self.rng.random()
        if r < 0.08:                      # broken marker
            return marked.replace("]]", "", 1)
        if r < 0.14:                      # silently changed quantity
            return re.sub(r"\[\[(\d)", lambda m: "[[" + str(int(m.group(1)) % 5 + 1),
                          marked, count=1)
        lead = self.rng.choice(["pls ", "Hey, ", "Quick request: ", "Hello! ", ""])
        body = marked[0].lower() + marked[1:] if lead else marked
        return lead + body
