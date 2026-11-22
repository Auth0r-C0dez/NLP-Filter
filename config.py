"""Central configuration for the bedroom-count training-data pipeline."""
import re

SEED = 42

LABEL_NAMES = ["0 bedroom", "1 bedroom", "2 bedrooms", "3 bedrooms",
               "4 bedrooms", "5 bedrooms", "not related to floor planning or architecture in general"]
NOT_RELATED_ID = 6
MAX_BEDROOM_CLASS = 5
OVERFLOW_POLICY = "fold"  # "fold" maps 6+ to class 5; "drop" excludes those rows
# Suggested balance: 80% floor planning, split evenly over classes 0..5; 20% unrelated.
TARGET_CLASS_DISTRIBUTION = {0: 0.1333333333, 1: 0.1333333333, 2: 0.1333333333,
                             3: 0.1333333333, 4: 0.1333333333, 5: 0.1333333335,
                             NOT_RELATED_ID: 0.20}
OVERSAMPLE_FACTOR = 4
NEGATIVE_LLM = False

# ----------------------------------------------------------------------------
# DECISION POINT: what counts as a bedroom?
#   * Everything in BEDROOM_TYPES is labelled as a bedroom (name, sampling weight).
#   * Everything in OTHER_ROOMS is NOT a bedroom.
# "study room", "servant room" and "home office" are NOT counted by default.
# To change the rule, move a room between the two lists and regenerate the data.
# ----------------------------------------------------------------------------
BEDROOM_TYPES = [
    ("bedroom", 5.0),
    ("master bedroom", 4.0),
    ("guest room", 3.0),
    ("guest bedroom", 1.5),
    ("kids room", 2.0),
    ("kids bedroom", 1.0),
    ("children's bedroom", 0.7),
    ("single bedroom", 0.7),
    ("double bedroom", 0.7),
    ("ensuite bedroom", 0.7),
    ("parents bedroom", 0.7),
]

OTHER_ROOMS = [
    "kitchen", "bathroom", "toilet", "living room", "dining room", "balcony",
    "garage", "store room", "pooja room", "study room", "servant room",
    "laundry room", "hall", "terrace", "home office", "porch", "utility area",
    "common bathroom",
]

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# Sanity-check regex: does this text mention a bedroom-type word at all?
# Built from BEDROOM_TYPES, so it follows the decision above.
BED_KEYWORD_RE = re.compile(
    "|".join(
        [r"bed", r"bhk", r"(?<![a-z])br\b"]
        + [re.escape(name).replace(r"\ ", r"\s+") for name, _ in BEDROOM_TYPES]
    ),
    re.I,
)

# ----------------------------------------------------------------------------
# Hard negatives: sentences with numbers that must NOT be counted as bedrooms.
# They are appended to some prompts, outside every span.
# ----------------------------------------------------------------------------
DISTRACTORS = [
    "The plot size is 30x40 feet.",
    "Total area should be around 1200 sq ft.",
    "I want it on 2 floors.",
    "Make it a 2 storey building.",
    "Budget is around 25 lakh.",
    "The house should face east.",
    "Add 3 windows in the living room.",
    "The plot is 1500 square feet.",
    "Keep the ceiling height at 10 feet.",
    "Use a 12 foot wide entrance.",
    "Total built-up area is about 2000 sq ft.",
    "Put the main door on the north side.",
]

# ----------------------------------------------------------------------------
# Hand-written sentence frames for the plain-template source.
# Each has exactly one {ROOMS} placeholder, no digits, no bedroom words.
# ----------------------------------------------------------------------------
HAND_FRAMES = [
    "Create an image of a floor plan which consists of {ROOMS}.",
    "Design a house layout with {ROOMS}.",
    "I need a floor plan having {ROOMS}.",
    "Generate a residential plan with {ROOMS}.",
    "Draw a floor plan for a house that has {ROOMS}.",
    "Make me a top-down floor plan including {ROOMS}.",
    "floor plan with {ROOMS}",
    "pls make a plan with {ROOMS}",
    "Can you generate a modern home layout containing {ROOMS}?",
    "Plan for a villa: {ROOMS}.",
    "Architect's request: produce a floor plan featuring {ROOMS}.",
    "I want a floor plan. It should have {ROOMS}.",
    "Generate a top-down floor plan image with {ROOMS}, clean white background.",
    "house plan - {ROOMS}",
    "Please design a compact apartment layout with {ROOMS}.",
    "Need a blueprint style plan, rooms required: {ROOMS}.",
    "Show me a floor plan that includes {ROOMS}.",
    "Sketch a family home floor plan with {ROOMS}.",
    "Create a minimalist floor plan image: {ROOMS}",
    "Generate floor plan for my new house. Requirements: {ROOMS}.",
    "Design an efficient single-storey layout comprising {ROOMS}.",
    "{ROOMS} - make a floor plan out of this.",
    "Floor plan needed: {ROOMS}!",
    "Could you please draw up a floor plan with {ROOMS} for me?",
    "Produce an architectural floor plan image with {ROOMS}.",
    "Build a clean, labelled floor plan for a home with {ROOMS}.",
    "Layout request for a small family: {ROOMS}.",
    "Give me a realistic floor plan with {ROOMS}",
    "Draft a floor plan showing {ROOMS}, doors and windows marked.",
    "My client wants a house with {ROOMS}. Generate the floor plan.",
    "Please create a floor plan. Rooms: {ROOMS}",
    "Can I get a simple floor plan with {ROOMS}?",
]

# ----------------------------------------------------------------------------
# LLM settings (offline data generation only; no LLM is used at inference time)
# ----------------------------------------------------------------------------
# Fill in model ids from the providers' current catalogs, e.g.
#   {"provider": "openrouter", "model": "<id from openrouter.ai/models>"}
#   {"provider": "nim",        "model": "<id from build.nvidia.com>"}
# Environment variables: OPENROUTER_API_KEY, NVIDIA_API_KEY
MODELS = [
]

STYLES = [
    "casual and chatty",
    "formal, like an architect's brief",
    "very short and blunt",
    "long and detailed",
    "texting style with minor typos",
    "Indian English",
    "polite request",
    "slightly rushed, like a transcribed voice note",
    "technical, using architectural terms",
    "Hinglish written in Latin letters",
]

# ----------------------------------------------------------------------------
# Dataset recipe
# ----------------------------------------------------------------------------
RATIOS = {"template": 0.30, "llm_frame": 0.40, "llm_full": 0.30}
FRAMES_PER_CALL = 15       # frames requested per LLM call (Method A)
ROWS_PER_FRAME = 12        # average rows generated per LLM frame (Method A)
REWRITES_PER_BASE = 2      # rewrites requested per base sentence (Method B)
NOISE_PROB = 0.35          # share of rows that get typo/casing noise
TYPO_WORD_PROB = 0.15      # per-word typo chance inside a noisy row
DISTRACTOR_PROB = 0.25     # share of rows with a hard-negative sentence
ZERO_ROW_PROB = 0.20       # richer no-bedroom floor-planning examples in candidates
VAL_FRAC = 0.10
TEST_FRAC = 0.10
WORKERS = 8                # parallel LLM calls
HOLDOUT_LLMS = []          # model ids whose rows go to test_unseen_llm.jsonl only
