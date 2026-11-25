# Bedroom Count Classifier Data Pipeline

This project creates labeled text examples for a model that classifies a prompt into one of seven classes. It does **not** train the classifier yet and it does not call an LLM when the classifier is used. LLMs can be used offline to create training text.

Read [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) for the full, plain-language architecture and walkthrough.

The initial classifier baseline is TF-IDF plus Logistic Regression. Setup, training, evaluation, and prediction commands are in [TFIDF_LOGISTIC_BASELINE.md](TFIDF_LOGISTIC_BASELINE.md).

The standalone classifier files are `train_tfidf_baseline.py` and `predict_tfidf_baseline.py`. The human-written `eval_set/real_test.jsonl` is only read when `--real-eval` is explicitly provided and should be kept out of training and tuning.

## Seven labels

| ID | Meaning |
|---:|---|
| 0 | Floor-plan request with zero bedrooms |
| 1 | One bedroom |
| 2 | Two bedrooms |
| 3 | Three bedrooms |
| 4 | Four bedrooms |
| 5 | Five or more bedrooms (default overflow rule folds 6+) |
| 6 | Not related to floor planning or architecture in general |

The default bedroom list is in `config.py`. The default decision counts “kids room” and “ensuite bedroom”; it does not count “study room”, “servant room”, or “home office”. Change the configuration and regenerate all data if this definition changes.

## Set up in VS Code on Windows

Open this project folder in VS Code, open a PowerShell terminal in the folder, and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, select the `.venv` interpreter in VS Code with **Python: Select Interpreter** and open a new terminal. The runtime dependencies are small; PyTorch/Transformers are not included because this repository only makes datasets.

## Run locally

```powershell
python test_pipeline.py
python build_dataset.py --no-llm --n-total 5000 --out-dir out_t
python build_dataset.py --mock-llm --n-total 3000 --out-dir out_m
```

All output paths are resolved from the project folder when you run these commands there. `--no-llm` still creates negative examples from local templates; it disables LLM-generated positive wording. `--mock-llm` exercises the LLM code paths without a network connection or API key.

### Real LLM generation

Add model IDs to `MODELS` in `config.py`, then set `OPENROUTER_API_KEY` and/or `NVIDIA_API_KEY` in the environment. Check provider terms and model licenses before using generated outputs. Real provider runs have not been verified by this project. Use a small run first:

```powershell
$env:OPENROUTER_API_KEY = "your-key"
python build_dataset.py --n-total 1000 --out-dir out_real_small
```

Optional unrelated prompts can be loaded from `--offtopic-file`. The file may be plain text with one prompt per line or JSONL containing `text`, `instruction`, or `prompt`. A local fallback is always mixed in. To also ask the configured LLM pool for cached negative variants, pass `--negative-llm`.

## Outputs

The chosen `--out-dir` receives:

- `train.jsonl`, `val.jsonl`, `test.jsonl`: one JSON object per line.
- `stats.json`: target and observed class counts, split/source counts, bedroom counts, LLM stats, and rejection reasons.
- `frames.jsonl`: accepted Method A frames when LLM frames are generated.
- `cache/` or `cache_mock/`: append-only raw LLM responses for resuming runs.

Every row has `text`, `label`, `label_name`, `source`, `llm`, and `group`. Floor-plan rows also have `spans` and `bedrooms`; negative rows have `spans: []` and `bedrooms: null`. Training should use only `text` and `label`. Keep the user's hand-written real evaluation prompts separate from generated data; never train on them or use them for tuning.

## Useful options

```text
--n-total N           final requested dataset size
--out-dir DIR         output directory (default: out)
--no-llm              use hand-written floor-plan examples only
--mock-llm            use deterministic offline fake LLM responses
--negative-llm        add cached LLM-authored negative examples
--offtopic-file FILE  local TXT or JSONL source for unrelated prompts
--ratios A,B,C        template, LLM frame, LLM rewrite proportions
--noise-prob FLOAT    share of rows to receive typo/casing noise
--val-frac FLOAT      validation group fraction (default: 0.10)
--test-frac FLOAT     test group fraction (default: 0.10)
--holdout-llms IDS    reserve complete generator groups by model id
```

See `PROJECT_DOCUMENTATION.md` for module responsibilities, data flow, design choices, limitations, and troubleshooting.
