"""Train and evaluate a seven-class TF-IDF + quantity-feature Logistic Regression baseline.

Improvements over the plain TF-IDF baseline:
  * a rule-based bedroom-count parser feeds explicit count features to the model
    (digits, number words, BHK/BR forms, "2+1 BHK", counts summed across mentions)
  * optional hyper-parameter tuning that uses the validation split ONLY
  * rule-only diagnostic accuracy and per-split error CSVs
"""
import argparse
import csv
import json
import re
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.pipeline import FeatureUnion, Pipeline

import config as cfg

PROJECT_DIR = Path(__file__).resolve().parent
CLASS_IDS = list(range(len(cfg.LABEL_NAMES)))

# --------------------------------------------------------------------------
# Bedroom-count parser (shared with predict_tfidf_baseline.py)
# --------------------------------------------------------------------------
NUMBER_WORDS = {
    "zero": 0, "no": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12,
}
BED_KEYWORDS = {"bedroom", "bedrooms", "bhk", "bhks", "br", "brs",
                "bdrm", "bdrms", "bedrm", "bedrms"}
# Words that end a number->bedroom lookback so "2 bathrooms and a bedroom"
# does not turn into "2 bedrooms".
BLOCKERS = {"and", "with", "plus", "bathroom", "bathrooms", "bath", "baths",
            "kitchen", "kitchens", "hall", "halls", "living", "room", "rooms",
            "study", "office", "garage", "balcony", "sqft", "sq", "ft"}
MAX_LOOKBACK = 4
MAX_REASONABLE = 20
TOTAL_BUCKETS = 7  # totals 0..5 get their own column, 6 means "6 or more"

_PLUS_RE = re.compile(r"(\d+)\s*\+\s*(\d+)(?=\s*-?\s*(?:bhk|br|bedroom))")
_SPLIT_BED_RE = re.compile(r"\bbed\s+rooms?\b")


def _normalise(text):
    text = text.lower().replace("\u2011", "-")
    text = _SPLIT_BED_RE.sub(
        lambda m: "bedrooms" if m.group(0).endswith("s") else "bedroom", text)
    return _PLUS_RE.sub(lambda m: str(int(m.group(1)) + int(m.group(2))), text)


def extract_bedrooms(text):
    """Return (total_bedrooms, mention_count, explicit_number_count)."""
    tokens = re.findall(r"\d+|[a-z]+", _normalise(text))
    total = mentions = explicit = 0
    for i, token in enumerate(tokens):
        if token not in BED_KEYWORDS:
            continue
        qty = None
        for j in range(i - 1, max(i - MAX_LOOKBACK, 0) - 1, -1):
            previous = tokens[j]
            if previous in BED_KEYWORDS or previous in BLOCKERS:
                break
            value = int(previous) if previous.isdigit() else NUMBER_WORDS.get(previous)
            if value is not None and value <= MAX_REASONABLE:
                qty = value
                break
        if qty is None:
            qty = 1
        else:
            explicit += 1
        total += qty
        mentions += 1
    return total, mentions, explicit


class QuantityFeatures(BaseEstimator, TransformerMixin):
    """Sparse features describing the parsed bedroom count."""

    def __init__(self, weight=3.0):
        self.weight = weight

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = np.zeros((len(X), TOTAL_BUCKETS + 3), dtype=np.float64)
        for index, text in enumerate(X):
            total, mentions, explicit = extract_bedrooms(text)
            rows[index, min(total, TOTAL_BUCKETS - 1)] = 1.0
            rows[index, TOTAL_BUCKETS] = 1.0 if mentions else 0.0
            rows[index, TOTAL_BUCKETS + 1] = 1.0 if mentions > 1 else 0.0
            rows[index, TOTAL_BUCKETS + 2] = 1.0 if explicit else 0.0
        return sparse.csr_matrix(rows * self.weight)


# --------------------------------------------------------------------------
# Data + model helpers
# --------------------------------------------------------------------------
def project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            text, label = row.get("text"), row.get("label")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_number}: text must be a non-empty string")
            if isinstance(label, bool) or not isinstance(label, int) or label not in CLASS_IDS:
                raise ValueError(f"{path}:{line_number}: label must be an integer from 0 to 6")
            rows.append((text, label))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def make_model(seed, c_value, min_df, qty_weight, use_char, class_weight):
    parts = [
        ("word", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                 ngram_range=(1, 3), min_df=min_df,
                                 max_features=200_000, sublinear_tf=True,
                                 token_pattern=r"(?u)\b\w+\b")),
    ]
    if use_char:
        parts.append(("character", TfidfVectorizer(
            analyzer="char_wb", lowercase=True, ngram_range=(3, 5),
            min_df=min_df, max_features=250_000, sublinear_tf=True)))
    parts.append(("quantity", QuantityFeatures(weight=qty_weight)))
    return Pipeline([
        ("features", FeatureUnion(parts)),
        ("classifier", LogisticRegression(C=c_value, max_iter=3000, solver="lbfgs",
                                          class_weight=class_weight,
                                          random_state=seed)),
    ])


def evaluate(model, rows):
    texts = [text for text, _ in rows]
    y_true = [label for _, label in rows]
    y_pred = model.predict(texts).tolist()
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
    report = classification_report(y_true, y_pred, labels=CLASS_IDS,
                                   target_names=cfg.LABEL_NAMES,
                                   output_dict=True, zero_division=0)
    return {
        "row_count": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_IDS,
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=CLASS_IDS,
                                      average="weighted", zero_division=0)),
        "per_class": {str(label): report[name]
                      for label, name in enumerate(cfg.LABEL_NAMES)},
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_axes": {
            "rows": "actual label ID (0 through 6)",
            "columns": "predicted label ID (0 through 6)",
            "label_names": cfg.LABEL_NAMES,
        },
    }


def rule_only_accuracy(rows):
    """Accuracy of 'parsed count -> class' on rows whose label is 0-5 (no ML)."""
    scored = [(min(extract_bedrooms(text)[0], cfg.MAX_BEDROOM_CLASS), label)
              for text, label in rows if label != cfg.NOT_RELATED_ID]
    if not scored:
        return None
    return float(sum(pred == label for pred, label in scored) / len(scored))


def write_confusion_csv(path, matrix):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["actual\\predicted", *CLASS_IDS])
        for label, row in zip(CLASS_IDS, matrix):
            writer.writerow([label, *row])


def write_error_csv(path, model, rows):
    texts = [text for text, _ in rows]
    probabilities = model.predict_proba(texts)
    predictions = probabilities.argmax(axis=1)
    classes = list(model.classes_)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["text", "actual", "predicted", "confidence", "parsed_bedrooms"])
        for (text, label), pred_index, scores in zip(rows, predictions, probabilities):
            predicted = int(classes[pred_index])
            if predicted != label:
                writer.writerow([text, label, predicted, f"{scores[pred_index]:.3f}",
                                 extract_bedrooms(text)[0]])


def tune(args, train_rows, val_rows):
    """Pick C and quantity weight by validation macro-F1 only."""
    c_grid, weight_grid = [0.5, 2.0, 8.0, 32.0], [1.0, 3.0, 6.0]
    train_texts = [text for text, _ in train_rows]
    train_labels = [label for _, label in train_rows]
    best, table = None, []
    for c_value, weight in product(c_grid, weight_grid):
        model = make_model(args.seed, c_value, args.min_df, weight,
                           not args.no_char, args.class_weight)
        model.fit(train_texts, train_labels)
        score = evaluate(model, val_rows)["macro_f1"]
        table.append({"C": c_value, "qty_weight": weight, "val_macro_f1": score})
        print(f"  tune C={c_value:<5} qty_weight={weight:<4} val macro-F1={score:.4f}")
        if best is None or score > best[0]:
            best = (score, c_value, weight, model)
    return best[1], best[2], best[3], table


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="out_t",
                        help="folder with train.jsonl, val.jsonl, and test.jsonl")
    parser.add_argument("--model-dir", default=None,
                        help="artifact output folder (default: models/tfidf_logreg_<data-folder>)")
    parser.add_argument("--real-eval", default=None,
                        help="optional separate JSONL evaluation file; never used for fitting or tuning")
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--c", type=float, default=8.0,
                        help="inverse regularization strength (ignored with --tune)")
    parser.add_argument("--qty-weight", type=float, default=3.0,
                        help="scale of the bedroom-count features (ignored with --tune)")
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--no-char", action="store_true",
                        help="drop character n-grams (faster, less typo-robust)")
    parser.add_argument("--class-weight", default=None, choices=[None, "balanced"])
    parser.add_argument("--tune", action="store_true",
                        help="grid-search C and quantity weight on the validation split only")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.c <= 0:
        raise ValueError("--c must be greater than 0")
    if args.min_df < 1:
        raise ValueError("--min-df must be at least 1")
    if args.qty_weight < 0:
        raise ValueError("--qty-weight must not be negative")

    data_dir = project_path(args.data_dir)
    train_path, val_path, test_path = (data_dir / name for name in
                                       ("train.jsonl", "val.jsonl", "test.jsonl"))
    for path in (train_path, val_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset split not found: {path}")
    train_rows, val_rows, test_rows = (load_jsonl(path) for path in
                                      (train_path, val_path, test_path))
    missing = sorted(set(CLASS_IDS) - {label for _, label in train_rows})
    if missing:
        raise ValueError(f"Training split is missing label IDs: {missing}")

    model_dir = (project_path(args.model_dir) if args.model_dir else
                 PROJECT_DIR / "models" / f"tfidf_logreg_{data_dir.name}")
    model_dir.mkdir(parents=True, exist_ok=True)

    tuning_table = None
    if args.tune:
        print("Tuning on the validation split only...")
        c_value, qty_weight, model, tuning_table = tune(args, train_rows, val_rows)
        print(f"Selected C={c_value}, qty_weight={qty_weight}")
    else:
        c_value, qty_weight = args.c, args.qty_weight
        model = make_model(args.seed, c_value, args.min_df, qty_weight,
                           not args.no_char, args.class_weight)
        model.fit([text for text, _ in train_rows], [label for _, label in train_rows])

    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)
    metrics = {
        "model": "word_char_tfidf_plus_quantity_features_logistic_regression",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "data_dir": str(data_dir),
        "training_file": str(train_path),
        "parameters": {"seed": args.seed, "C": c_value, "qty_weight": qty_weight,
                       "min_df": args.min_df, "max_iter": 3000,
                       "word_ngrams": [1, 3],
                       "character_ngrams": None if args.no_char else [3, 5],
                       "class_weight": args.class_weight},
        "tuning_table": tuning_table,
        "rule_only_accuracy_labels_0_to_5": {
            "validation": rule_only_accuracy(val_rows),
            "test": rule_only_accuracy(test_rows),
        },
        "splits": {
            "train": {"row_count": len(train_rows),
                      "label_counts": {str(i): sum(label == i for _, label in train_rows)
                                       for i in CLASS_IDS}},
            "validation": evaluate(model, val_rows),
            "test": evaluate(model, test_rows),
        },
        "real_evaluation": None,
    }
    if args.real_eval:
        real_path = project_path(args.real_eval)
        if real_path in {train_path.resolve(), val_path.resolve(), test_path.resolve()}:
            raise ValueError("--real-eval must be separate from train/val/test data")
        real_rows = load_jsonl(real_path)
        missing_real = sorted(set(CLASS_IDS) - {label for _, label in real_rows})
        metrics["real_evaluation"] = {
            "file": str(real_path),
            "note": "Evaluation only; these rows were not used for fitting or model selection.",
            "missing_label_ids": missing_real,
            "rule_only_accuracy_labels_0_to_5": rule_only_accuracy(real_rows),
            **evaluate(model, real_rows),
        }

    for split, rows in (("validation", val_rows), ("test", test_rows)):
        write_confusion_csv(model_dir / f"confusion_matrix_{split}.csv",
                            metrics["splits"][split]["confusion_matrix"])
        write_error_csv(model_dir / f"errors_{split}.csv", model, rows)
    if metrics["real_evaluation"]:
        write_confusion_csv(model_dir / "confusion_matrix_real_evaluation.csv",
                            metrics["real_evaluation"]["confusion_matrix"])
        write_error_csv(model_dir / "errors_real_evaluation.csv", model, real_rows)

    metadata = {
        "model_type": metrics["model"],
        "model_file": model_path.name,
        "feature_module": "train_tfidf_baseline",
        "label_names": {str(i): name for i, name in enumerate(cfg.LABEL_NAMES)},
        "input_field": "text",
        "prediction_field": "label",
        "training_data": str(train_path),
        "real_evaluation_used_for_training": False,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model and reports to: {model_dir}")
    for split in ("validation", "test"):
        report = metrics["splits"][split]
        print(f"{split:12} accuracy={report['accuracy']:.4f} macro-F1={report['macro_f1']:.4f}")
    rule = metrics["rule_only_accuracy_labels_0_to_5"]
    print(f"rule-only parser accuracy (labels 0-5): val={rule['validation']:.4f} test={rule['test']:.4f}")
    if metrics["real_evaluation"]:
        report = metrics["real_evaluation"]
        print(f"real eval    accuracy={report['accuracy']:.4f} macro-F1={report['macro_f1']:.4f}")


if __name__ == "__main__":
    # Re-import under the module name so the pickled model references
    # `train_tfidf_baseline.QuantityFeatures` (loadable by the predict script)
    # instead of `__main__.QuantityFeatures`.
    import train_tfidf_baseline
    train_tfidf_baseline.main()