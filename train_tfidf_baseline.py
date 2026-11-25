"""Train and evaluate a seven-class TF-IDF + Logistic Regression baseline."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.pipeline import FeatureUnion, Pipeline

import config as cfg

PROJECT_DIR = Path(__file__).resolve().parent
CLASS_IDS = list(range(len(cfg.LABEL_NAMES)))


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


def make_model(seed, c_value, min_df):
    # Word features capture room phrases and counts; character features help with
    # the typo noise already present in the generated dataset.
    features = FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                 ngram_range=(1, 2), min_df=min_df,
                                 max_features=200_000, sublinear_tf=True,
                                 token_pattern=r"(?u)\b\w+\b")),
        ("character", TfidfVectorizer(analyzer="char_wb", lowercase=True,
                                      ngram_range=(3, 5), min_df=min_df,
                                      max_features=250_000, sublinear_tf=True)),
    ])
    return Pipeline([
        ("tfidf", features),
        ("classifier", LogisticRegression(C=c_value, max_iter=2000,
                                           solver="lbfgs", random_state=seed)),
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


def write_confusion_csv(path, matrix):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["actual\\predicted", *CLASS_IDS])
        for label, row in zip(CLASS_IDS, matrix):
            writer.writerow([label, *row])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="out_t",
                        help="folder with train.jsonl, val.jsonl, and test.jsonl")
    parser.add_argument("--model-dir", default=None,
                        help="artifact output folder (default: models/tfidf_logreg_<data-folder>)")
    parser.add_argument("--real-eval", default=None,
                        help="optional separate JSONL evaluation file; never used for fitting")
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--c", type=float, default=2.0,
                        help="inverse regularization strength for Logistic Regression")
    parser.add_argument("--min-df", type=int, default=2,
                        help="ignore features appearing in fewer than this many training rows")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.c <= 0:
        raise ValueError("--c must be greater than 0")
    if args.min_df < 1:
        raise ValueError("--min-df must be at least 1")

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
    model = make_model(args.seed, args.c, args.min_df)
    model.fit([text for text, _ in train_rows], [label for _, label in train_rows])

    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)
    metrics = {
        "model": "word_and_character_tfidf_plus_logistic_regression",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "data_dir": str(data_dir),
        "training_file": str(train_path),
        "parameters": {"seed": args.seed, "C": args.c, "min_df": args.min_df,
                       "max_iter": 2000, "word_ngrams": [1, 2],
                       "character_ngrams": [3, 5]},
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
            **evaluate(model, real_rows),
        }

    for split in ("validation", "test"):
        write_confusion_csv(model_dir / f"confusion_matrix_{split}.csv",
                            metrics["splits"][split]["confusion_matrix"])
    if metrics["real_evaluation"]:
        write_confusion_csv(model_dir / "confusion_matrix_real_evaluation.csv",
                            metrics["real_evaluation"]["confusion_matrix"])

    metadata = {
        "model_type": metrics["model"],
        "model_file": model_path.name,
        "label_names": {str(i): name for i, name in enumerate(cfg.LABEL_NAMES)},
        "input_field": "text",
        "prediction_field": "label",
        "training_data": str(train_path),
        "real_evaluation_used_for_training": False,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model and reports to: {model_dir}")
    for split, key in (("validation", "validation"), ("test", "test")):
        report = metrics["splits"][key]
        print(f"{split:12} accuracy={report['accuracy']:.4f} macro-F1={report['macro_f1']:.4f}")
    if metrics["real_evaluation"]:
        report = metrics["real_evaluation"]
        print(f"real eval    accuracy={report['accuracy']:.4f} macro-F1={report['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
