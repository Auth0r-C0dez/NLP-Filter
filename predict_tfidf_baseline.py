"""Predict bedroom-count classes with a saved TF-IDF baseline."""
import argparse
import json
from pathlib import Path

import joblib

PROJECT_DIR = Path(__file__).resolve().parent


def project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("texts", nargs="*", help="one or more prompts to classify")
    parser.add_argument("--model-dir", default="models/tfidf_logreg_out_t")
    parser.add_argument("--input-file", default=None,
                        help="UTF-8 text file with one prompt per line")
    return parser.parse_args()


def main():
    args = parse_args()
    prompts = list(args.texts)
    if args.input_file:
        input_path = project_path(args.input_file)
        prompts.extend(line.strip() for line in input_path.read_text(encoding="utf-8").splitlines()
                       if line.strip())
    if not prompts:
        raise ValueError("Provide prompt text or use --input-file")

    model_dir = project_path(args.model_dir)
    model_path = model_dir / "model.joblib"
    metadata_path = model_dir / "metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Model artifacts not found in {model_dir}; train the model first")
    # Only load a joblib model that you created and trust; joblib uses pickle internally.
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    label_names = {int(key): value for key, value in metadata["label_names"].items()}
    predictions = model.predict(prompts).tolist()
    probabilities = model.predict_proba(prompts)
    for text, label, scores in zip(prompts, predictions, probabilities):
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:3]
        result = {
            "text": text,
            "label": int(label),
            "label_name": label_names[int(label)],
            "top_3": [{"label": int(i), "label_name": label_names[int(i)],
                       "probability": float(score)} for i, score in ranked],
        }
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
