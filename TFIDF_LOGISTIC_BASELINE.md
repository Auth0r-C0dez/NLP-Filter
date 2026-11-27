# TF-IDF + Logistic Regression Baseline

This guide covers the first classifier for the bedroom-count project. It is a simple local baseline that can be trained and used before trying a larger model such as DeBERTa-v3.

## 1. What this baseline does

The model reads one text prompt and predicts one of seven labels:

| ID | Label |
|---:|---|
| 0 | 0 bedroom |
| 1 | 1 bedroom |
| 2 | 2 bedrooms |
| 3 | 3 bedrooms |
| 4 | 4 bedrooms |
| 5 | 5 bedrooms (five or more when overflow policy is `fold`) |
| 6 | not related to floor planning or architecture in general |

The model uses two TF-IDF views of each prompt:

- **Word TF-IDF** uses single words and two-word phrases. It can learn patterns such as “3 bedrooms”, “studio apartment”, and room names.
- **Character TF-IDF** uses character pieces of length 3 to 5. It can recognize slightly misspelled words such as `bedrom` or `bedrrom` as similar to `bedroom`.

Logistic Regression learns how those text features relate to each label. It returns a predicted class and class probabilities. There is no LLM call in training or prediction.

The initial settings are deliberately fixed in the script: word n-grams `(1, 2)`, character n-grams `(3, 5)`, `min_df=2`, `C=2.0`, `max_iter=2000`, and seed `42`. This gives a repeatable starting point, not a tuned final model.

## 2. Files

| File | Role |
|---|---|
| `train_tfidf_baseline.py` | Loads `train.jsonl`, fits the vectorizer and classifier, evaluates validation/test, and saves the model plus reports. It can optionally evaluate a separate JSONL file. |
| `predict_tfidf_baseline.py` | Loads a saved model and predicts labels for supplied text. |
| `requirements.txt` | Includes scikit-learn and joblib for training and prediction. |
| `config.py` | Supplies the seven label names and default random seed. |

All paths in the examples below are relative to this project folder. The scripts resolve relative paths from the folder containing the scripts, so commands also work when launched from another current directory.

## 3. Install dependencies in VS Code

Open this folder in VS Code. In its PowerShell terminal, create and activate the virtual environment, then install the requirements:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

In VS Code, run **Python: Select Interpreter** and select this project's `.venv`. The existing environment may need the new packages even if it was already created:

```powershell
python -m pip install -r requirements.txt
```

The new direct packages are `scikit-learn` and `joblib`. They are CPU-friendly and do not require a GPU. The runtime classifier does not need PyTorch or Transformers.

## 4. Make sure the dataset is ready

The training script expects these three files inside the dataset folder:

```text
out_t/
  train.jsonl
  val.jsonl
  test.jsonl
```

The rows must have at least a non-empty `text` field and an integer `label` from 0 to 6. Additional data-generation fields such as `spans`, `bedrooms`, `source`, `llm`, and `group` are ignored by the classifier. The script checks that the training split contains every class.

Create the template-only dataset if it does not exist or needs to be regenerated:

```powershell
python build_dataset.py --no-llm --n-total 5000 --out-dir out_t
```

Do not run the data builder into the same folder while model training is using it. Keep a copy of the selected data if you later change the bedroom definition or overflow policy, because those changes alter labels.

## 5. Train the baseline

From the project folder:

```powershell
python train_tfidf_baseline.py --data-dir out_t --model-dir models/tfidf_logreg_out_t
```

The model is fitted only to `out_t/train.jsonl`. `val.jsonl` and `test.jsonl` are used only for reporting. The command prints accuracy and macro-F1 for both. Macro-F1 gives equal importance to all seven labels, so it is useful alongside accuracy for this task.

Training creates this folder:

```text
models/tfidf_logreg_out_t/
  model.joblib
  metadata.json
  metrics.json
  confusion_matrix_validation.csv
  confusion_matrix_test.csv
```

`model.joblib` holds the complete text vectorizers and Logistic Regression classifier. `metadata.json` records the label mapping and input/output fields. `metrics.json` contains accuracy, macro-F1, weighted-F1, per-class scores, and the confusion matrices. CSV confusion matrices use actual labels as rows and predicted labels as columns; row and column IDs follow the 0–6 order shown above.

You can change the output folder or a small number of fixed baseline settings:

```powershell
python train_tfidf_baseline.py --data-dir out_m --model-dir models/tfidf_logreg_out_m
python train_tfidf_baseline.py --data-dir out_t --c 1.0 --min-df 2
```

Use `out_t` for the first baseline because it has only template-generated floor-plan wording and built-in negatives. `out_m` is a useful second comparison, but it contains wording from the offline mock and its artifacts. These are not results from real LLMs.

## 6. Predict a prompt

After training, pass one or more prompts as quoted command-line text:

```powershell
python predict_tfidf_baseline.py --model-dir models/tfidf_logreg_out_t "Design a home with two bedrooms and a kitchen."
```

The program prints one JSON object per prompt with the predicted ID, label name, and three most likely classes. Example shape:

```json
{"text":"Design a home with two bedrooms and a kitchen.","label":2,"label_name":"2 bedrooms","top_3":[...]}
```

For multiple prompts, use a UTF-8 text file with one prompt per line:

```powershell
python predict_tfidf_baseline.py --model-dir models/tfidf_logreg_out_t --input-file prompts.txt
```

The prediction script does not need the original JSONL datasets after the model folder has been saved. Only load `model.joblib` files you trust; joblib uses Python pickle internally.

## 7. Evaluate on the human-written set only at the end

The project currently has `eval_set/real_test.jsonl` with 112 examples, 16 for each label. The training program does not read it by default. Keep it separate from training and all model-setting decisions.

After you have finished choosing the baseline settings and do not plan to tune from the result, you can run the one-time held-out evaluation:

```powershell
python train_tfidf_baseline.py --data-dir out_t --model-dir models/tfidf_logreg_out_t --real-eval eval_set/real_test.jsonl
```

The real file is only passed to `predict` and metrics code; it is never used in `fit`. Its metrics are added under `real_evaluation` in `metrics.json`, and a separate confusion matrix is saved. Do not change `C`, n-grams, the data, or label rules based on this result and then report the same set as an untouched final test. If you need to tune or compare many models, create another held-out set or use validation data for those decisions.

The examples in this folder contain `text` and `label`, which is the format the script expects. Do not copy this file into `out_t` or include it in the generated training splits.

## 8. Read the metrics

- **Accuracy**: fraction of prompts assigned the exact correct class.
- **Macro-F1**: F1 is calculated per class, then the seven scores are averaged equally. This helps reveal a class that performs poorly even when it is less common.
- **Per-class precision**: among examples predicted as a class, how many were correct.
- **Per-class recall**: among examples truly in a class, how many were found.
- **Confusion matrix**: each row is the true label and each column is the model prediction. Large off-diagonal values show which labels are confused.

Review errors between 4 and 5, and between bedroom classes and class 6. In particular, check whether hard negatives that mention “bedroom” are mistaken for a count request. Synthetic test performance is only a pipeline baseline; the human-written prompts are a better estimate of actual usefulness.

## 9. What comes after this baseline

The intended stronger model is DeBERTa-v3, starting with a small checkpoint and trying a base checkpoint only if the available compute and evaluation results justify it. That is a separate fine-tuning step and is not implemented by these two scripts. The baseline gives you a fast reference score and a working local prediction interface before introducing a transformer model.

When adding DeBERTa later, keep the same data splits, use only `text` and `label`, select settings with validation data, save the label mapping with the checkpoint, and keep the human-written evaluation set untouched until the agreed final evaluation.

## 10. Troubleshooting

**`ModuleNotFoundError: sklearn` or `joblib`**

Select the project `.venv` in VS Code, activate it, and run `pip install -r requirements.txt`.

**Missing `train.jsonl`, `val.jsonl`, or `test.jsonl`**

Generate them with the data-builder command in section 4 or point `--data-dir` at the correct output folder.

**Training split missing a class**

The script stops instead of silently training an incomplete classifier. Check `stats.json`, regenerate the dataset with all labels represented, and run training again.

**Model artifact not found during prediction**

Train first and pass the same `--model-dir` to the predictor that you passed to the training script.
