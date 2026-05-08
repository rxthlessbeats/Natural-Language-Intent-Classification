# ATIS intent classification

Reproducible training and evaluation for the ATIS intent dataset using **PyTorch** and **scikit-learn**: **TF‑IDF logistic regression**, **CNN** (word / char / SentencePiece BPE).

Project notes:
- **Pydantic** validates experiment configuration (`config.yaml`).
- **pre-commit** runs formatting/lint checks locally.

Data exploration and visualization stays in `notebooks/deep_learning.ipynb`.

## Requirements

- Python 3.10+
- CUDA optional (CPU works; set `ATIS_DEVICE=cpu` in `.env` if needed)

## Project layout

```
atis_intent_classification/
  config.yaml
  pyproject.toml          # ruff & mypy only
  requirements.txt
  data/
  runs/                   # training/eval outputs (gitignored)
    <run_name>/
      experiment.yaml
      model.pt
      bundle.pkl
      metrics.json
      evaluate_metrics.json
      evaluate_confusion_matrix.json
      evaluate_classification_report.txt
      evaluate_classification_report.json
  notebooks/deep_learning.ipynb   # exploration only
  atis_intent/
    __main__.py
    cli.py
    config.py
    data.py
    entities.py
    tokenization.py
    models.py
    train.py
    evaluate.py
```

## Setup

From this directory (`atis_intent_classification`):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and adjust paths if your data or caches live elsewhere.

Optional dev tools (lint / types): `pip install ruff mypy types-PyYAML`.

### Data

Place RASA-format JSON under `data/` (default):

- `data/train.json`
- `data/test.json`

Or set `ATIS_DATA_DIR` to the folder that contains those files.

#### Data format (RASA NLU JSON)

This project expects the **Rasa NLU** JSON structure with examples under `rasa_nlu_data.common_examples`.

Minimal example:

```json
{
  "rasa_nlu_data": {
    "common_examples": [
      {
        "text": "show me flights from boston to denver",
        "intent": "flight",
        "entities": [
          { "start": 20, "end": 26, "value": "boston", "entity": "city_name" },
          { "start": 30, "end": 36, "value": "denver", "entity": "city_name" }
        ]
      }
    ]
  }
}
```

Notes:
- **`text`** and **`intent`** are required for training/evaluation.
- **`entities`** are optional, but if present they’re used to build entity resources for masking/collapse.

## Configuration

All preprocessing and training options live in **`config.yaml`** at the project root (Pydantic-validated). Important fields:

| Section | Role |
|--------|------|
| `preprocess` | Intent filter, train/val split, masking buckets, tokenizer (word/char/bpe), TF‑IDF, GloVe, sequence lengths |
| `train` | `recipe` (`tfidf_lr` / `textcnn` / `charcnn`), epochs, LR, loss (`ce` / `focal`), class weights, CNN widths |

Environment overrides (prefix **`ATIS_`**): `DATA_DIR`, `VECTOR_CACHE`, `SENTENCEPIECE_DIR`, `RUNS_DIR`, `DEVICE`. See `.env.example`.

## Train

```bash
python -m atis_intent train --config config.yaml
```

Artifacts are written under `runs/<run_name>/` (default name includes timestamp):

- `experiment.yaml` — resolved config
- `model.pt` — PyTorch weights
- `bundle.pkl` — metadata for evaluation (vectorizer and/or vocab, tokenizer type, etc.)
- `metrics.json` — test accuracy / F1

**GloVe** is downloaded on first use into `ATIS_VECTOR_CACHE` (default `.vector_cache`). **SentencePiece** models default to `.spm/` and are copied into the run folder for evaluation.

## Evaluate

```bash
python -m atis_intent evaluate --run-dir runs/<your_run>
```

Recomputes metrics on the test split from `bundle.pkl` / `experiment.yaml` and writes `evaluate_metrics.json` in the same run directory.

## Development

```bash
pre-commit install
pre-commit run --all-files
```
