# ATIS intent classification

Reproducible training and evaluation for the ATIS intent dataset using **PyTorch** and **scikit-learn** (no TensorFlow): TF‑IDF logistic regression, Kim-style **TextCNN** (word / char / SentencePiece BPE), and a **char-level CNN** recipe (TextCNN + character tokenizer). Data exploration stays in `notebooks/deep_learning.ipynb`.

## Requirements

- Python 3.10+
- CUDA optional (CPU works; set `ATIS_DEVICE=cpu` in `.env` if needed)

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

## Project layout

```
atis_intent_classification/
  config.yaml
  pyproject.toml          # ruff & mypy only
  requirements.txt
  data/
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
