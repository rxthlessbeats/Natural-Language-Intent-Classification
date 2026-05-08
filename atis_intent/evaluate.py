"""Load a saved run bundle and evaluate on the test split."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from atis_intent.config import Settings, load_experiment_config, resolve_data_path
from atis_intent.data import load_rasa_json
from atis_intent.entities import EntityResources
from atis_intent.models import LogisticRegression, TextCNN, build_embedding
from atis_intent.tokenization import (
    CharTokenizer,
    SentencePieceTokenizer,
    Vocabulary,
    WordTokenizer,
)


def _json_ready(obj: Any) -> Any:
    """Convert numpy scalars / nested structures for json.dump."""
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list) or isinstance(obj, tuple):
        return [_json_ready(x) for x in obj]
    if isinstance(obj, np.integer) or isinstance(obj, np.floating):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved ATIS model on test JSON")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Directory from python -m atis_intent train (model.pt, bundle.pkl)",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Optional override experiment.yaml path"
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    cfg_path = Path(args.config) if args.config else run_dir / "experiment.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    exp = load_experiment_config(cfg_path)
    settings = Settings()
    pp = exp.preprocess

    with open(run_dir / "bundle.pkl", "rb") as f:
        bundle = pickle.load(f)

    test_path = Path(bundle["test_json"])
    if not test_path.is_file():
        test_path = resolve_data_path(settings, pp.test_json)

    test_df = load_rasa_json(test_path)
    if pp.strip_whitespace:
        test_df["text"] = test_df["text"].str.strip()

    le_classes = bundle["label_classes"]
    class_to_idx = {c: i for i, c in enumerate(le_classes)}
    test_df = test_df[test_df["intent"].isin(class_to_idx)].reset_index(drop=True)
    y_true = np.array([class_to_idx[i] for i in test_df["intent"]], dtype=np.int64)

    device = torch.device(
        settings.device
        if torch.cuda.is_available() and settings.device.startswith("cuda")
        else "cpu"
    )

    recipe = bundle["recipe"]
    if recipe == "tfidf_lr":
        tfidf: TfidfVectorizer = bundle["tfidf_vectorizer"]
        X = torch.tensor(tfidf.transform(test_df["text"]).toarray(), dtype=torch.float32)
        model = LogisticRegression(bundle["linear_in_features"], bundle["num_classes"])
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
        model = model.to(device).eval()
        with torch.no_grad():
            pred = model(X.to(device)).argmax(1).cpu().numpy()
    else:
        train_path = Path(bundle["train_json"])
        if not train_path.is_file():
            train_path = resolve_data_path(settings, pp.train_json)
        mask_buckets = frozenset(pp.mask_entity_buckets)
        entities = EntityResources(
            train_path, mask_buckets, collapse_multiword=pp.collapse_multiword
        )
        tok_type = bundle["tokenizer_type"]
        use_mask = pp.textcnn_use_masking and tok_type in ("word", "bpe")
        tokenizer: WordTokenizer | CharTokenizer | SentencePieceTokenizer
        if tok_type == "word":
            tokenizer = WordTokenizer(entities, mask=use_mask)
        elif tok_type == "char":
            tokenizer = CharTokenizer()
        else:
            sp = Path(bundle["sentencepiece_model"])
            tokenizer = SentencePieceTokenizer(sp, entities, mask=use_mask).load()

        vocab = Vocabulary()
        vocab.itos = bundle["vocab_itos"]
        vocab.stoi = {w: i for i, w in enumerate(vocab.itos)}

        fs = tuple(bundle["filter_sizes"])
        nf = bundle["num_filters"]
        max_len = bundle["max_len"]
        min_len = max(fs)
        embed_kind = bundle["embedding_type"]
        edim = bundle["glove_dim"] if embed_kind == "glove" else bundle["embed_dim_learned"]
        embedding = build_embedding(
            vocab,
            embed_kind,
            embed_dim=edim,
            glove_name=bundle["glove_name"],
            glove_dim=bundle["glove_dim"],
            freeze=bundle["freeze_pretrained"],
            cache=settings.vector_cache,
        )
        model = TextCNN(
            embedding,
            bundle["num_classes"],
            filter_sizes=fs,
            num_filters=nf,
            dropout=exp.train.textcnn_dropout,
        )
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
        model = model.to(device).eval()

        preds = []
        pad_id = vocab.pad_id
        with torch.no_grad():
            for text in test_df["text"]:
                toks = tokenizer.tokenize(text)[:max_len]
                ids = [vocab.stoi.get(t, vocab.unk_id) for t in toks]
                ltok = min(len(ids), max_len)
                w = max(ltok, min_len)
                row = torch.full((1, w), pad_id, dtype=torch.long)
                if ltok > 0:
                    row[0, :ltok] = torch.tensor(ids[:ltok], dtype=torch.long)
                logits = model(row.to(device), None)
                preds.append(int(logits.argmax(1).item()))
        pred = np.array(preds, dtype=np.int64)

    acc = accuracy_score(y_true, pred)
    mf1 = f1_score(y_true, pred, average="macro", zero_division=0)
    wf1 = f1_score(y_true, pred, average="weighted", zero_division=0)
    print(f"test_accuracy={acc:.4f}  macro_f1={mf1:.4f}  weighted_f1={wf1:.4f}")

    labels_all = list(range(len(le_classes)))
    report_text = classification_report(
        y_true,
        pred,
        labels=labels_all,
        target_names=le_classes,
        zero_division=0,
    )
    print(report_text)

    report_dict = classification_report(
        y_true,
        pred,
        labels=labels_all,
        target_names=le_classes,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, pred, labels=labels_all)

    metrics_payload = {
        "test_accuracy": float(acc),
        "test_macro_f1": float(mf1),
        "test_weighted_f1": float(wf1),
    }
    out = run_dir / "evaluate_metrics.json"
    out.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    cm_path = run_dir / "evaluate_confusion_matrix.json"
    cm_payload = {
        "label_ids": labels_all,
        "label_names": list(le_classes),
        "matrix": cm.tolist(),
    }
    cm_path.write_text(json.dumps(cm_payload, indent=2), encoding="utf-8")

    report_txt_path = run_dir / "evaluate_classification_report.txt"
    report_txt_path.write_text(report_text, encoding="utf-8")

    report_json_path = run_dir / "evaluate_classification_report.json"
    report_json_path.write_text(json.dumps(_json_ready(report_dict), indent=2), encoding="utf-8")

    print(f"Wrote {out}, {cm_path}, {report_txt_path}, {report_json_path}")


if __name__ == "__main__":
    main()
