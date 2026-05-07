"""Train ATIS models (TF-IDF LR, TextCNN, char-CNN), validate, test, save run bundle."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import shutil
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from atis_intent.config import Settings, load_experiment_config, resolve_data_path
from atis_intent.data import prepare_frames
from atis_intent.entities import EntityResources
from atis_intent.models import LogisticRegression, TextCNN, build_embedding
from atis_intent.tokenization import (
    CharTokenizer,
    SentencePieceTokenizer,
    Vocabulary,
    WordTokenizer,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _per_recipe_defaults(recipe: str, tokenizer: str) -> dict[str, Any]:
    if recipe == "charcnn" or tokenizer == "char":
        return {"filter_sizes": (3, 5, 7), "num_filters": 256, "max_len": 200}
    return {"filter_sizes": (2, 3, 4, 5), "num_filters": 128, "max_len": 50}


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


def class_weights_from_labels(
    y: torch.Tensor, num_classes: int, mode: str | None
) -> torch.Tensor | None:
    if mode is None:
        return None
    counts = torch.bincount(y, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    inv = 1.0 / counts
    if mode == "inv":
        w = inv * (num_classes / inv.sum())
    elif mode == "dampened_inv":
        beta = 0.999
        eff = 1.0 - torch.pow(beta, counts)
        w = (1.0 / eff) * (num_classes / (1.0 / eff).sum())
    else:
        raise ValueError(mode)
    return w


def build_criterion(
    loss_type: str,
    class_weights: torch.Tensor | None,
    focal_gamma: float,
    device: torch.device,
) -> nn.Module:
    if loss_type == "ce":
        w = class_weights.to(device) if class_weights is not None else None
        return nn.CrossEntropyLoss(weight=w)
    if loss_type == "focal":
        w = class_weights.to(device) if class_weights is not None else None
        return FocalLoss(gamma=focal_gamma, weight=w).to(device)
    raise ValueError(loss_type)


class TensorDatasetShuffled(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


def make_tensor_loader(X: torch.Tensor, y: torch.Tensor, batch: int, shuffle: bool) -> DataLoader:
    ds = TensorDatasetShuffled(X, y)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


class SeqDataset(Dataset):
    def __init__(
        self, texts: pd.Series, labels: np.ndarray, tokenize_fn, vocab: Vocabulary, max_len: int
    ):
        self.texts = texts.tolist()
        self.labels = labels
        self.tokenize_fn = tokenize_fn
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        toks = self.tokenize_fn(self.texts[i])[: self.max_len]
        ids = [self.vocab.stoi.get(t, self.vocab.unk_id) for t in toks]
        return torch.tensor(ids, dtype=torch.long), len(ids), int(self.labels[i])


def collate_pad(batch, pad_id: int, min_len: int = 1):
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    lens = [max(b[1], min_len) for b in batch]
    max_l = min(max(len(b[0]) for b in batch), max(lens) if lens else 1)
    max_l = max(max_l, min_len)
    out = torch.full((len(batch), max_l), pad_id, dtype=torch.long)
    for i, (ids, _, _) in enumerate(batch):
        L = min(len(ids), max_l)
        out[i, :L] = ids[:L]
    return out, torch.tensor([min(len(b[0]), max_l) for b in batch], dtype=torch.long), labels


def make_seq_loader(
    df: pd.DataFrame,
    tokenize_fn,
    vocab: Vocabulary,
    max_len: int,
    min_len: int,
    batch: int,
    shuffle: bool,
) -> DataLoader:
    ds = SeqDataset(df["text"], df["label"].values, tokenize_fn, vocab, max_len)
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        collate_fn=lambda b: collate_pad(b, vocab.pad_id, min_len),
    )


def train_dense(
    model: nn.Module,
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_te: torch.Tensor,
    y_te: torch.Tensor,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    sched_patience: int,
    sched_factor: float,
    sched_min_lr: float,
    criterion: nn.Module,
    batch_size: int,
    name: str = "tfidf_lr",
) -> tuple[nn.Module, list[dict], float, float, float, np.ndarray]:
    model = model.to(device)
    optimiser = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimiser, mode="max", factor=sched_factor, patience=sched_patience, min_lr=sched_min_lr
    )
    tr_loader = make_tensor_loader(X_tr, y_tr, batch_size, shuffle=True)
    val_loader = make_tensor_loader(X_val, y_val, batch_size, shuffle=False)

    best_val_f1, best_state, no_improve = -1.0, None, 0
    history: list[dict] = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_preds, tr_true = [], []
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimiser.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimiser.step()
            tr_loss += loss.item() * len(yb)
            tr_preds.append(logits.detach().argmax(1).cpu())
            tr_true.append(yb.cpu())
        tr_loss /= len(tr_loader.dataset)
        tr_preds_t = torch.cat(tr_preds).numpy()
        tr_true_t = torch.cat(tr_true).numpy()
        tr_acc = accuracy_score(tr_true_t, tr_preds_t)
        tr_f1 = f1_score(tr_true_t, tr_preds_t, average="macro", zero_division=0)

        model.eval()
        val_loss = 0.0
        val_preds, val_true = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb_d, yb_d = Xb.to(device), yb.to(device)
                logits = model(Xb_d)
                val_loss += criterion(logits, yb_d).item() * len(yb)
                val_preds.append(logits.argmax(1).cpu())
                val_true.append(yb)
        val_loss /= len(val_loader.dataset)
        val_preds_t = torch.cat(val_preds).numpy()
        val_true_t = torch.cat(val_true).numpy()
        val_f1 = f1_score(val_true_t, val_preds_t, average="macro", zero_division=0)
        val_acc = accuracy_score(val_true_t, val_preds_t)

        scheduler.step(val_f1)
        history.append(
            {
                "epoch": epoch,
                "tr_loss": tr_loss,
                "val_loss": val_loss,
                "tr_acc": tr_acc,
                "val_acc": val_acc,
                "tr_macro_f1": tr_f1,
                "val_macro_f1": val_f1,
            }
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  [{name}] ep {epoch:3d}  loss={tr_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}"
            )

    assert best_state is not None
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    te_preds = []
    with torch.no_grad():
        for Xb, _ in make_tensor_loader(X_te, y_te, batch_size, shuffle=False):
            te_preds.append(model(Xb.to(device)).argmax(1).cpu())
    te_preds_t = torch.cat(te_preds).numpy()
    te_true_t = y_te.numpy()
    te_acc = accuracy_score(te_true_t, te_preds_t)
    te_mf1 = f1_score(te_true_t, te_preds_t, average="macro", zero_division=0)
    te_wf1 = f1_score(te_true_t, te_preds_t, average="weighted", zero_division=0)
    elapsed = time.time() - t0
    print(
        f"\n  [{name}]  test_acc={te_acc:.4f}  macro_f1={te_mf1:.4f}  weighted_f1={te_wf1:.4f}  time={elapsed:.1f}s"
    )
    return model, history, te_acc, te_mf1, te_wf1, te_preds_t


def train_seq(
    model: nn.Module,
    name: str,
    tr_loader: DataLoader,
    val_loader: DataLoader,
    te_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    sched_patience: int,
    sched_factor: float,
    sched_min_lr: float,
    criterion: nn.Module,
    y_te: np.ndarray,
) -> tuple[nn.Module, list[dict], float, float, float, np.ndarray]:
    model = model.to(device)
    optimiser = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimiser, mode="max", factor=sched_factor, patience=sched_patience, min_lr=sched_min_lr
    )

    best_val_f1, best_state, no_improve = -1.0, None, 0
    history: list[dict] = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_preds, tr_true = [], []
        for ids, lengths, labels in tr_loader:
            ids, lengths, labels = ids.to(device), lengths.to(device), labels.to(device)
            optimiser.zero_grad()
            logits = model(ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()
            tr_loss += loss.item() * len(labels)
            tr_preds.append(logits.detach().argmax(1).cpu())
            tr_true.append(labels.cpu())
        tr_loss /= len(tr_loader.dataset)
        tr_preds_t = torch.cat(tr_preds).numpy()
        tr_true_t = torch.cat(tr_true).numpy()
        tr_acc = accuracy_score(tr_true_t, tr_preds_t)
        tr_f1 = f1_score(tr_true_t, tr_preds_t, average="macro", zero_division=0)

        model.eval()
        val_loss = 0.0
        val_preds, val_true = [], []
        with torch.no_grad():
            for ids, lengths, labels in val_loader:
                ids_d, lens_d, labels_d = ids.to(device), lengths.to(device), labels.to(device)
                logits = model(ids_d, lens_d)
                val_loss += criterion(logits, labels_d).item() * len(labels)
                val_preds.append(logits.argmax(1).cpu())
                val_true.append(labels)
        val_loss /= len(val_loader.dataset)
        val_preds_t = torch.cat(val_preds).numpy()
        val_true_t = torch.cat(val_true).numpy()
        val_f1 = f1_score(val_true_t, val_preds_t, average="macro", zero_division=0)
        val_acc = accuracy_score(val_true_t, val_preds_t)

        scheduler.step(val_f1)
        history.append(
            {
                "epoch": epoch,
                "tr_loss": tr_loss,
                "val_loss": val_loss,
                "tr_acc": tr_acc,
                "val_acc": val_acc,
                "tr_macro_f1": tr_f1,
                "val_macro_f1": val_f1,
            }
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  [{name}] ep {epoch:3d}  loss={tr_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}"
            )

    assert best_state is not None
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    te_preds, te_true_acc = [], []
    with torch.no_grad():
        for ids, lengths, labels in te_loader:
            te_preds.append(model(ids.to(device), lengths.to(device)).argmax(1).cpu())
            te_true_acc.append(labels)
    te_preds_t = torch.cat(te_preds).numpy()
    te_true_t = torch.cat(te_true_acc).numpy()
    te_acc = accuracy_score(te_true_t, te_preds_t)
    te_mf1 = f1_score(te_true_t, te_preds_t, average="macro", zero_division=0)
    te_wf1 = f1_score(te_true_t, te_preds_t, average="weighted", zero_division=0)
    elapsed = time.time() - t0
    print(
        f"\n  [{name}]  test_acc={te_acc:.4f}  macro_f1={te_mf1:.4f}  weighted_f1={te_wf1:.4f}  time={elapsed:.1f}s"
    )
    return model, history, te_acc, te_mf1, te_wf1, te_preds_t


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train ATIS intent model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to experiment YAML")
    args = parser.parse_args(argv)

    settings = Settings()
    exp = load_experiment_config(args.config)
    pp = exp.preprocess
    tr_cfg = exp.train

    train_path = resolve_data_path(settings, pp.train_json)
    test_path = resolve_data_path(settings, pp.test_json)
    if not train_path.is_file():
        raise FileNotFoundError(f"Train JSON not found: {train_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Test JSON not found: {test_path}")

    _set_seed(tr_cfg.seed)
    device = torch.device(
        settings.device
        if torch.cuda.is_available() and settings.device.startswith("cuda")
        else "cpu"
    )
    print(f"Device: {device}")

    train_df_tr, val_df, test_df, le, num_classes = prepare_frames(
        train_path,
        test_path,
        pp.intent_filter_shared_only,
        pp.val_fraction,
        pp.split_seed,
    )
    if pp.strip_whitespace:
        for _df in (train_df_tr, val_df, test_df):
            _df["text"] = _df["text"].str.strip()
    print(
        f"Train: {len(train_df_tr):,}  Val: {len(val_df):,}  Test: {len(test_df):,}  classes: {num_classes}"
    )

    mask_buckets = frozenset(pp.mask_entity_buckets)
    entities = EntityResources(train_path, mask_buckets, collapse_multiword=pp.collapse_multiword)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = tr_cfg.run_name or f"{tr_cfg.recipe}_{stamp}"
    run_dir = (settings.runs_dir / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    exp.dump_yaml(run_dir / "experiment.yaml")

    recipe = tr_cfg.recipe
    bundle: dict[str, Any] = {
        "recipe": recipe,
        "label_classes": le.classes_.tolist(),
        "num_classes": num_classes,
        "train_json": str(train_path),
        "test_json": str(test_path),
    }

    if recipe == "tfidf_lr":
        if pp.tfidf_use_masking:

            def _analyze(s: str) -> list[str]:
                return entities.simple_tokenize(s, apply_mask=True)

            # Custom `analyzer`: do not pass `strip_accents` (sklearn ignores or warns).
            tfidf = TfidfVectorizer(
                analyzer=_analyze,
                ngram_range=(pp.tfidf_ngram_min, pp.tfidf_ngram_max),
                max_features=pp.tfidf_max_features,
                sublinear_tf=pp.tfidf_sublinear_tf,
            )
        else:
            tfidf = TfidfVectorizer(
                analyzer="word",
                ngram_range=(pp.tfidf_ngram_min, pp.tfidf_ngram_max),
                max_features=pp.tfidf_max_features,
                sublinear_tf=pp.tfidf_sublinear_tf,
                strip_accents=pp.tfidf_strip_accents,
                token_pattern=r"(?u)\b\w+\b",
                lowercase=pp.lowercase,
            )
        tfidf.fit(train_df_tr["text"])
        X_tr = torch.tensor(tfidf.transform(train_df_tr["text"]).toarray(), dtype=torch.float32)
        X_val = torch.tensor(tfidf.transform(val_df["text"]).toarray(), dtype=torch.float32)
        X_te = torch.tensor(tfidf.transform(test_df["text"]).toarray(), dtype=torch.float32)
        y_tr = torch.tensor(train_df_tr["label"].values, dtype=torch.long)
        y_val = torch.tensor(val_df["label"].values, dtype=torch.long)
        y_te = torch.tensor(test_df["label"].values, dtype=torch.long)

        in_features = X_tr.shape[1]
        model = LogisticRegression(in_features, num_classes)
        cw = class_weights_from_labels(y_tr, num_classes, tr_cfg.class_weight_mode)
        crit = build_criterion(tr_cfg.loss_type, cw, tr_cfg.focal_gamma, device)

        model, _hist, te_acc, te_mf1, te_wf1, te_pred = train_dense(
            model,
            X_tr,
            y_tr,
            X_val,
            y_val,
            X_te,
            y_te,
            device,
            tr_cfg.max_epochs,
            tr_cfg.early_stopping_patience,
            tr_cfg.lr,
            tr_cfg.weight_decay,
            tr_cfg.lr_scheduler_patience,
            tr_cfg.lr_scheduler_factor,
            tr_cfg.lr_scheduler_min_lr,
            crit,
            tr_cfg.batch_size,
            name="tfidf_lr",
        )

        bundle["tfidf_vectorizer"] = tfidf
        bundle["linear_in_features"] = in_features
        torch.save(model.state_dict(), run_dir / "model.pt")
    else:
        tok_type = "char" if recipe == "charcnn" else pp.tokenizer
        use_mask = pp.textcnn_use_masking and tok_type in ("word", "bpe")

        tokenizer: WordTokenizer | CharTokenizer | SentencePieceTokenizer
        if tok_type == "word":
            tokenizer = WordTokenizer(entities, mask=use_mask)
        elif tok_type == "char":
            tokenizer = CharTokenizer()
        else:
            mask_tag = "masked" if use_mask else "raw"
            spm_path = (
                settings.sentencepiece_dir / f"atis_bpe_{mask_tag}_v{pp.bpe_vocab_size}.model"
            ).resolve()
            tokenizer = SentencePieceTokenizer(spm_path, entities, mask=use_mask)
            if not spm_path.is_file():
                print(f"Training SentencePiece → {spm_path.name} …")
                tokenizer.train(
                    train_df_tr["text"].tolist(),
                    vocab_size=pp.bpe_vocab_size,
                    model_type=pp.bpe_model_type,
                    character_coverage=pp.bpe_character_coverage,
                    hard_vocab_limit=pp.sentencepiece_hard_vocab_limit,
                )
            else:
                tokenizer.load()
            spm_dest = run_dir / spm_path.name
            shutil.copy2(spm_path, spm_dest)
            bundle["sentencepiece_model"] = str(spm_dest)

        dflt = _per_recipe_defaults(recipe, tok_type)
        if recipe == "charcnn":
            max_len = pp.max_len_char
        elif tok_type == "char":
            max_len = dflt["max_len"]
        else:
            max_len = pp.max_len_word
        fs = (
            tuple(tr_cfg.textcnn_filter_sizes)
            if tr_cfg.textcnn_filter_sizes
            else dflt["filter_sizes"]
        )
        nf = (
            tr_cfg.textcnn_num_filters
            if tr_cfg.textcnn_num_filters is not None
            else dflt["num_filters"]
        )

        train_tokens = [tokenizer.tokenize(t) for t in train_df_tr["text"]]
        vocab = Vocabulary().build(train_tokens, min_freq=1)
        print(
            f"Tokenizer={tok_type}  vocab={len(vocab):,}  filters={fs}  num_filters={nf}  max_len={max_len}"
        )

        embed_kind = pp.embedding_type
        embed_dim = pp.glove_dim if embed_kind == "glove" else pp.embed_dim_learned
        embedding = build_embedding(
            vocab,
            embed_kind,
            embed_dim=embed_dim,
            glove_name=pp.glove_name,
            glove_dim=pp.glove_dim,
            freeze=pp.freeze_pretrained,
            cache=settings.vector_cache,
        )
        cnn = TextCNN(
            embedding, num_classes, filter_sizes=fs, num_filters=nf, dropout=tr_cfg.textcnn_dropout
        )
        min_len = max(fs)
        tr_seq = make_seq_loader(
            train_df_tr, tokenizer.tokenize, vocab, max_len, min_len, tr_cfg.batch_size, True
        )
        val_seq = make_seq_loader(
            val_df, tokenizer.tokenize, vocab, max_len, min_len, tr_cfg.batch_size, False
        )
        te_seq = make_seq_loader(
            test_df, tokenizer.tokenize, vocab, max_len, min_len, tr_cfg.batch_size, False
        )

        y_tr_t = torch.tensor(train_df_tr["label"].values, dtype=torch.long)
        cw = class_weights_from_labels(y_tr_t, num_classes, tr_cfg.class_weight_mode)
        crit = build_criterion(tr_cfg.loss_type, cw, tr_cfg.focal_gamma, device)

        run_label = f"CNN [{tok_type}+{embed_kind} k={'_'.join(map(str, fs))}]"
        model, _hist, te_acc, te_mf1, te_wf1, te_pred = train_seq(
            cnn,
            run_label,
            tr_seq,
            val_seq,
            te_seq,
            device,
            tr_cfg.max_epochs,
            tr_cfg.early_stopping_patience,
            tr_cfg.lr,
            tr_cfg.weight_decay,
            tr_cfg.lr_scheduler_patience,
            tr_cfg.lr_scheduler_factor,
            tr_cfg.lr_scheduler_min_lr,
            crit,
            test_df["label"].values,
        )

        bundle["tokenizer_type"] = tok_type
        bundle["vocab_itos"] = vocab.itos
        bundle["filter_sizes"] = list(fs)
        bundle["num_filters"] = nf
        bundle["max_len"] = max_len
        bundle["embedding_type"] = embed_kind
        bundle["glove_name"] = pp.glove_name
        bundle["glove_dim"] = pp.glove_dim
        bundle["embed_dim_learned"] = pp.embed_dim_learned
        bundle["freeze_pretrained"] = pp.freeze_pretrained
        torch.save(model.state_dict(), run_dir / "model.pt")

    metrics = {
        "test_accuracy": float(te_acc),
        "test_macro_f1": float(te_mf1),
        "test_weighted_f1": float(te_wf1),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with open(run_dir / "bundle.pkl", "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
