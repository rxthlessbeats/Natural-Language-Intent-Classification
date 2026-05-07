"""Load ATIS JSON, optional intent filter, stratified train/val split, labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_rasa_json(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for ex in data["rasa_nlu_data"]["common_examples"]:
        rows.append({"text": ex["text"].strip(), "intent": ex["intent"].strip()})
    return pd.DataFrame(rows)


def apply_intent_filter_shared_only(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared = set(train_df["intent"]) & set(test_df["intent"])
    tr = train_df[train_df["intent"].isin(shared)].reset_index(drop=True)
    te = test_df[test_df["intent"].isin(shared)].reset_index(drop=True)
    return tr, te


def fit_label_encoder(train_df: pd.DataFrame, test_df: pd.DataFrame) -> LabelEncoder:
    le = LabelEncoder()
    intents = np.sort(pd.unique(pd.concat([train_df["intent"], test_df["intent"]])))
    le.fit(intents)
    return le


def stratified_val_split(
    df: pd.DataFrame,
    val_frac: float = 0.2,
    seed: int = 42,
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    single = df.groupby(label_col).filter(lambda g: len(g) == 1)
    multi = df.groupby(label_col).filter(lambda g: len(g) > 1)
    tr_m, va_m = train_test_split(
        multi,
        test_size=val_frac,
        random_state=seed,
        stratify=multi[label_col],
    )
    tr = pd.concat([tr_m, single]).reset_index(drop=True)
    va = va_m.reset_index(drop=True)
    return tr, va


def prepare_frames(
    train_path: Path,
    test_path: Path,
    intent_filter_shared_only: bool,
    val_fraction: float,
    split_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder, int]:
    train_df = load_rasa_json(train_path)
    test_df = load_rasa_json(test_path)
    if intent_filter_shared_only:
        train_df, test_df = apply_intent_filter_shared_only(train_df, test_df)
    le = fit_label_encoder(train_df, test_df)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["label"] = le.transform(train_df["intent"])
    test_df["label"] = le.transform(test_df["intent"])
    train_df_tr, val_df = stratified_val_split(train_df, val_frac=val_fraction, seed=split_seed)
    num_classes = len(le.classes_)
    return train_df_tr, val_df, test_df, le, num_classes
