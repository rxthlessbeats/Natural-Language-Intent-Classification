"""Pydantic configuration: environment settings and experiment YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Recipe = Literal["tfidf_lr", "textcnn", "charcnn"]
Tokenizer = Literal["word", "char", "bpe"]
EmbeddingType = Literal["learned", "glove"]
LossType = Literal["ce", "focal"]
ClassWeightMode = Literal["inv", "dampened_inv"]


class Settings(BaseSettings):
    """Paths and device from environment / `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="ATIS_", extra="ignore")

    data_dir: Path = Path("data")
    vector_cache: Path = Path(".vector_cache")
    sentencepiece_dir: Path = Path(".spm")
    runs_dir: Path = Path("runs")
    device: str = "cuda"


class PreprocessConfig(BaseModel):
    """All preprocessing knobs (defaults match `notebooks/deep_learning.ipynb`)."""

    train_json: str = "train.json"
    test_json: str = "test.json"
    intent_filter_shared_only: bool = False
    val_fraction: float = Field(0.2, gt=0, lt=1)
    split_seed: int = 42
    lowercase: bool = True
    strip_whitespace: bool = True
    collapse_multiword: bool = True
    mask_entity_buckets: list[str] = Field(
        default_factory=lambda: [
            "city",
            "airline",
            "airport-code",
            "airport-name",
            "state",
            "date",
            "time",
            "numeric",
        ]
    )
    tfidf_use_masking: bool = True
    textcnn_use_masking: bool = False
    tokenizer: Tokenizer = "word"
    bpe_vocab_size: int = Field(3000, ge=100)
    bpe_model_type: str = "bpe"
    bpe_character_coverage: float = Field(1.0, gt=0, le=1)
    sentencepiece_hard_vocab_limit: bool = False
    max_len_word: int = Field(50, ge=1)
    max_len_char: int = Field(200, ge=1)
    tfidf_max_features: int = Field(8000, ge=1)
    tfidf_ngram_min: int = Field(1, ge=1)
    tfidf_ngram_max: int = Field(2, ge=1)
    tfidf_sublinear_tf: bool = True
    tfidf_strip_accents: str | None = "unicode"
    # Shared stop-word removal settings for both TF-IDF and CNN tokenization.
    remove_stopwords: bool = False
    stopword_reserve: list[str] = Field(
        default_factory=lambda: [
            "from",
            "to",
            "on",
            "in",
            "at",
            "between",
            "and",
            "or",
            "via",
            "than",
            "my",
            "would",
            "like",
            "need",
            "show",
            "list",
            "give",
        ]
    )
    embedding_type: EmbeddingType = "glove"
    glove_name: str = "6B"
    glove_dim: int = Field(100, ge=1)
    freeze_pretrained: bool = False
    embed_dim_learned: int = Field(128, ge=1)
    # After stratified train/val split: duplicate train rows with token-dropout aug (notebook §1.6).
    random_deletion: bool = False
    random_deletion_p: float = Field(0.14, ge=0, lt=0.15)


class TrainConfig(BaseModel):
    recipe: Recipe = "textcnn"
    seed: int = 42
    batch_size: int = Field(64, ge=1)
    max_epochs: int = Field(40, ge=1)
    early_stopping_patience: int = Field(10, ge=0)
    lr: float = Field(3e-3, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    lr_scheduler_patience: int = Field(2, ge=0)
    lr_scheduler_factor: float = Field(0.5, gt=0, lt=1)
    lr_scheduler_min_lr: float = Field(1e-6, ge=0)
    loss_type: LossType = "focal"
    class_weight_mode: ClassWeightMode | None = "dampened_inv"
    focal_gamma: float = Field(2.0, ge=0)
    textcnn_filter_sizes: list[int] | None = None
    textcnn_num_filters: int | None = None
    textcnn_dropout: float = Field(0.5, ge=0, le=1)
    run_name: str | None = None


class ExperimentConfig(BaseModel):
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)

    @model_validator(mode="after")
    def _recipe_tokenizer(self) -> ExperimentConfig:
        r = self.train.recipe
        if r == "charcnn":
            return self
        if r == "textcnn":
            if self.preprocess.tokenizer not in ("word", "char", "bpe"):
                raise ValueError("tokenizer must be word|char|bpe for textcnn")
        return self

    def dump_yaml(self, path: Path) -> None:
        path.write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load experiment YAML into a validated ExperimentConfig (or defaults)."""
    p = Path(path)
    if not p.is_file():
        return ExperimentConfig()
    data: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return ExperimentConfig(**data)


def resolve_data_path(settings: Settings, rel: str) -> Path:
    """Resolve a data path relative to Settings.data_dir."""
    p = Path(rel)
    return p if p.is_absolute() else (settings.data_dir / p).resolve()
