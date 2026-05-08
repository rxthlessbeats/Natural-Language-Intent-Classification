"""GloVe loading, embeddings, TextCNN, and dense logistic regression head."""

from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from urllib.request import urlretrieve

import torch
import torch.nn as nn
import torch.nn.functional as F

from atis_intent.tokenization import Vocabulary

_GLOVE_ZIP_URL = {
    "42B": "http://nlp.stanford.edu/data/glove.42B.300d.zip",
    "840B": "http://nlp.stanford.edu/data/glove.840B.300d.zip",
    "twitter.27B": "http://nlp.stanford.edu/data/glove.twitter.27B.zip",
    "6B": "http://nlp.stanford.edu/data/glove.6B.zip",
}


def load_glove_vectors(
    name: str = "6B",
    dim: int = 100,
    cache: Path | None = None,
) -> SimpleNamespace:
    """Load (or download+cache) GloVe vectors and return stoi+tensor."""
    cache = Path(cache or Path(".vector_cache").resolve())
    cache.mkdir(parents=True, exist_ok=True)
    if name not in _GLOVE_ZIP_URL:
        raise ValueError(f"Unknown GloVe bundle {name!r}; choose from {list(_GLOVE_ZIP_URL)}")
    txt_name = f"glove.{name}.{dim}d.txt"
    path_txt = cache / txt_name
    path_pt = cache / f"{txt_name}.pt"
    if path_pt.is_file():
        _itos, stoi, vectors, _d = torch.load(path_pt, map_location="cpu", weights_only=False)
        return SimpleNamespace(stoi=stoi, vectors=vectors)
    if not path_txt.is_file():
        url = _GLOVE_ZIP_URL[name]
        zip_name = url.rsplit("/", 1)[-1]
        dest_zip = cache / zip_name
        if not dest_zip.is_file():
            print(f"Downloading {zip_name} (first run only) ...")
            urlretrieve(url, dest_zip)
        with zipfile.ZipFile(dest_zip, "r") as zf:
            zf.extractall(cache)
    if not path_txt.is_file():
        raise FileNotFoundError(f"Expected {path_txt} in the archive; check name/dim.")
    itos: list[str] = []
    rows: list[list[float]] = []
    with open(path_txt, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) != dim + 1:
                continue
            itos.append(parts[0])
            rows.append([float(x) for x in parts[1:]])
    vectors = torch.tensor(rows, dtype=torch.float32)
    stoi = {w: i for i, w in enumerate(itos)}
    torch.save((itos, stoi, vectors, dim), path_pt)
    return SimpleNamespace(stoi=stoi, vectors=vectors)


def build_embedding(
    vocab: Vocabulary,
    kind: Literal["learned", "glove"],
    embed_dim: int,
    glove_name: str,
    glove_dim: int,
    freeze: bool,
    cache: Path,
) -> nn.Embedding:
    """Create an embedding layer (learned or GloVe-initialized) for a vocab."""
    n = len(vocab)
    emb = nn.Embedding(n, embed_dim, padding_idx=vocab.pad_id)
    if kind == "learned":
        nn.init.uniform_(emb.weight, -0.25, 0.25)
        with torch.no_grad():
            emb.weight[vocab.pad_id].zero_()
        return emb
    g = load_glove_vectors(glove_name, glove_dim, cache=cache)
    mat = torch.randn(n, glove_dim) * 0.25
    hits = 0
    for w, i in vocab.stoi.items():
        if w in {vocab.PAD, vocab.UNK}:
            continue
        j = g.stoi.get(w)
        if j is not None:
            mat[i] = g.vectors[j]
            hits += 1
    emb.weight.data.copy_(mat)
    if freeze:
        emb.weight.requires_grad = False
    with torch.no_grad():
        emb.weight[vocab.pad_id].zero_()
    print(f"[GloVe] hits={hits}/{n - 2}  coverage={hits / max(n - 2, 1):.1%}")
    return emb


class TextCNN(nn.Module):
    def __init__(
        self,
        embedding: nn.Embedding,
        num_classes: int,
        filter_sizes: tuple[int, ...] = (2, 3, 4),
        num_filters: int = 128,
        dropout: float = 0.5,
    ):
        """Initialize a Kim-style TextCNN classifier."""
        super().__init__()
        self.embedding = embedding
        embed_dim = embedding.embedding_dim
        self.filter_sizes = tuple(filter_sizes)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, k) for k in self.filter_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(self.filter_sizes), num_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        for conv in self.convs:
            nn.init.kaiming_uniform_(conv.weight, nonlinearity="relu")

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Compute class logits from token id sequences."""
        x = self.embedding(ids).transpose(1, 2)
        pooled = [F.relu(conv(x)).max(dim=2).values for conv in self.convs]
        out = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(out))


class LogisticRegression(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        """Initialize a single-layer linear classifier."""
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits from dense features."""
        return self.fc(x)
