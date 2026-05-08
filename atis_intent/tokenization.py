"""Tokenizers and vocabulary (notebook §4a)."""

from __future__ import annotations

from pathlib import Path

import sentencepiece as spm

from atis_intent.entities import EntityResources


class WordTokenizer:
    def __init__(self, entities: EntityResources, mask: bool, stopwords: set[str] | None = None):
        """Create a word tokenizer with optional masking/stopword removal."""
        self._e = entities
        self.mask = mask
        self.stopwords = stopwords

    def tokenize(self, text: str) -> list[str]:
        """Tokenize a text string into word-level tokens."""
        toks = self._e.word_full_tokenize(text, apply_mask=self.mask)
        if not self.stopwords:
            return toks
        return [t for t in toks if t not in self.stopwords]


class CharTokenizer:
    def tokenize(self, text: str) -> list[str]:
        """Tokenize a text string into character tokens."""
        return list(text.lower())


class SentencePieceTokenizer:
    def __init__(
        self,
        model_path: Path,
        entities: EntityResources,
        mask: bool,
        stopwords: set[str] | None = None,
    ):
        """Create a SentencePiece tokenizer wrapper (train/load/tokenize)."""
        self.model_path = model_path
        self._e = entities
        self.mask = mask
        self.stopwords = stopwords
        self._sp: spm.SentencePieceProcessor | None = None

    def train(
        self,
        corpus: list[str],
        vocab_size: int,
        model_type: str = "bpe",
        character_coverage: float = 1.0,
        hard_vocab_limit: bool = False,
        user_defined_symbols: list[str] | None = None,
    ) -> SentencePieceTokenizer:
        """Train a SentencePiece model on a corpus and load it."""
        from atis_intent.entities import SENTENCEPIECE_USER_DEFINED_SYMBOLS

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        processed = [self._preprocess(s) for s in corpus]
        corpus_file = self.model_path.with_suffix(".txt")
        corpus_file.write_text("\n".join(processed), encoding="utf-8")
        uds = (
            user_defined_symbols
            if user_defined_symbols is not None
            else SENTENCEPIECE_USER_DEFINED_SYMBOLS
        )
        spm.SentencePieceTrainer.Train(
            input=str(corpus_file),
            model_prefix=str(self.model_path.with_suffix("")),
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            pad_id=0,
            unk_id=1,
            bos_id=-1,
            eos_id=-1,
            user_defined_symbols=uds,
            hard_vocab_limit=hard_vocab_limit,
        )
        return self.load()

    def load(self) -> SentencePieceTokenizer:
        """Load the SentencePiece model from disk."""
        self._sp = spm.SentencePieceProcessor(model_file=str(self.model_path))
        return self

    def tokenize(self, text: str) -> list[str]:
        """Tokenize a text string into SentencePiece subword tokens."""
        if self._sp is None:
            self.load()
        assert self._sp is not None
        pre = self._preprocess(text)
        return self._sp.encode(pre, out_type=str)

    def _preprocess(self, text: str) -> str:
        """Preprocess text for SentencePiece (masking + optional stopwords)."""
        pre = self._e.preprocess_for_sentencepiece(text, self.mask)
        if not self.stopwords:
            return pre
        toks = pre.split()
        toks = [t for t in toks if t not in self.stopwords]
        return " ".join(toks)


class Vocabulary:
    PAD, UNK = "<pad>", "<unk>"

    def __init__(self):
        """Create an empty vocabulary with PAD/UNK specials."""
        self.itos: list[str] = []
        self.stoi: dict[str, int] = {}

    def build(self, token_lists: list[list[str]], min_freq: int = 1) -> Vocabulary:
        """Build vocab mappings from a tokenized corpus."""
        from collections import Counter

        ctr: Counter[str] = Counter()
        for toks in token_lists:
            ctr.update(toks)
        specials = [self.PAD, self.UNK]
        rest = sorted([w for w, c in ctr.items() if c >= min_freq and w not in specials])
        self.itos = specials + rest
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        return self

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[self.PAD]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.UNK]
