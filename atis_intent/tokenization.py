"""Tokenizers and vocabulary (notebook §4a)."""

from __future__ import annotations

from pathlib import Path

import sentencepiece as spm

from atis_intent.entities import EntityResources


class WordTokenizer:
    def __init__(self, entities: EntityResources, mask: bool):
        self._e = entities
        self.mask = mask

    def tokenize(self, text: str) -> list[str]:
        return self._e.word_full_tokenize(text, apply_mask=self.mask)


class CharTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return list(text.lower())


class SentencePieceTokenizer:
    def __init__(
        self,
        model_path: Path,
        entities: EntityResources,
        mask: bool,
    ):
        self.model_path = model_path
        self._e = entities
        self.mask = mask
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
        from atis_intent.entities import SENTENCEPIECE_USER_DEFINED_SYMBOLS

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        processed = [self._e.preprocess_for_sentencepiece(s, self.mask) for s in corpus]
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
        self._sp = spm.SentencePieceProcessor(model_file=str(self.model_path))
        return self

    def tokenize(self, text: str) -> list[str]:
        if self._sp is None:
            self.load()
        assert self._sp is not None
        pre = self._e.preprocess_for_sentencepiece(text, self.mask)
        return self._sp.encode(pre, out_type=str)


class Vocabulary:
    PAD, UNK = "<pad>", "<unk>"

    def __init__(self):
        self.itos: list[str] = []
        self.stoi: dict[str, int] = {}

    def build(self, token_lists: list[list[str]], min_freq: int = 1) -> Vocabulary:
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
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[self.PAD]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.UNK]
