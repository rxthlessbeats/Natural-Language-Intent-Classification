"""Entity sets from train annotations, multi-word collapse, masking (notebook §1.5.5)."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

# TF-IDF / simple path: bracket placeholders + alnum/underscore tokens (no isolated punct)
WORD_RE_TFIDF = re.compile(r"\[[a-z_]+\]|[a-zA-Z0-9_]+")

# TextCNN word path: also split stray punctuation
WORD_RE_FULL = re.compile(r"\[[a-z_]+\]|[a-zA-Z0-9_]+|[^\sa-zA-Z0-9_]")

WORD_CHUNK = re.compile(r"[a-zA-Z0-9]+")
NUMERIC = re.compile(r"^\d+$")

_MASK_TOKEN = {
    "city": "[city]",
    "airline": "[airline]",
    "airport-code": "[airport_code]",
    "airport-name": "[airport_name]",
    "state": "[state]",
    "date": "[date]",
    "time": "[time]",
    "numeric": "[num]",
}

SENTENCEPIECE_USER_DEFINED_SYMBOLS = list(_MASK_TOKEN.values())


def collect_entity_values(json_path: Path) -> dict[str, set]:
    """Collect entity values by entity name from a Rasa NLU JSON file."""
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    out: dict[str, set] = defaultdict(set)
    for ex in raw["rasa_nlu_data"]["common_examples"]:
        for ent in ex.get("entities", []):
            out[ent["entity"]].add(ent["value"].strip().lower())
    return dict(out)


def gather(ent2vals: dict, suffix: str) -> set:
    """Union entity values whose key matches or ends with the given suffix."""
    bag: set = set()
    for k, v in ent2vals.items():
        if k == suffix or k.endswith("." + suffix):
            bag |= v
    return bag


class EntityResources:
    """Built from train JSON; drives collapse, bucketing, and optional masking."""

    def __init__(
        self,
        train_json: Path,
        mask_buckets: frozenset[str],
        collapse_multiword: bool = True,
    ):
        ent2vals = collect_entity_values(train_json)
        self.mask_buckets = mask_buckets
        self.collapse_multiword = collapse_multiword

        self.known_cities = gather(ent2vals, "city_name")
        self.known_airline_names = gather(ent2vals, "airline_name")
        self.known_airline_codes = gather(ent2vals, "airline_code")
        self.known_airport_names = gather(ent2vals, "airport_name")
        self.known_airport_codes = gather(ent2vals, "airport_code")
        self.known_state_names = gather(ent2vals, "state_name")
        self.known_state_codes = gather(ent2vals, "state_code")
        self.known_day_names = gather(ent2vals, "day_name")
        self.known_day_numbers = gather(ent2vals, "day_number")
        self.known_month_names = gather(ent2vals, "month_name")
        self.known_years = gather(ent2vals, "year")
        self.known_today_rel = gather(ent2vals, "today_relative")
        self.known_times = gather(ent2vals, "time")
        self.known_periods = gather(ent2vals, "period_of_day")
        self.known_start_times = gather(ent2vals, "start_time")
        self.known_end_times = gather(ent2vals, "end_time")

        def to_tok(p: str) -> str:
            return "_".join(WORD_CHUNK.findall(p.lower()))

        self.cities_tok = {to_tok(c) for c in self.known_cities}
        self.airlines_tok = {to_tok(a) for a in self.known_airline_names | self.known_airline_codes}
        self.airport_codes_tok = {to_tok(a) for a in self.known_airport_codes}
        self.airport_names_tok = {to_tok(a) for a in self.known_airport_names}
        self.states_tok = {to_tok(s) for s in self.known_state_names | self.known_state_codes}
        self.date_tok = (
            {to_tok(d) for d in self.known_day_names}
            | {to_tok(d) for d in self.known_day_numbers}
            | {to_tok(d) for d in self.known_month_names}
            | {to_tok(d) for d in self.known_years}
            | {to_tok(d) for d in self.known_today_rel}
        )
        self.time_tok = (
            {to_tok(t) for t in self.known_times}
            | {to_tok(t) for t in self.known_periods}
            | {to_tok(t) for t in self.known_start_times}
            | {to_tok(t) for t in self.known_end_times}
        )

        phrases = sorted(
            {
                p
                for s in (
                    self.known_cities,
                    self.known_airline_names,
                    self.known_airport_names,
                    self.known_state_names,
                    self.known_day_names,
                    self.known_day_numbers,
                    self.known_month_names,
                    self.known_today_rel,
                    self.known_times,
                    self.known_periods,
                    self.known_start_times,
                    self.known_end_times,
                )
                for p in s
                if " " in p
            },
            key=len,
            reverse=True,
        )
        self._multi_re = (
            re.compile(
                r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b",
                flags=re.IGNORECASE,
            )
            if phrases and collapse_multiword
            else None
        )

    def collapse_multiword_text(self, text: str) -> str:
        if self._multi_re is None:
            return text

        def repl(m: re.Match[str]) -> str:
            return "_".join(WORD_CHUNK.findall(m.group(0).lower()))

        return self._multi_re.sub(repl, text)

    def bucket(self, w: str) -> str:
        if w in self.date_tok:
            return "date"
        if w in self.time_tok:
            return "time"
        if NUMERIC.match(w):
            return "numeric"
        if w in self.cities_tok:
            return "city"
        if w in self.airlines_tok:
            return "airline"
        if w in self.airport_codes_tok:
            return "airport-code"
        if w in self.airport_names_tok:
            return "airport-name"
        if w in self.states_tok:
            return "state"
        return "general-english"

    def mask_token(self, tok: str, apply_mask: bool) -> str:
        if not apply_mask or not self.mask_buckets:
            return tok
        b = self.bucket(tok)
        if b in self.mask_buckets:
            return _MASK_TOKEN[b]
        return tok

    def simple_tokenize(self, text: str, apply_mask: bool) -> list[str]:
        """Tokenizer for TF-IDF when masking (matches notebook `simple_tokenize`)."""
        t = self.collapse_multiword_text(text)
        if True:  # lowercase always for this path in notebook
            t = t.lower()
        toks = WORD_RE_TFIDF.findall(t)
        return [self.mask_token(x, apply_mask) for x in toks]

    def word_full_tokenize(self, text: str, apply_mask: bool) -> list[str]:
        t = self.collapse_multiword_text(text.lower())
        toks = WORD_RE_FULL.findall(t)
        return [self.mask_token(x, apply_mask) for x in toks]

    def preprocess_for_sentencepiece(self, text: str, apply_mask: bool) -> str:
        t = self.collapse_multiword_text(text.lower())
        toks = WORD_RE_FULL.findall(t)
        masked = [self.mask_token(x, apply_mask) for x in toks]
        return " ".join(masked)
