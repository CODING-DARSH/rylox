from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")


def tokenize(text: str) -> list[str]:
    """Split code/docstring text into lowercase word/identifier tokens.

    Emits both the full identifier and its underscore-separated parts —
    e.g. `login_user` produces `login_user`, `login`, and `user`. Without
    this, a query for "login" would never match a real chunk called
    `login_user` or `check_login_password`, since those are single tokens
    as far as a naive identifier regex is concerned. snake_case is the
    Python norm, so this split matters far more here than it would for a
    typical natural-language BM25 use case.
    """
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        word = match.group(0).lower()
        tokens.append(word)
        if "_" in word:
            tokens.extend(part for part in word.split("_") if part)
    return tokens


class EmptyCorpusError(Exception):
    """Raised when trying to build a BM25 index from zero documents."""


@dataclass(frozen=True)
class BM25SearchResult:
    index: int  # position in the original documents list passed to build_bm25_index
    score: float  # BM25 score, higher is more relevant, unbounded (not 0-1)


class BM25Index:
    def __init__(self, model: BM25Okapi, size: int) -> None:
        self._model = model
        self._size = size

    @property
    def size(self) -> int:
        return self._size

    def search(self, query: str, top_k: int) -> list[BM25SearchResult]:
        if top_k <= 0 or self._size == 0:
            return []

        query_tokens = tokenize(query)
        scores = self._model.get_scores(query_tokens)

        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        top = ranked[:top_k]
        return [BM25SearchResult(index=idx, score=float(score)) for idx, score in top]


def build_bm25_index(documents: list[str]) -> BM25Index:
    if not documents:
        raise EmptyCorpusError("cannot build a BM25 index from zero documents")

    tokenized = [tokenize(doc) for doc in documents]
    model = BM25Okapi(tokenized)
    return BM25Index(model=model, size=len(documents))