from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np


class EmptyIndexError(Exception):
    """Raised when trying to build an index from zero vectors."""


class DimensionMismatchError(Exception):
    """Raised when input vectors don't all share the same dimension."""


@dataclass(frozen=True)
class SearchResult:
    index: int  # position in the original vectors list passed to build_index
    score: float  # cosine similarity, higher is more similar


class VectorStore:
    """Cosine-similarity search over a fixed set of vectors.

    Vectors are L2-normalized before insertion so that FAISS's inner-product
    index (`IndexFlatIP`) computes cosine similarity directly.
    """

    def __init__(self, index: faiss.IndexFlatIP, dimension: int) -> None:
        self._index = index
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    def search(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        if len(query_vector) != self._dimension:
            raise DimensionMismatchError(
                f"query vector has dimension {len(query_vector)}, "
                f"index expects {self._dimension}"
            )
        if top_k <= 0:
            return []

        query = _normalize(np.array([query_vector], dtype=np.float32))
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(SearchResult(index=int(idx), score=float(score)))
        return results


def build_index(vectors: list[list[float]]) -> VectorStore:
    if not vectors:
        raise EmptyIndexError("cannot build a vector index from zero vectors")

    dimension = len(vectors[0])
    for vector in vectors:
        if len(vector) != dimension:
            raise DimensionMismatchError(
                f"all vectors must share the same dimension; "
                f"got {dimension} and {len(vector)}"
            )

    matrix = _normalize(np.array(vectors, dtype=np.float32))
    index = faiss.IndexFlatIP(dimension)
    index.add(matrix)
    return VectorStore(index=index, dimension=dimension)


def _normalize(matrix: Any) -> Any:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid divide-by-zero for a zero vector
    return matrix / norms