from __future__ import annotations

from dataclasses import dataclass

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusedResult:
    index: int  # the chunk index shared across all input rankings
    score: float  # fused RRF score, higher is more relevant
    sources: frozenset[str]  # which ranking(s) contributed this result, e.g. {"dense", "sparse"}


def reciprocal_rank_fusion(
    rankings: dict[str, list[int]], k: int = DEFAULT_RRF_K
) -> list[FusedResult]:
    """Fuse multiple rankings of the same items by rank position.

    `rankings` maps a source name (e.g. "dense", "sparse") to an ordered
    list of item indices, best match first. Rank-based fusion is used
    instead of combining raw scores directly, since cosine similarity
    (dense) and BM25 scores (sparse) live on entirely different,
    non-comparable scales — only their relative ordering is meaningful.

    RRF score for an item = sum over every ranking it appears in of
    1 / (k + rank), where rank is 1-indexed. An item missing from a
    ranking simply contributes nothing from that ranking, rather than
    being penalized with a worst-case rank.
    """
    scores: dict[int, float] = {}
    sources: dict[int, set[str]] = {}

    for source_name, ranked_indices in rankings.items():
        for position, item_index in enumerate(ranked_indices):
            rank = position + 1
            contribution = 1.0 / (k + rank)
            scores[item_index] = scores.get(item_index, 0.0) + contribution
            sources.setdefault(item_index, set()).add(source_name)

    fused = [
        FusedResult(index=idx, score=score, sources=frozenset(sources[idx]))
        for idx, score in scores.items()
    ]
    fused.sort(key=lambda r: r.score, reverse=True)
    return fused