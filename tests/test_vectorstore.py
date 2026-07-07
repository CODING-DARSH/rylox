from __future__ import annotations

import pytest

from rylox.vectorstore import DimensionMismatchError, EmptyIndexError, build_index


def test_build_index_rejects_empty_vectors() -> None:
    with pytest.raises(EmptyIndexError):
        build_index([])


def test_build_index_rejects_mismatched_dimensions() -> None:
    with pytest.raises(DimensionMismatchError):
        build_index([[1.0, 2.0], [1.0, 2.0, 3.0]])


def test_build_index_reports_correct_dimension_and_size() -> None:
    store = build_index([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert store.dimension == 2
    assert store.size == 3


def test_search_rejects_wrong_query_dimension() -> None:
    store = build_index([[1.0, 0.0, 0.0]])
    with pytest.raises(DimensionMismatchError):
        store.search([1.0, 0.0], top_k=1)


def test_exact_match_scores_highest() -> None:
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    store = build_index(vectors)
    results = store.search([1.0, 0.0, 0.0], top_k=3)
    assert results[0].index == 0
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_similar_vectors_rank_above_dissimilar() -> None:
    vectors = [
        [1.0, 0.0, 0.0],  # near-identical to query
        [0.9, 0.1, 0.0],
        [0.0, 0.0, 1.0],  # orthogonal to query
    ]
    store = build_index(vectors)
    results = store.search([1.0, 0.0, 0.0], top_k=3)
    ordered_indices = [r.index for r in results]
    assert ordered_indices.index(0) < ordered_indices.index(2)
    assert ordered_indices.index(1) < ordered_indices.index(2)


def test_orthogonal_vector_scores_near_zero() -> None:
    store = build_index([[1.0, 0.0], [0.0, 1.0]])
    results = store.search([1.0, 0.0], top_k=2)
    orthogonal = next(r for r in results if r.index == 1)
    assert orthogonal.score == pytest.approx(0.0, abs=1e-5)


def test_top_k_larger_than_index_size_clamps_to_available() -> None:
    store = build_index([[1.0, 0.0], [0.0, 1.0]])
    results = store.search([1.0, 0.0], top_k=100)
    assert len(results) == 2


def test_top_k_zero_returns_empty_list() -> None:
    store = build_index([[1.0, 0.0], [0.0, 1.0]])
    assert store.search([1.0, 0.0], top_k=0) == []


def test_negative_top_k_returns_empty_list() -> None:
    store = build_index([[1.0, 0.0], [0.0, 1.0]])
    assert store.search([1.0, 0.0], top_k=-5) == []


def test_single_vector_index_search_returns_that_vector() -> None:
    store = build_index([[3.0, 4.0]])
    results = store.search([1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].index == 0


def test_zero_vector_does_not_crash_normalization() -> None:
    """A zero vector has no direction; the normalizer must not divide by
    zero and crash — it should still return a valid, searchable index."""
    store = build_index([[0.0, 0.0], [1.0, 0.0]])
    results = store.search([1.0, 0.0], top_k=2)
    assert len(results) == 2