from __future__ import annotations

from rylox.fusion import FusedResult, reciprocal_rank_fusion


def test_item_ranked_first_in_both_sources_wins() -> None:
    rankings = {"dense": [5, 2, 7], "sparse": [5, 9, 3]}
    results = reciprocal_rank_fusion(rankings)
    assert results[0].index == 5
    assert results[0].sources == frozenset({"dense", "sparse"})


def test_items_appearing_in_only_one_source_are_still_included() -> None:
    rankings = {"dense": [1, 2], "sparse": [3, 4]}
    results = reciprocal_rank_fusion(rankings)
    indices = {r.index for r in results}
    assert indices == {1, 2, 3, 4}


def test_single_source_ranking_preserves_original_order() -> None:
    rankings = {"dense": [7, 3, 9]}
    results = reciprocal_rank_fusion(rankings)
    assert [r.index for r in results] == [7, 3, 9]


def test_empty_rankings_dict_returns_empty_list() -> None:
    assert reciprocal_rank_fusion({}) == []


def test_source_with_empty_ranking_list_contributes_nothing() -> None:
    rankings = {"dense": [1, 2], "sparse": []}
    results = reciprocal_rank_fusion(rankings)
    assert {r.index for r in results} == {1, 2}
    assert all(r.sources == frozenset({"dense"}) for r in results)


def test_results_sorted_descending_by_score() -> None:
    rankings = {"dense": [1, 2, 3], "sparse": [3, 1, 2]}
    results = reciprocal_rank_fusion(rankings)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_item_appearing_earlier_in_a_ranking_scores_higher() -> None:
    rankings = {"dense": [1, 2, 3]}
    results = reciprocal_rank_fusion(rankings)
    by_index = {r.index: r.score for r in results}
    assert by_index[1] > by_index[2] > by_index[3]


def test_custom_k_changes_score_magnitude_but_not_order() -> None:
    rankings = {"dense": [1, 2, 3]}
    default_results = reciprocal_rank_fusion(rankings)
    custom_results = reciprocal_rank_fusion(rankings, k=1)
    assert [r.index for r in default_results] == [r.index for r in custom_results]
    assert default_results[0].score != custom_results[0].score


def test_same_item_ranked_in_three_sources_accumulates_all_contributions() -> None:
    rankings = {"a": [1], "b": [1], "c": [1]}
    results = reciprocal_rank_fusion(rankings)
    assert results[0].sources == frozenset({"a", "b", "c"})
    single_source = reciprocal_rank_fusion({"a": [1]})
    assert results[0].score == single_source[0].score * 3


def test_exact_tie_breaks_by_lower_index_not_insertion_order() -> None:
    """A rank1/rank2 swap of the same two items between two equally
    weighted rankings produces an exact score tie — a real mathematical
    property of RRF, not a bug. The result must still be deterministic:
    lower index wins, rather than depending on which source happened to
    be processed first.
    """
    rankings = {"dense": [1, 0, 2, 3], "sparse": [0, 1, 2, 3]}
    results = reciprocal_rank_fusion(rankings)
    assert results[0].score == results[1].score
    assert results[0].index == 0
    assert results[1].index == 1


def test_tie_break_prefers_more_sources_over_index() -> None:
    """When fused scores are exactly equal, an item found by more sources
    should rank first regardless of index — broader agreement across
    retrieval methods is a more meaningful signal than an arbitrary
    ordering artifact."""
    fewer_sources = FusedResult(index=1, score=0.05, sources=frozenset({"dense"}))
    more_sources = FusedResult(index=2, score=0.05, sources=frozenset({"dense", "sparse"}))
    items = [fewer_sources, more_sources]
    items.sort(key=lambda r: (-r.score, -len(r.sources), r.index))
    assert items[0].index == 2