from __future__ import annotations

from rylox.fusion import reciprocal_rank_fusion


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