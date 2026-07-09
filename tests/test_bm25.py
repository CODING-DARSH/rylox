from __future__ import annotations

import pytest

from rylox.bm25 import EmptyCorpusError, build_bm25_index, tokenize


def test_tokenize_splits_identifiers_and_lowercases() -> None:
    assert tokenize("def login_user(username, password):") == [
        "def",
        "login_user",
        "login",
        "user",
        "username",
        "password",
    ]


def test_tokenize_emits_both_full_identifier_and_underscore_parts() -> None:
    assert tokenize("check_login_password") == [
        "check_login_password",
        "check",
        "login",
        "password",
    ]


def test_tokenize_splits_numbers_separately() -> None:
    assert tokenize("x1 = 42") == ["x1", "42"]


def test_tokenize_ignores_punctuation_and_whitespace() -> None:
    assert tokenize("a.b, c:d;") == ["a", "b", "c", "d"]


def test_tokenize_empty_string_returns_empty_list() -> None:
    assert tokenize("") == []


def test_build_bm25_index_rejects_empty_corpus() -> None:
    with pytest.raises(EmptyCorpusError):
        build_bm25_index([])


def test_build_bm25_index_reports_correct_size() -> None:
    index = build_bm25_index(["def a(): pass", "def b(): pass"])
    assert index.size == 2


def test_document_matching_all_query_terms_ranks_highest() -> None:
    docs = [
        "def login(username, password):\n    check_credentials(username, password)",
        "def add(a, b):\n    return a + b",
        "class UserSession:\n    def login(self):\n        pass",
    ]
    index = build_bm25_index(docs)
    results = index.search("login username password", top_k=3)
    assert results[0].index == 0


def test_document_with_no_matching_terms_scores_zero() -> None:
    docs = ["def login(): pass", "def add(a, b): return a + b"]
    index = build_bm25_index(docs)
    results = index.search("login", top_k=2)
    unrelated = next(r for r in results if r.index == 1)
    assert unrelated.score == 0.0


def test_top_k_larger_than_corpus_clamps_to_available() -> None:
    index = build_bm25_index(["a", "b"])
    results = index.search("a", top_k=100)
    assert len(results) == 2


def test_top_k_zero_returns_empty_list() -> None:
    index = build_bm25_index(["a", "b"])
    assert index.search("a", top_k=0) == []


def test_negative_top_k_returns_empty_list() -> None:
    index = build_bm25_index(["a", "b"])
    assert index.search("a", top_k=-1) == []


def test_empty_query_does_not_crash() -> None:
    index = build_bm25_index(["def a(): pass", "def b(): pass"])
    results = index.search("", top_k=2)
    assert len(results) == 2


def test_repeating_a_term_increases_score_but_with_diminishing_returns() -> None:
    """BM25's whole point vs. naive term counting: relevance grows with
    term frequency, but saturates — it shouldn't be linear.

    Two variables are deliberately controlled here, both discovered by a
    first attempt at this test failing for the wrong reason:
    - Document length is held constant (all docs are 10 tokens via 'pad'
      padding), so length normalization doesn't confound the measurement.
    - 'login' is kept a minority term (3 of 8 documents), since BM25's IDF
      goes negative for a term appearing in a majority of the corpus —
      that would invert the entire experiment.
    """
    docs = [
        "login pad pad pad pad pad pad pad pad pad",
        "login login pad pad pad pad pad pad pad pad",
        "login login login login pad pad pad pad pad pad",
        "pad pad pad pad pad pad pad pad pad pad",
        "pad pad pad pad pad pad pad pad pad pad",
        "pad pad pad pad pad pad pad pad pad pad",
        "pad pad pad pad pad pad pad pad pad pad",
        "pad pad pad pad pad pad pad pad pad pad",
    ]
    index = build_bm25_index(docs)
    results = index.search("login", top_k=8)
    scores = {r.index: r.score for r in results}

    assert scores[0] < scores[1] < scores[2]
    first_gain = scores[1] - scores[0]
    second_gain = scores[2] - scores[1]
    assert second_gain < first_gain


def test_short_document_scores_higher_than_long_document_with_same_term_count() -> None:
    """Document-length normalization: the same single mention of the query
    term should count for more in a short, focused document than buried in
    a long one — that's the `b` parameter's whole purpose."""
    short_doc = "login"
    long_doc = "login " + "padding word here filler text more content " * 20
    index = build_bm25_index([short_doc, long_doc, "unrelated filler content"])
    results = index.search("login", top_k=2)
    scores = {r.index: r.score for r in results}
    assert scores[0] > scores[1]


def test_idf_can_be_exactly_zero_on_a_two_document_corpus() -> None:
    """Regression/documentation test: classical BM25 IDF is
    ln((N - n + 0.5) / (n + 0.5)), which crosses exactly zero when a term
    appears in almost exactly half the corpus. With N=2, n=1, that's
    ln(1.5/1.5) = 0. This is real, inherent BM25 behavior on tiny corpora
    (see KNOWN_LIMITATIONS.md) — locked in here so nobody mistakes it for
    a regression if `rank_bm25`'s formula ever changes, in either direction.
    """
    docs = ["login credentials here", "completely different unrelated words"]
    index = build_bm25_index(docs)
    results = index.search("login", top_k=2)
    assert {r.score for r in results} == {0.0}



    docs = [
        "unrelated content here",
        "login login login",
        "login mentioned once",
    ]
    index = build_bm25_index(docs)
    results = index.search("login", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)