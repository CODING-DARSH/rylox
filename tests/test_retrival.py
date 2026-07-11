from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from rylox import retrieval
from rylox.config import RyloxConfig
from rylox.indexer import run_index


class _FakeEmbedder:
    """Deterministic fake: vector = [text length, count of 'x' characters]."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t)), float(t.count("x"))] for t in texts]


def _write(repo: Path, relpath: str, content: str) -> None:
    path = repo / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_first_update_embeds_every_file(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "def a():\n    pass\n")
    _write(tmp_path, "b.py", "def b():\n    pass\n")
    run_index(tmp_path, RyloxConfig())

    fake = _FakeEmbedder("fake")
    with patch("rylox.retrieval.get_embedder", return_value=fake):
        report = retrieval.update_embeddings(tmp_path, RyloxConfig())

    assert report.embedded_files == 2
    assert report.reused_files == 0
    assert len(fake.calls) == 2


def test_second_update_with_no_changes_reuses_everything(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "def a():\n    pass\n")
    run_index(tmp_path, RyloxConfig())

    fake = _FakeEmbedder("fake")
    with patch("rylox.retrieval.get_embedder", return_value=fake):
        retrieval.update_embeddings(tmp_path, RyloxConfig())
        fake.calls.clear()
        report = retrieval.update_embeddings(tmp_path, RyloxConfig())

    assert report.embedded_files == 0
    assert report.reused_files == 1
    assert fake.calls == []


def test_only_changed_file_is_reembedded(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "def a():\n    pass\n")
    _write(tmp_path, "b.py", "def b():\n    pass\n")
    run_index(tmp_path, RyloxConfig())

    fake = _FakeEmbedder("fake")
    with patch("rylox.retrieval.get_embedder", return_value=fake):
        retrieval.update_embeddings(tmp_path, RyloxConfig())

        _write(tmp_path, "a.py", "def a():\n    return 1\n")
        run_index(tmp_path, RyloxConfig())

        fake.calls.clear()
        report = retrieval.update_embeddings(tmp_path, RyloxConfig())

    assert report.embedded_files == 1
    assert report.reused_files == 1
    assert len(fake.calls) == 1
    assert "return 1" in fake.calls[0][0]


def test_deleted_file_removed_from_embeddings(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "def a():\n    pass\n")
    _write(tmp_path, "b.py", "def b():\n    pass\n")
    run_index(tmp_path, RyloxConfig())

    fake = _FakeEmbedder("fake")
    with patch("rylox.retrieval.get_embedder", return_value=fake):
        retrieval.update_embeddings(tmp_path, RyloxConfig())

        (tmp_path / "b.py").unlink()
        run_index(tmp_path, RyloxConfig())

        report = retrieval.update_embeddings(tmp_path, RyloxConfig())

    assert report.deleted_files == 1
    assert report.reused_files == 1


def test_model_change_forces_full_reembed(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "def a():\n    pass\n")
    run_index(tmp_path, RyloxConfig())

    fake = _FakeEmbedder("model-a")
    with patch("rylox.retrieval.get_embedder", return_value=fake):
        retrieval.update_embeddings(tmp_path, RyloxConfig())

    cfg = RyloxConfig()
    cfg = replace(cfg, embedding=replace(cfg.embedding, model="model-b"))

    fake2 = _FakeEmbedder("model-b")
    with patch("rylox.retrieval.get_embedder", return_value=fake2):
        report = retrieval.update_embeddings(tmp_path, cfg)

    assert report.model_changed is True
    assert report.embedded_files == 1
    assert report.reused_files == 0


def test_load_chunk_vector_index_without_embeddings_raises(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "def a():\n    pass\n")
    run_index(tmp_path, RyloxConfig())

    with pytest.raises(retrieval.NoEmbeddedChunksError):
        retrieval.load_chunk_vector_index(tmp_path)


def test_search_returns_chunks_ranked_by_similarity(tmp_path: Path) -> None:
    # Realistic corpus size matters here: with only 2 documents, BM25's IDF
    # formula can degenerate to exactly zero for a term appearing in half
    # the corpus (see KNOWN_LIMITATIONS.md), silently making this test pass
    # off the dense signal alone without ever exercising sparse retrieval.
    _write(tmp_path, "a.py", "def xxxxx():\n    pass\n")
    _write(tmp_path, "b.py", "def y():\n    pass\n")
    _write(tmp_path, "c.py", "def z():\n    pass\n")
    _write(tmp_path, "d.py", "def w():\n    pass\n")
    run_index(tmp_path, RyloxConfig())

    fake = _FakeEmbedder("fake")
    with patch("rylox.retrieval.get_embedder", return_value=fake):
        retrieval.update_embeddings(tmp_path, RyloxConfig())
        results = retrieval.search(tmp_path, RyloxConfig(), "xxxxxxxxxx", top_k=4)

    assert len(results) == 4
    assert results[0].chunk.name == "xxxxx"


def test_search_sparse_signal_overrides_a_wrong_dense_ranking(tmp_path: Path) -> None:
    """A real adversarial case: dense actively prefers the wrong chunk,
    and only BM25 (lexical match on 'login'/'username'/'password') can
    correct it. Proves the two signals are genuinely combined through the
    real search() path, not just that fusion.py works in isolation.

    Two things had to be fixed to make this construction actually work,
    both discovered empirically rather than assumed:

    1. A simple rank1<->rank2 swap of the same two items between dense and
       sparse produces an EXACT tie under RRF (traced and confirmed with
       real numbers: both land on 0.03252247488101534) — a general
       property of rank-based fusion (see test_fusion.py's k-invariance
       test for the related 1st/last-rank case), not a bug. A fourth
       chunk ('another_function') is given a small, genuine lexical
       overlap so it lands at sparse rank 2, pushing 'unrelated_thing' to
       sparse rank 3 — breaking the exact symmetry with a real margin.
    2. That overlap word must not land at exactly 2-of-4 documents, or it
       hits the same BM25 IDF zero-crossing documented in
       KNOWN_LIMITATIONS.md (n/N = 0.5 exactly zeroes a term's IDF). A
       fifth, fully unrelated filler file keeps that ratio at 2-of-5.
    """
    _write(
        tmp_path,
        "auth.py",
        "def login_user(username, password):\n"
        "    check_login_username_password(username, password)\n",
    )
    _write(tmp_path, "b.py", "def unrelated_thing():\n    pass\n")
    _write(
        tmp_path,
        "c.py",
        "def another_function():\n    password_placeholder = None\n    do_something_else()\n",
    )
    _write(tmp_path, "d.py", "def yet_another():\n    more_unrelated_code()\n")
    _write(tmp_path, "e.py", "def fifth_filler():\n    nothing_related_here()\n")
    run_index(tmp_path, RyloxConfig())

    class _AdversarialEmbedder:
        """Dense is mildly mistaken, not maximally inverted: it ranks
        'unrelated_thing' 1st and the correct 'login_user' 2nd.
        """

        def __init__(self, model: str) -> None:
            self.model = model

        def embed(self, texts: list[str]) -> list[list[float]]:
            vectors = []
            for t in texts:
                if "unrelated_thing" in t or t == "login username password":
                    vectors.append([1.0, 0.0])
                elif "login_user" in t:
                    vectors.append([0.9, 0.1])
                elif "another_function" in t:
                    vectors.append([0.5, 0.5])
                elif "yet_another" in t:
                    vectors.append([0.3, 0.7])
                else:
                    vectors.append([0.2, 0.8])
            return vectors

    adversarial = _AdversarialEmbedder("adversarial")
    with patch("rylox.retrieval.get_embedder", return_value=adversarial):
        retrieval.update_embeddings(tmp_path, RyloxConfig())

        # Confirm the adversarial setup actually works as intended: dense
        # alone really does pick the wrong chunk, so the test below is
        # meaningful rather than accidentally already correct.
        index = retrieval.load_chunk_vector_index(tmp_path)
        query_vector = adversarial.embed(["login username password"])[0]
        dense_only = index.store.search(query_vector, top_k=5)
        assert index.resolve(dense_only[0].index).name == "unrelated_thing"

        results = retrieval.search(tmp_path, RyloxConfig(), "login username password", top_k=5)

    assert results[0].chunk.name == "login_user"
    assert "sparse" in results[0].sources
    # A real margin, not a coin-flip tie-break: the fix must leave login_user
    # strictly ahead, not merely first due to insertion-order tie-breaking.
    assert results[0].score > results[1].score


def test_search_top_k_respected(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "def a():\n    pass\ndef b():\n    pass\ndef c():\n    pass\n")
    run_index(tmp_path, RyloxConfig())

    fake = _FakeEmbedder("fake")
    with patch("rylox.retrieval.get_embedder", return_value=fake):
        retrieval.update_embeddings(tmp_path, RyloxConfig())
        results = retrieval.search(tmp_path, RyloxConfig(), "query", top_k=1)

    assert len(results) == 1


def test_search_with_expansion_includes_direct_callee_only(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "auth.py",
        "def login():\n    check_credentials()\n\n"
        "def check_credentials():\n    return db_lookup()\n\n"
        "def db_lookup():\n    return True\n\n"
        "def unrelated():\n    pass\n",
    )
    run_index(tmp_path, RyloxConfig())

    fake = _FakeEmbedder("fake")
    with patch("rylox.retrieval.get_embedder", return_value=fake):
        retrieval.update_embeddings(tmp_path, RyloxConfig())
        primary, expanded = retrieval.search_with_expansion(
            tmp_path, RyloxConfig(), "login", top_k=1
        )

    assert primary[0].chunk.name == "login"
    expanded_names = {e.chunk.name for e in expanded}
    assert expanded_names == {"check_credentials"}  # not db_lookup — two hops away


def test_search_with_expansion_scores_lower_than_primary(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "auth.py",
        "def login():\n    check_credentials()\n\ndef check_credentials():\n    return True\n",
    )
    run_index(tmp_path, RyloxConfig())

    fake = _FakeEmbedder("fake")
    with patch("rylox.retrieval.get_embedder", return_value=fake):
        retrieval.update_embeddings(tmp_path, RyloxConfig())
        primary, expanded = retrieval.search_with_expansion(
            tmp_path, RyloxConfig(), "login", top_k=1
        )

    assert expanded[0].score < primary[0].score
    assert expanded[0].reason == "callee of login"


def test_search_with_expansion_empty_primary_returns_empty_expansion(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "def a():\n    pass\n")
    run_index(tmp_path, RyloxConfig())

    fake = _FakeEmbedder("fake")
    with patch("rylox.retrieval.get_embedder", return_value=fake):
        retrieval.update_embeddings(tmp_path, RyloxConfig())
        primary, expanded = retrieval.search_with_expansion(
            tmp_path, RyloxConfig(), "query", top_k=0
        )

    assert primary == []
    assert expanded == []