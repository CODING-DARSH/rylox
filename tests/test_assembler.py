from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from rylox import retrieval
from rylox.assembler import assemble_context
from rylox.config import RyloxConfig
from rylox.errors import BudgetTooSmallError
from rylox.indexer import run_index


def _write(repo: Path, relpath: str, content: str) -> None:
    path = repo / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class _FakeEmbedder:
    def __init__(self, model: str) -> None:
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), float(t.count("x"))] for t in texts]


def _char_count(text: str) -> int:
    """Deterministic stand-in for a real tokenizer — no network involved."""
    return len(text)


def test_generous_budget_includes_entry_point_and_callee(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "auth.py",
        "def login():\n    check_credentials()\n\n"
        "def check_credentials():\n    return True\n\n"
        "def unrelated():\n    pass\n",
    )
    run_index(tmp_path, RyloxConfig())

    with patch("rylox.retrieval.get_embedder", return_value=_FakeEmbedder("fake")):
        retrieval.update_embeddings(tmp_path, RyloxConfig())
        ctx = assemble_context(
            tmp_path, RyloxConfig(), "login", max_tokens=1000, top_k=1, token_counter=_char_count
        )

    assert [e.chunk.name for e in ctx.entry_points] == ["login"]
    assert [e.chunk.name for e in ctx.execution_flow] == ["check_credentials"]
    assert ctx.execution_flow[0].reason == "callee of login"
    assert ctx.skipped == []
    assert ctx.total_tokens == sum(e.tokens for e in ctx.included)


def test_unrelated_chunk_never_appears(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "auth.py",
        "def login():\n    check_credentials()\n\n"
        "def check_credentials():\n    return True\n\n"
        "def unrelated():\n    pass\n",
    )
    run_index(tmp_path, RyloxConfig())

    with patch("rylox.retrieval.get_embedder", return_value=_FakeEmbedder("fake")):
        retrieval.update_embeddings(tmp_path, RyloxConfig())
        ctx = assemble_context(
            tmp_path, RyloxConfig(), "login", max_tokens=1000, top_k=1, token_counter=_char_count
        )

    names = {e.chunk.name for e in ctx.included} | {e.chunk.name for e in ctx.skipped}
    assert "unrelated" not in names


def test_chunk_that_does_not_fit_is_skipped_not_truncated(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "auth.py",
        "def login():\n    check_credentials()\n\ndef check_credentials():\n    return True\n",
    )
    run_index(tmp_path, RyloxConfig())

    with patch("rylox.retrieval.get_embedder", return_value=_FakeEmbedder("fake")):
        retrieval.update_embeddings(tmp_path, RyloxConfig())
        entry_only_budget = _char_count("def login():\n    check_credentials()")
        ctx = assemble_context(
            tmp_path,
            RyloxConfig(),
            "login",
            max_tokens=entry_only_budget,
            top_k=1,
            token_counter=_char_count,
        )

    assert [e.chunk.name for e in ctx.entry_points] == ["login"]
    assert ctx.execution_flow == []
    assert [e.chunk.name for e in ctx.skipped] == ["check_credentials"]
    assert ctx.skipped[0].chunk.content == "def check_credentials():\n    return True"


def test_never_exceeds_max_tokens(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "auth.py",
        "def login():\n    check_credentials()\n\n"
        "def check_credentials():\n    return True\n\n"
        "def helper_one():\n    pass\n\n"
        "def helper_two():\n    pass\n",
    )
    run_index(tmp_path, RyloxConfig())

    with patch("rylox.retrieval.get_embedder", return_value=_FakeEmbedder("fake")):
        retrieval.update_embeddings(tmp_path, RyloxConfig())
        entry_size = _char_count("def login():\n    check_credentials()")
        # Budgets at or above the entry point's own size: below that is the
        # separately-tested BudgetTooSmallError path, not what this test
        # is checking (that the greedy fill never overshoots the budget).
        for budget in (entry_size, entry_size + 20, entry_size + 50, 500):
            ctx = assemble_context(
                tmp_path,
                RyloxConfig(),
                "login",
                max_tokens=budget,
                top_k=1,
                token_counter=_char_count,
            )
            assert ctx.total_tokens <= budget


def test_primary_entry_point_too_large_raises_budget_too_small_error(tmp_path: Path) -> None:
    _write(tmp_path, "auth.py", "def login():\n    check_credentials()\n")
    run_index(tmp_path, RyloxConfig())

    with patch("rylox.retrieval.get_embedder", return_value=_FakeEmbedder("fake")):
        retrieval.update_embeddings(tmp_path, RyloxConfig())
        with pytest.raises(BudgetTooSmallError):
            assemble_context(
                tmp_path, RyloxConfig(), "login", max_tokens=1, top_k=1, token_counter=_char_count
            )


def test_related_files_groups_by_path_without_duplicates(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "auth.py",
        "def login():\n    check_credentials()\n\ndef check_credentials():\n    return True\n",
    )
    run_index(tmp_path, RyloxConfig())

    with patch("rylox.retrieval.get_embedder", return_value=_FakeEmbedder("fake")):
        retrieval.update_embeddings(tmp_path, RyloxConfig())
        ctx = assemble_context(
            tmp_path, RyloxConfig(), "login", max_tokens=1000, top_k=1, token_counter=_char_count
        )

    assert len(ctx.related_files) == 1
    assert ctx.related_files[0].endswith("auth.py")


def test_no_search_results_produces_empty_context(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "def a():\n    pass\n")
    run_index(tmp_path, RyloxConfig())

    with patch("rylox.retrieval.get_embedder", return_value=_FakeEmbedder("fake")):
        retrieval.update_embeddings(tmp_path, RyloxConfig())
        ctx = assemble_context(
            tmp_path, RyloxConfig(), "query", max_tokens=1000, top_k=0, token_counter=_char_count
        )

    assert ctx.entry_points == []
    assert ctx.execution_flow == []
    assert ctx.supporting_symbols == []
    assert ctx.total_tokens == 0


def test_count_tokens_does_not_touch_network_until_actually_called() -> None:
    """Importing the module must never trigger tiktoken's network
    download — only actually calling count_tokens() should, and no test
    here does that."""
    import rylox.assembler as assembler_mod

    assert assembler_mod._ENCODING is None