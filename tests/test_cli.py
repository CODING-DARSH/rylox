"""CLI interface tests: command signatures, options, exit codes."""

from __future__ import annotations

import pytest
import typer.main
from typer.testing import CliRunner

from rylox import __version__
from rylox.cli import app

runner = CliRunner()


def test_no_args_shows_help() -> None:
    # Click's `no_args_is_help=True` exits with code 2 (treated as a usage
    # nudge, not a clean success) — we assert on the help content, not 0.
    result = runner.invoke(app, [])
    assert "index" in result.output
    assert "context" in result.output
    assert "clean" in result.output
    assert "doctor" in result.output


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_exactly_four_commands_registered() -> None:
    click_app = typer.main.get_command(app)
    assert set(click_app.commands.keys()) == {"index", "context", "clean", "doctor"}


def test_index_runs_full_pipeline_and_reports_summary(tmp_path: object) -> None:
    from pathlib import Path
    from unittest.mock import patch

    class _FakeEmbedder:
        def __init__(self, model: str) -> None:
            self.model = model

        def embed(self, texts: list) -> list:
            return [[float(len(t)), 0.0] for t in texts]

    repo = Path(str(tmp_path))
    (repo / "a.py").write_text("def a():\n    pass\n", encoding="utf-8")

    with patch("rylox.retrieval.get_embedder", return_value=_FakeEmbedder("fake")):
        result = runner.invoke(app, ["index", "--repo", str(repo)])

    assert result.exit_code == 0
    assert "chunks: 1 changed" in result.output
    assert "embeddings: 1 files embedded" in result.output
    assert (repo / "rylox.toml").exists()
    assert (repo / ".rylox" / "index.json").exists()
    assert (repo / ".rylox" / "embeddings.json").exists()


def test_index_second_run_reuses_unchanged_embeddings(tmp_path: object) -> None:
    from pathlib import Path
    from unittest.mock import patch

    class _FakeEmbedder:
        def __init__(self, model: str) -> None:
            self.model = model

        def embed(self, texts: list) -> list:
            return [[float(len(t)), 0.0] for t in texts]

    repo = Path(str(tmp_path))
    (repo / "a.py").write_text("def a():\n    pass\n", encoding="utf-8")

    with patch("rylox.retrieval.get_embedder", return_value=_FakeEmbedder("fake")):
        runner.invoke(app, ["index", "--repo", str(repo)])
        result = runner.invoke(app, ["index", "--repo", str(repo)])

    assert result.exit_code == 0
    assert "chunks: 0 changed, 1 unchanged" in result.output
    assert "embeddings: 0 files embedded, 1 reused" in result.output


def test_context_requires_task_argument() -> None:
    result = runner.invoke(app, ["context"])
    assert result.exit_code != 0


def test_context_rejects_non_markdown_format(tmp_path: object) -> None:
    result = runner.invoke(
        app, ["context", "some task", "--repo", str(tmp_path), "--format", "json"]
    )
    assert result.exit_code != 0
    assert "markdown" in result.output.lower()


def test_context_without_prior_index_raises_index_not_found(tmp_path: object) -> None:
    from rylox.errors import IndexNotFoundError

    result = runner.invoke(app, ["context", "how does auth work", "--repo", str(tmp_path)])
    # CliRunner invokes the raw Typer app, bypassing cli.py's run() wrapper
    # (which is what turns RyloxError into a clean printed message + clean
    # exit) — so here we assert on the exception type Click/Typer itself
    # captures. run()'s clean-message/exit-code behavior is covered
    # separately, since that wrapper only matters for the real installed
    # `rylox` command, not for direct app() invocation.
    assert isinstance(result.exception, IndexNotFoundError)


def test_run_wrapper_exits_cleanly_on_rylox_error_no_traceback(tmp_path, capsys) -> None:
    """Regression test for a real bug: run() used to `raise typer.Exit(...)`
    after catching RyloxError, but by that point app() has already fully
    unwound and Click's own exception-handling context is no longer active
    — so typer.Exit became an ordinary uncaught exception with a full
    printed traceback, instead of a clean process exit. run() must use
    sys.exit() instead, which the interpreter always handles silently.
    Confirmed via the real installed `rylox` binary before this fix landed.
    """
    import sys

    from rylox.cli import run

    sys.argv = ["rylox", "context", "how does auth work", "--repo", str(tmp_path)]
    with pytest.raises(SystemExit) as exc_info:
        run()

    assert exc_info.value.code == 3  # IndexNotFoundError.exit_code
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "Traceback" not in captured.err


def test_clean_on_missing_cache_reports_nothing_to_clean(tmp_path: object) -> None:
    result = runner.invoke(app, ["clean", "--repo", str(tmp_path)])
    assert result.exit_code == 0
    assert "nothing to clean" in result.output


def test_clean_removes_existing_cache(tmp_path: object) -> None:
    from pathlib import Path

    from rylox.cache import ensure_cache_dir

    ensure_cache_dir(Path(str(tmp_path)))
    result = runner.invoke(app, ["clean", "--repo", str(tmp_path)])
    assert result.exit_code == 0
    assert not (Path(str(tmp_path)) / ".rylox").exists()


def test_doctor_runs_and_reports_checks(tmp_path: object) -> None:
    result = runner.invoke(app, ["doctor", "--repo", str(tmp_path)])
    assert result.exit_code == 0
    assert "python_version" in result.output
    assert "tree_sitter" in result.output
    assert "cache_manifest" in result.output


def test_index_rejects_nonexistent_repo_path() -> None:
    result = runner.invoke(app, ["index", "--repo", "/definitely/does/not/exist"])
    assert result.exit_code != 0