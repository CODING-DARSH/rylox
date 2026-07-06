"""CLI interface tests: command signatures, options, exit codes."""

from __future__ import annotations

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


def test_index_stub_runs_and_reports_not_implemented(tmp_path: object) -> None:
    result = runner.invoke(app, ["index", "--repo", str(tmp_path)])
    assert result.exit_code == 1
    assert "not implemented" in result.output


def test_context_requires_task_argument() -> None:
    result = runner.invoke(app, ["context"])
    assert result.exit_code != 0


def test_context_rejects_non_markdown_format(tmp_path: object) -> None:
    result = runner.invoke(
        app, ["context", "some task", "--repo", str(tmp_path), "--format", "json"]
    )
    assert result.exit_code != 0
    assert "markdown" in result.output.lower()


def test_context_stub_runs_and_reports_not_implemented(tmp_path: object) -> None:
    result = runner.invoke(app, ["context", "how does auth work", "--repo", str(tmp_path)])
    assert result.exit_code == 1
    assert "not implemented" in result.output


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