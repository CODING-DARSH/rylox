from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import typer

from rylox import __version__
from rylox.doctor import run_doctor
from rylox.errors import RyloxError

app = typer.Typer(
    name="rylox",
    help=(
        "Local repository context engine — turns a repo + task into a "
        "token-budgeted markdown package."
    ),
    add_completion=False,
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"rylox {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the Rylox version and exit.",
    ),
) -> None:
    """Rylox: build the best local repository context assembler."""


@app.command()
def index(
    repo: Path = typer.Option(
        Path("."), "--repo", exists=True, file_okay=False, help="Repository root to index."
    ),
) -> None:
    """Build or incrementally update the local `.rylox/` index."""
    _not_implemented("index")


@app.command()
def context(
    task: str = typer.Argument(..., help="The task/question to build context for."),
    max_tokens: int = typer.Option(
        16000, "--max-tokens", min=1, help="Hard token budget for the assembled context."
    ),
    output_format: str = typer.Option(
        "markdown",
        "--format",
        help="Output format. Only 'markdown' is supported in v0.1.",
    ),
    repo: Path = typer.Option(
        Path("."), "--repo", exists=True, file_okay=False, help="Repository root."
    ),
) -> None:
    """Assemble a token-budgeted, relationship-aware context package for TASK."""
    if output_format != "markdown":
        raise typer.BadParameter("Only --format markdown is supported in v0.1.")
    _not_implemented("context")


@app.command()
def clean(
    repo: Path = typer.Option(
        Path("."), "--repo", exists=True, file_okay=False, help="Repository root."
    ),
) -> None:
    """Delete the `.rylox/` cache and start fresh."""
    from rylox.cache import cache_dir

    directory = cache_dir(repo)
    if not directory.exists():
        typer.echo(f"nothing to clean: {directory} does not exist")
        return
    shutil.rmtree(directory)
    typer.echo(f"removed {directory}")


@app.command()
def doctor(
    repo: Path = typer.Option(
        Path("."), "--repo", exists=True, file_okay=False, help="Repository root."
    ),
) -> None:
    """Run environment/health checks and report each pass/fail individually."""
    results = run_doctor(repo)

    symbols = {"pass": "[ok]", "fail": "[FAIL]", "skip": "[skip]"}
    for result in results:
        typer.echo(f"{symbols[result.status]} {result.name}: {result.detail}")

    if any(r.status == "fail" for r in results):
        raise typer.Exit(code=1)


def _not_implemented(command: str) -> None:
    """Placeholder body for Phase 1 stubs.

    Deliberately visible and non-zero-exit rather than a silent no-op, so
    running any command today tells you honestly that it's unbuilt instead
    of pretending to succeed.
    """
    typer.echo(f"rylox {command}: not implemented yet (Phase 1 skeleton)", err=True)
    raise typer.Exit(code=1)


def run() -> None:
    """Entry point wrapper: turn RyloxError into a clean message + exit code."""
    try:
        app()
    except RyloxError as err:
        typer.echo(f"error: {err.message}", err=True)
        raise typer.Exit(code=err.exit_code) from err


if __name__ == "__main__":
    run()