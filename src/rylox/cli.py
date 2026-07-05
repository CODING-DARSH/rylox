"""Rylox CLI — exactly four commands (spec §9). No fifth command in v0.1.

Phase 1 goal: lock in the *interface* (command names, options, exit codes,
error presentation) before any real logic exists. Every command below is a
stub that returns a clearly-labeled "not implemented" result rather than
doing nothing silently — that way `rylox --help` and each subcommand's
`--help` are already correct and testable, and later phases fill bodies in
without touching signatures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from rylox import __version__
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
    _not_implemented("clean")


@app.command()
def doctor() -> None:
    """Run environment/health checks and report each pass/fail individually."""
    _not_implemented("doctor")


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
