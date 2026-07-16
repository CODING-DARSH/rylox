"""Rylox CLI — exactly four commands. No fifth command in v0.1."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional

import typer

from rylox import __version__
from rylox.assembler import assemble_context
from rylox.config import load_or_write_defaults
from rylox.doctor import run_doctor
from rylox.errors import RyloxError
from rylox.indexer import run_index
from rylox.renderer import write_context_file
from rylox.retrieval import update_embeddings

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
    config = load_or_write_defaults(repo)

    chunk_report = run_index(repo, config)
    typer.echo(
        f"chunks: {chunk_report.changed} changed, {chunk_report.unchanged} unchanged, "
        f"{chunk_report.deleted} deleted ({chunk_report.total_files} files tracked)"
    )
    for warning in chunk_report.parse_errors:
        typer.echo(f"[warn] {warning}", err=True)
    for skipped in chunk_report.skipped_too_large:
        typer.echo(f"[warn] skipped (too large): {skipped}", err=True)
    for skipped in chunk_report.skipped_symlink_escape:
        typer.echo(f"[warn] skipped (symlink escapes repo): {skipped}", err=True)

    embed_report = update_embeddings(repo, config)
    if embed_report.model_changed:
        typer.echo("embedding model changed since last run — full re-embed performed")
    typer.echo(
        f"embeddings: {embed_report.embedded_files} files embedded, "
        f"{embed_report.reused_files} reused, {embed_report.deleted_files} deleted "
        f"({embed_report.total_chunks} chunks total)"
    )


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

    config = load_or_write_defaults(repo)
    ctx = assemble_context(repo, config, task, max_tokens=max_tokens)
    output_path = write_context_file(ctx, repo)

    typer.echo(f"wrote {output_path}")
    typer.echo(
        f"{len(ctx.included)} chunk(s) included, {len(ctx.related_files)} file(s) touched, "
        f"{ctx.total_tokens} / {max_tokens} tokens"
    )
    if ctx.skipped:
        typer.echo(
            f"{len(ctx.skipped)} chunk(s) skipped (did not fit budget)", err=True
        )


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


def run() -> None:
    """Entry point wrapper: turn RyloxError into a clean message + exit code.

    Uses sys.exit(), not `raise typer.Exit(...)`: by the time RyloxError
    propagates out of app() and is caught here, we're outside Click's own
    active command-invocation context, so typer.Exit/click.exceptions.Exit
    no longer gets silently handled by Click — it becomes an ordinary
    uncaught exception with a full printed traceback. sys.exit() raises
    SystemExit, which the interpreter always handles silently regardless.
    """
    try:
        app()
    except RyloxError as err:
        typer.echo(f"error: {err.message}", err=True)
        sys.exit(err.exit_code)


if __name__ == "__main__":
    run()