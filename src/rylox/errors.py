"""Shared exception types.

Every command-level failure in Rylox should raise (or be wrapped into) one of
these, so the CLI layer has one place that decides how errors are printed and
which process exit code is returned. This keeps "fail loudly, fail clearly"
(see spec §8, §12) consistent across commands instead of ad hoc per-command.
"""

from __future__ import annotations


class RyloxError(Exception):
    """Base class for all expected/handled Rylox failures.

    Anything raised as RyloxError is treated as a *known* failure mode: the
    CLI catches it, prints `err.message` cleanly (no traceback), and exits
    with `err.exit_code`. Unexpected exceptions are left to propagate as
    tracebacks — that distinction is deliberate, so silent/ambiguous failures
    are never mistaken for handled ones.
    """

    exit_code: int = 1

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigError(RyloxError):
    """Invalid or unreadable rylox.toml. Must fail at command start (§8)."""

    exit_code: int = 2


class IndexNotFoundError(RyloxError):
    """`context` was run without a prior `index`, or the cache is missing."""

    exit_code: int = 3


class IndexCorruptError(RyloxError):
    """`.rylox/` cache exists but its manifest is unreadable/corrupt."""

    exit_code: int = 4


class BudgetTooSmallError(RyloxError):
    """Primary entry point alone doesn't fit --max-tokens (§6 defined failure)."""

    exit_code: int = 5
