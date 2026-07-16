"""rylox.toml schema, loading, validation, and default-writing (spec §8).

Design notes:
- Config is optional at the CLI level — commands that need it call
  `load_or_write_defaults()`, which writes a default file on first run
  (matching spec §8: "rylox init ... or first `rylox index` in a repo
  without one writes sensible defaults").
- Every field has a validated, typed home in `RyloxConfig` and its nested
  dataclasses. Nothing downstream should ever read a raw dict out of the
  parsed TOML — going through these dataclasses is what makes "fail loudly
  at command start" (§8) enforceable in one place instead of scattered
  `config["embedding"]["model"]` lookups with no validation.
- `provider` in `[embedding]` is validated against
  `embedding.SUPPORTED_PROVIDERS` (currently just "huggingface"). This is
  the config-side half of the Embedder abstraction described in
  embedding.py — see that module for why other provider names are
  rejected rather than silently accepted in v0.1.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from rylox.embedding import SUPPORTED_PROVIDERS
from rylox.errors import ConfigError

CONFIG_FILENAME = "rylox.toml"

DEFAULT_EMBEDDING_PROVIDER = "huggingface"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CALL_DEPTH = 1
DEFAULT_MAX_TOKENS = 16000
DEFAULT_OUTPUT_FORMAT = "markdown"
DEFAULT_IGNORE_PATTERNS = [".git", "*.min.js", "node_modules", "__pycache__"]
DEFAULT_MAX_FILE_SIZE_MB = 5

SUPPORTED_OUTPUT_FORMATS = ("markdown",)

def _toml_string_array(items: list[str]) -> str:
    quoted = ", ".join(f'"{item}"' for item in items)
    return f"[{quoted}]"


_DEFAULT_TOML = f"""\
[embedding]
model = "{DEFAULT_EMBEDDING_MODEL}"     # local, ONNX-runnable by default
provider = "{DEFAULT_EMBEDDING_PROVIDER}"  # only "huggingface" is supported in v0.1

[retrieval]
call_depth = {DEFAULT_CALL_DEPTH}          # hardcoded to 1 in v0.1, forward-compat only

[budget]
max_tokens = {DEFAULT_MAX_TOKENS}      # default; overridable via --max-tokens

[output]
format = "{DEFAULT_OUTPUT_FORMAT}"   # only supported value in v0.1

[ignore]
patterns = {_toml_string_array(DEFAULT_IGNORE_PATTERNS)}
respect_gitignore = true
max_file_size_mb = {DEFAULT_MAX_FILE_SIZE_MB}  # files larger than this are skipped, not indexed
"""


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = DEFAULT_EMBEDDING_MODEL
    provider: str = DEFAULT_EMBEDDING_PROVIDER


@dataclass(frozen=True)
class RetrievalConfig:
    call_depth: int = DEFAULT_CALL_DEPTH


@dataclass(frozen=True)
class BudgetConfig:
    max_tokens: int = DEFAULT_MAX_TOKENS


@dataclass(frozen=True)
class OutputConfig:
    format: str = DEFAULT_OUTPUT_FORMAT


@dataclass(frozen=True)
class IgnoreConfig:
    patterns: list[str] = field(default_factory=lambda: list(DEFAULT_IGNORE_PATTERNS))
    respect_gitignore: bool = True
    max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB


@dataclass(frozen=True)
class RyloxConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    ignore: IgnoreConfig = field(default_factory=IgnoreConfig)


def config_path(repo: Path) -> Path:
    return repo / CONFIG_FILENAME


def write_default_config(repo: Path) -> Path:
    """Write rylox.toml with sensible defaults. Does not overwrite an existing file."""
    path = config_path(repo)
    if path.exists():
        raise ConfigError(f"{path} already exists; refusing to overwrite it.")
    path.write_text(_DEFAULT_TOML, encoding="utf-8")
    return path


def load_config(repo: Path) -> RyloxConfig:
    """Load and validate rylox.toml from `repo`. Raises ConfigError on any problem.

    Fails loudly and immediately (§8: "Invalid config values fail loudly at
    command start, not partway through indexing") — every validation error
    below is raised the moment it's detected, with a message specific
    enough to fix without re-reading the spec.
    """
    path = config_path(repo)
    if not path.exists():
        raise ConfigError(
            f"no {CONFIG_FILENAME} found in {repo}. Run `rylox index` to create one "
            "with defaults, or add rylox.toml yourself."
        )

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc

    return _parse(raw, source=path)


def load_or_write_defaults(repo: Path) -> RyloxConfig:
    """Load rylox.toml, writing defaults first if it doesn't exist yet (§8)."""
    if not config_path(repo).exists():
        write_default_config(repo)
    return load_config(repo)


def _parse(raw: dict[str, Any], *, source: Path) -> RyloxConfig:
    embedding_raw = _section(raw, "embedding", source)
    retrieval_raw = _section(raw, "retrieval", source)
    budget_raw = _section(raw, "budget", source)
    output_raw = _section(raw, "output", source)
    ignore_raw = _section(raw, "ignore", source)

    embedding = EmbeddingConfig(
        model=_get(embedding_raw, "model", DEFAULT_EMBEDDING_MODEL, str, source, "embedding.model"),
        provider=_get(
            embedding_raw,
            "provider",
            DEFAULT_EMBEDDING_PROVIDER,
            str,
            source,
            "embedding.provider",
        ),
    )
    if embedding.provider not in SUPPORTED_PROVIDERS:
        raise ConfigError(
            f"{source}: embedding.provider = '{embedding.provider}' is not supported in v0.1. "
            f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}. "
            "Other providers are planned post-v0.1 (see roadmap) but are not runnable yet — "
            "v0.1 is local-only by design (spec §13)."
        )

    retrieval = RetrievalConfig(
        call_depth=_get(
            retrieval_raw, "call_depth", DEFAULT_CALL_DEPTH, int, source, "retrieval.call_depth"
        ),
    )
    if retrieval.call_depth != 1:
        raise ConfigError(
            f"{source}: retrieval.call_depth = {retrieval.call_depth} is not supported in v0.1. "
            "call_depth is hardcoded to 1 (spec §4/§8) — the field exists for forward "
            "compatibility only and is not yet user-adjustable."
        )

    budget = BudgetConfig(
        max_tokens=_get(
            budget_raw, "max_tokens", DEFAULT_MAX_TOKENS, int, source, "budget.max_tokens"
        ),
    )
    if budget.max_tokens <= 0:
        raise ConfigError(f"{source}: budget.max_tokens must be a positive integer.")

    output = OutputConfig(
        format=_get(output_raw, "format", DEFAULT_OUTPUT_FORMAT, str, source, "output.format"),
    )
    if output.format not in SUPPORTED_OUTPUT_FORMATS:
        raise ConfigError(
            f"{source}: output.format = '{output.format}' is not supported in v0.1. "
            f"Supported formats: {', '.join(SUPPORTED_OUTPUT_FORMATS)}."
        )

    ignore = IgnoreConfig(
        patterns=_get_str_list(
            ignore_raw, "patterns", list(DEFAULT_IGNORE_PATTERNS), source, "ignore.patterns"
        ),
        respect_gitignore=_get(
            ignore_raw, "respect_gitignore", True, bool, source, "ignore.respect_gitignore"
        ),
        max_file_size_mb=_get(
            ignore_raw,
            "max_file_size_mb",
            DEFAULT_MAX_FILE_SIZE_MB,
            int,
            source,
            "ignore.max_file_size_mb",
        ),
    )
    if ignore.max_file_size_mb <= 0:
        raise ConfigError(f"{source}: ignore.max_file_size_mb must be a positive integer.")

    return RyloxConfig(
        embedding=embedding, retrieval=retrieval, budget=budget, output=output, ignore=ignore
    )


def _section(raw: dict[str, Any], name: str, source: Path) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{source}: [{name}] must be a table, got {type(value).__name__}.")
    return value


_T = TypeVar("_T", str, int, bool)


def _get(
    section: dict[str, Any],
    key: str,
    default: _T,
    expected_type: type[_T],
    source: Path,
    dotted: str,
) -> _T:
    if key not in section:
        return default
    value = section[key]
    # bool is a subclass of int in Python; guard against `max_tokens = true` sneaking through.
    if expected_type is int and isinstance(value, bool):
        raise ConfigError(f"{source}: {dotted} must be an integer, got a boolean.")
    if not isinstance(value, expected_type):
        raise ConfigError(
            f"{source}: {dotted} must be of type {expected_type.__name__}, "
            f"got {type(value).__name__}."
        )
    return value


def _get_str_list(
    section: dict[str, Any], key: str, default: list[str], source: Path, dotted: str
) -> list[str]:
    if key not in section:
        return default
    value = section[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{source}: {dotted} must be a list of strings.")
    return value