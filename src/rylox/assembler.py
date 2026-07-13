from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from rylox import retrieval
from rylox.cache import CachedChunk
from rylox.config import RyloxConfig
from rylox.errors import BudgetTooSmallError

# Lazily loaded so importing this module never triggers a network call.
# tiktoken's cl100k_base encoding data isn't bundled in the package — it's
# downloaded once on first real use and cached locally afterward, the same
# one-time-download-then-fully-offline tradeoff already accepted for the
# embedding model (spec §13).
_ENCODING: Any = None


def count_tokens(text: str) -> int:
    global _ENCODING
    if _ENCODING is None:
        import tiktoken

        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return len(_ENCODING.encode(text))


TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class AssembledEntry:
    chunk: CachedChunk
    reason: str  # "direct match" | "caller of X" | "callee of X"
    tokens: int


@dataclass
class AssembledContext:
    query: str
    max_tokens: int
    entry_points: list[AssembledEntry] = field(default_factory=list)
    execution_flow: list[AssembledEntry] = field(default_factory=list)
    supporting_symbols: list[AssembledEntry] = field(default_factory=list)
    skipped: list[AssembledEntry] = field(default_factory=list)
    total_tokens: int = 0

    @property
    def included(self) -> list[AssembledEntry]:
        return self.entry_points + self.execution_flow + self.supporting_symbols

    @property
    def related_files(self) -> list[str]:
        seen: list[str] = []
        for entry in self.included:
            if entry.chunk.path not in seen:
                seen.append(entry.chunk.path)
        return seen


def assemble_context(
    repo: Path,
    config: RyloxConfig,
    query: str,
    max_tokens: int,
    top_k: int = 3,
    token_counter: TokenCounter = count_tokens,
) -> AssembledContext:
    """Assemble a token-budgeted context package for `query`.

    Assembly order (spec §5): primary entry point(s) first, then execution
    flow (one-hop callees), then supporting symbols (everything else the
    graph surfaced — one-hop callers and other references). Chunks are
    greedily filled into the budget in that fixed order; a chunk that
    doesn't fit is skipped, never truncated. If the primary entry point
    alone exceeds `max_tokens`, that's a defined failure (BudgetTooSmallError),
    not silent partial/broken output.
    """
    primary, expanded = retrieval.search_with_expansion(repo, config, query, top_k)

    entry_candidates = [
        AssembledEntry(chunk=r.chunk, reason="direct match", tokens=token_counter(r.chunk.content))
        for r in primary
    ]
    execution_flow_candidates = [
        AssembledEntry(chunk=e.chunk, reason=e.reason, tokens=token_counter(e.chunk.content))
        for e in expanded
        if e.reason.startswith("callee of")
    ]
    supporting_candidates = [
        AssembledEntry(chunk=e.chunk, reason=e.reason, tokens=token_counter(e.chunk.content))
        for e in expanded
        if not e.reason.startswith("callee of")
    ]

    if entry_candidates and entry_candidates[0].tokens > max_tokens:
        raise BudgetTooSmallError(
            f"the primary entry point ({entry_candidates[0].chunk.name}, "
            f"{entry_candidates[0].tokens} tokens) alone exceeds --max-tokens={max_tokens}."
        )

    result = AssembledContext(query=query, max_tokens=max_tokens)
    remaining = max_tokens

    for entry in entry_candidates:
        if entry.tokens <= remaining:
            result.entry_points.append(entry)
            remaining -= entry.tokens
            result.total_tokens += entry.tokens
        else:
            result.skipped.append(entry)

    for entry in execution_flow_candidates:
        if entry.tokens <= remaining:
            result.execution_flow.append(entry)
            remaining -= entry.tokens
            result.total_tokens += entry.tokens
        else:
            result.skipped.append(entry)

    for entry in supporting_candidates:
        if entry.tokens <= remaining:
            result.supporting_symbols.append(entry)
            remaining -= entry.tokens
            result.total_tokens += entry.tokens
        else:
            result.skipped.append(entry)

    return result