from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field

from rylox.cache import CachedChunk

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_KEYWORDS = frozenset(keyword.kwlist)


def _identifiers(text: str) -> set[str]:
    return {tok for tok in _IDENTIFIER_RE.findall(text) if tok not in _KEYWORDS}


# A name that resolves to more than this many chunks repo-wide is treated
# as too generic/ambiguous to be a meaningful reference target at all, and
# is excluded from graph-building entirely — see build_reference_graph.
_MAX_NAME_AMBIGUITY = 3


@dataclass
class ReferenceGraph:
    """Forward and reverse reference edges between chunks, by index.

    A single mechanism stands in for both the call graph and the import
    graph (spec §4): chunk i references chunk j if j's name appears as an
    identifier inside i's content — this covers direct calls (`foo()`),
    constructor/class usage (`Foo()`), and references brought in via
    import (since the imported name still has to appear as a plain
    identifier somewhere to be used). It does not distinguish *why* a
    name is referenced, only that it is — narrower than resolving actual
    import statements and call targets precisely, but requires no module
    resolution or scope analysis to get right.
    """

    forward: dict[int, set[int]] = field(default_factory=dict)  # chunk -> chunks it references
    reverse: dict[int, set[int]] = field(default_factory=dict)  # chunk -> chunks that reference it


def build_reference_graph(chunks: list[CachedChunk]) -> ReferenceGraph:
    """Build the reference graph.

    Names that map to more than _MAX_NAME_AMBIGUITY chunks are excluded as
    valid reference targets entirely, not just capped. This matters
    concretely for real Python codebases: a class defining its own
    `__init__` (or `name`, `fail`, `convert`, or any other extremely
    common method name) would otherwise appear to "reference" every other
    same-named method anywhere in the repo, purely because they share a
    name — confirmed directly against a real codebase, where a single
    class's own constructor definition produced dozens of false "callee"
    edges to unrelated classes' unrelated constructors, consuming the
    entire token budget on noise instead of relevant code. A name that
    maps to only a handful of chunks is a specific, meaningful signal; a
    name that maps to dozens carries essentially none.
    """
    name_to_indices: dict[str, list[int]] = {}
    for i, chunk in enumerate(chunks):
        name_to_indices.setdefault(chunk.name, []).append(i)

    unambiguous_names = {
        name: indices
        for name, indices in name_to_indices.items()
        if len(indices) <= _MAX_NAME_AMBIGUITY
    }

    graph = ReferenceGraph()
    for i, chunk in enumerate(chunks):
        referenced_names = _identifiers(chunk.content)
        targets: set[int] = set()
        for name in referenced_names:
            for j in unambiguous_names.get(name, ()):
                if j != i:
                    targets.add(j)
        if targets:
            graph.forward[i] = targets
            for j in targets:
                graph.reverse.setdefault(j, set()).add(i)

    return graph


@dataclass(frozen=True)
class ExpandedChunk:
    index: int
    reason: str  # e.g. "caller of login" / "callee of login"


def expand_one_hop(
    top_indices: list[int], graph: ReferenceGraph, chunks: list[CachedChunk]
) -> list[ExpandedChunk]:
    """From the given top-ranked chunk indices, expand exactly one hop in
    each direction (callers and callees/references), skipping anything
    already in the top-ranked set. No transitive/multi-hop expansion.
    """
    already = set(top_indices)
    seen: set[int] = set()
    expanded: list[ExpandedChunk] = []

    for idx in top_indices:
        target_name = chunks[idx].name

        for callee_idx in sorted(graph.forward.get(idx, ())):
            if callee_idx in already or callee_idx in seen:
                continue
            seen.add(callee_idx)
            expanded.append(ExpandedChunk(index=callee_idx, reason=f"callee of {target_name}"))

        for caller_idx in sorted(graph.reverse.get(idx, ())):
            if caller_idx in already or caller_idx in seen:
                continue
            seen.add(caller_idx)
            expanded.append(ExpandedChunk(index=caller_idx, reason=f"caller of {target_name}"))

    return expanded