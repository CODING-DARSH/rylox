"""Python chunking via tree-sitter (spec §2).

Scope for this module, deliberately narrow:
  - integrate tree-sitter-python and parse .py source into an AST
  - chunk at function / method / class granularity (never file, never line)
  - store per-chunk metadata: file path, line range, parent class, docstring
  - skip a file tree-sitter can't parse without crashing the whole run

Explicitly NOT in this module yet (later phases): content hashing for
incremental re-index, the `.rylox/` cache, and CLI wiring for `rylox index`.
`parse_source` and `parse_file` below are self-contained and testable without
any of that machinery existing yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

_PY_LANGUAGE = Language(tspython.language(), "python")


def _make_parser() -> Parser:
    parser = Parser()
    parser.set_language(_PY_LANGUAGE)
    return parser


ChunkKind = str  # "function" | "method" | "class" — kept as str for py3.9-friendly simplicity


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of code, per spec §2's chunk metadata list."""

    path: Path
    kind: ChunkKind
    name: str
    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive
    parent_class: Optional[str]
    docstring: Optional[str]


@dataclass
class ParseResult:
    """Outcome of parsing one file: either chunks, or a recorded error.

    `error` being set means the file was skipped, not that the whole
    operation failed (spec §12: malformed files are logged and skipped,
    never fatal for the whole `index` run). Callers should check `error`
    and continue on to the next file rather than propagating an exception.
    """

    chunks: list[Chunk]
    error: Optional[str] = None


def parse_source(source: str, path: Path) -> list[Chunk]:
    """Parse Python source text into function/method/class chunks.

    Pure function, no filesystem access — this is what makes chunking
    behavior directly unit-testable against hand-written source strings,
    independent of `parse_file`'s I/O and error handling.
    """
    parser = _make_parser()
    tree = parser.parse(source.encode("utf-8"))
    lines = source.splitlines()
    chunks: list[Chunk] = []
    _walk(tree.root_node, path, lines, parent_class=None, chunks=chunks)
    return chunks


def parse_file(path: Path) -> ParseResult:
    """Read and parse a single .py file. Never raises — malformed or
    unreadable files come back as a ParseResult with `error` set and an
    empty chunk list, per spec §12.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return ParseResult(chunks=[], error=f"{path}: could not read file ({exc})")

    try:
        chunks = parse_source(source, path)
    except Exception as exc:  # noqa: BLE001 — deliberately broad: any parser
        # failure must degrade to "skip this file", never crash the whole
        # index run (spec §12). Narrowing this to tree_sitter's specific
        # exception types would risk missing a case and defeating the point.
        return ParseResult(chunks=[], error=f"{path}: failed to parse ({exc})")

    return ParseResult(chunks=chunks, error=None)


def _walk(
    node: Node,
    path: Path,
    lines: list[str],
    *,
    parent_class: Optional[str],
    chunks: list[Chunk],
) -> None:
    for child in node.children:
        if child.type == "class_definition":
            name = _node_name(child)
            chunks.append(
                Chunk(
                    path=path,
                    kind="class",
                    name=name,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    parent_class=parent_class,
                    docstring=_docstring(child, lines),
                )
            )
            # Recurse with this class as the new parent_class, so nested
            # function_definitions inside it are chunked as methods.
            _walk(child, path, lines, parent_class=name, chunks=chunks)

        elif child.type == "function_definition":
            name = _node_name(child)
            chunks.append(
                Chunk(
                    path=path,
                    kind="method" if parent_class is not None else "function",
                    name=name,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    parent_class=parent_class,
                    docstring=_docstring(child, lines),
                )
            )
            # Recurse without a parent_class: a function nested inside
            # another function is chunked as its own top-level-style
            # "function" (v0.1 doesn't track function-in-function nesting).
            _walk(child, path, lines, parent_class=None, chunks=chunks)

        else:
            _walk(child, path, lines, parent_class=parent_class, chunks=chunks)


def _node_name(def_node: Node) -> str:
    name_node = def_node.child_by_field_name("name")
    if name_node is None:
        return "<anonymous>"
    return name_node.text.decode("utf-8")


def _docstring(def_node: Node, lines: list[str]) -> Optional[str]:
    """Extract a def/class's own docstring: the first statement in its body,
    if that statement is a bare string expression.
    """
    body = def_node.child_by_field_name("body")
    if body is None or body.child_count == 0:
        return None

    first_statement = body.children[0]
    if first_statement.type != "expression_statement":
        return None
    if first_statement.child_count == 0:
        return None

    string_node = first_statement.children[0]
    if string_node.type != "string":
        return None

    raw = string_node.text.decode("utf-8")
    return _strip_string_literal(raw)


def _strip_string_literal(raw: str) -> str:
    """Strip Python string-literal quoting/prefixes to get the docstring body."""
    text = raw.strip()
    for prefix in ("u", "U", "r", "R", "b", "B", "rb", "Rb", "rB", "RB", "br", "Br", "bR", "BR"):
        if text.startswith(prefix) and text[len(prefix) :].startswith(('"', "'")):
            text = text[len(prefix) :]
            break
    for quote in ('"""', "'''", '"', "'"):
        if text.startswith(quote) and text.endswith(quote) and len(text) >= 2 * len(quote):
            return text[len(quote) : -len(quote)].strip()
    return text.strip()