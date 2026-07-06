from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from rylox.chunking import Chunk
from rylox.errors import IndexCorruptError, IndexNotFoundError

CACHE_DIRNAME = ".rylox"
INDEX_FILENAME = "index.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CachedChunk:
    path: str
    kind: str
    name: str
    start_line: int
    end_line: int
    parent_class: Optional[str]
    docstring: Optional[str]
    content: str

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> CachedChunk:
        return cls(
            path=chunk.path.as_posix(),
            kind=chunk.kind,
            name=chunk.name,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            parent_class=chunk.parent_class,
            docstring=chunk.docstring,
            content=chunk.content,
        )

    def to_chunk(self) -> Chunk:
        return Chunk(
            path=Path(self.path),
            kind=self.kind,  # type: ignore[arg-type]
            name=self.name,
            start_line=self.start_line,
            end_line=self.end_line,
            parent_class=self.parent_class,
            docstring=self.docstring,
            content=self.content,
        )


@dataclass
class FileEntry:
    hash: str
    chunks: list[CachedChunk] = field(default_factory=list)


@dataclass
class IndexManifest:
    schema_version: int = SCHEMA_VERSION
    files: dict[str, FileEntry] = field(default_factory=dict)


def cache_dir(repo: Path) -> Path:
    return repo / CACHE_DIRNAME


def index_path(repo: Path) -> Path:
    return cache_dir(repo) / INDEX_FILENAME


def ensure_cache_dir(repo: Path) -> Path:
    directory = cache_dir(repo)
    directory.mkdir(parents=True, exist_ok=True)
    gitignore = directory / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n", encoding="utf-8")
    return directory


def save_index(repo: Path, manifest: IndexManifest) -> None:
    ensure_cache_dir(repo)
    target = index_path(repo)
    payload = _serialize(manifest)

    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".index-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_name, target)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def load_index(repo: Path) -> IndexManifest:
    target = index_path(repo)
    if not target.exists():
        raise IndexNotFoundError(
            f"no index found at {target}. Run `rylox index` first."
        )

    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexCorruptError(f"{target} is unreadable or corrupt: {exc}") from exc

    try:
        return _deserialize(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise IndexCorruptError(f"{target} has an unexpected structure: {exc}") from exc


def load_or_empty(repo: Path) -> IndexManifest:
    try:
        return load_index(repo)
    except IndexNotFoundError:
        return IndexManifest()


def _serialize(manifest: IndexManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "files": {
            relpath: {
                "hash": entry.hash,
                "chunks": [asdict(c) for c in entry.chunks],
            }
            for relpath, entry in manifest.files.items()
        },
    }


def _deserialize(raw: dict[str, Any]) -> IndexManifest:
    schema_version = raw["schema_version"]
    if schema_version != SCHEMA_VERSION:
        raise IndexCorruptError(
            f"index schema_version {schema_version} is not supported "
            f"(expected {SCHEMA_VERSION}); run `rylox clean` and re-index."
        )

    files: dict[str, FileEntry] = {}
    for relpath, entry_raw in raw["files"].items():
        chunks = [CachedChunk(**c) for c in entry_raw["chunks"]]
        files[relpath] = FileEntry(hash=entry_raw["hash"], chunks=chunks)

    return IndexManifest(schema_version=schema_version, files=files)