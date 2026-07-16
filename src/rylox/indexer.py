from __future__ import annotations

import fnmatch
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from rylox import cache, chunking
from rylox.config import RyloxConfig


@dataclass
class IndexReport:
    total_files: int = 0
    changed: int = 0
    unchanged: int = 0
    deleted: int = 0
    parse_errors: list[str] = field(default_factory=list)
    skipped_too_large: list[str] = field(default_factory=list)
    skipped_symlink_escape: list[str] = field(default_factory=list)


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_gitignore_patterns(repo: Path) -> list[str]:
    gitignore = repo / ".gitignore"
    if not gitignore.exists():
        return []
    patterns = []
    for line in gitignore.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(line.rstrip("/"))
    return patterns


def _is_ignored(rel_posix: str, patterns: list[str]) -> bool:
    parts = rel_posix.split("/")
    for pattern in patterns:
        if "/" in pattern:
            if fnmatch.fnmatch(rel_posix, pattern) or fnmatch.fnmatch(rel_posix, pattern + "/*"):
                return True
        else:
            if any(fnmatch.fnmatch(part, pattern) for part in parts):
                return True
    return False


def iter_python_files(repo: Path, config: RyloxConfig, report: IndexReport) -> dict[str, Path]:
    repo_real = repo.resolve()
    patterns = list(config.ignore.patterns)
    if config.ignore.respect_gitignore:
        patterns += _load_gitignore_patterns(repo)

    found: dict[str, Path] = {}
    # Sorted explicitly: repo.rglob()'s traversal order is OS/filesystem
    # dependent (confirmed differing between Linux and Windows in practice,
    # not just in theory) and is never guaranteed by Python's docs. Without
    # this sort, chunk index assignment — and therefore BM25/FAISS
    # tie-breaking behavior — silently varies by platform, breaking the
    # "deterministic ordering given the same index and query" requirement.
    for path in sorted(repo.rglob("*.py")):
        rel = path.relative_to(repo).as_posix()

        if _is_ignored(rel, patterns):
            continue

        if path.is_symlink():
            target_real = path.resolve()
            if not target_real.is_relative_to(repo_real):
                report.skipped_symlink_escape.append(rel)
                continue

        try:
            size = path.stat().st_size
        except OSError:
            continue
        max_bytes = config.ignore.max_file_size_mb * 1024 * 1024
        if size > max_bytes:
            report.skipped_too_large.append(rel)
            continue

        found[rel] = path

    return found


def run_index(repo: Path, config: RyloxConfig) -> IndexReport:
    manifest = cache.load_or_empty(repo)
    report = IndexReport()

    current_files = iter_python_files(repo, config, report)
    report.total_files = len(current_files)

    current_relpaths = set(current_files.keys())
    previous_relpaths = set(manifest.files.keys())

    for stale in previous_relpaths - current_relpaths:
        del manifest.files[stale]
        report.deleted += 1

    for relpath, abspath in current_files.items():
        file_hash = hash_file(abspath)
        existing = manifest.files.get(relpath)
        if existing is not None and existing.hash == file_hash:
            report.unchanged += 1
            continue

        parse_result = chunking.parse_file(abspath)
        if parse_result.error is not None:
            report.parse_errors.append(parse_result.error)

        manifest.files[relpath] = cache.FileEntry(
            hash=file_hash,
            chunks=[cache.CachedChunk.from_chunk(c) for c in parse_result.chunks],
        )
        report.changed += 1

    cache.save_index(repo, manifest)
    return report