from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rylox.errors import IndexCorruptError, IndexNotFoundError

# Keep in sync with `requires-python` in pyproject.toml.
MIN_PYTHON = (3, 9)

CheckStatus = Literal["pass", "fail", "skip"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    detail: str


def run_doctor(repo: Path) -> list[CheckResult]:
    return [
        _check_python_version(),
        _check_tree_sitter(),
        _check_faiss(),
        _check_embedding_model(),
        _check_cache_manifest(repo),
    ]


def _check_python_version() -> CheckResult:
    current = sys.version_info[:2]
    if current >= MIN_PYTHON:
        return CheckResult(
            "python_version",
            "pass",
            f"Python {sys.version.split()[0]} (>= {'.'.join(map(str, MIN_PYTHON))} required)",
        )
    return CheckResult(
        "python_version",
        "fail",
        f"Python {sys.version.split()[0]} is below the required "
        f"{'.'.join(map(str, MIN_PYTHON))} floor",
    )


def _check_tree_sitter() -> CheckResult:
    try:
        from rylox import chunking

        chunking.parse_source("def _rylox_doctor_probe():\n    pass\n", Path("_probe.py"))
    except Exception as exc:  # noqa: BLE001 - any failure here is a real doctor finding
        return CheckResult("tree_sitter", "fail", f"tree-sitter-python not usable: {exc}")
    return CheckResult("tree_sitter", "pass", "tree-sitter-python importable and parsing")


def _check_faiss() -> CheckResult:
    try:
        from rylox.vectorstore import build_index

        store = build_index([[1.0, 0.0], [0.0, 1.0]])
        results = store.search([1.0, 0.0], top_k=1)
        if not results or results[0].index != 0:
            return CheckResult("faiss", "fail", "faiss round-trip returned an unexpected result")
    except Exception as exc:  # noqa: BLE001 - any failure here is a real doctor finding
        return CheckResult("faiss", "fail", f"faiss not usable: {exc}")
    return CheckResult("faiss", "pass", "faiss importable, index/search round-trip works")


def _check_embedding_model() -> CheckResult:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return CheckResult("embedding_model", "fail", "sentence-transformers is not importable")
    return CheckResult(
        "embedding_model",
        "skip",
        "sentence-transformers is importable; actual model download/inference "
        "not checked here (requires network, too slow for a health check) — "
        "runs on the torch backend, see KNOWN_LIMITATIONS.md",
    )


def _check_cache_manifest(repo: Path) -> CheckResult:
    from rylox import cache

    try:
        manifest = cache.load_index(repo)
    except IndexNotFoundError:
        return CheckResult("cache_manifest", "skip", "no index yet — run `rylox index`")
    except IndexCorruptError as exc:
        return CheckResult("cache_manifest", "fail", str(exc))
    return CheckResult(
        "cache_manifest", "pass", f"index readable, {len(manifest.files)} file(s) tracked"
    )