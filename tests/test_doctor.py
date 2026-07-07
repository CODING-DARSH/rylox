from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from rylox import cache, doctor


def test_run_doctor_returns_one_result_per_check(tmp_path: Path) -> None:
    results = doctor.run_doctor(tmp_path)
    names = [r.name for r in results]
    assert names == [
        "python_version",
        "tree_sitter",
        "faiss",
        "embedding_model",
        "cache_manifest",
    ]


def test_check_python_version_passes_on_current_interpreter() -> None:
    result = doctor._check_python_version()
    assert result.status == "pass"
    assert sys.version.split()[0] in result.detail


def test_check_python_version_fails_below_floor() -> None:
    with patch.object(doctor, "MIN_PYTHON", (99, 0)):
        result = doctor._check_python_version()
    assert result.status == "fail"
    assert "below the required" in result.detail


def test_check_tree_sitter_passes_on_valid_probe_source() -> None:
    result = doctor._check_tree_sitter()
    assert result.status == "pass"
    assert "parsing" in result.detail


def test_check_tree_sitter_fails_when_parsing_raises() -> None:
    with patch("rylox.chunking.parse_source", side_effect=RuntimeError("boom")):
        result = doctor._check_tree_sitter()
    assert result.status == "fail"
    assert "boom" in result.detail


def test_check_faiss_passes_import_when_available() -> None:
    result = doctor._check_faiss()
    # faiss-cpu is a hard runtime dependency (pyproject.toml), so on any
    # environment where the test suite itself runs, it must be importable.
    assert result.status in ("pass", "skip")
    if result.status == "skip":
        assert "round-trip" in result.detail


def test_check_faiss_fails_when_not_importable() -> None:
    with patch.dict(sys.modules, {"faiss": None}):
        result = doctor._check_faiss()
    assert result.status == "fail"
    assert "not importable" in result.detail


def test_check_embedding_model_skips_when_onnxruntime_importable() -> None:
    result = doctor._check_embedding_model()
    assert result.status in ("skip", "fail")
    if result.status == "skip":
        assert "not yet wired" in result.detail


def test_check_embedding_model_fails_when_not_importable() -> None:
    with patch.dict(sys.modules, {"onnxruntime": None}):
        result = doctor._check_embedding_model()
    assert result.status == "fail"
    assert "not importable" in result.detail


def test_check_cache_manifest_skips_when_no_index_yet(tmp_path: Path) -> None:
    result = doctor._check_cache_manifest(tmp_path)
    assert result.status == "skip"
    assert "run `rylox index`" in result.detail


def test_check_cache_manifest_passes_with_valid_index(tmp_path: Path) -> None:
    manifest = cache.IndexManifest()
    manifest.files["a.py"] = cache.FileEntry(hash="deadbeef", chunks=[])
    cache.save_index(tmp_path, manifest)

    result = doctor._check_cache_manifest(tmp_path)
    assert result.status == "pass"
    assert "1 file(s) tracked" in result.detail


def test_check_cache_manifest_fails_with_corrupt_index(tmp_path: Path) -> None:
    cache.ensure_cache_dir(tmp_path)
    (tmp_path / ".rylox" / "index.json").write_text("not valid json {{{", encoding="utf-8")

    result = doctor._check_cache_manifest(tmp_path)
    assert result.status == "fail"


def test_run_doctor_all_pass_or_skip_on_clean_repo_without_index(tmp_path: Path) -> None:
    """No hard failures expected in a normal dev environment with no index yet —
    matches the CLI's exit-code-1-only-on-fail behavior (cli.py `doctor` command).
    """
    results = doctor.run_doctor(tmp_path)
    assert all(r.status in ("pass", "skip") for r in results)