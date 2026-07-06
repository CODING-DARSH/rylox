from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from rylox import cache
from rylox.config import RyloxConfig
from rylox.indexer import MAX_FILE_SIZE_BYTES, hash_file, run_index


def _write(repo: Path, relpath: str, content: str) -> None:
    path = repo / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_first_run_indexes_all_files_as_changed(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "def a():\n    pass\n")
    _write(tmp_path, "b.py", "def b():\n    pass\n")
    report = run_index(tmp_path, RyloxConfig())
    assert report.total_files == 2
    assert report.changed == 2
    assert report.unchanged == 0


def test_second_run_with_no_changes_reports_all_unchanged(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "def a():\n    pass\n")
    run_index(tmp_path, RyloxConfig())
    report = run_index(tmp_path, RyloxConfig())
    assert report.changed == 0
    assert report.unchanged == 1


def test_modified_file_is_reindexed(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "def a():\n    pass\n")
    run_index(tmp_path, RyloxConfig())

    _write(tmp_path, "a.py", "def a():\n    return 1\n")
    report = run_index(tmp_path, RyloxConfig())
    assert report.changed == 1
    assert report.unchanged == 0

    manifest = cache.load_index(tmp_path)
    assert "return 1" in manifest.files["a.py"].chunks[0].content


def test_deleted_file_is_removed_from_manifest(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "def a():\n    pass\n")
    run_index(tmp_path, RyloxConfig())

    (tmp_path / "a.py").unlink()
    report = run_index(tmp_path, RyloxConfig())
    assert report.deleted == 1
    assert report.total_files == 0

    manifest = cache.load_index(tmp_path)
    assert "a.py" not in manifest.files


def test_unchanged_file_chunks_survive_in_manifest_across_runs(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "def a():\n    pass\n")
    _write(tmp_path, "b.py", "def b():\n    pass\n")
    run_index(tmp_path, RyloxConfig())

    _write(tmp_path, "a.py", "def a():\n    return 1\n")
    run_index(tmp_path, RyloxConfig())

    manifest = cache.load_index(tmp_path)
    assert manifest.files["b.py"].chunks[0].name == "b"


def test_default_ignore_patterns_exclude_common_directories(tmp_path: Path) -> None:
    _write(tmp_path, "real.py", "def real():\n    pass\n")
    _write(tmp_path, "node_modules/vendor.py", "def vendor():\n    pass\n")
    _write(tmp_path, "__pycache__/cached.py", "def cached():\n    pass\n")
    report = run_index(tmp_path, RyloxConfig())
    manifest = cache.load_index(tmp_path)
    assert report.total_files == 1
    assert list(manifest.files.keys()) == ["real.py"]


def test_gitignore_patterns_respected_when_enabled(tmp_path: Path) -> None:
    _write(tmp_path, "keep.py", "def keep():\n    pass\n")
    _write(tmp_path, "build/generated.py", "def generated():\n    pass\n")
    (tmp_path / ".gitignore").write_text("build\n", encoding="utf-8")

    report = run_index(tmp_path, RyloxConfig())
    manifest = cache.load_index(tmp_path)
    assert report.total_files == 1
    assert list(manifest.files.keys()) == ["keep.py"]


def test_gitignore_ignored_when_respect_gitignore_is_false(tmp_path: Path) -> None:
    _write(tmp_path, "keep.py", "def keep():\n    pass\n")
    _write(tmp_path, "build/generated.py", "def generated():\n    pass\n")
    (tmp_path / ".gitignore").write_text("build\n", encoding="utf-8")

    config = RyloxConfig()
    config = replace(config, ignore=replace(config.ignore, respect_gitignore=False))
    report = run_index(tmp_path, config)
    assert report.total_files == 2


def test_oversized_file_is_skipped_with_warning(tmp_path: Path) -> None:
    big = tmp_path / "big.py"
    big.write_bytes(b"x = 1\n" * (MAX_FILE_SIZE_BYTES // 6 + 1000))
    report = run_index(tmp_path, RyloxConfig())
    assert report.total_files == 0
    assert "big.py" in report.skipped_too_large


def test_symlink_escaping_repo_root_is_skipped(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_target.py"
    outside.write_text("def outside():\n    pass\n", encoding="utf-8")
    link = tmp_path / "link.py"
    try:
        os.symlink(outside, link)
    except OSError as exc:
        pytest.skip(f"cannot create symlinks in this environment: {exc}")
    _write(tmp_path, "normal.py", "def normal():\n    pass\n")

    try:
        report = run_index(tmp_path, RyloxConfig())
        assert "link.py" in report.skipped_symlink_escape
        assert report.total_files == 1
    finally:
        outside.unlink(missing_ok=True)


def test_malformed_file_is_recorded_and_does_not_crash(tmp_path: Path) -> None:
    _write(tmp_path, "broken.py", "def f(:\n    !!! not python ###\n")
    report = run_index(tmp_path, RyloxConfig())
    assert report.total_files == 1
    assert isinstance(report.parse_errors, list)


def test_hash_file_is_stable_for_same_content(tmp_path: Path) -> None:
    path = tmp_path / "a.py"
    path.write_text("def a():\n    pass\n", encoding="utf-8")
    assert hash_file(path) == hash_file(path)


def test_hash_file_differs_for_different_content(tmp_path: Path) -> None:
    path1 = tmp_path / "a.py"
    path2 = tmp_path / "b.py"
    path1.write_text("def a():\n    pass\n", encoding="utf-8")
    path2.write_text("def b():\n    return 1\n", encoding="utf-8")
    assert hash_file(path1) != hash_file(path2)