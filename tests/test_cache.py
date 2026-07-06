from __future__ import annotations

from pathlib import Path

import pytest

from rylox import cache
from rylox.chunking import Chunk
from rylox.errors import IndexCorruptError, IndexNotFoundError


def test_ensure_cache_dir_creates_directory_and_gitignore(tmp_path: Path) -> None:
    directory = cache.ensure_cache_dir(tmp_path)
    assert directory.exists()
    assert (directory / ".gitignore").read_text(encoding="utf-8") == "*\n"


def test_ensure_cache_dir_does_not_overwrite_existing_gitignore(tmp_path: Path) -> None:
    directory = cache.ensure_cache_dir(tmp_path)
    (directory / ".gitignore").write_text("custom\n", encoding="utf-8")
    cache.ensure_cache_dir(tmp_path)
    assert (directory / ".gitignore").read_text(encoding="utf-8") == "custom\n"


def test_load_index_missing_raises_index_not_found(tmp_path: Path) -> None:
    with pytest.raises(IndexNotFoundError):
        cache.load_index(tmp_path)


def test_load_or_empty_returns_empty_manifest_when_missing(tmp_path: Path) -> None:
    manifest = cache.load_or_empty(tmp_path)
    assert manifest.files == {}
    assert manifest.schema_version == cache.SCHEMA_VERSION


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    manifest = cache.IndexManifest()
    manifest.files["a.py"] = cache.FileEntry(
        hash="deadbeef",
        chunks=[
            cache.CachedChunk(
                path="a.py",
                kind="function",
                name="f",
                start_line=1,
                end_line=2,
                parent_class=None,
                docstring=None,
                content="def f():\n    pass",
            )
        ],
    )
    cache.save_index(tmp_path, manifest)

    loaded = cache.load_index(tmp_path)
    assert loaded.files["a.py"].hash == "deadbeef"
    assert loaded.files["a.py"].chunks[0].name == "f"


def test_cached_chunk_round_trips_to_chunk() -> None:
    original = Chunk(
        path=Path("a.py"),
        kind="method",
        name="bar",
        start_line=3,
        end_line=5,
        parent_class="Foo",
        docstring="doc",
        content="def bar(self):\n    pass",
    )
    cached = cache.CachedChunk.from_chunk(original)
    restored = cached.to_chunk()
    assert restored == original


def test_corrupt_json_raises_index_corrupt_error(tmp_path: Path) -> None:
    cache.ensure_cache_dir(tmp_path)
    (tmp_path / ".rylox" / "index.json").write_text("not valid json {{{", encoding="utf-8")
    with pytest.raises(IndexCorruptError):
        cache.load_index(tmp_path)


def test_unsupported_schema_version_raises_index_corrupt_error(tmp_path: Path) -> None:
    cache.ensure_cache_dir(tmp_path)
    (tmp_path / ".rylox" / "index.json").write_text(
        '{"schema_version": 999, "files": {}}', encoding="utf-8"
    )
    with pytest.raises(IndexCorruptError):
        cache.load_index(tmp_path)


def test_missing_required_field_raises_index_corrupt_error(tmp_path: Path) -> None:
    cache.ensure_cache_dir(tmp_path)
    (tmp_path / ".rylox" / "index.json").write_text('{"schema_version": 1}', encoding="utf-8")
    with pytest.raises(IndexCorruptError):
        cache.load_index(tmp_path)


def test_save_index_does_not_leave_temp_file_behind(tmp_path: Path) -> None:
    manifest = cache.IndexManifest()
    cache.save_index(tmp_path, manifest)
    remaining = list((tmp_path / ".rylox").glob(".index-*.tmp"))
    assert remaining == []


def test_save_index_overwrites_previous_content(tmp_path: Path) -> None:
    manifest1 = cache.IndexManifest()
    manifest1.files["a.py"] = cache.FileEntry(hash="h1", chunks=[])
    cache.save_index(tmp_path, manifest1)

    manifest2 = cache.IndexManifest()
    manifest2.files["b.py"] = cache.FileEntry(hash="h2", chunks=[])
    cache.save_index(tmp_path, manifest2)

    loaded = cache.load_index(tmp_path)
    assert "a.py" not in loaded.files
    assert "b.py" in loaded.files