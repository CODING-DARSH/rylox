from __future__ import annotations

from pathlib import Path

import pytest

from rylox import cache
from rylox.errors import IndexCorruptError, IndexNotFoundError


def test_load_embeddings_missing_raises_index_not_found(tmp_path: Path) -> None:
    with pytest.raises(IndexNotFoundError):
        cache.load_embeddings(tmp_path)


def test_load_embeddings_or_empty_returns_empty_with_given_model(tmp_path: Path) -> None:
    manifest = cache.load_embeddings_or_empty(tmp_path, model="some/model")
    assert manifest.files == {}
    assert manifest.model == "some/model"


def test_save_and_load_embeddings_round_trip(tmp_path: Path) -> None:
    manifest = cache.EmbeddingManifest(model="some/model")
    manifest.files["a.py"] = cache.EmbeddedFileEntry(hash="h1", vectors=[[0.1, 0.2], [0.3, 0.4]])
    cache.save_embeddings(tmp_path, manifest)

    loaded = cache.load_embeddings(tmp_path)
    assert loaded.model == "some/model"
    assert loaded.files["a.py"].hash == "h1"
    assert loaded.files["a.py"].vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_corrupt_embeddings_json_raises_index_corrupt_error(tmp_path: Path) -> None:
    cache.ensure_cache_dir(tmp_path)
    (tmp_path / ".rylox" / "embeddings.json").write_text("not json {{{", encoding="utf-8")
    with pytest.raises(IndexCorruptError):
        cache.load_embeddings(tmp_path)


def test_unsupported_embeddings_schema_version_raises_index_corrupt_error(
    tmp_path: Path,
) -> None:
    cache.ensure_cache_dir(tmp_path)
    (tmp_path / ".rylox" / "embeddings.json").write_text(
        '{"schema_version": 999, "model": "x", "files": {}}', encoding="utf-8"
    )
    with pytest.raises(IndexCorruptError):
        cache.load_embeddings(tmp_path)


def test_embeddings_missing_field_raises_index_corrupt_error(tmp_path: Path) -> None:
    cache.ensure_cache_dir(tmp_path)
    (tmp_path / ".rylox" / "embeddings.json").write_text(
        '{"schema_version": 1}', encoding="utf-8"
    )
    with pytest.raises(IndexCorruptError):
        cache.load_embeddings(tmp_path)


def test_save_embeddings_overwrites_previous_content(tmp_path: Path) -> None:
    manifest1 = cache.EmbeddingManifest(model="m1")
    manifest1.files["a.py"] = cache.EmbeddedFileEntry(hash="h1", vectors=[[0.1]])
    cache.save_embeddings(tmp_path, manifest1)

    manifest2 = cache.EmbeddingManifest(model="m2")
    manifest2.files["b.py"] = cache.EmbeddedFileEntry(hash="h2", vectors=[[0.2]])
    cache.save_embeddings(tmp_path, manifest2)

    loaded = cache.load_embeddings(tmp_path)
    assert loaded.model == "m2"
    assert "a.py" not in loaded.files
    assert "b.py" in loaded.files