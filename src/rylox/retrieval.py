from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rylox import cache
from rylox.cache import CachedChunk
from rylox.config import RyloxConfig
from rylox.embedding import get_embedder
from rylox.errors import IndexNotFoundError
from rylox.vectorstore import VectorStore, build_index


@dataclass
class EmbeddingUpdateReport:
    embedded_files: int = 0
    reused_files: int = 0
    deleted_files: int = 0
    total_chunks: int = 0
    model_changed: bool = False


def update_embeddings(repo: Path, config: RyloxConfig) -> EmbeddingUpdateReport:
    """Bring the embedding cache in line with the current chunk index.

    Only files whose content hash changed since the last embedding run are
    re-embedded. Files whose hash matches keep their previously computed
    vectors untouched. This never re-chunks or re-indexes anything — it
    only reads whatever `rylox index` already produced.
    """
    chunk_manifest = cache.load_index(repo)
    embedding_manifest = cache.load_embeddings_or_empty(repo, model=config.embedding.model)
    report = EmbeddingUpdateReport()

    if embedding_manifest.model != config.embedding.model:
        embedding_manifest = cache.EmbeddingManifest(model=config.embedding.model)
        report.model_changed = True

    current_relpaths = set(chunk_manifest.files.keys())
    for stale in set(embedding_manifest.files.keys()) - current_relpaths:
        del embedding_manifest.files[stale]
        report.deleted_files += 1

    embedder = get_embedder(config.embedding.provider, config.embedding.model)

    for relpath, file_entry in chunk_manifest.files.items():
        existing = embedding_manifest.files.get(relpath)
        if existing is not None and existing.hash == file_entry.hash:
            report.reused_files += 1
            report.total_chunks += len(existing.vectors)
            continue

        contents = [chunk.content for chunk in file_entry.chunks]
        vectors = embedder.embed(contents) if contents else []
        embedding_manifest.files[relpath] = cache.EmbeddedFileEntry(
            hash=file_entry.hash, vectors=vectors
        )
        report.embedded_files += 1
        report.total_chunks += len(vectors)

    cache.save_embeddings(repo, embedding_manifest)
    return report


@dataclass
class ChunkVectorIndex:
    store: VectorStore
    chunks: list[CachedChunk] = field(default_factory=list)

    def resolve(self, vector_index: int) -> CachedChunk:
        return self.chunks[vector_index]


class NoEmbeddedChunksError(Exception):
    """Raised when there is nothing to build a vector index from."""


def load_chunk_vector_index(repo: Path) -> ChunkVectorIndex:
    """Build a searchable index from whatever embeddings are on disk.

    Call `update_embeddings` first if the index needs to reflect the
    latest repo state — this function is read-only.
    """
    chunk_manifest = cache.load_index(repo)
    try:
        embedding_manifest = cache.load_embeddings(repo)
    except IndexNotFoundError as exc:
        raise NoEmbeddedChunksError(
            "no embedded chunks found. Run indexing and the embedding update first."
        ) from exc

    vectors: list[list[float]] = []
    chunks: list[CachedChunk] = []
    for relpath, file_entry in chunk_manifest.files.items():
        embedded = embedding_manifest.files.get(relpath)
        if embedded is None:
            continue
        for chunk, vector in zip(file_entry.chunks, embedded.vectors):
            vectors.append(vector)
            chunks.append(chunk)

    if not vectors:
        raise NoEmbeddedChunksError(
            "no embedded chunks found. Run indexing and the embedding update first."
        )

    return ChunkVectorIndex(store=build_index(vectors), chunks=chunks)


def search(
    repo: Path, config: RyloxConfig, query: str, top_k: int
) -> list[tuple[CachedChunk, float]]:
    """Embed `query` and return the top_k most similar chunks currently indexed."""
    vector_index = load_chunk_vector_index(repo)
    embedder = get_embedder(config.embedding.provider, config.embedding.model)
    query_vector = embedder.embed([query])[0]

    results = vector_index.store.search(query_vector, top_k=top_k)
    return [(vector_index.resolve(r.index), r.score) for r in results]