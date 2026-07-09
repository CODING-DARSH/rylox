from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rylox import cache
from rylox.bm25 import BM25Index, build_bm25_index
from rylox.cache import CachedChunk
from rylox.config import RyloxConfig
from rylox.embedding import get_embedder
from rylox.errors import IndexNotFoundError
from rylox.fusion import reciprocal_rank_fusion
from rylox.vectorstore import VectorStore, build_index

# How many candidates to pull from each of dense/sparse before fusing.
# Wider than the final top_k so RRF has enough overlap to work with —
# fusing two lists that are each already clipped to top_k loses exactly
# the candidates that would have ranked second-best in one source but
# still relevant overall.
_CANDIDATE_POOL_MULTIPLIER = 5
_MIN_CANDIDATE_POOL = 50


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
    bm25: BM25Index
    chunks: list[CachedChunk] = field(default_factory=list)

    def resolve(self, index: int) -> CachedChunk:
        return self.chunks[index]


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

    bm25 = build_bm25_index([chunk.content for chunk in chunks])
    return ChunkVectorIndex(store=build_index(vectors), bm25=bm25, chunks=chunks)


@dataclass(frozen=True)
class FusedSearchResult:
    chunk: CachedChunk
    score: float
    sources: frozenset[str]  # {"dense"}, {"sparse"}, or {"dense", "sparse"}


def search(
    repo: Path, config: RyloxConfig, query: str, top_k: int
) -> list[FusedSearchResult]:
    """Return the top_k chunks for `query`, fusing dense (FAISS) and sparse
    (BM25) retrieval via Reciprocal Rank Fusion rather than either alone.
    """
    index = load_chunk_vector_index(repo)

    pool_size = min(index.store.size, max(top_k * _CANDIDATE_POOL_MULTIPLIER, _MIN_CANDIDATE_POOL))

    embedder = get_embedder(config.embedding.provider, config.embedding.model)
    query_vector = embedder.embed([query])[0]
    dense_hits = index.store.search(query_vector, top_k=pool_size)
    sparse_hits = index.bm25.search(query, top_k=pool_size)

    fused = reciprocal_rank_fusion(
        {
            "dense": [hit.index for hit in dense_hits],
            "sparse": [hit.index for hit in sparse_hits],
        }
    )

    return [
        FusedSearchResult(chunk=index.resolve(r.index), score=r.score, sources=r.sources)
        for r in fused[:top_k]
    ]