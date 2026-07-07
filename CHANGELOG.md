# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog and this project follows Semantic Versioning.

---

## [0.0.2] - Unreleased

### Added

- Added `rylox.toml` provider abstraction for embeddings: an `Embedder`
  interface with a pluggable provider registry, so future providers
  (Ollama, OpenAI, Voyage, etc.) can be added without touching retrieval
  code. Only `"huggingface"` is currently supported and validated.
- Added real local embedding via `sentence-transformers`: repository
  chunks and search queries are turned into vectors using
  `BAAI/bge-small-en-v1.5` by default, fully offline after the model is
  first downloaded.
- Added a FAISS-backed vector store with cosine-similarity search,
  including dimension-mismatch and empty-index validation.
- Added persistent, incremental embedding storage
  (`.rylox/embeddings.json`): embeddings are computed once and reused on
  future runs — only files whose content actually changed are re-embedded.
  Deleted files are cleaned up automatically, and a changed embedding
  model correctly triggers a full re-embed.
- Added the `rylox index` command's real implementation: it now builds
  and incrementally updates the on-disk chunk index and embedding cache,
  reporting a summary of files changed, unchanged, deleted, and embedded.
- Added the `rylox doctor` command for diagnosing local environments and
  repository health, with environment validation for the installed
  Python version, Tree-sitter parser availability, FAISS (via a real
  build-and-search round-trip), the embedding library, and the on-disk
  cache/index manifest (missing or corrupted state detected explicitly).
- Added the `rylox clean` command's real implementation: deletes the
  `.rylox/` cache directory, or reports there was nothing to clean.
- Added a `rylox.toml` config schema and loader: typed sections for
  embedding, retrieval, budget, output, and ignore settings, with
  immediate, specific validation errors rather than silent defaults or
  partial failures.
- Added atomic, crash-safe writes for both the chunk index and the
  embedding cache (write-to-temp-file-then-rename), so an interrupted
  `rylox index` run can never leave a corrupted cache behind.
- Added chunk-level source text capture (`content`) alongside existing
  metadata (path, line range, parent class, docstring), so later stages
  don't need to re-open files to retrieve code text.
- Added decorator handling to the chunker: a decorated function or class
  is chunked as one unit including its decorator line(s), rather than
  starting at the `def`/`class` line and silently dropping the decorator.
- Added UTF-8 BOM stripping on file read, preventing a leading BOM
  character from silently corrupting the first chunk of a file.
- Added CLI and unit test coverage across configuration, chunking,
  caching, indexing, embedding, vector search, retrieval, and diagnostics.

### Changed

- Improved the CLI with built-in environment diagnostics (`rylox doctor`)
  for easier troubleshooting.
- Expanded automated test coverage for CLI commands, configuration
  loading, chunking edge cases (decorators, `async def`, nested classes,
  CRLF line endings, malformed/unreadable files), and cache management
  workflows.

### Notes

Embedding inference currently runs on the standard PyTorch backend rather
than ONNX Runtime. An ONNX backend was attempted and reverted after a
confirmed, real incompatibility between `torch`, `optimum`, and
`transformers` versions (see `KNOWN_LIMITATIONS.md`); `onnxruntime`
remains a declared dependency but is not currently exercised.

## [0.0.1] - 2026-07-06

### Added

- Initial project structure and Python packaging.
- Command-line interface built with Typer.
- Configuration system with `rylox.toml`.
- Tree-sitter based Python parsing.
- Function, method, and class level semantic chunking.
- Persistent on-disk index cache.
- Incremental repository indexing using SHA-256 content hashing.
- Configurable ignore patterns and `.gitignore` support.
- Cross-platform test suite.
- Continuous Integration with Ruff, mypy, and pytest.
- Repository documentation and contribution guidelines.
- GitHub issue forms, pull request template, and CODEOWNERS.

### Notes

The project is currently under active development. While the indexing
foundation is complete, context generation and retrieval capabilities are
still being implemented as part of the planned v0.1.0 milestone.