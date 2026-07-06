# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog and this project follows Semantic Versioning.

---
## [0.0.2] - Unreleased

### Added

- Added the `rylox doctor` command for diagnosing local environments and repository health.
- Added environment validation for the installed Python version, Tree-sitter parser availability, FAISS, and ONNX Runtime.
- Added cache and index manifest validation to detect missing or corrupted repository indexes.
- Added clear pass, fail, and skip status reporting for diagnostic checks.
- Added CLI test coverage for the `doctor` command, including diagnostic output verification.
- Expanded cache management test coverage with validation for missing and existing local cache cleanup.

### Changed

- Improved the CLI with built-in environment diagnostics for easier troubleshooting.
- Expanded automated test coverage for CLI commands and cache management workflows.

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