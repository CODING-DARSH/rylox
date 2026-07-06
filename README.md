# Rylox

**Rylox** is a local, vendor-independent repository context engine for Python codebases.

It builds a structured index of your repository and generates relationship-aware,
token-budgeted context that can be used with any large language model. Rylox
operates entirely offline during indexing and retrieval—no API keys, network
requests, or hosted services are required.

Instead of relying on simple keyword search or sending entire repositories to an
LLM, Rylox analyzes source code structure, extracts meaningful semantic units,
and retrieves only the code that is relevant to a given task.

```bash
rylox index
rylox context "How does authentication work?" --max-tokens 20000
```

The generated output is a structured Markdown document that can be pasted
directly into any LLM while staying within the requested token budget.

---

## Features

- 🚀 Incremental repository indexing using SHA-256 content hashing
- 🌳 Tree-sitter powered Python parsing
- 📦 Function, method, and class-level semantic chunking
- 📄 Rich metadata extraction (paths, line ranges, parent classes, and docstrings)
- 💾 Persistent on-disk index cache
- 📁 Configurable ignore patterns with `.gitignore` support
- 🔒 Offline-first architecture with no external dependencies during retrieval
- 🧪 Comprehensive unit test coverage
- 🖥️ Cross-platform support

---

## Philosophy

Rylox is designed around three principles:

- **Offline First** — Repository indexing and retrieval never require an internet connection.
- **Vendor Independent** — Generated context works with any language model instead of being tied to a specific provider.
- **Deterministic Retrieval** — Context generation is reproducible and based on repository structure rather than opaque LLM reasoning.

---

## Installation

```bash
pip install -e ".[dev]"
```

---

## Development

Run the project's quality checks before submitting changes.

```bash
ruff check .
mypy src
pytest
```

---

```

---

## License

Rylox is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.