# Rylox

**Rylox** is a fast, local repository context engine for Python codebases.

It analyzes your repository, builds a structured semantic index, and retrieves
relationship-aware, token-budgeted context that can be used with any large
language model.

Unlike keyword search or sending entire repositories to an LLM, Rylox
understands the structure of your codebase and selects only the code that is
relevant to the task. Indexing and retrieval run completely offline—no API
keys, cloud services, or vendor-specific integrations required.

```bash
rylox index
rylox context "How does authentication work?" --max-tokens 20000
```

The generated output is a structured Markdown document that can be pasted
directly into ChatGPT, Claude, Gemini, Codex, or any other LLM while remaining
within the requested token budget.

---

## Why Rylox?

Large repositories quickly exceed an LLM's context window. Traditional keyword
search often retrieves unrelated files, while embedding-based approaches require
external services or non-deterministic ranking.

Rylox takes a different approach by analyzing the repository's structure,
extracting semantic code units, and building deterministic context from actual
relationships inside the codebase.

---

## Features

- Incremental indexing using SHA-256 content hashing
- Tree-sitter powered Python parsing
- Semantic chunking at the function, method, and class level
- Relationship-aware context retrieval
- Token-budgeted context generation
- Rich metadata extraction
  - file paths
  - line ranges
  - parent classes
  - signatures
  - docstrings
- Persistent on-disk index cache
- `.gitignore` and configurable ignore pattern support
- Offline-first architecture
- Cross-platform support
- Comprehensive unit tests

---

## Example

Index a repository:

```bash
rylox index
```

Generate context for an LLM:

```bash
rylox context \
  "How does authentication work?" \
  --max-tokens 20000
```

Rylox returns a structured Markdown document containing only the most relevant
parts of the repository while respecting the requested token budget.

---

## Philosophy

Rylox is built around three core principles.

### Offline First

Repository indexing and retrieval never require an internet connection.

### Vendor Independent

The generated context is provider-agnostic and works with any language model.

### Deterministic Retrieval

Context generation is reproducible and driven by repository structure rather
than opaque ranking heuristics.

---

## Installation

Install from source:

```bash
pip install -e ".[dev]"
```

---

## Supported Language

Current release:

- Python

Future releases are designed to support additional programming languages through
Tree-sitter grammars.

---

## License

Rylox is licensed under the Apache License 2.0. See the
[LICENSE](LICENSE) file for details.