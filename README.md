# Rylox

Local, vendor-independent repository context engine. Turns a codebase and a
task description into a structured, token-budgeted markdown context package
— no LLM in the retrieval path, no network calls, no API keys.

```bash
rylox index
rylox context "How does authentication work?" --max-tokens 20000
# → checkout_context.md: relevant, relationship-aware, budget-respecting,
#   paste-ready into any LLM
```

## Status

**Phase 1 — skeleton.** The CLI interface (`index`, `context`, `clean`,
`doctor`) is locked in and tested; command bodies are stubs. See
[`ROADMAP.md`](./ROADMAP.md) for the full build plan.

## Development

```bash
pip install -e ".[dev]"
ruff check .
mypy src
pytest
```

## License

Apache License 2.0 — see [`LICENSE`](./LICENSE).