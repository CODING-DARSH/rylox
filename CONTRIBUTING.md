# Contributing to Rylox

First off, thank you for considering contributing to Rylox!

Rylox is an offline, vendor-independent repository context engine focused on correctness, deterministic behavior, and maintainable code. Whether you're fixing a bug, improving documentation, or implementing a new feature, your contribution is appreciated.

---

## Development Setup

Clone the repository and install the project in editable mode with development dependencies.

```bash
git clone https://github.com/<your-username>/rylox.git
cd rylox

pip install -e ".[dev]"
```

---

## Running Quality Checks

Before opening a pull request, ensure all quality checks pass.

```bash
ruff check . --fix
mypy src
pytest -v --cov=rylox --cov-report=term-missing
```

Pull requests that fail linting, type checking, or tests should be fixed before review.

---

## Code Style

Please follow the existing project style.

- Add type hints to public functions.
- Keep functions focused and easy to understand.
- Prefer readable code over clever code.
- Add or update tests for every bug fix or feature.
- Avoid introducing unnecessary dependencies.
- Keep changes scoped to a single logical purpose.

Formatting and linting are enforced using **Ruff**, while static type checking is performed using **mypy**.

---

## Pull Requests

Before opening a pull request:

- Ensure all quality checks pass.
- Add tests for new functionality or bug fixes.
- Update documentation if user-facing behavior changes.
- Keep commits focused and use clear commit messages.
- Follow Conventional Commits where possible:

  - `feat:` — New feature
  - `fix:` — Bug fix
  - `docs:` — Documentation changes
  - `test:` — Tests
  - `refactor:` — Code refactoring
  - `chore:` — Maintenance tasks

A good pull request should clearly explain:

- What changed.
- Why the change was necessary.
- Any important implementation details.
- Reference related issues when applicable.

---

## Reporting Bugs

When reporting a bug, please include:

- Operating system
- Python version
- Steps to reproduce
- Expected behavior
- Actual behavior
- Full error message or traceback (if available)

Providing a minimal reproducible example makes issues much easier to investigate.

---

## Feature Requests

Feature requests are always welcome.

When proposing a new feature, please explain:

- The problem you're trying to solve.
- Why the feature would benefit Rylox users.
- Any potential design or implementation ideas.

Opening an issue for discussion before starting implementation is encouraged for larger features.

---

## Project Philosophy

Rylox is built around a few core principles:

- **Offline First** — Indexing and retrieval should never require network access.
- **Vendor Independent** — Generated context should work with any language model.
- **Deterministic** — Identical inputs should produce identical outputs.
- **Well Tested** — New functionality should include appropriate automated test coverage.

Contributions should align with these principles whenever possible.

---
---

## Recognition

We believe contributors should receive proper credit for their work.

After a pull request is merged:

- Your name or GitHub username will be added to **`CONTRIBUTORS.md`**.
- Your contribution will be credited in the release notes for the version in which it is included.
- Major contributions will be acknowledged alongside the corresponding feature or improvement whenever possible.

Every contribution—whether it's code, tests, documentation, or bug fixes—is valued and appreciated.

---

## Questions

If you're unsure about an implementation or would like feedback before starting work, feel free to open an issue for discussion.

Every contribution, no matter how small, is appreciated. Thank you for helping make Rylox better!