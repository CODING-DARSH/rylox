"""Embedding provider abstraction.

Rylox's retrieval code should only ever depend on the `Embedder` protocol
below — never on a specific library (sentence-transformers, an HTTP client
for Ollama, etc.) directly. That keeps "how embeddings are produced" a
swappable implementation detail behind one interface, so adding a new
provider later (Ollama, OpenAI, Voyage...) means adding a new class here,
not touching retrieval logic elsewhere in the codebase.

v0.1 scope note: only "huggingface" is registered and runnable. This
matches spec §13's "no network dependency at runtime for the default
configuration" — a provider that calls out to a hosted API would break
that guarantee, so such providers are deliberately not wired up yet, even
though the interface already supports them. Registering a provider name in
`_PROVIDERS` without a real, spec-compliant implementation would be worse
than not having the abstraction at all: it would silently promise
something Rylox doesn't do.
"""

from __future__ import annotations

from typing import Callable, Protocol


class Embedder(Protocol):
    """Anything that turns text into vectors.

    Implementations are added under `_PROVIDERS` below and selected via
    `[embedding].provider` in rylox.toml. Retrieval code (Phase 5) should
    request an Embedder through `get_embedder()` and never import a
    specific implementation directly.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input text, same order."""
        ...


class HuggingFaceONNXEmbedder:
    """Local, ONNX-runnable embedding via sentence-transformers/onnxruntime.

    The only Embedder implementation shipped in v0.1. Real model loading
    and inference land in Phase 5 (Dense retrieval) — this class exists now
    so the config schema and provider-selection logic have a real target,
    but `embed()` intentionally raises until Phase 5 fills it in.
    """

    def __init__(self, model: str) -> None:
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "HuggingFaceONNXEmbedder.embed() is implemented in Phase 5 (Dense retrieval)."
        )


# Provider name -> implementation. This is the ONLY place that needs a new
# entry when a future provider (ollama, openai, voyage, ...) is added.
# Adding an entry here is a deliberate, reviewable act — not something a
# user can trigger just by writing a new string into rylox.toml.
_PROVIDERS: dict[str, Callable[..., Embedder]] = {
    "huggingface": HuggingFaceONNXEmbedder,
}

SUPPORTED_PROVIDERS = tuple(sorted(_PROVIDERS))


def get_embedder(provider: str, model: str) -> Embedder:
    """Instantiate the Embedder for a validated provider name.

    Callers (i.e. the config loader) are expected to have already validated
    `provider` against SUPPORTED_PROVIDERS and raised a clear ConfigError if
    not — this function's KeyError is a last-resort internal safety net,
    not the primary user-facing error path.
    """
    try:
        embedder_cls = _PROVIDERS[provider]
    except KeyError as exc:
        raise KeyError(
            f"no Embedder registered for provider '{provider}'; "
            f"supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
        ) from exc
    return embedder_cls(model=model)