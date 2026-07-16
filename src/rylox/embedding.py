"""Embedding provider abstraction.

Rylox's retrieval code only ever depends on the `Embedder` protocol below —
never on a specific library (sentence-transformers, an HTTP client for
Ollama, etc.) directly. Adding a new provider later means adding a new
class here, not touching retrieval logic elsewhere.

Only "huggingface" is registered and runnable right now — it's the only
provider that keeps embedding fully local with no network calls at query
time. Other provider names are rejected by the config loader.
"""

from __future__ import annotations

from typing import Callable, Protocol

from rylox.errors import EmbeddingModelUnavailableError


class Embedder(Protocol):
    """Anything that turns text into vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input text, same order."""
        ...


# Loaded sentence-transformers models are cached by model name, since
# constructing HuggingFaceONNXEmbedder per call/config-load shouldn't force
# a full model reload each time.
_MODEL_CACHE: dict[str, object] = {}


class HuggingFaceONNXEmbedder:
    """Local embedding via sentence-transformers, no network calls at query time.

    The model itself is downloaded once (cached by the HF ecosystem's own
    tooling, typically under ~/.cache/huggingface) the first time a given
    model name is used; after that, loading and inference are fully local.
    """

    def __init__(self, model: str) -> None:
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model = _MODEL_CACHE.get(self.model)
        if model is None:
            from sentence_transformers import SentenceTransformer

            # NOTE: backend="onnx" is intentionally not used here. It routes
            # through optimum's ONNX exporter, which on real testing failed
            # with an ImportError reaching into a private, version-specific
            # torch API (torch.onnx.symbolic_opset14._attention_scale) that
            # doesn't exist across all torch releases. That's a genuine
            # dependency incompatibility, confirmed on a real machine, not a
            # bug in this code. Using the default torch backend until the
            # optimum/torch/transformers version triangle is pinned to a
            # combination that's actually verified to work end-to-end.
            try:
                model = SentenceTransformer(self.model)
            except Exception as exc:  # noqa: BLE001 - any load failure here
                # needs to surface as a clean, actionable message, not a raw
                # HTTPError/OSError traceback bubbling up from huggingface_hub.
                raise EmbeddingModelUnavailableError(
                    f"could not load embedding model '{self.model}' (needs a "
                    f"one-time network download on first use, or a local "
                    f"cache): {exc}"
                ) from exc
            _MODEL_CACHE[self.model] = model

        vectors = model.encode(texts, convert_to_numpy=True)  # type: ignore[attr-defined]
        return [vector.tolist() for vector in vectors]


_PROVIDERS: dict[str, Callable[..., Embedder]] = {
    "huggingface": HuggingFaceONNXEmbedder,
}

SUPPORTED_PROVIDERS = tuple(sorted(_PROVIDERS))


def get_embedder(provider: str, model: str) -> Embedder:
    """Instantiate the Embedder for a validated provider name.

    Callers (the config loader) are expected to have already validated
    `provider` against SUPPORTED_PROVIDERS — this function's KeyError is a
    last-resort internal safety net, not the primary user-facing error path.
    """
    try:
        embedder_cls = _PROVIDERS[provider]
    except KeyError as exc:
        raise KeyError(
            f"no Embedder registered for provider '{provider}'; "
            f"supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
        ) from exc
    return embedder_cls(model=model)