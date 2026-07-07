from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rylox.embedding import SUPPORTED_PROVIDERS, get_embedder

_RUN_REAL_MODEL_TESTS = os.environ.get("RYLOX_TEST_REAL_MODEL") == "1"


def test_supported_providers_contains_only_huggingface() -> None:
    assert SUPPORTED_PROVIDERS == ("huggingface",)


def test_get_embedder_unregistered_provider_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        get_embedder("ollama", "bge-m3")


def test_embed_empty_list_returns_empty_list_without_loading_model() -> None:
    embedder = get_embedder("huggingface", "some/model")
    assert embedder.embed([]) == []


def test_embed_returns_list_of_lists_matching_input_count() -> None:
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model) as mock_cls:
        import rylox.embedding as embedding_mod

        embedding_mod._MODEL_CACHE.clear()
        embedder = get_embedder("huggingface", "fake/model-for-test")
        vectors = embedder.embed(["hello", "world"])

        assert len(vectors) == 2
        assert vectors[0] == pytest.approx([0.1, 0.2, 0.3])
        assert vectors[1] == pytest.approx([0.4, 0.5, 0.6])
        mock_cls.assert_called_once_with("fake/model-for-test")


def test_model_is_loaded_only_once_across_multiple_embed_calls() -> None:
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[0.1, 0.2]])

    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model) as mock_cls:
        import rylox.embedding as embedding_mod

        embedding_mod._MODEL_CACHE.clear()
        embedder = get_embedder("huggingface", "fake/reuse-model")
        embedder.embed(["first call"])
        embedder.embed(["second call"])

        assert mock_cls.call_count == 1


@pytest.mark.skipif(
    not _RUN_REAL_MODEL_TESTS,
    reason=(
        "downloads a real model over the network and runs real ONNX inference; "
        "set RYLOX_TEST_REAL_MODEL=1 to run this test explicitly"
    ),
)
def test_real_huggingface_embedder_produces_correct_dimension_vectors() -> None:
    """Not mocked — this actually downloads BAAI/bge-small-en-v1.5 (once,
    cached afterward) and runs it through the ONNX backend. Every other
    test in this file mocks SentenceTransformer, which proves the code
    around the model call is correct but never proves the real model
    loads and produces real embeddings. This test is the one that does.
    """
    embedder = get_embedder("huggingface", "BAAI/bge-small-en-v1.5")
    vectors = embedder.embed(["def login():\n    pass", "def logout():\n    pass"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 384  # BAAI/bge-small-en-v1.5's embedding dimension
    assert len(vectors[1]) == 384
    assert vectors[0] != vectors[1]  # different inputs must not collapse to the same vector