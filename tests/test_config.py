from __future__ import annotations

from pathlib import Path

import pytest

from rylox import config as config_mod
from rylox.embedding import SUPPORTED_PROVIDERS, get_embedder
from rylox.errors import ConfigError


def test_write_default_config_creates_file(tmp_path: Path) -> None:
    path = config_mod.write_default_config(tmp_path)
    assert path.exists()
    assert path.name == "rylox.toml"


def test_write_default_config_refuses_to_overwrite(tmp_path: Path) -> None:
    config_mod.write_default_config(tmp_path)
    with pytest.raises(ConfigError):
        config_mod.write_default_config(tmp_path)


def test_load_config_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        config_mod.load_config(tmp_path)


def test_load_or_write_defaults_writes_then_loads(tmp_path: Path) -> None:
    cfg = config_mod.load_or_write_defaults(tmp_path)
    assert (tmp_path / "rylox.toml").exists()
    assert cfg.embedding.model == config_mod.DEFAULT_EMBEDDING_MODEL
    assert cfg.embedding.provider == "huggingface"
    assert cfg.retrieval.call_depth == 1
    assert cfg.budget.max_tokens == config_mod.DEFAULT_MAX_TOKENS
    assert cfg.output.format == "markdown"
    assert cfg.ignore.respect_gitignore is True


def test_default_config_round_trips_through_loader(tmp_path: Path) -> None:
    """The exact string we write as defaults must itself be valid + accepted."""
    config_mod.write_default_config(tmp_path)
    cfg = config_mod.load_config(tmp_path)
    assert cfg == config_mod.RyloxConfig()


def test_invalid_toml_raises_config_error(tmp_path: Path) -> None:
    (tmp_path / "rylox.toml").write_text("this is not [valid toml", encoding="utf-8")
    with pytest.raises(ConfigError):
        config_mod.load_config(tmp_path)


def test_unsupported_embedding_provider_rejected(tmp_path: Path) -> None:
    (tmp_path / "rylox.toml").write_text(
        '[embedding]\nprovider = "openai"\nmodel = "text-embedding-3-small"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="not supported"):
        config_mod.load_config(tmp_path)


def test_call_depth_other_than_one_rejected(tmp_path: Path) -> None:
    (tmp_path / "rylox.toml").write_text("[retrieval]\ncall_depth = 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="hardcoded to 1"):
        config_mod.load_config(tmp_path)


def test_negative_max_tokens_rejected(tmp_path: Path) -> None:
    (tmp_path / "rylox.toml").write_text("[budget]\nmax_tokens = -5\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        config_mod.load_config(tmp_path)


def test_non_markdown_output_format_rejected(tmp_path: Path) -> None:
    (tmp_path / "rylox.toml").write_text('[output]\nformat = "json"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="not supported"):
        config_mod.load_config(tmp_path)


def test_wrong_type_for_max_tokens_rejected(tmp_path: Path) -> None:
    (tmp_path / "rylox.toml").write_text('[budget]\nmax_tokens = "a lot"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="must be of type int"):
        config_mod.load_config(tmp_path)


def test_bool_rejected_for_int_field(tmp_path: Path) -> None:
    (tmp_path / "rylox.toml").write_text("[budget]\nmax_tokens = true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="boolean"):
        config_mod.load_config(tmp_path)


def test_ignore_patterns_must_be_strings(tmp_path: Path) -> None:
    (tmp_path / "rylox.toml").write_text("[ignore]\npatterns = [1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="list of strings"):
        config_mod.load_config(tmp_path)


def test_section_must_be_a_table(tmp_path: Path) -> None:
    (tmp_path / "rylox.toml").write_text('embedding = "nope"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a table"):
        config_mod.load_config(tmp_path)


def test_partial_config_falls_back_to_defaults_for_missing_fields(tmp_path: Path) -> None:
    (tmp_path / "rylox.toml").write_text("[budget]\nmax_tokens = 4000\n", encoding="utf-8")
    cfg = config_mod.load_config(tmp_path)
    assert cfg.budget.max_tokens == 4000
    assert cfg.embedding.model == config_mod.DEFAULT_EMBEDDING_MODEL
    assert cfg.output.format == "markdown"


def test_supported_providers_contains_only_huggingface() -> None:
    assert SUPPORTED_PROVIDERS == ("huggingface",)


def test_get_embedder_returns_huggingface_embedder() -> None:
    embedder = get_embedder("huggingface", "BAAI/bge-small-en-v1.5")
    assert embedder.model == "BAAI/bge-small-en-v1.5"


def test_get_embedder_unregistered_provider_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        get_embedder("ollama", "bge-m3")


def test_huggingface_embedder_embed_not_yet_implemented() -> None:
    """Phase 5 fills this in; Phase 2 only wires the interface + config validation."""
    embedder = get_embedder("huggingface", "BAAI/bge-small-en-v1.5")
    with pytest.raises(NotImplementedError):
        embedder.embed(["some text"])
