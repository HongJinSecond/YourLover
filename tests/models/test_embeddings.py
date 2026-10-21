"""Embedding 工厂测试：三种提供方构造与禁用行为，全部离线。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def make_config(**overrides):
    values = {
        "EMBEDDING_PROVIDER": "ollama",
        "EMBEDDING_MODEL_NAME": "emb-model",
        "EMBEDDING_API_BASE": "http://localhost:11500",
        "EMBEDDING_OPENAI_MODEL_ID": "text-embedding-x",
        "EMBEDDING_OPENAI_BASE_URL": "https://emb.example.test/v1",
        "EMBEDDING_OPENAI_API_KEY": "emb-key",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_missing_provider_defaults_to_ollama(monkeypatch):
    import agent.models.embeddings as module

    monkeypatch.setattr(module, "_resolve_config", lambda config=None: SimpleNamespace())
    assert module.get_embedding_provider() == "ollama"
    assert module.is_embedding_enabled() is True


def test_none_provider_raises_disabled_error():
    from agent.models.embeddings import EmbeddingDisabledError, create_embeddings

    with pytest.raises(EmbeddingDisabledError):
        create_embeddings(make_config(EMBEDDING_PROVIDER="none"))


def test_ollama_provider_uses_independent_base_url():
    from agent.models.embeddings import create_embeddings

    embeddings = create_embeddings(make_config())
    assert embeddings.model == "emb-model"
    assert embeddings.base_url == "http://localhost:11500"


def test_ollama_provider_requires_model_name():
    from agent.models.embeddings import create_embeddings

    with pytest.raises(ValueError, match="EMBEDDING_MODEL_NAME"):
        create_embeddings(make_config(EMBEDDING_MODEL_NAME=" "))


def test_ollama_provider_defaults_base_url_when_blank():
    from agent.models.embeddings import DEFAULT_OLLAMA_BASE_URL, create_embeddings

    embeddings = create_embeddings(make_config(EMBEDDING_API_BASE=""))
    assert embeddings.base_url == DEFAULT_OLLAMA_BASE_URL


def test_openai_provider_uses_independent_url_and_key():
    from agent.models.embeddings import create_embeddings

    embeddings = create_embeddings(make_config(EMBEDDING_PROVIDER="openai"))
    assert embeddings.model == "text-embedding-x"
    assert embeddings.openai_api_base == "https://emb.example.test/v1"
    assert embeddings.openai_api_key.get_secret_value() == "emb-key"


def test_openai_provider_requires_model_id_and_key():
    from agent.models.embeddings import create_embeddings

    with pytest.raises(ValueError, match="EMBEDDING_OPENAI_MODEL_ID"):
        create_embeddings(make_config(EMBEDDING_PROVIDER="openai", EMBEDDING_OPENAI_MODEL_ID=""))
    with pytest.raises(ValueError, match="Embedding API Key"):
        create_embeddings(make_config(EMBEDDING_PROVIDER="openai", EMBEDDING_OPENAI_API_KEY=""))


def test_openai_provider_rejects_invalid_base_url():
    from agent.models.embeddings import create_embeddings

    with pytest.raises(ValueError):
        create_embeddings(
            make_config(EMBEDDING_PROVIDER="openai", EMBEDDING_OPENAI_BASE_URL="not-a-url")
        )


def test_unknown_provider_raises_value_error():
    from agent.models.embeddings import create_embeddings

    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        create_embeddings(make_config(EMBEDDING_PROVIDER="azure"))


def test_shared_config_instance_is_used_when_none(monkeypatch):
    import agent.models.embeddings as module
    import utils.config_manager as config_module

    sentinel = SimpleNamespace(
        EMBEDDING_PROVIDER="ollama",
        EMBEDDING_MODEL_NAME="shared-model",
        EMBEDDING_API_BASE="http://localhost:11434",
    )
    monkeypatch.setattr(config_module, "config", sentinel)
    embeddings = module.create_embeddings(None)
    assert embeddings.model == "shared-model"
