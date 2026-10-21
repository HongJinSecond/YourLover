"""Embedding 提供方工厂：本地 Ollama / OpenAI 兼容接口 / 不启用。

Embedding 与聊天模型解耦：`EMBEDDING_PROVIDER` 独立选择向量化服务，
不随 `LLM_PROVIDER` 联动。可选值：

    - "ollama": 本地 Ollama（模型 EMBEDDING_MODEL_NAME，地址 EMBEDDING_API_BASE，
      留空默认 http://localhost:11434；与聊天服务的 api_base 相互独立）
    - "openai": OpenAI 兼容接口（模型 EMBEDDING_OPENAI_MODEL_ID，
      地址 EMBEDDING_OPENAI_BASE_URL，密钥 EMBEDDING_OPENAI_API_KEY；
      与聊天服务的 OPENAI_* 配置相互独立）
    - "none":   不启用 embedding。长期记忆走 JSON+关键词检索（Plan B），
      RAG 知识库整体禁用。

旧配置缺少 EMBEDDING_PROVIDER 时默认 "ollama"，保持历史行为。
构造 Embeddings 实例不发送网络请求；首次向量化时才连接服务。
"""

from __future__ import annotations

from typing import Any

EMBEDDING_PROVIDER_OLLAMA = "ollama"
EMBEDDING_PROVIDER_OPENAI = "openai"
EMBEDDING_PROVIDER_NONE = "none"

EMBEDDING_PROVIDERS = (
    EMBEDDING_PROVIDER_OLLAMA,
    EMBEDDING_PROVIDER_OPENAI,
    EMBEDDING_PROVIDER_NONE,
)

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class EmbeddingDisabledError(RuntimeError):
    """EMBEDDING_PROVIDER 为 none 时抛出：当前配置不使用任何 embedding 服务。"""


def _resolve_config(config: Any | None) -> Any:
    """返回显式传入的配置；未传入时使用共享 config 实例。"""
    if config is not None:
        return config
    from utils.config_manager import config as shared

    return shared


def get_embedding_provider(config: Any | None = None) -> str:
    """读取 embedding 提供方；旧配置缺少该键时默认 ollama（保持原行为）。"""
    settings = _resolve_config(config)
    provider = str(getattr(settings, "EMBEDDING_PROVIDER", "") or "").strip().lower()
    if not provider:
        return EMBEDDING_PROVIDER_OLLAMA
    return provider


def is_embedding_enabled(config: Any | None = None) -> bool:
    """embedding 是否启用（provider 不为 none）。"""
    return get_embedding_provider(config) != EMBEDDING_PROVIDER_NONE


def create_embeddings(config: Any | None = None) -> Any:
    """按 EMBEDDING_PROVIDER 构造 LangChain Embeddings 实例。

    参数：
        config: 配置对象；None 时使用共享 config。

    返回：
        OllamaEmbeddings 或 OpenAIEmbeddings 实例（构造时不联网）。

    异常：
        EmbeddingDisabledError: provider 为 none，调用方应整体禁用向量功能。
        ValueError: provider 未知或必填配置缺失。
    """
    settings = _resolve_config(config)
    provider = get_embedding_provider(settings)

    if provider == EMBEDDING_PROVIDER_NONE:
        raise EmbeddingDisabledError(
            "Embedding 未启用（EMBEDDING_PROVIDER=none）：向量记忆与 RAG 已禁用，"
            "长期记忆使用 JSON+关键词检索。"
        )

    if provider == EMBEDDING_PROVIDER_OLLAMA:
        model = str(getattr(settings, "EMBEDDING_MODEL_NAME", "") or "").strip()
        if not model:
            raise ValueError("请在 config.json 中填写 EMBEDDING_MODEL_NAME（Ollama embedding 模型名）")
        base_url = (
            str(getattr(settings, "EMBEDDING_API_BASE", "") or "").strip().rstrip("/")
            or DEFAULT_OLLAMA_BASE_URL
        )
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=model, base_url=base_url)

    if provider == EMBEDDING_PROVIDER_OPENAI:
        model = str(getattr(settings, "EMBEDDING_OPENAI_MODEL_ID", "") or "").strip()
        base_url = str(getattr(settings, "EMBEDDING_OPENAI_BASE_URL", "") or "").strip()
        api_key = str(getattr(settings, "EMBEDDING_OPENAI_API_KEY", "") or "").strip()
        if not model:
            raise ValueError("请在 config.json 中填写 EMBEDDING_OPENAI_MODEL_ID（Embedding 模型 ID）")
        if not api_key:
            raise ValueError("请填写 Embedding API Key（无鉴权的本地兼容服务可填写任意非空值）")
        # Embedding 使用独立的地址与密钥，不读取聊天服务的 OPENAI_* 配置。
        from agent.models.OpenAIModels import normalize_base_url

        base_url = normalize_base_url(base_url)
        # 兼容用户直接粘贴完整接口地址。
        if base_url.endswith("/embeddings"):
            base_url = base_url[: -len("/embeddings")]
        from langchain_openai import OpenAIEmbeddings

        # check_embedding_ctx_length=False：第三方兼容服务不做本地 tiktoken
        # 切分，直接发送原文，由服务端处理长度限制。
        return OpenAIEmbeddings(
            model=model,
            base_url=base_url,
            api_key=api_key,
            check_embedding_ctx_length=False,
            timeout=60.0,
            max_retries=1,
        )

    raise ValueError(
        f"不支持的 EMBEDDING_PROVIDER：{provider}（可选 {'/'.join(EMBEDDING_PROVIDERS)}）"
    )


__all__ = [
    "DEFAULT_OLLAMA_BASE_URL",
    "EMBEDDING_PROVIDERS",
    "EMBEDDING_PROVIDER_NONE",
    "EMBEDDING_PROVIDER_OLLAMA",
    "EMBEDDING_PROVIDER_OPENAI",
    "EmbeddingDisabledError",
    "create_embeddings",
    "get_embedding_provider",
    "is_embedding_enabled",
]
