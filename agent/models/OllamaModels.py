"""Ollama chat configuration and embedding provider shared by memory and RAG."""

from typing import Any

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings as OllamaEmbeddings

from utils.config_manager import config

# Ollama 默认 num_ctx=4096：本项目 system prompt 约 3400 token，留给生成的
# 空间极小；qwen3 系列默认开启思考模式，几百 token 的 <think> 一旦撞顶
# （done_reason=length）正文为空，表现为"空回复"。因此默认调大窗口并关闭
# 思考，两者均可在 config.json 覆盖（OLLAMA_NUM_CTX / OLLAMA_THINKING）。
_DEFAULT_NUM_CTX = 8192


def get_ollama_llm(model_name: str, base_url: str | None = None) -> ChatOllama:
    """构造聊天 LLM：扩大上下文窗口、默认关闭 qwen3 思考模式。"""
    kwargs: dict[str, Any] = {
        "model": model_name,
        "num_ctx": int(getattr(config, "OLLAMA_NUM_CTX", _DEFAULT_NUM_CTX)),
        # think=false 对不支持思考的模型（如 qwen2.5）是无害 no-op；
        # 如需让 qwen3 思考，把 config.json 的 OLLAMA_THINKING 改为 true
        # （同时建议调大 OLLAMA_NUM_CTX，否则思考 token 容易撞顶截断）。
        "reasoning": bool(getattr(config, "OLLAMA_THINKING", False)),
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOllama(**kwargs)
