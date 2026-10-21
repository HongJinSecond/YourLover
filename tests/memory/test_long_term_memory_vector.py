"""LongTermMemory 向量主路径 + JSON 降级路径的单元测试。"""

from __future__ import annotations

import pytest

from agent.memory.llm_extractor import LLMMemoryEntry
from agent.memory.long_term_memory import LongTermMemory
from agent.memory.store import FileMemoryStore
from agent.memory.vector_memory_store import VectorMemoryStore, VectorStoreError


def test_vector_store_uses_current_json_configuration(tmp_path, monkeypatch):
    import json
    from unittest.mock import Mock

    import langchain_chroma
    import langchain_ollama

    import utils.config_manager as config_module

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "CHROMA_PATH": str(tmp_path / "custom-chroma"),
                "EMBEDDING_MODEL_NAME": "custom-embedding-model",
                "EMBEDDING_API_BASE": "http://localhost:11500",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "config", config_module.ConfigManager(path))
    chroma = Mock()
    embeddings = Mock(return_value=object())
    monkeypatch.setattr(langchain_chroma, "Chroma", chroma)
    monkeypatch.setattr(langchain_ollama, "OllamaEmbeddings", embeddings)
    VectorMemoryStore()
    embeddings.assert_called_once_with(
        model="custom-embedding-model", base_url="http://localhost:11500"
    )
    assert chroma.call_args.kwargs["persist_directory"] == str(tmp_path / "custom-chroma")


def test_vector_store_raises_when_embedding_disabled(tmp_path, monkeypatch):
    import json

    import utils.config_manager as config_module

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"CHROMA_PATH": str(tmp_path / "custom-chroma"), "EMBEDDING_PROVIDER": "none"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "config", config_module.ConfigManager(path))
    with pytest.raises(VectorStoreError, match="Embedding"):
        VectorMemoryStore()


class BrokenVectorStore:
    """测试桩：模拟 Chroma 不可用（初始化或写入失败）。"""

    def __init__(self, fail_on_write: bool = True) -> None:
        self.fail_on_write = fail_on_write
        self.writes: list = []

    def add_entries(self, user_id, entries, session_id=None):
        if self.fail_on_write:
            raise VectorStoreError("Chroma 写入失败")
        self.writes.append((user_id, entries, session_id))
        return ["vec-1"]

    def search(self, user_id, query, top_k=5):
        raise VectorStoreError("Chroma 检索失败")


def test_write_llm_memories_without_vector_store_falls_back_to_json(tmp_path):
    memory = LongTermMemory(store=FileMemoryStore(tmp_path / "users"))
    ids = memory.write_llm_memories("user-a", [LLMMemoryEntry("semantic", "用户偏好：喜欢喝咖啡")])

    assert len(ids) == 1
    stored = memory.list("user-a")
    assert len(stored) == 1
    assert "喝咖啡" in stored[0].content
    assert "semantic" in stored[0].tags


def test_write_llm_memories_with_empty_list_writes_nothing(tmp_path):
    memory = LongTermMemory(store=FileMemoryStore(tmp_path / "users"))
    assert memory.write_llm_memories("user-a", []) == []
    assert memory.list("user-a") == []


def test_vector_write_failure_falls_back_to_json(tmp_path):
    memory = LongTermMemory(
        store=FileMemoryStore(tmp_path / "users"),
        vector_store=BrokenVectorStore(fail_on_write=True),
    )
    ids = memory.write_llm_memories("user-a", [LLMMemoryEntry("semantic", "用户偏好：喜欢喝咖啡")])

    assert len(ids) == 1
    stored = memory.list("user-a")
    assert any("喝咖啡" in item.content for item in stored)


def test_retrieve_context_falls_back_to_keyword_when_vector_search_fails(tmp_path):
    memory = LongTermMemory(
        store=FileMemoryStore(tmp_path / "users"),
        vector_store=BrokenVectorStore(fail_on_write=True),
    )
    # 预置 JSON 记忆（向量检索失败时降级关键词应能召回）
    from agent.memory.schema import SemanticMemory

    memory.store.add_memory(
        "user-a",
        SemanticMemory(user_id="user-a", content="用户偏好：喜欢喝咖啡"),
    )

    context = memory.retrieve_context("user-a", "喝咖啡")
    assert "喝咖啡" in context


def test_format_context_groups_by_type_with_dates():
    entries = [
        {
            "content": "用户偏好：喜欢喝咖啡",
            "memory_type": "semantic",
            "created_date": "2026-09-01",
        },
        {"content": "用户因考试压力大", "memory_type": "episodic", "created_date": "2026-09-01"},
        {"content": "以后先安抚情绪", "memory_type": "procedural", "created_date": "2026-09-02"},
    ]
    context = VectorMemoryStore.format_context(entries)

    assert "[2026-09-01] 用户偏好：喜欢喝咖啡" in context
    assert "[2026-09-02] 以后先安抚情绪" in context
    assert "Semantic memories:" in context
    assert "Episodic memories:" in context
    assert "Procedural memories:" in context
