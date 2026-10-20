"""长期记忆的 Chroma 向量存储。

负责把 LLM 提炼出的长期记忆条目向量化并持久化到 Chroma，
检索时按 user_id 过滤后做向量相似度召回。日期作为元数据随条目存储，
注入 Prompt 时展示，让 LLM 知道"这条回忆是哪一天说的"。

实现参考 agent/tools/RAGTool.py：使用 langchain_chroma 的 Chroma +
OllamaEmbeddings（embedding 模型见 config.json 的 EMBEDDING_MODEL_NAME）。

任何初始化/写入/检索失败都会抛出 VectorStoreError，由上层
LongTermMemory 降级到 JSON + 关键词检索，不中断聊天。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agent.memory.llm_extractor import LLMMemoryEntry


class VectorStoreError(RuntimeError):
    """Chroma 向量存储在初始化、写入或检索失败时抛出。"""


class VectorMemoryStore:
    """Chroma 向量化长期记忆存储。

    每个条目：
        - 文本：LLM 提炼后的记忆内容（作为 embedding 载体）
        - 元数据：user_id / memory_type / created_date / session_id / created_at
    检索时通过 where={"user_id": user_id} 隔离不同用户。
    """

    def __init__(
        self,
        collection_name: str = "long_term_memory",
        persist_directory: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        """初始化 Chroma 集合（不发送 embedding 请求，查询时才调用模型）。

        参数：
            collection_name: Chroma 集合名，默认 long_term_memory（与 RAG_DB 隔离）
            persist_directory: Chroma 持久化目录，默认 config.CHROMA_PATH
            embedding_model: embedding 模型名，默认 config.EMBEDDING_MODEL_NAME
        """
        try:
            from langchain_chroma import Chroma

            from agent.models.OllamaModels import OllamaEmbeddings
            from utils.config_manager import config
        except ImportError as exc:
            raise VectorStoreError(f"Chroma 依赖未安装: {exc}") from exc

        persist_directory = persist_directory or config.CHROMA_PATH
        embedding_model = embedding_model or config.EMBEDDING_MODEL_NAME
        try:
            self.collection = Chroma(
                collection_name=collection_name,
                embedding_function=OllamaEmbeddings(model=embedding_model),
                persist_directory=persist_directory,
            )
        except Exception as exc:
            raise VectorStoreError(f"Chroma 初始化失败: {exc}") from exc

    def add_entries(
        self,
        user_id: str,
        entries: list[LLMMemoryEntry],
        session_id: str | None = None,
    ) -> list[str]:
        """写入一批长期记忆条目，返回生成的文档 ID 列表。

        日期由系统统一盖章（created_date = 当前日期），不依赖模型生成。
        """
        if not entries:
            return []
        created_at = datetime.now()
        created_date = created_at.strftime("%Y-%m-%d")
        documents = [entry.content for entry in entries]
        metadatas: list[dict[str, Any]] = []
        for entry in entries:
            metadata: dict[str, Any] = {
                "user_id": user_id,
                "memory_type": entry.memory_type,
                "created_date": created_date,
                "created_at": created_at.isoformat(),
            }
            if session_id is not None:
                metadata["session_id"] = session_id
            metadatas.append(metadata)
        try:
            return self.collection.add_texts(documents, metadatas=metadatas)
        except Exception as exc:
            raise VectorStoreError(f"Chroma 写入失败: {exc}") from exc

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """按 query 做向量召回，返回 top_k 条记忆（含日期/类型元数据）。

        返回项：{"content", "memory_type", "created_date", "score"}。
        """
        if top_k < 1 or not query.strip():
            return []
        try:
            results = self.collection.similarity_search_with_score(
                query=query,
                k=top_k,
                filter={"user_id": user_id},
            )
        except Exception as exc:
            raise VectorStoreError(f"Chroma 检索失败: {exc}") from exc

        entries: list[dict[str, Any]] = []
        for document, score in results:
            metadata = document.metadata or {}
            if metadata.get("user_id") != user_id:
                continue
            entries.append(
                {
                    "content": document.page_content,
                    "memory_type": metadata.get("memory_type", ""),
                    "created_date": metadata.get("created_date", ""),
                    "score": float(score),
                }
            )
        return entries

    @staticmethod
    def format_context(entries: list[dict[str, Any]]) -> str:
        """把检索结果格式化为可注入 system prompt 的字符串。

        每条格式：- [2026-09-01] 记忆内容（带日期，让 LLM 知道是哪天说的）。
        按记忆类型归入小节（semantic / episodic / procedural）。
        """
        sections: list[str] = []
        for memory_type, label in (
            ("semantic", "Semantic memories:"),
            ("episodic", "Episodic memories:"),
            ("procedural", "Procedural memories:"),
        ):
            lines = []
            for entry in entries:
                if entry.get("memory_type") != memory_type:
                    continue
                date = entry.get("created_date") or "未知日期"
                lines.append(f"- [{date}] {entry.get('content', '')}")
            if lines:
                sections.append(label + "\n" + "\n".join(lines))
        return "\n\n".join(sections)


__all__ = ["VectorMemoryStore", "VectorStoreError"]
