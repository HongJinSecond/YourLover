"""短期记忆管理器：Session 存取与短期窗口裁剪的统一入口。

短期记忆负责"同一个 Session 内的连续性"：
    - 会话（Session）的创建、查询、重命名、删除；
    - 会话原始消息（MemoryItem）的持久化与恢复；
    - 会话摘要的更新；
    - 短期窗口裁剪（超限时按完整轮次保留最近消息）。

内部组合 SessionStore（SQLite）与 window.py 的裁剪函数；
调用方（ChatAgent / ChatWindow / 测试）只需面向本类，不直接触碰底层文件。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from agent.memory.MemoryItem import MemoryItem
from agent.memory.session_store import (
    SessionRecord,
    SessionSnapshot,
    SessionStore,
    SessionStoreError,
)
from agent.memory.window import (
    Message,
    MessageList,
    select_messages_for_memory_write,
    trim_messages,
)


class ShortTermMemory:
    """短期记忆管理类：对外暴露短期记忆的全部能力。

    职责边界：
        1. 管理 SessionStore 的生命周期并转发会话 CRUD；
        2. 提供按"轮"保存与恢复 MemoryItem 的便捷方法；
        3. 封装短期窗口裁剪（trim_window / select_memory_writes）。

    不负责：
        - 摘要文本的生成策略（由 ChatAgent 决定怎么拼摘要）；
        - 长期记忆候选提取（见 LongTermMemory）；
        - 用户画像（见 UserProfileManager）。
    """

    def __init__(
        self,
        session_store: SessionStore | None = None,
        database_path: str | Path | None = None,
        *,
        max_messages: int = 80,
        keep_recent_turns: int = 20,
        timeout: float = 5.0,
    ) -> None:
        """初始化短期记忆管理器。

        参数：
            session_store: 已构造的 SessionStore；传入时忽略 database_path/timeout。
            database_path: SQLite 数据库路径，默认 data/memory/sessions.sqlite3。
            max_messages: 短期窗口消息上限（裁剪默认值）。
            keep_recent_turns: 裁剪时默认保留的最近完整轮次数。
            timeout: SQLite 锁等待秒数。
        """
        if session_store is not None:
            self.session_store = session_store
        else:
            self.session_store = SessionStore(database_path, timeout=timeout)
        self.max_messages = max_messages  # 短期窗口消息上限（默认裁剪参数）
        self.keep_recent_turns = keep_recent_turns  # 默认保留的最近轮次数

    # ------------------------------------------------------------------
    # 会话管理（Session CRUD）
    # ------------------------------------------------------------------

    def create_session(
        self,
        user_id: str,
        session_id: str,
        title: str = "",
    ) -> SessionRecord:
        """创建 Session；已存在时保留原数据并返回现有记录。"""
        return self.session_store.create_session(user_id, session_id, title)

    def get_session(self, user_id: str, session_id: str) -> SessionRecord | None:
        """读取一个 Session；不存在时返回 None。"""
        return self.session_store.get_session(user_id, session_id)

    def list_sessions(self, user_id: str) -> list[SessionRecord]:
        """列出用户的全部 Session，按最近更新时间从新到旧排序。"""
        return self.session_store.list_sessions(user_id)

    def rename_session(self, user_id: str, session_id: str, title: str) -> bool:
        """更新 Session 展示标题；Session 不存在时返回 False。"""
        return self.session_store.update_session_title(user_id, session_id, title)

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """删除 Session 及其全部消息（外键级联删除）。"""
        return self.session_store.delete_session(user_id, session_id)

    # ------------------------------------------------------------------
    # 短期记忆读写（MemoryItem 持久化 / 恢复 / 摘要）
    # ------------------------------------------------------------------

    def load_session(
        self,
        user_id: str,
        session_id: str,
        *,
        recent_limit: int | None = None,
    ) -> SessionSnapshot | None:
        """加载 Session 快照：返回会话摘要 + 最近（或全部）MemoryItem。

        用于会话切换或应用重启后恢复短期状态；Session 不存在时返回 None。
        """
        return self.session_store.load_session(
            user_id,
            session_id,
            recent_limit=recent_limit if recent_limit is not None else self.max_messages,
        )

    def load_messages(
        self,
        user_id: str,
        session_id: str,
        *,
        limit: int | None = None,
    ) -> list[MemoryItem]:
        """读取 Session 的 MemoryItem 列表（按会话顺序）。"""
        return self.session_store.load_memory_items(user_id, session_id, limit=limit)

    def load_item_records(
        self,
        user_id: str,
        session_id: str,
        *,
        limit: int | None = None,
    ) -> list:
        """读取带事件 ID 和序号的完整记录（StoredMemoryItem 列表）。

        供需要 sequence_no / event_id 的调用方使用（如摘要进度同步）。
        """
        return self.session_store.load_memory_item_records(user_id, session_id, limit=limit)

    def save_turn(
        self,
        user_id: str,
        session_id: str,
        items: Sequence[MemoryItem],
        *,
        event_ids: Sequence[str | None] | None = None,
    ) -> list[str]:
        """保存一轮对话产生的 MemoryItem（用户/AI/工具事件）。

        同一 event_id 重复保存时为更新而非重复插入；返回事件 ID 列表。
        """
        return self.session_store.save_memory_items(
            user_id,
            session_id,
            items,
            event_ids=event_ids,
        )

    def update_summary(
        self,
        user_id: str,
        session_id: str,
        summary: str,
        summarized_through_sequence: int,
    ) -> bool:
        """更新会话摘要和已摘要进度；Session 不存在时返回 False。"""
        return self.session_store.update_session_summary(
            user_id,
            session_id,
            summary,
            summarized_through_sequence,
        )

    def get_last_user_activity_time(self, user_id: str) -> float | None:
        """查询用户跨全部会话的最后一次发言时间戳；从未发言返回 None。

        供时间感知（system prompt 注入"上次用户回复时间"）和
        主动对话引擎（静默计时起点）使用。
        """
        return self.session_store.get_last_user_activity_time(user_id)

    # ------------------------------------------------------------------
    # 短期窗口裁剪
    # ------------------------------------------------------------------

    def trim_window(
        self,
        messages: Sequence[Message],
        max_messages: int | None = None,
        keep_recent_turns: int | None = None,
    ) -> tuple[MessageList, MessageList]:
        """裁剪消息列表，返回 (保留消息, 被裁剪消息)。

        未指定参数时使用构造时配置的默认值。裁剪规则见 window.trim_messages：
        固定保留头部 SystemMessage，按完整轮次保留最近若干轮，
        保证工具调用结果与其发起消息不被拆散。
        """
        return trim_messages(
            messages,
            self.max_messages if max_messages is None else max_messages,
            self.keep_recent_turns if keep_recent_turns is None else keep_recent_turns,
        )

    def select_memory_writes(self, removed_messages: Sequence[Message]) -> MessageList:
        """从被裁剪的消息中筛选可喂给长期记忆分类器的消息。

        排除 SystemMessage 与 RemoveMessage（裁剪标记本身无内容）。
        """
        return select_messages_for_memory_write(removed_messages)

    # ------------------------------------------------------------------
    # 便捷属性：透传底层存储信息（供测试与调试）
    # ------------------------------------------------------------------

    @property
    def database_path(self) -> Path:
        """短期记忆 SQLite 数据库文件路径。"""
        return self.session_store.database_path

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return (
            f"ShortTermMemory(database_path={str(self.database_path)!r}, "
            f"max_messages={self.max_messages}, "
            f"keep_recent_turns={self.keep_recent_turns})"
        )


# 便于外部按名导入的兼容别名（避免到处写 try/except 导入）
__all__ = [
    "ShortTermMemory",
    "SessionRecord",
    "SessionSnapshot",
    "SessionStoreError",
    "MemoryItem",
]
