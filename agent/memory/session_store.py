"""基于 SQLite 的短期会话记忆存储。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from langchain_core.messages import message_to_dict, messages_from_dict

from agent.memory.MemoryItem import MemoryItem


def _utc_now_iso() -> str:
    """生成当前 UTC 时间，供 session 创建和更新时间使用。"""
    return datetime.now(timezone.utc).isoformat()


class SessionStoreError(RuntimeError):
    """Session 存储异常：用于统一包装数据库和消息反序列化错误。"""


@dataclass(frozen=True)
class SessionRecord:
    """Session 基本记录：保存会话归属、摘要和摘要处理进度。"""

    user_id: str  # Session 所属用户
    session_id: str  # Session 唯一标识
    title: str  # Session 展示标题
    summary: str  # 已压缩的短期会话摘要
    summarized_through_sequence: int  # 摘要已处理到的消息序号
    created_at: str  # Session 创建时间
    updated_at: str  # Session 最近更新时间


@dataclass(frozen=True)
class StoredMemoryItem:
    """已存储的 MemoryItem：补充事件 ID、Session 归属和消息顺序。"""

    event_id: str  # 数据库中的事件唯一标识
    user_id: str  # 事件所属用户
    session_id: str  # 事件所属 Session
    sequence_no: int  # 事件在 Session 内的顺序
    item: MemoryItem  # 原始短期记忆事件


@dataclass(frozen=True)
class SessionSnapshot:
    """Session 快照：一次返回 Session 信息和按顺序排列的 MemoryItem。"""

    session: SessionRecord  # Session 基本信息和摘要
    items: tuple[StoredMemoryItem, ...]  # 本次加载到的短期记忆事件


class SessionStore:
    """Session 存储类：负责短期会话和 MemoryItem 的 SQLite 持久化。

    本类只处理 Session 的创建、查询、摘要更新和 MemoryItem 存取，
    不负责上下文裁剪、摘要生成或长期记忆提取。
    """

    def __init__(
        self,
        database_path: str | Path | None = None,
        timeout: float = 5.0,
    ) -> None:
        """初始化存储：确定数据库路径并创建所需数据表。"""
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        if database_path is None:
            database_path = (
                Path(__file__).resolve().parents[2] / "data" / "memory" / "sessions.sqlite3"
            )

        self.database_path = Path(database_path)  # SQLite 数据库文件路径
        self.timeout = float(timeout)  # 数据库锁等待时间，单位为秒
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def create_session(
        self,
        user_id: str,
        session_id: str,
        title: str = "",
    ) -> SessionRecord:
        """创建 Session；已存在时保留原数据并直接返回现有记录。"""
        self._require_id(user_id, "user_id")
        self._require_id(session_id, "session_id")
        now = _utc_now_iso()

        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO chat_sessions (
                        user_id, session_id, title, summary,
                        summarized_through_sequence, created_at, updated_at
                    ) VALUES (?, ?, ?, '', 0, ?, ?)
                    """,
                    (user_id, session_id, title, now, now),
                )
        except sqlite3.DatabaseError as exc:
            raise SessionStoreError("Failed to create session") from exc

        session = self.get_session(user_id, session_id)
        if session is None:  # 正常写入后必须能查询到，防止数据库状态异常
            raise SessionStoreError("Session was not created")
        return session

    def get_session(self, user_id: str, session_id: str) -> SessionRecord | None:
        """读取一个 Session；不存在时返回 None。"""
        self._require_id(user_id, "user_id")
        self._require_id(session_id, "session_id")

        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT user_id, session_id, title, summary,
                           summarized_through_sequence, created_at, updated_at
                    FROM chat_sessions
                    WHERE user_id = ? AND session_id = ?
                    """,
                    (user_id, session_id),
                ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise SessionStoreError("Failed to load session") from exc
        return self._session_from_row(row) if row is not None else None

    def list_sessions(self, user_id: str) -> list[SessionRecord]:
        """列出用户的全部 Session，按最近更新时间从新到旧排序。"""
        self._require_id(user_id, "user_id")

        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT user_id, session_id, title, summary,
                           summarized_through_sequence, created_at, updated_at
                    FROM chat_sessions
                    WHERE user_id = ?
                    ORDER BY updated_at DESC
                    """,
                    (user_id,),
                ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise SessionStoreError("Failed to list sessions") from exc
        return [self._session_from_row(row) for row in rows]

    def save_memory_item(
        self,
        user_id: str,
        session_id: str,
        item: MemoryItem,
        *,
        event_id: str | None = None,
        sequence_no: int | None = None,
    ) -> str:
        """保存一条 MemoryItem；相同 event_id 再次保存时更新原记录。"""
        event_ids = self.save_memory_items(
            user_id,
            session_id,
            [item],
            event_ids=[event_id] if event_id is not None else None,
            start_sequence=sequence_no,
        )
        return event_ids[0]

    def save_memory_items(
        self,
        user_id: str,
        session_id: str,
        items: Sequence[MemoryItem],
        *,
        event_ids: Sequence[str | None] | None = None,
        start_sequence: int | None = None,
    ) -> list[str]:
        """批量保存 MemoryItem；同一批次使用一个事务并保持原始顺序。"""
        self._require_id(user_id, "user_id")
        self._require_id(session_id, "session_id")
        if event_ids is not None and len(event_ids) != len(items):
            raise ValueError("event_ids length must match items length")
        if start_sequence is not None and start_sequence < 1:
            raise ValueError("start_sequence must be greater than or equal to 1")
        if not items:
            return []
        if any(not isinstance(item, MemoryItem) for item in items):
            raise TypeError("items must contain only MemoryItem instances")

        generated_ids = [
            self._event_id(item, event_ids[index] if event_ids else None)
            for index, item in enumerate(items)
        ]
        if len(set(generated_ids)) != len(generated_ids):
            raise ValueError("event_ids must be unique within one batch")
        now = _utc_now_iso()

        try:
            with self._connect() as connection:
                self._ensure_session(connection, user_id, session_id, now)
                next_sequence = (
                    start_sequence
                    if start_sequence is not None
                    else self._next_sequence(connection, user_id, session_id)
                )
                for item, item_event_id in zip(items, generated_ids, strict=True):
                    inserted = self._save_item(
                        connection,
                        user_id,
                        session_id,
                        item_event_id,
                        next_sequence,
                        item,
                    )
                    if inserted:
                        next_sequence += 1
                connection.execute(
                    """
                    UPDATE chat_sessions
                    SET updated_at = ?
                    WHERE user_id = ? AND session_id = ?
                    """,
                    (now, user_id, session_id),
                )
        except (sqlite3.DatabaseError, TypeError, ValueError) as exc:
            if isinstance(exc, (TypeError, ValueError)):
                raise
            raise SessionStoreError("Failed to save MemoryItem") from exc
        return generated_ids

    def load_memory_items(
        self,
        user_id: str,
        session_id: str,
        *,
        limit: int | None = None,
        after_sequence: int | None = None,
    ) -> list[MemoryItem]:
        """读取 MemoryItem；可按消息序号过滤，并可只取最近若干条。"""
        return [
            record.item
            for record in self.load_memory_item_records(
                user_id,
                session_id,
                limit=limit,
                after_sequence=after_sequence,
            )
        ]

    def get_last_user_activity_time(self, user_id: str) -> float | None:
        """查询某用户跨全部会话的最后一次发言时间戳（speaker='User'）。

        供时间感知与主动对话机制使用：
            - 注入 system prompt 的"上次用户回复时间"（跨会话取最近一条）；
            - ProactiveEngine 启动时初始化静默计时的起点。
        用户从未发言时返回 None。
        """
        self._require_id(user_id, "user_id")
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT MAX(speak_time) AS last_time
                    FROM session_memory_items
                    WHERE user_id = ? AND speaker = 'User'
                    """,
                    (user_id,),
                ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise SessionStoreError("Failed to query last user activity time") from exc
        value = row["last_time"] if row is not None else None
        return float(value) if value is not None else None

    def load_memory_item_records(
        self,
        user_id: str,
        session_id: str,
        *,
        limit: int | None = None,
        after_sequence: int | None = None,
    ) -> list[StoredMemoryItem]:
        """读取带事件 ID 和顺序信息的 MemoryItem 记录，并按会话顺序返回。"""
        self._require_id(user_id, "user_id")
        self._require_id(session_id, "session_id")
        if limit is not None and limit < 1:
            raise ValueError("limit must be greater than or equal to 1")
        if after_sequence is not None and after_sequence < 0:
            raise ValueError("after_sequence must be greater than or equal to 0")

        conditions = ["user_id = ?", "session_id = ?"]
        parameters: list[Any] = [user_id, session_id]
        if after_sequence is not None:
            conditions.append("sequence_no > ?")
            parameters.append(after_sequence)

        # 最近 N 条需先倒序查询，再恢复成会话中的正序。
        order = "DESC" if limit is not None else "ASC"
        limit_clause = " LIMIT ?" if limit is not None else ""
        if limit is not None:
            parameters.append(limit)

        query = f"""
            SELECT event_id, user_id, session_id, sequence_no,
                   item_type, speaker, message_json, love,
                   ai_emo_json, user_emo_json, speak_time, importance
            FROM session_memory_items
            WHERE {" AND ".join(conditions)}
            ORDER BY sequence_no {order}{limit_clause}
        """

        try:
            with self._connect() as connection:
                rows = connection.execute(query, parameters).fetchall()
            records = [self._memory_item_from_row(row) for row in rows]
        except (
            sqlite3.DatabaseError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise SessionStoreError("Failed to load MemoryItem") from exc
        return list(reversed(records)) if limit is not None else records

    def load_session(
        self,
        user_id: str,
        session_id: str,
        *,
        recent_limit: int | None = None,
    ) -> SessionSnapshot | None:
        """加载 Session 快照；返回摘要以及最近或全部 MemoryItem。"""
        session = self.get_session(user_id, session_id)
        if session is None:
            return None
        records = self.load_memory_item_records(
            user_id,
            session_id,
            limit=recent_limit,
        )
        return SessionSnapshot(session=session, items=tuple(records))

    def update_session_summary(
        self,
        user_id: str,
        session_id: str,
        summary: str,
        summarized_through_sequence: int,
    ) -> bool:
        """更新会话摘要和处理进度；Session 不存在时返回 False。"""
        self._require_id(user_id, "user_id")
        self._require_id(session_id, "session_id")
        if summarized_through_sequence < 0:
            raise ValueError("summarized_through_sequence must be greater than or equal to 0")

        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    UPDATE chat_sessions
                    SET summary = ?, summarized_through_sequence = ?, updated_at = ?
                    WHERE user_id = ? AND session_id = ?
                    """,
                    (
                        summary,
                        summarized_through_sequence,
                        _utc_now_iso(),
                        user_id,
                        session_id,
                    ),
                )
                return cursor.rowcount > 0
        except sqlite3.DatabaseError as exc:
            raise SessionStoreError("Failed to update session summary") from exc

    def update_session_title(
        self,
        user_id: str,
        session_id: str,
        title: str,
    ) -> bool:
        """更新 Session 展示标题；Session 不存在时返回 False。"""
        self._require_id(user_id, "user_id")
        self._require_id(session_id, "session_id")
        if not isinstance(title, str):
            raise TypeError("title must be a string")

        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    UPDATE chat_sessions
                    SET title = ?, updated_at = ?
                    WHERE user_id = ? AND session_id = ?
                    """,
                    (title, _utc_now_iso(), user_id, session_id),
                )
                return cursor.rowcount > 0
        except sqlite3.DatabaseError as exc:
            raise SessionStoreError("Failed to update session title") from exc

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """删除一个 Session；关联的 MemoryItem 由外键级联删除。"""
        self._require_id(user_id, "user_id")
        self._require_id(session_id, "session_id")

        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    DELETE FROM chat_sessions
                    WHERE user_id = ? AND session_id = ?
                    """,
                    (user_id, session_id),
                )
                return cursor.rowcount > 0
        except sqlite3.DatabaseError as exc:
            raise SessionStoreError("Failed to delete session") from exc

    def _initialize_database(self) -> None:
        """初始化数据库：启用 WAL，并创建 Session 与 MemoryItem 数据表。"""
        try:
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        title TEXT NOT NULL DEFAULT '',
                        summary TEXT NOT NULL DEFAULT '',
                        summarized_through_sequence INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (user_id, session_id)
                    );

                    CREATE TABLE IF NOT EXISTS session_memory_items (
                        event_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        sequence_no INTEGER NOT NULL,
                        item_type TEXT NOT NULL,
                        speaker TEXT NOT NULL,
                        message_json TEXT NOT NULL,
                        love INTEGER NOT NULL DEFAULT 0,
                        ai_emo_json TEXT,
                        user_emo_json TEXT,
                        speak_time REAL NOT NULL,
                        importance REAL NOT NULL DEFAULT 0.5,
                        UNIQUE (user_id, session_id, sequence_no),
                        FOREIGN KEY (user_id, session_id)
                            REFERENCES chat_sessions (user_id, session_id)
                            ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_session_memory_items_order
                    ON session_memory_items (user_id, session_id, sequence_no);
                    """
                )
                connection.execute("PRAGMA user_version = 1")
        except sqlite3.DatabaseError as exc:
            raise SessionStoreError("Failed to initialize session database") from exc

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """创建独立 SQLite 连接，避免多个 AgentWorker 跨线程共享连接。"""
        connection = sqlite3.connect(self.database_path, timeout=self.timeout)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _ensure_session(
        self,
        connection: sqlite3.Connection,
        user_id: str,
        session_id: str,
        now: str,
    ) -> None:
        """确保目标 Session 存在；首次保存消息时允许自动创建 Session。"""
        connection.execute(
            """
            INSERT OR IGNORE INTO chat_sessions (
                user_id, session_id, title, summary,
                summarized_through_sequence, created_at, updated_at
            ) VALUES (?, ?, '', '', 0, ?, ?)
            """,
            (user_id, session_id, now, now),
        )

    def _save_item(
        self,
        connection: sqlite3.Connection,
        user_id: str,
        session_id: str,
        event_id: str,
        requested_sequence: int,
        item: MemoryItem,
    ) -> bool:
        """在当前事务中新增或更新 MemoryItem，并防止 event_id 跨 Session 冲突。"""
        existing = connection.execute(
            """
            SELECT user_id, session_id, sequence_no
            FROM session_memory_items
            WHERE event_id = ?
            """,
            (event_id,),
        ).fetchone()
        if existing is not None and (
            existing["user_id"] != user_id or existing["session_id"] != session_id
        ):
            raise ValueError("event_id already belongs to another session")

        sequence_no = existing["sequence_no"] if existing is not None else requested_sequence
        message_json = json.dumps(
            message_to_dict(item.message),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        ai_emo_json = self._optional_json(item.ai_emo)
        user_emo_json = self._optional_json(item.user_emo)

        connection.execute(
            """
            INSERT INTO session_memory_items (
                event_id, user_id, session_id, sequence_no,
                item_type, speaker, message_json, love,
                ai_emo_json, user_emo_json, speak_time, importance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_id) DO UPDATE SET
                item_type = excluded.item_type,
                speaker = excluded.speaker,
                message_json = excluded.message_json,
                love = excluded.love,
                ai_emo_json = excluded.ai_emo_json,
                user_emo_json = excluded.user_emo_json,
                speak_time = excluded.speak_time,
                importance = excluded.importance
            """,
            (
                event_id,
                user_id,
                session_id,
                sequence_no,
                item.type,
                item.speaker,
                message_json,
                item.love,
                ai_emo_json,
                user_emo_json,
                float(item.speak_time),
                float(item.importance),
            ),
        )
        return existing is None

    def _next_sequence(
        self,
        connection: sqlite3.Connection,
        user_id: str,
        session_id: str,
    ) -> int:
        """计算 Session 的下一个消息序号，保证新增事件顺序连续。"""
        row = connection.execute(
            """
            SELECT COALESCE(MAX(sequence_no), 0) + 1 AS next_sequence
            FROM session_memory_items
            WHERE user_id = ? AND session_id = ?
            """,
            (user_id, session_id),
        ).fetchone()
        return int(row["next_sequence"])

    def _event_id(self, item: MemoryItem, event_id: str | None) -> str:
        """优先使用调用方或 LangChain 消息 ID，否则生成新的事件 ID。"""
        candidate = event_id or getattr(item.message, "id", None) or uuid4().hex
        candidate = str(candidate).strip()
        if not candidate:
            raise ValueError("event_id must not be empty")
        return candidate

    def _memory_item_from_row(self, row: sqlite3.Row) -> StoredMemoryItem:
        """把数据库记录恢复成 StoredMemoryItem 和 LangChain 消息对象。"""
        message_payload = json.loads(row["message_json"])
        messages = messages_from_dict([message_payload])
        if len(messages) != 1:
            raise ValueError("message_json must contain exactly one message")
        item = MemoryItem(
            type=row["item_type"],
            speaker=row["speaker"],
            message=messages[0],
            love=int(row["love"]),
            ai_emo=self._load_optional_json(row["ai_emo_json"]),
            user_emo=self._load_optional_json(row["user_emo_json"]),
            speak_time=float(row["speak_time"]),
            importance=float(row["importance"]),
        )
        return StoredMemoryItem(
            event_id=row["event_id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            sequence_no=int(row["sequence_no"]),
            item=item,
        )

    def _session_from_row(self, row: sqlite3.Row) -> SessionRecord:
        """把数据库记录转换成不可变的 SessionRecord。"""
        return SessionRecord(
            user_id=row["user_id"],
            session_id=row["session_id"],
            title=row["title"],
            summary=row["summary"],
            summarized_through_sequence=int(row["summarized_through_sequence"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _optional_json(value: Any) -> str | None:
        """把可选元数据编码为 JSON；None 保持为空。"""
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _load_optional_json(value: str | None) -> Any:
        """把数据库中的可选 JSON 恢复为 Python 数据。"""
        return json.loads(value) if value is not None else None

    @staticmethod
    def _require_id(value: str, field_name: str) -> None:
        """校验用户和 Session 标识，避免空值进入数据库。"""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must not be empty")
