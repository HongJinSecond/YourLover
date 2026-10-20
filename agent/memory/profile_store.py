"""基于 SQLite 的用户画像存储。

本模块负责三类持久化数据：
1. user_profiles：当前生效的用户画像；
2. profile_update_candidates：大模型或规则生成的画像更新候选；
3. profile_change_logs：每次实际画像修改的审计日志。

ProfileStore 不调用大模型。它会再次执行字段白名单和 actor 权限校验，
即使上层服务出现错误，也不允许 MODEL 直接修改用户画像。
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent.memory.user_profile import (
    ProfileActor,
    ProfileCandidateStatus,
    ProfileChangeLog,
    StoredProfileCandidate,
    UserProfile,
    get_profile_field_policy,
    normalize_profile_actor,
    profile_now_iso,
    validate_profile_update,
)


class ProfileStoreError(RuntimeError):
    """画像数据库读取、写入或反序列化失败时抛出的统一异常。"""


class ProfileConflictError(ProfileStoreError):
    """画像 revision 已变化或候选状态不允许当前操作时抛出的异常。"""


class ProfileStore:
    """画像存储类：负责画像、待确认候选和修改日志的 SQLite 存取。

    本类负责持久化和最终写入校验，不负责理解用户自然语言，也不负责决定
    模型候选是否应该自动应用；这些业务判断由 UserProfileManager 完成。
    """

    def __init__(
        self,
        database_path: str | Path | None = None,
        timeout: float = 5.0,
    ) -> None:
        """初始化画像存储，并创建数据库目录和数据表。"""
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")

        if database_path is None:
            database_path = (
                Path(__file__).resolve().parents[2] / "data" / "memory" / "profiles.sqlite3"
            )

        self.database_path = Path(database_path)  # 画像 SQLite 文件路径
        self.timeout = float(timeout)  # 数据库锁等待秒数
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def create_profile(self, user_id: str) -> UserProfile:
        """创建一个空画像；用户已经存在时保留原画像并直接返回。"""
        self._validate_user_id(user_id)
        now = profile_now_iso()

        try:
            with self._connect() as connection:
                self._ensure_profile(connection, user_id, now)
        except sqlite3.DatabaseError as exc:
            raise ProfileStoreError("创建用户画像失败") from exc

        profile = self.get_profile(user_id)
        if profile is None:
            raise ProfileStoreError("画像创建后无法读取")
        return profile

    def get_profile(self, user_id: str) -> UserProfile | None:
        """读取当前生效画像；用户不存在时返回 None。"""
        self._validate_user_id(user_id)

        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT user_id, name, nickname, preferred_address,
                           identity, language, timezone, onboarding_completed,
                           revision, created_at, updated_at
                    FROM user_profiles
                    WHERE user_id = ?
                    """,
                    (user_id,),
                ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise ProfileStoreError("读取用户画像失败") from exc

        if row is None:
            return None
        return self._profile_from_row(row)

    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """读取用户画像；不存在时创建空画像。"""
        profile = self.get_profile(user_id)
        if profile is not None:
            return profile
        return self.create_profile(user_id)

    def save_candidate(
        self,
        user_id: str,
        field_name: str,
        proposed_value: Any,
        source_actor: ProfileActor | str,
        confidence: float,
        reason: str,
        needs_user_confirmation: bool,
        source_session_id: str | None = None,
        source_message_id: str | None = None,
        source_excerpt: str | None = None,
        status: ProfileCandidateStatus | str = ProfileCandidateStatus.PENDING,
        resolution_reason: str | None = None,
        candidate_id: str | None = None,
    ) -> StoredProfileCandidate:
        """保存画像候选，不在此方法中自动修改画像。

        字段必须属于画像白名单，但候选来源可以是 MODEL。MODEL 候选只会保存
        为待确认或拒绝状态，真正应用时仍需 USER 或 SYSTEM 权限校验。
        """
        self._validate_user_id(user_id)
        get_profile_field_policy(field_name)
        normalized_actor = normalize_profile_actor(source_actor)
        normalized_status = self._normalize_status(status)

        if confidence < 0 or confidence > 1:
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must not be empty")

        if candidate_id is None:
            candidate_id = "profile_candidate_" + uuid4().hex
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("candidate_id must not be empty")

        now = profile_now_iso()
        resolved_at = None
        if normalized_status != ProfileCandidateStatus.PENDING:
            resolved_at = now

        try:
            value_json = self._serialize_value(proposed_value)
            with self._connect() as connection:
                self._ensure_profile(connection, user_id, now)
                connection.execute(
                    """
                    INSERT INTO profile_update_candidates (
                        candidate_id, user_id, field_name, proposed_value_json,
                        source_actor, confidence, reason,
                        needs_user_confirmation, status,
                        source_session_id, source_message_id, source_excerpt,
                        created_at, resolved_at, resolution_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate_id,
                        user_id,
                        field_name,
                        value_json,
                        normalized_actor.value,
                        float(confidence),
                        reason.strip(),
                        int(needs_user_confirmation),
                        normalized_status.value,
                        source_session_id,
                        source_message_id,
                        source_excerpt,
                        now,
                        resolved_at,
                        resolution_reason,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ProfileStoreError("画像候选 ID 已存在或数据不完整") from exc
        except (sqlite3.DatabaseError, TypeError, ValueError) as exc:
            if isinstance(exc, (TypeError, ValueError)):
                raise
            raise ProfileStoreError("保存画像候选失败") from exc

        candidate = self.get_candidate(candidate_id)
        if candidate is None:
            raise ProfileStoreError("候选保存后无法读取")
        return candidate

    def get_candidate(self, candidate_id: str) -> StoredProfileCandidate | None:
        """按候选 ID 读取记录；不存在时返回 None。"""
        self._validate_candidate_id(candidate_id)

        try:
            with self._connect() as connection:
                row = self._load_candidate_row(connection, candidate_id)
            if row is None:
                return None
            return self._candidate_from_row(row)
        except (sqlite3.DatabaseError, json.JSONDecodeError, ValueError) as exc:
            raise ProfileStoreError("读取画像候选失败") from exc

    def list_candidates(
        self,
        user_id: str,
        status: ProfileCandidateStatus | str | None = None,
    ) -> list[StoredProfileCandidate]:
        """列出用户候选；可按 pending/applied/rejected/expired 过滤。"""
        self._validate_user_id(user_id)
        parameters: list[Any] = [user_id]
        condition = "user_id = ?"

        if status is not None:
            normalized_status = self._normalize_status(status)
            condition += " AND status = ?"
            parameters.append(normalized_status.value)

        try:
            with self._connect() as connection:
                rows = connection.execute(
                    f"""
                    SELECT candidate_id, user_id, field_name,
                           proposed_value_json, source_actor, confidence,
                           reason, needs_user_confirmation, status,
                           source_session_id, source_message_id, source_excerpt,
                           created_at, resolved_at, resolution_reason
                    FROM profile_update_candidates
                    WHERE {condition}
                    ORDER BY created_at DESC
                    """,
                    parameters,
                ).fetchall()
            candidates: list[StoredProfileCandidate] = []
            for row in rows:
                candidates.append(self._candidate_from_row(row))
            return candidates
        except (sqlite3.DatabaseError, json.JSONDecodeError, ValueError) as exc:
            raise ProfileStoreError("列出画像候选失败") from exc

    def apply_candidate(
        self,
        user_id: str,
        candidate_id: str,
        approved_by: ProfileActor | str,
        resolution_reason: str | None = None,
    ) -> UserProfile:
        """应用待处理候选，并在同一事务中写入画像、状态和修改日志。"""
        self._validate_user_id(user_id)
        self._validate_candidate_id(candidate_id)
        normalized_actor = normalize_profile_actor(approved_by)
        now = profile_now_iso()

        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = self._load_candidate_row(connection, candidate_id)
                if row is None or row["user_id"] != user_id:
                    raise ProfileConflictError("画像候选不存在或不属于当前用户")
                if row["status"] != ProfileCandidateStatus.PENDING.value:
                    raise ProfileConflictError("只有 pending 候选可以应用")

                field_name = row["field_name"]
                proposed_value = self._deserialize_value(row["proposed_value_json"])
                cleaned_value = validate_profile_update(
                    field_name,
                    proposed_value,
                    normalized_actor,
                )
                self._ensure_profile(connection, user_id, now)
                profile_row = self._load_profile_row(connection, user_id)
                old_value = self._profile_value_from_row(profile_row, field_name)

                self._write_profile_value(
                    connection,
                    user_id,
                    field_name,
                    cleaned_value,
                    now,
                    expected_revision=None,
                )
                final_reason = resolution_reason or row["reason"]
                connection.execute(
                    """
                    UPDATE profile_update_candidates
                    SET status = ?, resolved_at = ?, resolution_reason = ?
                    WHERE candidate_id = ?
                    """,
                    (
                        ProfileCandidateStatus.APPLIED.value,
                        now,
                        final_reason,
                        candidate_id,
                    ),
                )
                self._insert_change_log(
                    connection,
                    user_id,
                    field_name,
                    old_value,
                    cleaned_value,
                    normalized_actor,
                    final_reason,
                    candidate_id,
                    now,
                )
        except ProfileConflictError:
            raise
        except (sqlite3.DatabaseError, json.JSONDecodeError) as exc:
            raise ProfileStoreError("应用画像候选失败") from exc

        profile = self.get_profile(user_id)
        if profile is None:
            raise ProfileStoreError("候选应用后无法读取画像")
        return profile

    def reject_candidate(
        self,
        user_id: str,
        candidate_id: str,
        resolution_reason: str,
    ) -> StoredProfileCandidate:
        """拒绝一个 pending 候选，并保留拒绝原因供审计。"""
        return self._resolve_candidate_without_applying(
            user_id,
            candidate_id,
            ProfileCandidateStatus.REJECTED,
            resolution_reason,
        )

    def expire_candidate(
        self,
        user_id: str,
        candidate_id: str,
        resolution_reason: str,
    ) -> StoredProfileCandidate:
        """把长期未处理的 pending 候选标记为过期。"""
        return self._resolve_candidate_without_applying(
            user_id,
            candidate_id,
            ProfileCandidateStatus.EXPIRED,
            resolution_reason,
        )

    def update_profile_field(
        self,
        user_id: str,
        field_name: str,
        value: Any,
        actor: ProfileActor | str,
        reason: str,
        expected_revision: int | None = None,
    ) -> UserProfile:
        """直接更新画像字段，适用于可信系统操作或用户明确设置。

        MODEL actor 会在 validate_profile_update 中被拒绝。expected_revision 可选，
        传入后只有数据库中的 revision 相等才允许更新，避免覆盖并发修改。
        """
        self._validate_user_id(user_id)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must not be empty")

        normalized_actor = normalize_profile_actor(actor)
        cleaned_value = validate_profile_update(field_name, value, normalized_actor)
        now = profile_now_iso()

        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._ensure_profile(connection, user_id, now)
                profile_row = self._load_profile_row(connection, user_id)
                old_value = self._profile_value_from_row(profile_row, field_name)
                self._write_profile_value(
                    connection,
                    user_id,
                    field_name,
                    cleaned_value,
                    now,
                    expected_revision,
                )
                self._insert_change_log(
                    connection,
                    user_id,
                    field_name,
                    old_value,
                    cleaned_value,
                    normalized_actor,
                    reason.strip(),
                    candidate_id=None,
                    created_at=now,
                )
        except ProfileConflictError:
            raise
        except sqlite3.DatabaseError as exc:
            raise ProfileStoreError("更新用户画像失败") from exc

        profile = self.get_profile(user_id)
        if profile is None:
            raise ProfileStoreError("画像更新后无法读取")
        return profile

    def list_change_logs(self, user_id: str) -> list[ProfileChangeLog]:
        """按时间从新到旧列出用户画像修改日志。"""
        self._validate_user_id(user_id)

        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT log_id, user_id, field_name, old_value_json,
                           new_value_json, actor, reason, candidate_id, created_at
                    FROM profile_change_logs
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    """,
                    (user_id,),
                ).fetchall()

            logs: list[ProfileChangeLog] = []
            for row in rows:
                logs.append(
                    ProfileChangeLog(
                        log_id=row["log_id"],
                        user_id=row["user_id"],
                        field_name=row["field_name"],
                        old_value=self._deserialize_value(row["old_value_json"]),
                        new_value=self._deserialize_value(row["new_value_json"]),
                        actor=ProfileActor(row["actor"]),
                        reason=row["reason"],
                        candidate_id=row["candidate_id"],
                        created_at=row["created_at"],
                    )
                )
            return logs
        except (sqlite3.DatabaseError, json.JSONDecodeError, ValueError) as exc:
            raise ProfileStoreError("读取画像修改日志失败") from exc

    def _initialize_database(self) -> None:
        """创建画像、候选和修改日志三张表及常用索引。"""
        try:
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        user_id TEXT PRIMARY KEY,
                        name TEXT,
                        nickname TEXT,
                        preferred_address TEXT,
                        identity TEXT,
                        language TEXT,
                        timezone TEXT,
                        onboarding_completed INTEGER NOT NULL DEFAULT 0,
                        revision INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS profile_update_candidates (
                        candidate_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        field_name TEXT NOT NULL,
                        proposed_value_json TEXT NOT NULL,
                        source_actor TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        reason TEXT NOT NULL,
                        needs_user_confirmation INTEGER NOT NULL DEFAULT 1,
                        status TEXT NOT NULL,
                        source_session_id TEXT,
                        source_message_id TEXT,
                        source_excerpt TEXT,
                        created_at TEXT NOT NULL,
                        resolved_at TEXT,
                        resolution_reason TEXT,
                        FOREIGN KEY (user_id)
                            REFERENCES user_profiles (user_id)
                            ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS profile_change_logs (
                        log_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        field_name TEXT NOT NULL,
                        old_value_json TEXT NOT NULL,
                        new_value_json TEXT NOT NULL,
                        actor TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        candidate_id TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (user_id)
                            REFERENCES user_profiles (user_id)
                            ON DELETE CASCADE,
                        FOREIGN KEY (candidate_id)
                            REFERENCES profile_update_candidates (candidate_id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_profile_candidates_user_status
                    ON profile_update_candidates (user_id, status, created_at);

                    CREATE INDEX IF NOT EXISTS idx_profile_logs_user_time
                    ON profile_change_logs (user_id, created_at);
                    """
                )
                connection.execute("PRAGMA user_version = 1")
        except sqlite3.DatabaseError as exc:
            raise ProfileStoreError("初始化画像数据库失败") from exc

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """创建独立连接，便于聊天工作线程并发访问同一个 SQLite 文件。"""
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

    def _ensure_profile(
        self,
        connection: sqlite3.Connection,
        user_id: str,
        now: str,
    ) -> None:
        """在当前事务中确保用户画像存在。"""
        connection.execute(
            """
            INSERT OR IGNORE INTO user_profiles (
                user_id, onboarding_completed, revision, created_at, updated_at
            ) VALUES (?, 0, 0, ?, ?)
            """,
            (user_id, now, now),
        )

    def _load_profile_row(
        self,
        connection: sqlite3.Connection,
        user_id: str,
    ) -> sqlite3.Row:
        """在当前连接中读取画像行；调用前应先确保画像存在。"""
        row = connection.execute(
            """
            SELECT user_id, name, nickname, preferred_address,
                   identity, language, timezone, onboarding_completed,
                   revision, created_at, updated_at
            FROM user_profiles
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
        if row is None:
            raise ProfileStoreError("用户画像不存在")
        return row

    def _load_candidate_row(
        self,
        connection: sqlite3.Connection,
        candidate_id: str,
    ) -> sqlite3.Row | None:
        """在当前连接中读取完整候选数据库行。"""
        return connection.execute(
            """
            SELECT candidate_id, user_id, field_name,
                   proposed_value_json, source_actor, confidence,
                   reason, needs_user_confirmation, status,
                   source_session_id, source_message_id, source_excerpt,
                   created_at, resolved_at, resolution_reason
            FROM profile_update_candidates
            WHERE candidate_id = ?
            """,
            (candidate_id,),
        ).fetchone()

    def _write_profile_value(
        self,
        connection: sqlite3.Connection,
        user_id: str,
        field_name: str,
        value: Any,
        now: str,
        expected_revision: int | None,
    ) -> None:
        """更新一个白名单字段，并令 revision 自动加一。"""
        policy = get_profile_field_policy(field_name)
        database_value = value
        if policy.value_type.value == "boolean" and value is not None:
            database_value = int(value)

        parameters: list[Any] = [database_value, now, user_id]
        revision_condition = ""
        if expected_revision is not None:
            revision_condition = " AND revision = ?"
            parameters.append(expected_revision)

        # field_name 来自固定白名单，不接受任意模型字符串，因此可安全拼接列名。
        cursor = connection.execute(
            f"""
            UPDATE user_profiles
            SET {field_name} = ?, revision = revision + 1, updated_at = ?
            WHERE user_id = ?{revision_condition}
            """,
            parameters,
        )
        if cursor.rowcount == 0:
            raise ProfileConflictError("画像 revision 已变化，请重新读取后再修改")

    def _insert_change_log(
        self,
        connection: sqlite3.Connection,
        user_id: str,
        field_name: str,
        old_value: Any,
        new_value: Any,
        actor: ProfileActor,
        reason: str,
        candidate_id: str | None,
        created_at: str,
    ) -> None:
        """在当前事务中写入一条不可变的画像修改日志。"""
        connection.execute(
            """
            INSERT INTO profile_change_logs (
                log_id, user_id, field_name, old_value_json,
                new_value_json, actor, reason, candidate_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "profile_log_" + uuid4().hex,
                user_id,
                field_name,
                self._serialize_value(old_value),
                self._serialize_value(new_value),
                actor.value,
                reason,
                candidate_id,
                created_at,
            ),
        )

    def _resolve_candidate_without_applying(
        self,
        user_id: str,
        candidate_id: str,
        status: ProfileCandidateStatus,
        resolution_reason: str,
    ) -> StoredProfileCandidate:
        """把 pending 候选改为 rejected 或 expired，但不修改画像。"""
        self._validate_user_id(user_id)
        self._validate_candidate_id(candidate_id)
        if status not in {
            ProfileCandidateStatus.REJECTED,
            ProfileCandidateStatus.EXPIRED,
        }:
            raise ValueError("status must be rejected or expired")
        if not isinstance(resolution_reason, str) or not resolution_reason.strip():
            raise ValueError("resolution_reason must not be empty")

        now = profile_now_iso()
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = self._load_candidate_row(connection, candidate_id)
                if row is None or row["user_id"] != user_id:
                    raise ProfileConflictError("画像候选不存在或不属于当前用户")
                if row["status"] != ProfileCandidateStatus.PENDING.value:
                    raise ProfileConflictError("只有 pending 候选可以处理")
                connection.execute(
                    """
                    UPDATE profile_update_candidates
                    SET status = ?, resolved_at = ?, resolution_reason = ?
                    WHERE candidate_id = ?
                    """,
                    (status.value, now, resolution_reason.strip(), candidate_id),
                )
        except ProfileConflictError:
            raise
        except sqlite3.DatabaseError as exc:
            raise ProfileStoreError("更新画像候选状态失败") from exc

        candidate = self.get_candidate(candidate_id)
        if candidate is None:
            raise ProfileStoreError("候选状态更新后无法读取")
        return candidate

    @staticmethod
    def _profile_from_row(row: sqlite3.Row) -> UserProfile:
        """把 user_profiles 数据库行转换为 UserProfile。"""
        return UserProfile(
            user_id=row["user_id"],
            name=row["name"],
            nickname=row["nickname"],
            preferred_address=row["preferred_address"],
            identity=row["identity"],
            language=row["language"],
            timezone=row["timezone"],
            onboarding_completed=bool(row["onboarding_completed"]),
            revision=int(row["revision"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _candidate_from_row(self, row: sqlite3.Row) -> StoredProfileCandidate:
        """把 profile_update_candidates 行转换为候选对象。"""
        return StoredProfileCandidate(
            candidate_id=row["candidate_id"],
            user_id=row["user_id"],
            field_name=row["field_name"],
            proposed_value=self._deserialize_value(row["proposed_value_json"]),
            source_actor=ProfileActor(row["source_actor"]),
            confidence=float(row["confidence"]),
            reason=row["reason"],
            needs_user_confirmation=bool(row["needs_user_confirmation"]),
            status=ProfileCandidateStatus(row["status"]),
            source_session_id=row["source_session_id"],
            source_message_id=row["source_message_id"],
            source_excerpt=row["source_excerpt"],
            created_at=row["created_at"],
            resolved_at=row["resolved_at"],
            resolution_reason=row["resolution_reason"],
        )

    @staticmethod
    def _profile_value_from_row(row: sqlite3.Row, field_name: str) -> Any:
        """读取画像字段值，并把 SQLite 整数布尔值还原为 bool。"""
        policy = get_profile_field_policy(field_name)
        value = row[field_name]
        if policy.value_type.value == "boolean" and value is not None:
            return bool(value)
        return value

    @staticmethod
    def _serialize_value(value: Any) -> str:
        """把候选值或日志值编码为 JSON。"""
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _deserialize_value(value: str) -> Any:
        """把数据库 JSON 值恢复为 Python 基础类型。"""
        return json.loads(value)

    @staticmethod
    def _normalize_status(
        status: ProfileCandidateStatus | str,
    ) -> ProfileCandidateStatus:
        """把字符串状态转换为 ProfileCandidateStatus。"""
        if isinstance(status, ProfileCandidateStatus):
            return status
        return ProfileCandidateStatus(status)

    @staticmethod
    def _validate_user_id(user_id: str) -> None:
        """确保 user_id 是非空字符串。"""
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must not be empty")

    @staticmethod
    def _validate_candidate_id(candidate_id: str) -> None:
        """确保 candidate_id 是非空字符串。"""
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("candidate_id must not be empty")
