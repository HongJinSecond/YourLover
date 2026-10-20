"""长期记忆数据模型定义。

该模块定义了用户长期记忆的核心数据结构，包括三类记忆：
    - SemanticMemory（语义记忆）：用户的偏好、状态、画像候选、事实陈述等。
    - EpisodicMemory（情节记忆）：一次完整对话的"问题-行动-结果"摘要。
    - ProceduralCandidate（程序性候选）：用户给出的未来交互规则候选项。

所有记忆对象均支持 to_dict / from_dict 序列化，并统一通过 UserMemoryFile
按用户维度持久化为 JSON 文件。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

# 记忆文件 schema 版本号，用于未来数据迁移
SCHEMA_VERSION = "0.1"


class MemoryType(str, Enum):
    """记忆类型枚举。

    - SEMANTIC: 语义记忆（偏好/状态/事实等结构化陈述）
    - EPISODIC: 情节记忆（一次对话的摘要）
    - PROCEDURAL: 程序性记忆候选（未来交互规则）
    """

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


class MemoryStatus(str, Enum):
    """记忆状态枚举。

    - ACTIVE: 活跃，正常参与检索
    - SUPERSEDED: 被新记忆取代，通常不再参与检索
    - DELETED: 软删除，保留数据但默认不返回
    """

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


def now_iso() -> str:
    """返回当前 UTC 时间的 ISO 8601 字符串，用于记忆的时间戳字段。"""
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    """生成全局唯一记忆 ID。

    格式为 `{prefix}_{时间戳}_{uuid前8位}`，prefix 用于区分记忆类型
    （sem/epi/proc），时间戳保证同一毫秒内大致有序，uuid 段避免冲突。
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return f"{prefix}_{timestamp}_{uuid4().hex[:8]}"


def _enum_value(value: Any) -> Any:
    """若传入枚举则返回其 value，否则原样返回。"""
    if isinstance(value, Enum):
        return value.value
    return value


def _serialize(value: Any) -> Any:
    """递归将记忆对象序列化为 JSON 可存储的原生类型。

    处理逻辑：
        - Enum 转为其 value
        - dataclass 转为字段名到值的字典
        - list/tuple 转为列表
        - dict 递归处理值
        - 其它类型原样返回
    """
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: _serialize(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


def _coerce_enum(enum_cls: type[Enum], value: Any) -> Enum:
    """将任意值强制转换为指定枚举类型。

    若已是该枚举实例则直接返回，否则用 enum_cls(value) 构造。
    用于 from_dict 时把字符串状态还原为 MemoryStatus。
    """
    if isinstance(value, enum_cls):
        return value
    return enum_cls(value)


@dataclass
class MemoryEvidence:
    """记忆证据：记录一条记忆来源于哪些消息。

    用于追溯记忆的出处，便于后续校验、去重与用户确认。
    字段：
        session_id: 来源会话 ID
        message_ids: 来源消息 ID 列表
        excerpt: 关键原文摘录
        metadata: 额外元数据
    """

    session_id: str | None = None
    message_ids: list[str] = field(default_factory=list)
    excerpt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MemoryEvidence":
        """从字典反序列化，data 为空时返回默认实例。"""
        if not data:
            return cls()
        return cls(
            session_id=data.get("session_id"),
            message_ids=list(data.get("message_ids") or []),
            excerpt=data.get("excerpt"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class SemanticMemory:
    """语义记忆：结构化的用户陈述。

    语义记忆描述用户的偏好、当前状态、画像候选或事实，是长期记忆中
    最常参与检索的部分。scope 字段决定其语义类别（preference/state/
    profile_candidate/fact），影响检索权重与生命周期（如 state 类有
    过期时间）。
    """

    user_id: str
    content: str
    scope: str = "fact"
    subject: str = "user"
    predicate: str = "fact"
    object: str | None = None
    confidence: float = 1.0
    importance: float = 0.5
    sensitivity: str = "normal"
    status: MemoryStatus = MemoryStatus.ACTIVE
    evidence: MemoryEvidence = field(default_factory=MemoryEvidence)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    last_used_at: str | None = None
    expires_at: str | None = None
    tags: list[str] = field(default_factory=list)
    embedding_text: str | None = None
    id: str = field(default_factory=lambda: _new_id("sem"))
    type: MemoryType = MemoryType.SEMANTIC

    def __post_init__(self) -> None:
        """初始化后处理：固定 type 为 SEMANTIC，规整 status 与 evidence，
        并在 embedding_text 为空时用 content 作为可嵌入文本。"""
        self.type = MemoryType.SEMANTIC
        self.status = _coerce_enum(MemoryStatus, self.status)
        if isinstance(self.evidence, dict):
            self.evidence = MemoryEvidence.from_dict(self.evidence)
        if self.embedding_text is None:
            self.embedding_text = self.content

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticMemory":
        """从字典反序列化，缺失字段使用默认值，id 缺失时生成新 ID。"""
        return cls(
            id=data.get("id") or _new_id("sem"),
            user_id=data["user_id"],
            content=data["content"],
            scope=data.get("scope", "fact"),
            subject=data.get("subject", "user"),
            predicate=data.get("predicate", "fact"),
            object=data.get("object"),
            confidence=float(data.get("confidence", 1.0)),
            importance=float(data.get("importance", 0.5)),
            sensitivity=data.get("sensitivity", "normal"),
            status=MemoryStatus(data.get("status", MemoryStatus.ACTIVE.value)),
            evidence=MemoryEvidence.from_dict(data.get("evidence")),
            created_at=data.get("created_at") or now_iso(),
            updated_at=data.get("updated_at") or now_iso(),
            last_used_at=data.get("last_used_at"),
            expires_at=data.get("expires_at"),
            tags=list(data.get("tags") or []),
            embedding_text=data.get("embedding_text"),
        )


@dataclass
class EpisodicMemory:
    """情节记忆：一次完整对话的"问题-行动-结果"摘要。

    记录用户提出的问题、Agent 采取的行动、最终结果与用户反馈，
    以及对话前后的情感状态和当时的好感度快照。recall_triggers 用于
    在未来相似场景下重新唤起该情节。
    """

    user_id: str
    summary: str
    session_id: str | None = None
    title: str | None = None
    turn_range: list[int] = field(default_factory=list)
    user_goal: str | None = None
    agent_actions: list[str] = field(default_factory=list)
    outcome: str | None = None
    emotion_before: dict[str, float] = field(default_factory=dict)
    emotion_after: dict[str, float] = field(default_factory=dict)
    love_snapshot: int | None = None
    tools_called: list[str] = field(default_factory=list)
    importance: float = 0.5
    confidence: float = 1.0
    status: MemoryStatus = MemoryStatus.ACTIVE
    evidence: MemoryEvidence = field(default_factory=MemoryEvidence)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    last_used_at: str | None = None
    recall_triggers: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: _new_id("epi"))
    type: MemoryType = MemoryType.EPISODIC

    def __post_init__(self) -> None:
        """初始化后处理：固定 type 为 EPISODIC，规整 status 与 evidence。"""
        self.type = MemoryType.EPISODIC
        self.status = _coerce_enum(MemoryStatus, self.status)
        if isinstance(self.evidence, dict):
            self.evidence = MemoryEvidence.from_dict(self.evidence)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodicMemory":
        """从字典反序列化，缺失字段使用默认值，id 缺失时生成新 ID。"""
        return cls(
            id=data.get("id") or _new_id("epi"),
            user_id=data["user_id"],
            summary=data["summary"],
            session_id=data.get("session_id"),
            title=data.get("title"),
            turn_range=list(data.get("turn_range") or []),
            user_goal=data.get("user_goal"),
            agent_actions=list(data.get("agent_actions") or []),
            outcome=data.get("outcome"),
            emotion_before=dict(data.get("emotion_before") or {}),
            emotion_after=dict(data.get("emotion_after") or {}),
            love_snapshot=data.get("love_snapshot"),
            tools_called=list(data.get("tools_called") or []),
            importance=float(data.get("importance", 0.5)),
            confidence=float(data.get("confidence", 1.0)),
            status=MemoryStatus(data.get("status", MemoryStatus.ACTIVE.value)),
            evidence=MemoryEvidence.from_dict(data.get("evidence")),
            created_at=data.get("created_at") or now_iso(),
            updated_at=data.get("updated_at") or now_iso(),
            last_used_at=data.get("last_used_at"),
            recall_triggers=list(data.get("recall_triggers") or []),
        )


@dataclass
class ProceduralCandidate:
    """程序性记忆候选：用户给出的未来交互规则。

    例如"以后回答要简短"等规则性陈述。该类记忆默认 needs_user_confirmation
    为 True，需经用户确认后才可提升为正式的交互规则。
    """

    user_id: str
    content: str
    rule_type: str = "candidate"
    reason: str | None = None
    needs_user_confirmation: bool = True
    confidence: float = 0.5
    importance: float = 0.5
    status: MemoryStatus = MemoryStatus.ACTIVE
    evidence: MemoryEvidence = field(default_factory=MemoryEvidence)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    tags: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: _new_id("proc"))
    type: MemoryType = MemoryType.PROCEDURAL

    def __post_init__(self) -> None:
        """初始化后处理：固定 type 为 PROCEDURAL，规整 status 与 evidence。"""
        self.type = MemoryType.PROCEDURAL
        self.status = _coerce_enum(MemoryStatus, self.status)
        if isinstance(self.evidence, dict):
            self.evidence = MemoryEvidence.from_dict(self.evidence)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProceduralCandidate":
        """从字典反序列化，缺失字段使用默认值，id 缺失时生成新 ID。"""
        return cls(
            id=data.get("id") or _new_id("proc"),
            user_id=data["user_id"],
            content=data["content"],
            rule_type=data.get("rule_type", "candidate"),
            reason=data.get("reason"),
            needs_user_confirmation=bool(data.get("needs_user_confirmation", True)),
            confidence=float(data.get("confidence", 0.5)),
            importance=float(data.get("importance", 0.5)),
            status=MemoryStatus(data.get("status", MemoryStatus.ACTIVE.value)),
            evidence=MemoryEvidence.from_dict(data.get("evidence")),
            created_at=data.get("created_at") or now_iso(),
            updated_at=data.get("updated_at") or now_iso(),
            tags=list(data.get("tags") or []),
        )


@dataclass
class UserMemoryIndex:
    """用户记忆文件的轻量索引，用于快速统计与压缩决策。

    字段：
        semantic_count: 活跃语义记忆数量
        episodic_count: 活跃情节记忆数量
        last_compacted_at: 上一次压缩时间
    """

    semantic_count: int = 0
    episodic_count: int = 0
    last_compacted_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "UserMemoryIndex":
        """从字典反序列化，data 为空时返回默认实例。"""
        if not data:
            return cls()
        return cls(
            semantic_count=int(data.get("semantic_count", 0)),
            episodic_count=int(data.get("episodic_count", 0)),
            last_compacted_at=data.get("last_compacted_at"),
        )


@dataclass
class UserMemoryFile:
    """单个用户的记忆文件：按用户维度聚合所有记忆。

    一个用户对应一个 JSON 文件，包含三类记忆列表、索引与时间戳。
    该类负责在加载时把 dict 形式的记忆项还原为对应 dataclass 实例，
    并在保存前刷新索引。
    """

    user_id: str
    schema_version: str = SCHEMA_VERSION
    semantic_memories: list[SemanticMemory] = field(default_factory=list)
    episodic_memories: list[EpisodicMemory] = field(default_factory=list)
    procedural_candidates: list[ProceduralCandidate] = field(default_factory=list)
    index: UserMemoryIndex = field(default_factory=UserMemoryIndex)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def __post_init__(self) -> None:
        """初始化后处理：将列表中的 dict 项还原为对应 dataclass 实例，
        并刷新索引统计。"""
        self.semantic_memories = [
            item if isinstance(item, SemanticMemory) else SemanticMemory.from_dict(item)
            for item in self.semantic_memories
        ]
        self.episodic_memories = [
            item if isinstance(item, EpisodicMemory) else EpisodicMemory.from_dict(item)
            for item in self.episodic_memories
        ]
        self.procedural_candidates = [
            item if isinstance(item, ProceduralCandidate) else ProceduralCandidate.from_dict(item)
            for item in self.procedural_candidates
        ]
        if isinstance(self.index, dict):
            self.index = UserMemoryIndex.from_dict(self.index)
        self.refresh_index()

    def refresh_index(self) -> None:
        """根据当前记忆列表重新计算索引中的活跃计数（排除已删除项）。"""
        self.index.semantic_count = sum(
            1 for item in self.semantic_memories if item.status != MemoryStatus.DELETED
        )
        self.index.episodic_count = sum(
            1 for item in self.episodic_memories if item.status != MemoryStatus.DELETED
        )

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典，输出前刷新索引。"""
        self.refresh_index()
        return {
            "schema_version": self.schema_version,
            "user_id": self.user_id,
            "semantic_memories": [item.to_dict() for item in self.semantic_memories],
            "episodic_memories": [item.to_dict() for item in self.episodic_memories],
            "procedural_candidates": [item.to_dict() for item in self.procedural_candidates],
            "index": self.index.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserMemoryFile":
        """从字典反序列化，将三类记忆列表逐项还原为 dataclass 实例。"""
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            user_id=data["user_id"],
            semantic_memories=[
                SemanticMemory.from_dict(item) for item in data.get("semantic_memories", [])
            ],
            episodic_memories=[
                EpisodicMemory.from_dict(item) for item in data.get("episodic_memories", [])
            ],
            procedural_candidates=[
                ProceduralCandidate.from_dict(item)
                for item in data.get("procedural_candidates", [])
            ],
            index=UserMemoryIndex.from_dict(data.get("index")),
            created_at=data.get("created_at") or now_iso(),
            updated_at=data.get("updated_at") or now_iso(),
        )
