"""用户画像的数据模型和字段权限定义。

本模块只负责定义“用户画像长什么样”和“谁能修改哪些字段”。
它不访问数据库，也不调用大模型。数据库操作放在 profile_store.py，
候选提取和审核流程放在 long_memory.py。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def profile_now_iso() -> str:
    """生成当前 UTC 时间，统一用于画像、候选和修改日志。"""
    return datetime.now(timezone.utc).isoformat()


class ProfileActor(str, Enum):
    """画像修改来源。

    USER 表示信息由用户明确表达或确认；SYSTEM 表示应用内部可信逻辑；
    MODEL 表示信息只是大模型推断。MODEL 只能创建候选，不能直接写画像。
    """

    USER = "user"
    SYSTEM = "system"
    MODEL = "model"


class ProfileCandidateStatus(str, Enum):
    """画像候选状态，用于记录候选从产生到处理的生命周期。"""

    PENDING = "pending"
    APPLIED = "applied"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ProfileValueType(str, Enum):
    """画像字段值类型；MVP 只需要文本和布尔值。"""

    TEXT = "text"
    BOOLEAN = "boolean"


class ProfileValidationError(ValueError):
    """画像字段或字段值不符合权限和格式要求时抛出的异常。"""


@dataclass(frozen=True)
class ProfileFieldPolicy:
    """单个画像字段的权限策略。

    field_name 是数据库列名；display_name 用于拼接模型上下文；
    value_type 决定值校验方式；allowed_actors 是允许最终写入该字段的来源；
    show_in_context 决定该字段是否会发送给聊天模型；max_length 限制文本长度。
    """

    field_name: str  # 数据库字段名
    display_name: str  # 面向提示词的中文名称
    value_type: ProfileValueType  # 字段值类型
    allowed_actors: tuple[ProfileActor, ...]  # 允许写入的来源
    show_in_context: bool = True  # 是否放入长期记忆上下文
    max_length: int = 100  # 文本字段最大长度
    allow_clear: bool = True  # 是否允许使用 None 清空字段


# 字段白名单是画像权限的唯一依据。大模型即使返回白名单外字段，
# UserProfileManager 和 ProfileStore 也会拒绝，避免仅靠提示词控制权限。
PROFILE_FIELD_POLICIES: dict[str, ProfileFieldPolicy] = {
    "name": ProfileFieldPolicy(
        field_name="name",
        display_name="姓名",
        value_type=ProfileValueType.TEXT,
        allowed_actors=(ProfileActor.USER, ProfileActor.SYSTEM),
    ),
    "nickname": ProfileFieldPolicy(
        field_name="nickname",
        display_name="昵称",
        value_type=ProfileValueType.TEXT,
        allowed_actors=(ProfileActor.USER, ProfileActor.SYSTEM),
    ),
    "preferred_address": ProfileFieldPolicy(
        field_name="preferred_address",
        display_name="希望的称呼",
        value_type=ProfileValueType.TEXT,
        allowed_actors=(ProfileActor.USER, ProfileActor.SYSTEM),
    ),
    "identity": ProfileFieldPolicy(
        field_name="identity",
        display_name="身份",
        value_type=ProfileValueType.TEXT,
        allowed_actors=(ProfileActor.USER, ProfileActor.SYSTEM),
        max_length=200,
    ),
    "language": ProfileFieldPolicy(
        field_name="language",
        display_name="常用语言",
        value_type=ProfileValueType.TEXT,
        allowed_actors=(ProfileActor.USER, ProfileActor.SYSTEM),
        max_length=50,
    ),
    "timezone": ProfileFieldPolicy(
        field_name="timezone",
        display_name="时区",
        value_type=ProfileValueType.TEXT,
        allowed_actors=(ProfileActor.USER, ProfileActor.SYSTEM),
        max_length=50,
    ),
    "onboarding_completed": ProfileFieldPolicy(
        field_name="onboarding_completed",
        display_name="初始化状态",
        value_type=ProfileValueType.BOOLEAN,
        allowed_actors=(ProfileActor.SYSTEM,),
        show_in_context=False,
        allow_clear=False,
    ),
}


def get_profile_field_policy(field_name: str) -> ProfileFieldPolicy:
    """读取字段策略；字段不在白名单时直接拒绝。"""
    policy = PROFILE_FIELD_POLICIES.get(field_name)
    if policy is None:
        raise ProfileValidationError(f"不允许修改画像字段: {field_name}")
    return policy


def normalize_profile_actor(actor: ProfileActor | str) -> ProfileActor:
    """把字符串来源转换为 ProfileActor，非法值会抛出明确异常。"""
    if isinstance(actor, ProfileActor):
        return actor
    try:
        return ProfileActor(actor)
    except ValueError as exc:
        raise ProfileValidationError(f"未知的画像修改来源: {actor}") from exc


def validate_profile_update(
    field_name: str,
    value: Any,
    actor: ProfileActor | str,
) -> Any:
    """校验画像字段、写入来源和值，并返回清洗后的值。

    该函数只做确定性校验，不信任大模型返回的字段名、actor 或数据类型。
    文本会去除首尾空白；允许清空的字段可传 None；布尔字段只接受 bool。
    """
    policy = get_profile_field_policy(field_name)
    normalized_actor = normalize_profile_actor(actor)

    if normalized_actor not in policy.allowed_actors:
        raise ProfileValidationError(f"来源 {normalized_actor.value} 无权修改字段 {field_name}")

    if value is None:
        if policy.allow_clear:
            return None
        raise ProfileValidationError(f"字段 {field_name} 不允许清空")

    if policy.value_type == ProfileValueType.BOOLEAN:
        if not isinstance(value, bool):
            raise ProfileValidationError(f"字段 {field_name} 必须是布尔值")
        return value

    if not isinstance(value, str):
        raise ProfileValidationError(f"字段 {field_name} 必须是文本")

    cleaned_value = value.strip()
    if not cleaned_value:
        raise ProfileValidationError(f"字段 {field_name} 不能为空文本")
    if len(cleaned_value) > policy.max_length:
        raise ProfileValidationError(f"字段 {field_name} 最多允许 {policy.max_length} 个字符")
    return cleaned_value


@dataclass
class UserProfile:
    """用户画像：保存稳定、结构化并且适合每轮注入的用户信息。"""

    user_id: str  # 用户唯一标识，由系统创建后不可修改
    name: str | None = None  # 用户姓名
    nickname: str | None = None  # 用户昵称
    preferred_address: str | None = None  # Agent 对用户的称呼
    identity: str | None = None  # 用户明确提供的身份描述
    language: str | None = None  # 常用语言
    timezone: str | None = None  # 用户时区
    onboarding_completed: bool = False  # 系统维护的初始化完成状态
    revision: int = 0  # 每次修改递增，用于发现并发覆盖
    created_at: str = field(default_factory=profile_now_iso)  # 创建时间
    updated_at: str = field(default_factory=profile_now_iso)  # 最近更新时间

    def to_dict(self) -> dict[str, Any]:
        """转换成普通字典，供数据库、测试和界面使用。"""
        return {
            "user_id": self.user_id,
            "name": self.name,
            "nickname": self.nickname,
            "preferred_address": self.preferred_address,
            "identity": self.identity,
            "language": self.language,
            "timezone": self.timezone,
            "onboarding_completed": self.onboarding_completed,
            "revision": self.revision,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class StoredProfileCandidate:
    """已保存的画像候选：包含权限来源、证据和当前处理状态。"""

    candidate_id: str  # 候选唯一 ID
    user_id: str  # 候选所属用户
    field_name: str  # 计划修改的画像字段
    proposed_value: Any  # 建议写入的值
    source_actor: ProfileActor  # 信息来源，不由模型自行决定系统权限
    confidence: float  # 模型对提取结果的置信度
    reason: str  # 产生候选的原因
    needs_user_confirmation: bool  # 是否必须等待用户确认
    status: ProfileCandidateStatus  # 候选当前状态
    source_session_id: str | None = None  # 来源 Session
    source_message_id: str | None = None  # 来源消息
    source_excerpt: str | None = None  # 来源原文
    created_at: str = field(default_factory=profile_now_iso)  # 创建时间
    resolved_at: str | None = None  # 应用、拒绝或过期时间
    resolution_reason: str | None = None  # 最终处理说明


@dataclass
class ProfileChangeLog:
    """画像修改日志：记录字段修改前后的值以及谁批准了修改。"""

    log_id: str  # 日志唯一 ID
    user_id: str  # 被修改的用户
    field_name: str  # 被修改字段
    old_value: Any  # 修改前的值
    new_value: Any  # 修改后的值
    actor: ProfileActor  # 最终批准写入的来源
    reason: str  # 修改原因
    candidate_id: str | None = None  # 对应候选；系统直接修改时为空
    created_at: str = field(default_factory=profile_now_iso)  # 修改时间
