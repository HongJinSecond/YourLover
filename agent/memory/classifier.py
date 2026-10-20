"""长期记忆分类器：从对话中提取记忆候选。

提供基于规则的 MVP 实现，负责：
    1. 判断哪些消息值得长期记忆（classify_memory_candidate）；
    2. 从用户消息提取语义记忆（extract_semantic_memories）；
    3. 汇总一次对话为情节记忆（summarize_episodic_memory）；
    4. 提取未来交互规则候选（extract_procedural_candidates）；
    5. 提取用户画像候选（extract_profile_candidates）。

该类只生成候选对象，不负责持久化、画像更新或好感度变更。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from agent.memory.profile_candidate import ProfileCandidate
from agent.memory.schema import EpisodicMemory, MemoryEvidence, ProceduralCandidate, SemanticMemory

# 用户当前状态类记忆的默认过期天数
STATE_TTL_DAYS = 30


@dataclass(frozen=True)
class MemoryCandidateDecision:
    """单条消息的记忆决策结果。

    字段：
        should_store: 是否值得长期存储
        memory_types: 候选记忆类型元组（semantic/episodic/procedural/profile_candidate）
        reason: 决策理由（人类可读）
        confidence: 决策置信度 0~1
        needs_user_confirmation: 是否需要用户确认后才可落地
    """

    should_store: bool
    memory_types: tuple[str, ...]
    reason: str
    confidence: float
    needs_user_confirmation: bool = False


@dataclass(frozen=True)
class _MessageView:
    """消息的统一视图，屏蔽 LangChain Message / dict / str 的差异。

    字段：
        role: 规范化角色（user/assistant/其它）
        content: 文本内容
        message_id: 消息 ID（可空）
        session_id: 所属会话 ID（可空）
        index: 在原消息列表中的位置
    """

    role: str
    content: str
    message_id: str | None
    session_id: str | None
    index: int


class MemoryClassifier:
    """基于规则的长期记忆候选提取器。

    该类只创建候选对象，不持久化、不修改用户画像、不更新好感度，
    以保持职责单一，便于后续替换为 LLM 驱动的提取器。
    """

    def __init__(self, state_ttl_days: int = STATE_TTL_DAYS) -> None:
        """初始化分类器。

        参数：
            state_ttl_days: state 类语义记忆的过期天数，必须 >=1。
        """
        if state_ttl_days < 1:
            raise ValueError("state_ttl_days must be greater than or equal to 1")
        self.state_ttl_days = state_ttl_days

    def classify_memory_candidate(self, messages: list[Any]) -> list[MemoryCandidateDecision]:
        """对一组消息做记忆决策，返回每条用户消息的存储建议。

        逻辑：
            - 跳过非 user 消息与空内容；
            - 模型推断类内容标记为需用户确认、不直接存储；
            - 命中偏好/状态/画像/事实/规则触发词时建议存储对应类型；
            - 若整组消息构成"问题-行动-结果"则追加 episodic 决策。
        """
        decisions: list[MemoryCandidateDecision] = []
        for message in _view_messages(messages):
            if message.role != "user" or not message.content:
                continue
            if _is_model_inference(message.content):
                decisions.append(
                    MemoryCandidateDecision(
                        should_store=False,
                        memory_types=("profile_candidate",),
                        reason="模型推断缺少用户直接证据，需要先向用户确认。",
                        confidence=0.4,
                        needs_user_confirmation=True,
                    )
                )
                continue

            scopes = _semantic_scopes(message.content)
            types: list[str] = ["semantic"] if scopes else []
            if _looks_procedural(message.content):
                types.append("procedural")
            if _profile_field_value(message.content) is not None:
                types.append("profile_candidate")
            if types:
                decisions.append(
                    MemoryCandidateDecision(
                        should_store=True,
                        # dict.fromkeys 保序去重
                        memory_types=tuple(dict.fromkeys(types)),
                        reason="用户提供了明确的身份、偏好、状态、目标或未来交互规则。",
                        confidence=0.95 if _has_strong_trigger(message.content) else 0.85,
                    )
                )

        if self._episodic_parts(messages) is not None:
            decisions.append(
                MemoryCandidateDecision(
                    should_store=True,
                    memory_types=("episodic",),
                    reason="对话包含明确的问题、Agent 行动和用户反馈结果。",
                    confidence=0.9,
                )
            )
        return decisions

    def extract_semantic_memories(self, messages: list[Any], user_id: str) -> list[SemanticMemory]:
        """从用户消息中提取语义记忆。

        对每条用户消息按 scope（preference/state/profile_candidate/fact）拆分，
        同一 (scope, content) 只保留一条。模型推断类消息被跳过。
        state 类记忆带过期时间，preference/profile_candidate 类重要性更高。
        """
        _require_user_id(user_id)
        memories: list[SemanticMemory] = []
        seen: set[tuple[str, str]] = set()
        for message in _view_messages(messages):
            if message.role != "user" or _is_model_inference(message.content):
                continue
            for scope in _semantic_scopes(message.content):
                content, predicate, value = _semantic_value(message.content, scope)
                key = (scope, content)
                if key in seen:
                    continue
                seen.add(key)
                memories.append(
                    SemanticMemory(
                        user_id=user_id,
                        scope=scope,
                        content=content,
                        predicate=predicate,
                        object=value,
                        confidence=0.98 if _has_strong_trigger(message.content) else 0.9,
                        importance=0.8 if scope in {"preference", "profile_candidate"} else 0.7,
                        expires_at=self._state_expiry() if scope == "state" else None,
                        tags=[scope],
                        evidence=_evidence(message),
                    )
                )
        return memories

    def summarize_episodic_memory(
        self,
        messages: list[Any],
        user_id: str,
        love_snapshot: int | None = None,
    ) -> EpisodicMemory | None:
        """将一次"问题-行动-结果"对话汇总为情节记忆。

        仅当消息中存在满足时序的 (问题, 行动, 结果) 三段时才生成：
            - 问题：用户消息含 _PROBLEM_WORDS；
            - 行动：assistant 在问题之后的非空回复；
            - 结果：用户消息含 _RESULT_WORDS 且位于行动之后。
        生成内容包含标题、摘要、目标、行动列表、结果、情感前后对比、
        召回触发词与证据（消息 ID/摘录）。
        """
        _require_user_id(user_id)
        parts = self._episodic_parts(messages)
        if parts is None:
            return None
        problem, actions, result, views = parts
        ids = [item.message_id for item in views if item.message_id]
        session_id = next((item.session_id for item in views if item.session_id), None)
        # 规则降级路径仅根据明确关键词记录情感线索。
        before = {"anxiety": 0.8} if _contains_any(problem, ("焦虑", "压力", "紧张")) else {}
        after = {"calm": 0.7} if _contains_any(result, ("好多", "缓解", "平静", "放心")) else {}
        triggers = _recall_triggers(problem)
        return EpisodicMemory(
            user_id=user_id,
            session_id=session_id,
            title=_episode_title(problem),
            summary=f"用户表达{_trim(problem)}；Agent {_trim('；'.join(actions))}；随后用户反馈{_trim(result)}。",
            turn_range=[views[0].index, views[-1].index],
            user_goal=_trim(problem),
            agent_actions=[_trim(action) for action in actions],
            outcome=_trim(result),
            emotion_before=before,
            emotion_after=after,
            love_snapshot=love_snapshot,
            importance=0.8,
            confidence=0.9,
            recall_triggers=triggers,
            evidence=MemoryEvidence(
                session_id=session_id,
                message_ids=ids,
                excerpt=" | ".join(_trim(item.content, 80) for item in views),
            ),
        )

    def extract_procedural_candidates(
        self, messages: list[Any], user_id: str
    ) -> list[ProceduralCandidate]:
        """从用户消息中提取未来交互规则候选。

        命中"以后/今后/每次 ... 要/记得/按 ..."等模式时生成一条候选，
        默认 needs_user_confirmation=True，需用户确认后才可生效。
        """
        _require_user_id(user_id)
        candidates: list[ProceduralCandidate] = []
        for message in _view_messages(messages):
            if message.role != "user" or _is_model_inference(message.content):
                continue
            if _looks_procedural(message.content):
                candidates.append(
                    ProceduralCandidate(
                        user_id=user_id,
                        content=_trim(message.content),
                        rule_type="interaction_rule",
                        reason="用户明确给出了未来交互或输出规则。",
                        needs_user_confirmation=True,
                        confidence=0.9,
                        importance=0.7,
                        evidence=_evidence(message),
                        tags=["rule_candidate"],
                    )
                )
        return candidates

    def extract_profile_candidates(
        self, messages: list[Any], user_id: str
    ) -> list[ProfileCandidate]:
        """从用户消息中提取画像更新候选。

        - 模型推断类内容生成 inferred_preference 候选，需用户确认；
        - 命中"别叫我X 叫我Y"、"我叫Y"、"我是Y"等模式时分别生成
          nickname/name/identity 候选。
        """
        _require_user_id(user_id)
        candidates: list[ProfileCandidate] = []
        for message in _view_messages(messages):
            if message.role != "user":
                continue
            if _is_model_inference(message.content):
                candidates.append(
                    ProfileCandidate(
                        user_id=user_id,
                        field="inferred_preference",
                        value=_trim(message.content),
                        reason="该内容来自模型推断而非用户明确陈述。",
                        needs_user_confirmation=True,
                        confidence=0.4,
                        evidence=_evidence(message),
                    )
                )
                continue
            profile = _profile_field_value(message.content)
            if profile is not None:
                field, value, reason = profile
                candidates.append(
                    ProfileCandidate(
                        user_id=user_id,
                        field=field,
                        value=value,
                        reason=reason,
                        evidence=_evidence(message),
                    )
                )
        return candidates

    def _state_expiry(self) -> str:
        """计算 state 类记忆的过期时间（当前时间 + state_ttl_days）。"""
        return (datetime.now(timezone.utc) + timedelta(days=self.state_ttl_days)).isoformat()

    def _episodic_parts(
        self, messages: list[Any]
    ) -> tuple[str, list[str], str, list[_MessageView]] | None:
        """检测消息中是否存在完整的"问题-行动-结果"三段。

        返回 (problem, actions, result, views) 元组；任一段缺失或时序错乱返回 None。
        - problem: 第一条含问题词的用户消息
        - actions: problem 与 result 之间的 assistant 回复内容列表
        - result: 行动之后第一条含结果词的用户消息
        - views: 该情节涉及的全部消息视图（含索引，用于 turn_range）
        """
        views = _view_messages(messages)
        problems = [
            item
            for item in views
            if item.role == "user" and _contains_any(item.content, _PROBLEM_WORDS)
        ]
        actions = [item for item in views if item.role == "assistant" and item.content.strip()]
        results = [
            item
            for item in views
            if item.role == "user" and _contains_any(item.content, _RESULT_WORDS)
        ]
        if not problems or not actions or not results:
            return None
        problem = problems[0]
        result = next((item for item in results if item.index > actions[0].index), None)
        if result is None or result.index <= problem.index:
            return None
        relevant_actions = [
            item.content for item in actions if problem.index < item.index < result.index
        ]
        if not relevant_actions:
            return None
        relevant_views = [item for item in views if problem.index <= item.index <= result.index]
        return problem.content, relevant_actions, result.content, relevant_views


# 问题词：用户表达困扰/负面情绪的关键词
_PROBLEM_WORDS = ("焦虑", "压力", "紧张", "难过", "失眠", "困扰", "问题", "怎么办")
# 结果词：用户表达问题被缓解/解决的关键词
_RESULT_WORDS = ("好多了", "好一些", "缓解", "平静", "有帮助", "有用", "解决了", "明白了", "谢谢")


def _view_messages(messages: Iterable[Any]) -> list[_MessageView]:
    """将任意消息序列转为统一的 _MessageView 列表，附带原始索引。"""
    return [_message_view(message, index) for index, message in enumerate(messages)]


def _message_view(message: Any, index: int) -> _MessageView:
    """把单条消息（str / dict / LangChain Message）归一化为 _MessageView。

    兼容三种输入：
        - str：视为 user 消息；
        - dict：取 role/content/id/metadata 等字段；
        - 对象：取 type/role/content/id/metadata/additional_kwargs 等属性。
    content 为列表时拼接为字符串（多模态消息的 text 部分）。
    """
    if isinstance(message, str):
        return _MessageView("user", message.strip(), None, None, index)
    if isinstance(message, dict):
        role = message.get("role") or message.get("type") or "user"
        content = message.get("content", "")
        message_id = message.get("id") or message.get("message_id")
        metadata = message.get("metadata") or message.get("additional_kwargs") or {}
        session_id = message.get("session_id") or metadata.get("session_id")
    else:
        role = getattr(message, "role", None) or getattr(message, "type", None)
        role = role or _role_from_class_name(message.__class__.__name__)
        content = getattr(message, "content", "")
        message_id = getattr(message, "id", None) or getattr(message, "message_id", None)
        metadata = (
            getattr(message, "metadata", None) or getattr(message, "additional_kwargs", None) or {}
        )
        session_id = getattr(message, "session_id", None) or metadata.get("session_id")
    if isinstance(content, list):
        # 多模态 content：拼接所有 text 部分
        content = " ".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content
        )
    return _MessageView(
        _normalize_role(str(role)), str(content).strip(), message_id, session_id, index
    )


def _role_from_class_name(name: str) -> str:
    """根据消息类名推断角色（HumanMessage->user，AI/Assistant->assistant）。"""
    lowered = name.lower()
    if "human" in lowered:
        return "user"
    if "ai" in lowered or "assistant" in lowered:
        return "assistant"
    return "user"


def _normalize_role(role: str) -> str:
    """将 role 字符串规范化为 user/assistant/原值。"""
    role = role.lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"ai", "assistant"}:
        return "assistant"
    return role


def _semantic_scopes(text: str) -> list[str]:
    """识别用户消息命中的语义 scope 列表。

    依次检测：
        - preference: 喜好/回答风格偏好
        - state: "我最近/最近我"等当前状态
        - profile_candidate: "我叫/我是"等画像信息
        - fact: "我想/目标/打算"等事实陈述
        - 若有强触发词（记住/请记得）但未命中上述，归为 fact
    """
    scopes: list[str] = []
    if re.search(r"(?:我喜欢|我不喜欢|喜欢你回答|回答.{0,8}(?:短|简洁)|别叫我|叫我)", text):
        scopes.append("preference")
    if re.search(r"(?:我最近|最近我)", text):
        scopes.append("state")
    if re.search(r"(?:我叫|我是)", text):
        scopes.append("profile_candidate")
    if re.search(r"(?:我想|我的目标是|我打算)", text):
        scopes.append("fact")
    if _has_strong_trigger(text) and not scopes:
        scopes.append("fact")
    return scopes


def _semantic_value(text: str, scope: str) -> tuple[str, str, str]:
    """根据 scope 生成语义记忆的 (content, predicate, object)。

    content 是面向检索的规范化文案，predicate 是关系谓词，
    object 是从原文清洗出的值（如偏好具体内容、状态原文等）。
    """
    cleaned = _clean_trigger(text)
    if scope == "preference":
        if re.search(r"(?:回答.{0,8}(?:短|简洁)|简短回答|回答短一点)", cleaned):
            value = "简短回答"
        else:
            value = cleaned
        return f"用户偏好：{value}。", "preference", value
    if scope == "state":
        return f"用户当前状态：{cleaned}。", "current_state", cleaned
    if scope == "profile_candidate":
        return f"用户画像候选：{cleaned}。", "profile", cleaned
    return f"用户陈述：{cleaned}。", "fact", cleaned


def _profile_field_value(text: str) -> tuple[str, str, str] | None:
    """从用户消息提取画像字段更新。

    依次匹配：
        - 称呼纠正："别叫我X 叫我Y" -> (nickname, Y, 理由)
        - 姓名："我叫Y" -> (name, Y, 理由)
        - 身份："我是Y" -> (identity, Y, 理由)
    无匹配返回 None。
    """
    correction = re.search(r"别叫我[^，。,.！!]*[，, ]*叫我(?P<value>[^，。,.！!]+)", text)
    if correction:
        return "nickname", correction.group("value").strip(), "用户明确纠正了称呼。"
    name = re.search(r"我叫(?P<value>[^，。,.！!]+)", text)
    if name:
        return "name", name.group("value").strip(), "用户明确提供了姓名。"
    identity = re.search(r"我是(?P<value>[^，。,.！!]+)", text)
    if identity:
        return "identity", identity.group("value").strip(), "用户明确提供了身份信息。"
    return None


def _looks_procedural(text: str) -> bool:
    """判断文本是否像未来交互规则（"以后/今后/每次 ... 要/记得/按 ..."）。"""
    return bool(re.search(r"(?:以后|今后|每次).{0,30}(?:要|记得|按照|按|先|不要)", text))


def _has_strong_trigger(text: str) -> bool:
    """判断文本是否包含强记忆触发词（记住/请记得/你要记得）。"""
    return bool(re.search(r"(?:记住|请记得|你要记得)", text))


def _is_model_inference(text: str) -> bool:
    """判断文本是否为模型推断而非用户直接陈述（含"模型猜测/推测用户"等）。"""
    return _contains_any(text, ("模型猜测", "模型推断", "推测用户", "猜测用户", "可能喜欢"))


def _clean_trigger(text: str) -> str:
    """去除文本开头的"请记住/记得"等触发词与标点，返回清洗后的内容。"""
    cleaned = re.sub(r"^\s*(?:请)?(?:记住|记得)[，,:：\s]*", "", text.strip())
    return cleaned.rstrip("。.!！ ")


def _evidence(message: _MessageView) -> MemoryEvidence:
    """根据消息视图构造记忆证据（含 session_id、message_id、原文摘录）。"""
    return MemoryEvidence(
        session_id=message.session_id,
        message_ids=[message.message_id] if message.message_id else [],
        excerpt=message.content,
    )


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    """判断 text 是否包含 words 中任意一个词。"""
    return any(word in text for word in words)


def _trim(text: str, limit: int = 100) -> str:
    """清洗并截断文本：合并空白、去首尾标点，超 limit 加省略号。"""
    cleaned = re.sub(r"\s+", " ", text).strip().strip("。.!！")
    return cleaned if len(cleaned) <= limit else f"{cleaned[: limit - 1]}…"


def _episode_title(problem: str) -> str:
    """根据问题内容生成情节标题（含考试/期末/复习则标记为考试压力支持）。"""
    if _contains_any(problem, ("考试", "期末", "复习")):
        return "考试压力支持对话"
    return "问题解决对话"


def _recall_triggers(problem: str) -> list[str]:
    """从问题文本中提取召回触发词，用于未来相似场景唤起该情节。

    优先返回命中的关键词（期末/考试/复习/焦虑/压力/失眠），
    都未命中时返回截断后的问题文本作为触发词。
    """
    triggers = [
        word for word in ("期末", "考试", "复习", "焦虑", "压力", "失眠") if word in problem
    ]
    return triggers or [_trim(problem, 20)]


def _require_user_id(user_id: str) -> None:
    """校验 user_id 非空，否则抛出 ValueError。"""
    if not user_id or not user_id.strip():
        raise ValueError("user_id must not be empty")
