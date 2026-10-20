"""用户画像管理器：画像读取、候选抽取与确认流程的统一入口。

用户画像负责"稳定、结构化、受权限控制"的用户信息（称呼、语言等），
与动态长期记忆（LongTermMemory）分离。

本模块把"大模型理解自然语言"和"系统最终写入画像"严格分开：
    - ProfileCandidateExtractor 只生成候选（接口）；
    - OllamaProfileCandidateExtractor 是默认 qwen2.5:3b 实现；
    - UserProfileManager 校验权限、证据和置信度；
    - ProfileStore 执行最终 SQLite 写入并记录审计日志。

模型永远只能产生候选；字段白名单和 actor 权限由 user_profile.py 的
确定性校验保证，不依赖提示词自觉。

未来接入其它本地模型或云 API 时，只需新增 ProfileCandidateExtractor 子类，
画像模型、数据库和权限策略不需要变化。

（本文件原名 long_memory.py，因名字与动态长期记忆模块容易混淆而改名；
原服务类名 LongMemoryService 同步改为 UserProfileManager。）
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from agent.memory.profile_candidate import ProfileCandidate
from agent.memory.profile_store import ProfileStore
from agent.memory.schema import MemoryEvidence
from agent.memory.user_profile import (
    ProfileActor,
    ProfileCandidateStatus,
    ProfileValidationError,
    StoredProfileCandidate,
    UserProfile,
    get_profile_field_policy,
    validate_profile_update,
)

DEFAULT_PROFILE_MODEL = "qwen2.5:3b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# 画像候选提取的触发规则：命中才调用候选模型，未命中直接跳过。
# 覆盖 user_profile.py 白名单的 6 个字段：
#   name / nickname / preferred_address（称呼类）、identity（身份）、
#   language（语言）、timezone（时区）。
# 说明：规则宁全勿缺——漏触发会导致画像信息不落库，误触发最多浪费一次
# LLM 调用（与旧的"每轮都调"相比仍是大幅裁剪）。
PROFILE_TRIGGER_PATTERNS: list[str] = [
    # 姓名 / 昵称 / 称呼
    r"我叫",
    r"我是",
    r"名字是",
    r"名字叫",
    r"叫我",
    r"别叫我",
    r"请叫我",
    r"可以叫我",
    r"以后叫我",
    r"就叫",
    r"称呼我",
    r"你叫我",
    # 身份
    r"我是一名",
    r"我是一位",
    r"我是做",
    # 语言
    r"说中文",
    r"说英文",
    r"说英语",
    r"用中文",
    r"用英文",
    r"用英语",
    r"讲中文",
    r"讲英文",
    r"讲英语",
    r"语言",
    # 时区
    r"时区",
]

# 预编译：把全部触发词合并为单一正则，一次 search 完成判定
_PROFILE_TRIGGER_RE = re.compile("|".join(PROFILE_TRIGGER_PATTERNS))


def matches_profile_trigger(text: str) -> bool:
    """判断用户消息是否命中画像候选触发词。

    命中才需要调用候选模型（LLM）；未命中直接跳过画像候选写入，
    避免每轮都发起一次 LLM 请求。空文本恒为 False。
    """
    if not text:
        return False
    return _PROFILE_TRIGGER_RE.search(text) is not None


class ProfileCandidateExtractionError(RuntimeError):
    """候选模型不可用、响应不是合法 JSON 或字段格式错误时抛出。"""


@dataclass
class ProfileCandidateProposal:
    """候选提取器输出：复用 ProfileCandidate，并明确记录信息来源。

    字段：
        candidate: 画像字段和值候选
        source_actor: USER 表示明确原话，MODEL 表示模型推断
        explicit_quote: 用户原文中的直接证据（自动应用的必要条件）
    """

    candidate: ProfileCandidate
    source_actor: ProfileActor
    explicit_quote: str | None = None


@dataclass
class ProfileProcessingResult:
    """单条画像候选的处理结果，供界面提示或后续确认流程使用。

    字段：
        field_name: 候选字段
        proposed_value: 候选值
        status: applied/pending/rejected 三种结局
        message: 人类可读处理说明
        candidate_id: 候选 ID（未进入数据库的非法字段为 None）
    """

    field_name: str
    proposed_value: Any
    status: ProfileCandidateStatus
    message: str
    candidate_id: str | None = None


class ProfileCandidateExtractor:
    """画像候选提取器接口。

    子类只负责把用户自然语言转换成候选，不能访问 ProfileStore。
    以后接 OpenAI、其它本地模型或规则模型时，实现 extract 方法即可。
    """

    def extract(
        self,
        user_id: str,
        user_message: str,
        session_id: str | None = None,
        message_id: str | None = None,
    ) -> list[ProfileCandidateProposal]:
        """从一条用户消息中提取零个或多个画像候选。"""
        raise NotImplementedError("子类必须实现 extract()")


class OllamaProfileCandidateExtractor(ProfileCandidateExtractor):
    """Ollama 候选提取器：默认使用 qwen2.5:3b 返回严格 JSON 候选。"""

    def __init__(
        self,
        model_name: str = DEFAULT_PROFILE_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: float = 30.0,
        max_candidates: int = 5,
    ) -> None:
        """保存 Ollama 配置；初始化时不发网络请求。

        参数：
            model_name: Ollama 模型名，默认 qwen2.5:3b。
            base_url: Ollama 服务地址。
            timeout: 单次候选提取超时时间（秒）。
            max_candidates: 单次最多接受的候选条数。
        """
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must not be empty")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        if max_candidates < 1:
            raise ValueError("max_candidates must be greater than or equal to 1")

        self.model_name = model_name.strip()  # Ollama 模型名
        self.base_url = base_url.rstrip("/")  # Ollama 服务地址
        self.timeout = float(timeout)  # 单次候选提取超时时间
        self.max_candidates = int(max_candidates)  # 防止一次生成过多候选

    def extract(
        self,
        user_id: str,
        user_message: str,
        session_id: str | None = None,
        message_id: str | None = None,
    ) -> list[ProfileCandidateProposal]:
        """调用 Ollama，并把 JSON 响应转换为画像候选。"""
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must not be empty")
        if not isinstance(user_message, str) or not user_message.strip():
            return []

        prompt = self._build_prompt(user_message)
        response_text = self._generate_response(prompt)
        response_payload = self._parse_json_response(response_text)
        raw_candidates = response_payload.get("candidates", [])
        if not isinstance(raw_candidates, list):
            raise ProfileCandidateExtractionError("模型返回的 candidates 必须是列表")

        proposals: list[ProfileCandidateProposal] = []
        for raw_candidate in raw_candidates[: self.max_candidates]:
            proposal = self._proposal_from_dict(
                raw_candidate,
                user_id,
                user_message,
                session_id,
                message_id,
            )
            if proposal is not None:
                proposals.append(proposal)
        return proposals

    def _build_prompt(self, user_message: str) -> str:
        """构建画像提取提示词，明确字段范围和 JSON 输出格式。"""
        return f"""/no_think
你是用户画像候选提取器，不是数据库管理员。
你的任务是判断用户原话是否明确提供了下面这些固定画像字段：
- name：姓名
- nickname：昵称
- preferred_address：希望助手如何称呼用户
- identity：用户明确描述的身份
- language：常用语言
- timezone：时区

规则：
1. 兴趣、爱好、讨厌的事、目标、当前情绪、回答风格不要放进固定画像；它们属于动态长期记忆。
2. user_id、revision、created_at、updated_at、onboarding_completed 都不能提取。
3. 用户明确说出的信息使用 source="user_explicit"，并从原文复制一段 evidence。
4. 如果只是从语气或上下文猜测，使用 source="model_inference"，必须 needs_confirmation=true。
5. 不要编造字段和值。没有合适候选时返回空列表。
6. 只输出 JSON，不要输出 Markdown、解释或思考过程。

输出格式：
{{
  "candidates": [
    {{
      "field": "nickname",
      "value": "阿辰",
      "reason": "用户明确要求使用该昵称",
      "source": "user_explicit",
      "confidence": 0.98,
      "needs_confirmation": false,
      "evidence": "以后叫我阿辰"
    }}
  ]
}}

用户原话：
{user_message}
"""

    def _generate_response(self, prompt: str) -> str:
        """生成候选文本；独立钩子允许兼容接口复用字段与证据校验。"""
        return self._call_ollama(prompt)

    def _call_ollama(self, prompt: str) -> str:
        """调用 Ollama /api/generate；使用 JSON 模式并关闭流式响应。"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": 0},
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib_request.Request(
            self.base_url + "/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib_request.urlopen(request, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
        except (urllib_error.URLError, TimeoutError, OSError) as exc:
            raise ProfileCandidateExtractionError(
                f"无法连接画像候选模型 {self.model_name}: {exc}"
            ) from exc

        try:
            response_payload = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise ProfileCandidateExtractionError("Ollama 返回了非法 JSON") from exc

        response_text = response_payload.get("response")
        if not isinstance(response_text, str):
            raise ProfileCandidateExtractionError("Ollama 响应缺少 response 文本")
        return response_text

    def _parse_json_response(self, response_text: str) -> dict[str, Any]:
        """解析模型 JSON；兼容模型偶尔添加代码块或思考标签。"""
        cleaned_text = response_text.strip()
        cleaned_text = cleaned_text.replace("```json", "")
        cleaned_text = cleaned_text.replace("```", "")

        try:
            payload = json.loads(cleaned_text)
        except json.JSONDecodeError:
            first_brace = cleaned_text.find("{")
            last_brace = cleaned_text.rfind("}")
            if first_brace < 0 or last_brace <= first_brace:
                raise ProfileCandidateExtractionError("候选模型没有返回 JSON 对象")
            try:
                payload = json.loads(cleaned_text[first_brace : last_brace + 1])
            except json.JSONDecodeError as exc:
                raise ProfileCandidateExtractionError("候选模型 JSON 无法解析") from exc

        if not isinstance(payload, dict):
            raise ProfileCandidateExtractionError("候选模型返回值必须是 JSON 对象")
        return payload

    def _proposal_from_dict(
        self,
        raw_candidate: Any,
        user_id: str,
        user_message: str,
        session_id: str | None,
        message_id: str | None,
    ) -> ProfileCandidateProposal | None:
        """校验单条模型结果并转换为内部候选对象。"""
        if not isinstance(raw_candidate, dict):
            return None

        field_name = raw_candidate.get("field")
        reason = raw_candidate.get("reason")
        source = raw_candidate.get("source")
        explicit_quote = raw_candidate.get("evidence")
        value = raw_candidate.get("value")

        if not isinstance(field_name, str) or not field_name.strip():
            return None
        if not isinstance(reason, str) or not reason.strip():
            reason = "候选模型提取了画像字段"
        if not isinstance(explicit_quote, str) or not explicit_quote.strip():
            explicit_quote = None
        else:
            explicit_quote = explicit_quote.strip()[:200]

        try:
            confidence = float(raw_candidate.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        # 模型不能声明自己是 SYSTEM。只有明确 user_explicit 才映射为 USER，
        # 其它未知来源一律按 MODEL 推断处理，并强制等待用户确认。
        if source == "user_explicit":
            source_actor = ProfileActor.USER
        else:
            source_actor = ProfileActor.MODEL

        needs_confirmation = bool(raw_candidate.get("needs_confirmation", True))
        if source_actor == ProfileActor.MODEL:
            needs_confirmation = True

        evidence = MemoryEvidence(
            session_id=session_id,
            message_ids=[message_id] if message_id else [],
            excerpt=user_message[:500],
            metadata={"explicit_quote": explicit_quote},
        )
        candidate = ProfileCandidate(
            user_id=user_id,
            field=field_name.strip(),
            value=value,
            reason=reason.strip()[:500],
            needs_user_confirmation=needs_confirmation,
            confidence=confidence,
            evidence=evidence,
        )
        return ProfileCandidateProposal(
            candidate=candidate,
            source_actor=source_actor,
            explicit_quote=explicit_quote,
        )


class LLMProfileCandidateExtractor(OllamaProfileCandidateExtractor):
    """通过已配置的聊天模型提取画像，复用现有 JSON 解析与候选审核结构。

    OpenAI 模式使用未绑定工具的 LLM，不需要额外的本地 Ollama 模型。
    """

    def __init__(self, llm: Any, max_candidates: int = 5) -> None:
        """注入聊天模型，构造时不发送请求。"""
        super().__init__(max_candidates=max_candidates)
        self.llm = llm

    def _generate_response(self, prompt: str) -> str:
        """请求普通 JSON 文本，避免依赖服务商额外实现 JSON mode。"""
        try:
            response = self.llm.invoke(
                [{"role": "user", "content": prompt.removeprefix("/no_think\n")}]
            )
        except Exception as exc:
            raise ProfileCandidateExtractionError("画像候选接口调用失败，已跳过本轮提取") from exc
        text = getattr(response, "content", "")
        if not isinstance(text, str) or not text.strip():
            raise ProfileCandidateExtractionError("画像候选接口返回了空文本")
        return text


class UserProfileManager:
    """用户画像管理类：画像读取、候选审核、确认和上下文生成的唯一业务入口。

    store 负责数据库，candidate_extractor 负责调用模型；本类决定候选应该
    自动应用、等待确认还是拒绝。

    典型用法（ChatAgent 每轮）：
        - 读：manager.get_profile_context(user_id) 注入 system prompt；
        - 写：manager.process_user_message(user_id, text, session_id, message_id)
          产生候选（内部调用 Ollama，失败会抛异常，调用方需捕获降级）。
    """

    def __init__(
        self,
        store: ProfileStore | None = None,
        candidate_extractor: ProfileCandidateExtractor | None = None,
        auto_apply_confidence: float = 0.85,
    ) -> None:
        """初始化管理器；默认使用独立 SQLite 和 Ollama qwen2.5:3b。

        参数：
            store: 画像持久化存储，默认 data/memory/profiles.sqlite3。
            candidate_extractor: 候选提取器，默认 Ollama 实现。
            auto_apply_confidence: 用户明确候选的自动应用置信度阈值。
        """
        if auto_apply_confidence < 0 or auto_apply_confidence > 1:
            raise ValueError("auto_apply_confidence must be between 0 and 1")

        self.store = store or ProfileStore()  # 用户画像持久化存储
        self.candidate_extractor = (
            candidate_extractor or OllamaProfileCandidateExtractor()
        )  # 可替换的候选提取器
        self.auto_apply_confidence = float(auto_apply_confidence)  # 用户明确候选的自动应用阈值

    # ------------------------------------------------------------------
    # 画像读取（每轮调用，仅读 SQLite，安全）
    # ------------------------------------------------------------------

    def get_profile(self, user_id: str) -> UserProfile:
        """读取用户画像；首次调用时自动创建空画像。"""
        return self.store.get_or_create_profile(user_id)

    def get_profile_context(self, user_id: str) -> str:
        """把非空且允许展示的画像字段格式化为模型上下文。

        直接拼进 system prompt；未设置字段不输出，无任何字段时返回空串。
        """
        profile = self.store.get_or_create_profile(user_id)
        profile_data = profile.to_dict()
        lines: list[str] = []

        for field_name in (
            "name",
            "nickname",
            "preferred_address",
            "identity",
            "language",
            "timezone",
        ):
            policy = get_profile_field_policy(field_name)
            value = profile_data.get(field_name)
            if policy.show_in_context and value is not None:
                lines.append(f"- {policy.display_name}：{value}")

        if not lines:
            return ""
        return "用户画像（来自用户明确提供或确认的信息）：\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # 候选流程（每轮回复后调用；内部走 Ollama，调用方需处理异常降级）
    # ------------------------------------------------------------------

    def process_user_message(
        self,
        user_id: str,
        user_message: str,
        session_id: str | None = None,
        message_id: str | None = None,
    ) -> list[ProfileProcessingResult]:
        """让候选模型处理用户消息，并逐条审核、保存或应用候选。

        返回每条候选的处理结果（applied / pending / rejected）。
        Ollama 不可用时抛 ProfileCandidateExtractionError，
        调用方必须捕获并降级，不能中断聊天主流程。
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must not be empty")
        if not isinstance(user_message, str) or not user_message.strip():
            return []

        self.store.get_or_create_profile(user_id)
        proposals = self.candidate_extractor.extract(
            user_id,
            user_message,
            session_id,
            message_id,
        )

        results: list[ProfileProcessingResult] = []
        for proposal in proposals:
            result = self._process_proposal(proposal, user_message)
            results.append(result)
        return results

    def get_pending_candidates(self, user_id: str) -> list[StoredProfileCandidate]:
        """读取等待用户确认的候选，供设置界面或聊天窗口展示。"""
        return self.store.list_candidates(user_id, ProfileCandidateStatus.PENDING)

    def confirm_candidate(
        self,
        user_id: str,
        candidate_id: str,
    ) -> UserProfile:
        """用户确认 pending 候选后，以 USER 权限应用画像更新。"""
        return self.store.apply_candidate(
            user_id,
            candidate_id,
            approved_by=ProfileActor.USER,
            resolution_reason="用户确认了画像候选",
        )

    def reject_candidate(
        self,
        user_id: str,
        candidate_id: str,
        reason: str = "用户拒绝了画像候选",
    ) -> StoredProfileCandidate:
        """拒绝 pending 候选，画像保持不变。"""
        return self.store.reject_candidate(user_id, candidate_id, reason)

    # ------------------------------------------------------------------
    # 直接修改（绕过候选流程，供 UI/系统使用）
    # ------------------------------------------------------------------

    def update_user_field(
        self,
        user_id: str,
        field_name: str,
        value: Any,
        reason: str,
    ) -> UserProfile:
        """以 USER 权限直接更新画像字段（供设置界面等用户手动修改场景）。

        MODEL 权限不可用；修改会写审计日志。
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must not be empty")
        return self.store.update_profile_field(
            user_id,
            field_name,
            value,
            ProfileActor.USER,
            reason.strip(),
        )

    def update_system_field(
        self,
        user_id: str,
        field_name: str,
        value: Any,
        reason: str,
    ) -> UserProfile:
        """以 SYSTEM 权限直接更新画像字段（供应用内部逻辑使用）。

        仅 SYSTEM 允许的字段（如 onboarding_completed）走此入口。
        """
        return self.store.update_profile_field(
            user_id,
            field_name,
            value,
            ProfileActor.SYSTEM,
            reason,
        )

    def list_change_logs(self, user_id: str) -> list:
        """列出画像修改审计日志（从新到旧），供画像溯源界面使用。"""
        return self.store.list_change_logs(user_id)

    # ------------------------------------------------------------------
    # 内部：候选审核
    # ------------------------------------------------------------------

    def _process_proposal(
        self,
        proposal: ProfileCandidateProposal,
        user_message: str,
    ) -> ProfileProcessingResult:
        """审核单条候选，并执行自动应用、等待确认或拒绝。"""
        candidate = proposal.candidate

        try:
            policy = get_profile_field_policy(candidate.field)
        except ProfileValidationError as exc:
            return ProfileProcessingResult(
                field_name=candidate.field,
                proposed_value=candidate.value,
                status=ProfileCandidateStatus.REJECTED,
                message=str(exc),
            )

        # MODEL 不能直接写。这里用 USER 权限预校验值，是为了确认该字段在
        # 用户确认后是否可应用；系统专属字段会在此被拒绝。
        try:
            validate_profile_update(candidate.field, candidate.value, ProfileActor.USER)
        except ProfileValidationError as exc:
            rejected = self.store.save_candidate(
                user_id=candidate.user_id,
                field_name=candidate.field,
                proposed_value=candidate.value,
                source_actor=proposal.source_actor,
                confidence=candidate.confidence,
                reason=candidate.reason,
                needs_user_confirmation=True,
                source_session_id=candidate.evidence.session_id,
                source_message_id=self._first_message_id(candidate),
                source_excerpt=candidate.evidence.excerpt,
                status=ProfileCandidateStatus.REJECTED,
                resolution_reason=str(exc),
            )
            return ProfileProcessingResult(
                field_name=candidate.field,
                proposed_value=candidate.value,
                status=ProfileCandidateStatus.REJECTED,
                message=str(exc),
                candidate_id=rejected.candidate_id,
            )

        explicit_evidence_is_valid = self._has_valid_explicit_evidence(
            proposal,
            user_message,
        )
        can_auto_apply = (
            proposal.source_actor == ProfileActor.USER
            and ProfileActor.USER in policy.allowed_actors
            and candidate.confidence >= self.auto_apply_confidence
            and not candidate.needs_user_confirmation
            and explicit_evidence_is_valid
        )

        stored = self.store.save_candidate(
            user_id=candidate.user_id,
            field_name=candidate.field,
            proposed_value=candidate.value,
            source_actor=proposal.source_actor,
            confidence=candidate.confidence,
            reason=candidate.reason,
            needs_user_confirmation=not can_auto_apply,
            source_session_id=candidate.evidence.session_id,
            source_message_id=self._first_message_id(candidate),
            source_excerpt=candidate.evidence.excerpt,
        )

        if can_auto_apply:
            self.store.apply_candidate(
                candidate.user_id,
                stored.candidate_id,
                approved_by=ProfileActor.USER,
                resolution_reason="用户原话明确且候选通过自动审核",
            )
            return ProfileProcessingResult(
                field_name=candidate.field,
                proposed_value=candidate.value,
                status=ProfileCandidateStatus.APPLIED,
                message="用户明确提供的信息已写入画像",
                candidate_id=stored.candidate_id,
            )

        return ProfileProcessingResult(
            field_name=candidate.field,
            proposed_value=candidate.value,
            status=ProfileCandidateStatus.PENDING,
            message="候选需要用户确认后才能写入画像",
            candidate_id=stored.candidate_id,
        )

    @staticmethod
    def _has_valid_explicit_evidence(
        proposal: ProfileCandidateProposal,
        user_message: str,
    ) -> bool:
        """确认直接证据和候选值都能在用户原话中找到。

        仅检查 evidence 是用户原文的子串仍不够，因为模型可能复制一小段原文，
        却另外编造 proposed_value。自动应用前还要求文本候选值出现在证据或原话中。
        经过规范化转换但原文中不存在的值会进入 pending，等待用户确认。
        """
        quote = proposal.explicit_quote
        if quote is None or not quote.strip():
            return False
        cleaned_quote = quote.strip()
        if cleaned_quote not in user_message:
            return False

        value = proposal.candidate.value
        if not isinstance(value, str) or not value.strip():
            return False
        cleaned_value = value.strip()
        return cleaned_value in cleaned_quote or cleaned_value in user_message

    @staticmethod
    def _first_message_id(candidate: ProfileCandidate) -> str | None:
        """读取候选证据中的第一条消息 ID。"""
        if not candidate.evidence.message_ids:
            return None
        return candidate.evidence.message_ids[0]


# 兼容别名：老代码引用 LongMemoryService 的地方可平滑迁移
LongMemoryService = UserProfileManager


__all__ = [
    "UserProfileManager",
    "LongMemoryService",  # 兼容别名
    "ProfileCandidateExtractor",
    "OllamaProfileCandidateExtractor",
    "LLMProfileCandidateExtractor",
    "ProfileCandidateExtractionError",
    "ProfileCandidateProposal",
    "ProfileProcessingResult",
    "DEFAULT_PROFILE_MODEL",
    "DEFAULT_OLLAMA_BASE_URL",
]
