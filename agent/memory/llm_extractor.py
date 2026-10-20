"""LLM 合并提取器：一次调用同时产出会话摘要与长期记忆条目。

短期摘要（session_summary）与长期记忆提取共用一次 LLM 调用，避免
"摘要调一次、长期记忆又调一次"的重复开销。

输出通过固定分隔符分为两段：
    <<<SESSION_SUMMARY>>>
    会话摘要正文...
    <<<LONG_TERM_MEMORIES>>>
    - [semantic] 提炼后的记忆内容
    - [episodic] 提炼后的记忆内容

长期记忆条目的存储日期由系统在写入时统一盖章（created_date），
而不是依赖模型生成，保证日期确定、可信。

本提取器只做"理解对话 → 输出候选"，不负责持久化；写入见
LongTermMemory.write_llm_memories / VectorMemoryStore。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# 输出分段分隔符（提示词与解析共用同一常量）
SUMMARY_MARKER = "<<<SESSION_SUMMARY>>>"
MEMORIES_MARKER = "<<<LONG_TERM_MEMORIES>>>"

# 记忆类型白名单：模型只能输出这三种类型
MEMORY_TYPES = ("semantic", "episodic", "procedural")


class MemoryExtractionError(RuntimeError):
    """LLM 合并提取失败（调用异常、返回非法格式或字段不合法）时抛出。"""


@dataclass(frozen=True)
class LLMMemoryEntry:
    """一条由 LLM 提炼的长期记忆候选。

    字段：
        memory_type: semantic / episodic / procedural（写入时存为元数据）
        content: 提炼后的记忆内容（不是原文）
    """

    memory_type: str
    content: str


@dataclass
class ExtractionResult:
    """一次合并提取的结果。

    字段：
        summary: 新的会话摘要（短期记忆用）
        memories: 长期记忆条目列表（可为空，表示本轮无可存记忆）
    """

    summary: str
    memories: list[LLMMemoryEntry] = field(default_factory=list)


class LLMMemoryExtractor:
    """用当前聊天 LLM 把被裁剪的对话提炼为摘要 + 长期记忆条目。

    职责边界：
        - 只做"对话 → (摘要, 记忆条目)"的转换；
        - 不持久化、不读数据库、不校验权限；
        - 判断"哪些值得存"由 LLM 依据提示词规则完成
          （强触发词：记住/记得/请记得/以后/每次；以及偏好、事实、
          重要情绪事件、未完成计划等）。
    """

    def __init__(
        self,
        llm: Any,
        *,
        max_candidates: int = 10,
        fragment_max_chars: int = 200,
        summary_max_chars: int = 1000,
    ) -> None:
        """初始化提取器。

        参数：
            llm: LangChain LLM 实例（需支持 invoke([SystemMessage, HumanMessage])）
            max_candidates: 单次最多接受的长期记忆条目数
            fragment_max_chars: 单条消息喂给提示词的长度上限
            summary_max_chars: 摘要安全上限（超长输出硬截断）
        """
        if llm is None:
            raise ValueError("llm must not be None")
        if max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        self.llm = llm
        self.max_candidates = int(max_candidates)
        self.fragment_max_chars = int(fragment_max_chars)
        self.summary_max_chars = int(summary_max_chars)

    def extract(
        self,
        messages: list[tuple[str, str, str | None]],
        previous_summary: str = "",
        now: datetime | None = None,
    ) -> ExtractionResult:
        """从带时间戳的对话片段中提取摘要与长期记忆。

        参数：
            messages: [(speaker, content, time_str), ...] 被裁剪的对话片段；
                time_str 为可读时间（如 "2026-09-01 10:30"），无则 None。
            previous_summary: 旧会话摘要，用于滚动合并。
            now: 提取时刻，用于提示词中的"当前时间"；缺省取当前 UTC+8 时间。

        返回 ExtractionResult；LLM 调用失败或输出无法解析时抛
        MemoryExtractionError，由调用方决定降级策略。
        """
        if not messages:
            return ExtractionResult(summary=previous_summary)

        now = now or datetime.now()
        prompt = self._build_prompt(messages, previous_summary, now)
        try:
            response = self.llm.invoke(
                [
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception as exc:
            # 上层会记录此异常；服务商错误正文可能回显密钥，不拼接正文。
            raise MemoryExtractionError(f"LLM 合并提取调用失败: {type(exc).__name__}") from exc

        text = str(getattr(response, "content", "") or "").strip()
        if not text:
            raise MemoryExtractionError("LLM 合并提取返回了空内容")
        return self._parse(text)

    # ------------------------------------------------------------------
    # 提示词构建
    # ------------------------------------------------------------------

    _SYSTEM_PROMPT = (
        "你是记忆整理助手。请把给定的【历史摘要】与【本轮被裁剪的对话】合并提炼，"
        "一次性输出两部分：\n"
        "1. 会话摘要：保留主要话题、关键约定、情绪状态、未完成事项，丢弃寒暄与重复；\n"
        "2. 长期记忆条目：判断哪些信息值得跨会话长期记住，不能什么都存。\n"
        '值得存的情况包括：用户明确说"记住/记得/请记得"、明确偏好、重要事实、'
        "身份信息、情绪波动较大或有影响的经历、明确的未来计划或约定、"
        '有教育意义的"问题-解决"过程。\n'
        "每条长期记忆必须是：提炼后的理解（不是原文照抄），标注类型：\n"
        "    [semantic]   语义记忆：偏好、事实、身份、当前状态\n"
        "    [episodic]   情节记忆：一次有起承转合的经历/情绪事件\n"
        '    [procedural] 程序性候选：用户给出的未来交互规则（如"以后/每次...要"）\n'
        "没有值得长期记住的内容时，第二部分输出空。\n"
        "严格遵守输出格式，不要输出解释或 Markdown：\n"
        + SUMMARY_MARKER
        + "\n<会话摘要>\n"
        + MEMORIES_MARKER
        + "\n- [类型] 记忆内容\n- [类型] 记忆内容"
    )

    def _build_prompt(
        self,
        messages: list[tuple[str, str, str | None]],
        previous_summary: str,
        now: datetime,
    ) -> str:
        """把历史摘要、当前时间与对话片段拼成用户提示词。"""
        lines: list[str] = []
        for speaker, content, time_str in messages:
            text = content[: self.fragment_max_chars]
            if time_str:
                lines.append(f"{speaker}（{time_str}）：{text}")
            else:
                lines.append(f"{speaker}：{text}")
        return (
            f"当前时间：{now.strftime('%Y-%m-%d %H:%M')}\n\n"
            f"【历史摘要】\n{previous_summary or '（无）'}\n\n"
            f"【本轮被裁剪的对话】\n" + "\n".join(lines)
        )

    # ------------------------------------------------------------------
    # 输出解析
    # ------------------------------------------------------------------

    def _parse(self, text: str) -> ExtractionResult:
        """解析模型输出，兼容漏标分隔符与少量格式瑕疵。"""
        summary = ""
        memories: list[LLMMemoryEntry] = []

        if SUMMARY_MARKER in text:
            summary_text = text.split(SUMMARY_MARKER, 1)[1]
        else:
            summary_text = text

        if MEMORIES_MARKER in summary_text:
            summary_part, memories_part = summary_text.split(MEMORIES_MARKER, 1)
            summary = summary_part.strip()
            memories = self._parse_memories(memories_part)
        else:
            # 缺少 MEMORIES 标记：前半段视为摘要（若 SUMMARY 标记存在），
            # 后半段尝试按行解析记忆条目
            summary = summary_text.strip()
            memories = self._parse_memories(summary_text)

        summary = summary[: self.summary_max_chars].strip()
        return ExtractionResult(summary=summary, memories=memories[: self.max_candidates])

    def _parse_memories(self, text: str) -> list[LLMMemoryEntry]:
        """从记忆段文本中解析 "- [类型] 内容" 条目。"""
        entries: list[LLMMemoryEntry] = []
        pattern = re.compile(r"^\s*[-*]?\s*\[(\w+)\]\s*(.+?)\s*$")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = pattern.match(line)
            if match is None:
                continue
            memory_type = match.group(1).strip().lower()
            if memory_type not in MEMORY_TYPES:
                continue
            content = match.group(2).strip()
            if content:
                entries.append(LLMMemoryEntry(memory_type=memory_type, content=content))
        return entries


__all__ = [
    "LLMMemoryEntry",
    "ExtractionResult",
    "LLMMemoryExtractor",
    "MemoryExtractionError",
    "SUMMARY_MARKER",
    "MEMORIES_MARKER",
]
