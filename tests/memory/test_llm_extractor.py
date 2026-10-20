"""LLM 合并提取器单元测试：解析、分隔符、错误处理。"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from agent.memory.llm_extractor import (
    LLMMemoryEntry,
    LLMMemoryExtractor,
    MemoryExtractionError,
)


class StubLLM:
    """测试 LLM：返回预设文本或抛异常。"""

    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.prompt = ""

    def invoke(self, messages):
        self.prompt = messages[-1]["content"]
        if self.error is not None:
            raise self.error
        return AIMessage(content=self.text)


def _extractor(stub: StubLLM) -> LLMMemoryExtractor:
    return LLMMemoryExtractor(stub)


def test_parses_summary_and_memories_separated_by_markers():
    llm = StubLLM(
        text=(
            "<<<SESSION_SUMMARY>>>\n"
            "用户聊了咖啡与考试。\n"
            "<<<LONG_TERM_MEMORIES>>>\n"
            "- [semantic] 用户偏好：喜欢喝咖啡\n"
            "- [episodic] 用户今天因考试压力很大\n"
            "- [procedural] 以后回答前先安抚情绪\n"
        )
    )
    result = _extractor(llm).extract([("用户", "喜欢喝咖啡", None)])

    assert result.summary == "用户聊了咖啡与考试。"
    assert result.memories == [
        LLMMemoryEntry("semantic", "用户偏好：喜欢喝咖啡"),
        LLMMemoryEntry("episodic", "用户今天因考试压力很大"),
        LLMMemoryEntry("procedural", "以后回答前先安抚情绪"),
    ]


def test_empty_memories_when_none_worth_storing():
    llm = StubLLM(
        text=("<<<SESSION_SUMMARY>>>\n日常闲聊，无重要信息。\n<<<LONG_TERM_MEMORIES>>>\n")
    )
    result = _extractor(llm).extract([("用户", "你好", None)])

    assert result.summary == "日常闲聊，无重要信息。"
    assert result.memories == []


def test_ignores_unknown_memory_types_and_malformed_lines():
    llm = StubLLM(
        text=(
            "<<<SESSION_SUMMARY>>>\n摘要\n"
            "<<<LONG_TERM_MEMORIES>>>\n"
            "- [weird_type] 这种类型要忽略\n"
            "没有方括号的行也忽略\n"
            "- [semantic] 这条有效\n"
        )
    )
    result = _extractor(llm).extract([("用户", "内容", None)])

    assert result.memories == [LLMMemoryEntry("semantic", "这条有效")]


def test_prompt_includes_current_time_previous_summary_and_speaker_time():
    llm = StubLLM(text="<<<SESSION_SUMMARY>>>\n摘要\n<<<LONG_TERM_MEMORIES>>>\n")
    extractor = _extractor(llm)
    from datetime import datetime

    now = datetime(2026, 9, 1, 10, 30)
    extractor.extract(
        [("用户", "我在准备考试", "2026-09-01 10:00")],
        previous_summary="旧摘要",
        now=now,
    )

    assert "当前时间：2026-09-01 10:30" in llm.prompt
    assert "旧摘要" in llm.prompt
    assert "用户（2026-09-01 10:00）：我在准备考试" in llm.prompt


def test_summary_truncated_to_safety_limit():
    llm = StubLLM(text="<<<SESSION_SUMMARY>>>\n" + "长" * 2000 + "\n<<<LONG_TERM_MEMORIES>>>\n")
    result = _extractor(llm).extract([("用户", "内容", None)])
    assert len(result.summary) <= 1000


def test_llm_failure_raises_memory_extraction_error():
    llm = StubLLM(error=RuntimeError("连接失败 sensitive-api-key"))
    try:
        _extractor(llm).extract([("用户", "内容", None)])
    except MemoryExtractionError as exc:
        assert "RuntimeError" in str(exc)
        assert "sensitive-api-key" not in str(exc)
    else:  # pragma: no cover
        raise AssertionError("应当抛出 MemoryExtractionError")


def test_empty_response_raises_memory_extraction_error():
    llm = StubLLM(text="")
    try:
        _extractor(llm).extract([("用户", "内容", None)])
    except MemoryExtractionError:
        pass
    else:  # pragma: no cover
        raise AssertionError("空响应应当抛出 MemoryExtractionError")


def test_empty_messages_returns_previous_summary():
    llm = StubLLM(text="不应被调用")
    result = _extractor(llm).extract([], previous_summary="旧摘要")
    assert result.summary == "旧摘要"
    assert result.memories == []
