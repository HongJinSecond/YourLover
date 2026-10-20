from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from agent.affinity.affinity_store import AffinityStore
from agent.affinity.service import AffinityService
from agent.agent.ChatAgent import ChatAgent
from agent.memory.llm_extractor import (
    ExtractionResult,
    LLMMemoryEntry,
    MemoryExtractionError,
)
from agent.memory.long_term_memory import LongTermMemory
from agent.memory.profile_candidate import ProfileCandidate
from agent.memory.profile_service import (
    ProfileCandidateExtractionError,
    ProfileCandidateExtractor,
    ProfileCandidateProposal,
    UserProfileManager,
)
from agent.memory.profile_store import ProfileStore
from agent.memory.schema import MemoryEvidence
from agent.memory.session_store import SessionStore
from agent.memory.short_term_memory import ShortTermMemory
from agent.memory.store import FileMemoryStore
from agent.memory.user_profile import ProfileActor
from agent.prompts import DEFAULT_PERSONA_PROMPT, GLOBAL_SYSTEM_RULES
from utils.config_manager import ConfigManager


class RecordingLLM:
    """测试 LLM：记录每次收到的消息，并返回固定回复。"""

    def __init__(self) -> None:
        self.calls: list[list] = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return AIMessage(content="好的，我会温柔地陪着你。")


class ToolCallingLLM(RecordingLLM):
    """测试 LLM：第一轮返回工具调用，第二轮返回普通回复。"""

    def invoke(self, messages):
        self.calls.append(list(messages))
        if len(self.calls) == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator",
                        "args": {"expression": "3**2+5"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="计算结果是 14。")


class StubCandidateExtractor(ProfileCandidateExtractor):
    """测试提取器：返回固定候选或抛固定异常，避免测试连接真实 Ollama。

    参数：
        proposals: 固定返回的候选列表（默认为空，即不产生候选）。
        error: 非 None 时 extract 抛出该异常（模拟 Ollama 不可用）。
    """

    def __init__(self, proposals=None, error: Exception | None = None) -> None:
        self.proposals = proposals or []
        self.error = error
        self.call_count = 0

    def extract(
        self,
        user_id: str,
        user_message: str,
        session_id: str | None = None,
        message_id: str | None = None,
    ) -> list[ProfileCandidateProposal]:
        self.call_count += 1
        if self.error is not None:
            raise self.error
        return self.proposals


class StubMemoryExtractor:
    """测试合并提取器：返回固定摘要+记忆条目，或抛异常模拟提取失败。

    参数：
        summary: 固定返回的会话摘要。
        memories: 固定返回的长期记忆条目列表。
        error: 非 None 时 extract 抛出该异常（模拟 LLM 不可用）。
    """

    def __init__(self, summary="", memories=None, error: Exception | None = None) -> None:
        self.summary = summary
        self.memories = memories or []
        self.error = error
        self.calls: list = []  # 记录每次 extract 的 (fragments, previous_summary)

    def extract(self, fragments, previous_summary=""):
        self.calls.append((fragments, previous_summary))
        if self.error is not None:
            raise self.error
        return ExtractionResult(summary=self.summary, memories=self.memories)


class FailingMemoryExtractor(StubMemoryExtractor):
    """默认测试提取器：始终抛异常，强制走规则提取 + 规则摘要降级路径。"""

    def __init__(self) -> None:
        super().__init__(error=MemoryExtractionError("测试环境禁用 LLM 提取"))


class AffectionateLLM(RecordingLLM):
    """测试 LLM：回复末尾带好感度增量标记（模拟 system prompt 约定格式）。

    好感度标记用双花括号 {{+N}} / {{-N}}，与情绪标记 <<Happy>> 的双尖括号
    分属两套符号，避免正则解析冲突。

    参数：
        delta: 附加的好感度增量（可为负，如 -3 → {{-3}}）。
    """

    def __init__(self, delta: int = 5) -> None:
        super().__init__()
        self.delta = delta

    def invoke(self, messages):
        self.calls.append(list(messages))
        sign = "+" if self.delta >= 0 else ""
        # 好感度标记用字面双花括号 {{+N}}，与情绪标记 <<Happy>> 符号分离
        content = "好的，我会温柔地陪着你。<<Happy>>" + "{{" + sign + str(self.delta) + "}}"
        return AIMessage(content=content)


def _agent(tmp_path, llm=None, *, extractor=None, **short_term_kwargs):
    """构造使用临时存储的 ChatAgent 与三大管理器。

    short_term_kwargs 可传 max_messages / keep_recent_turns 定制窗口参数。
    画像候选提取器默认用桩（不产生候选），需要候选行为的测试单独替换
    agent.user_profile_manager.candidate_extractor。
    合并提取器默认 FailingMemoryExtractor（强制走规则降级路径），
    需要 LLM 提取行为的测试传入 StubMemoryExtractor。
    好感度增量由回复中的 {{+N}} 标记驱动（ChatAgent 解析，无独立判定器）；
    默认 RecordingLLM 回复不带标记（好感度不变），带标记行为用 AffectionateLLM。
    """
    long_term = LongTermMemory(store=FileMemoryStore(tmp_path / "users"))
    short_term = ShortTermMemory(
        session_store=SessionStore(tmp_path / "sessions.sqlite3"),
        **short_term_kwargs,
    )
    profile = UserProfileManager(
        store=ProfileStore(tmp_path / "profiles.sqlite3"),
        candidate_extractor=StubCandidateExtractor(),
    )
    affinity = AffinityService(store=AffinityStore(tmp_path / "affinity.sqlite3"))
    agent = ChatAgent(
        llm or RecordingLLM(),
        ConfigManager(),
        {"configurable": {"thread_id": "default"}},
        short_term_memory=short_term,
        long_term_memory=long_term,
        user_profile_manager=profile,
        affinity_service=affinity,
        memory_extractor=extractor or FailingMemoryExtractor(),
    )
    return agent, long_term, affinity


def test_injected_configuration_controls_summary_and_fragment_limits(tmp_path):
    """摘要成功和降级分支都使用注入配置，不读取全局参数。"""
    for label, extractor in [
        ("success", StubMemoryExtractor(summary="一二三四五六七八九十")),
        ("fallback", FailingMemoryExtractor()),
    ]:
        agent, _, _ = _agent(
            tmp_path / label,
            extractor=extractor,
            max_messages=4,
            keep_recent_turns=1,
        )
        agent.config.update({"SUMMARY_MAX_CHARS": 7, "SUMMARY_FRAGMENT_MAX_CHARS": 3}, save=False)
        for turn in ["第一轮较长消息", "第二轮较长消息", "第三轮较长消息"]:
            state = agent.invoke({"user_input": turn, "user_id": "user-a", "session_id": label})
        assert extractor.calls
        assert all(
            len(content) <= 3 for content in [fragment[1] for fragment in extractor.calls[0][0]]
        )
        assert len(state["session_summary"]) == 7


def test_persona_is_snapshotted_per_turn_and_global_rules_survive_tool_calls(tmp_path):
    class ChangingPersonaLLM(ToolCallingLLM):
        def invoke(self, messages):
            if not self.calls:
                agent.config.update({"CUSTOM_PERSONA_PROMPT": "下一轮的摄影师阿林"}, save=False)
            return super().invoke(messages)

    llm = ChangingPersonaLLM()
    agent, _, _ = _agent(tmp_path, llm)
    agent.config.update(
        {"PERSONA_PROMPT_MODE": "custom", "CUSTOM_PERSONA_PROMPT": "本轮的老师小陈"}, save=False
    )
    payload = {"user_input": "计算一下 3**2+5", "user_id": "user-a", "session_id": "prompt-test"}
    agent.invoke(payload)
    assert len(llm.calls) == 2
    for messages in llm.calls:
        assert "本轮的老师小陈" in messages[0].content
        assert "下一轮的摄影师阿林" not in messages[0].content
        assert messages[0].content.endswith(GLOBAL_SYSTEM_RULES)
    agent.invoke({**payload, "user_input": "你好"})
    assert "下一轮的摄影师阿林" in llm.calls[-1][0].content
    assert "本轮的老师小陈" not in llm.calls[-1][0].content
    agent.config.update({"PERSONA_PROMPT_MODE": "default"}, save=False)
    agent.invoke({**payload, "user_input": "再见"})
    assert DEFAULT_PERSONA_PROMPT in llm.calls[-1][0].content
    assert llm.calls[-1][0].content.endswith(GLOBAL_SYSTEM_RULES)


def test_llm_extraction_writes_memory_and_recalled_in_new_session(tmp_path):
    """LLM 合并提取产出的长期记忆应写入并在新会话被召回。"""
    llm = RecordingLLM()
    extractor = StubMemoryExtractor(
        summary="摘要：用户聊了咖啡话题。",
        memories=[LLMMemoryEntry("semantic", "用户偏好：喜欢喝咖啡")],
    )
    agent, long_term, affinity = _agent(
        tmp_path, llm, extractor=extractor, max_messages=4, keep_recent_turns=1
    )
    session1 = {"configurable": {"thread_id": "session-1", "user_id": "user-a"}}
    agent.invoke({"user_input": "我先说个事"}, session=session1)
    agent.invoke({"user_input": "接着说"}, session=session1)
    third = agent.invoke({"user_input": "记住我喜欢喝咖啡"}, session=session1)

    # 第三轮触发裁剪：记忆条目已写入（无向量库 → JSON 降级），摘要来自 LLM
    memories = long_term.list("user-a")
    assert len(memories) >= 1
    assert any("喝咖啡" in memory.content for memory in memories)
    assert third["session_summary"] == "摘要：用户聊了咖啡话题。"

    # 新会话按关键词召回
    second = agent.invoke(
        {
            "user_input": "今天要喝咖啡吗",
            "user_id": "user-a",
            "session_id": "session-2",
        }
    )
    assert "喝咖啡" in second["memory_context"]
    system_prompt = llm.calls[-1][0]
    assert isinstance(system_prompt, SystemMessage)
    assert "喝咖啡" in system_prompt.content
    # 当前时间已注入 system prompt
    assert "当前时间：" in system_prompt.content
    assert affinity.get_events("user-a")[-1].event["type"] == "memory_write_completed"


def test_long_term_memory_is_isolated_by_user_id(tmp_path):
    """用户 A 的记忆不应被用户 B 检索到。"""
    llm = RecordingLLM()
    extractor = StubMemoryExtractor(
        summary="摘要：A 用户对话。",
        memories=[LLMMemoryEntry("semantic", "用户偏好：喜欢喝咖啡")],
    )
    agent, long_term, _ = _agent(
        tmp_path, llm, extractor=extractor, max_messages=4, keep_recent_turns=1
    )
    session_a = {"configurable": {"thread_id": "session-a", "user_id": "user-a"}}
    agent.invoke({"user_input": "第一轮"}, session=session_a)
    agent.invoke({"user_input": "第二轮"}, session=session_a)
    agent.invoke({"user_input": "第三轮"}, session=session_a)  # 触发裁剪，写入 A 的记忆

    state_b = agent.invoke(
        {
            "user_input": "喝咖啡",
            "user_id": "user-b",
            "session_id": "session-b",
        }
    )

    assert state_b["memory_context"] == ""
    assert long_term.list("user-b") == []
    assert "喝咖啡" not in llm.calls[-1][0].content


def test_same_session_refreshes_long_term_memory_every_turn(tmp_path):
    """同一 Session 的后续轮次也应刷新长期记忆检索（每轮刷新约定）。"""
    llm = RecordingLLM()
    agent, long_term, _ = _agent(tmp_path, llm, max_messages=4, keep_recent_turns=1)
    # 预置一条记忆，验证每轮检索都会刷新（而非只在加载时）
    long_term.write_llm_memories("user-a", [LLMMemoryEntry("semantic", "用户偏好：喜欢喝咖啡")])
    session = {"configurable": {"thread_id": "same", "user_id": "user-a"}}

    first = agent.invoke({"user_input": "喝咖啡"}, session=session)
    second = agent.invoke({"user_input": "咖啡"}, session=session)

    assert "喝咖啡" in first["memory_context"]
    assert "喝咖啡" in second["memory_context"]


def test_profile_context_is_injected_into_system_prompt(tmp_path):
    """用户画像字段应每轮注入 system prompt。"""
    llm = RecordingLLM()
    agent, _, _ = _agent(tmp_path, llm)

    agent.user_profile_manager.update_user_field(
        "user-a", "preferred_address", "队长", "测试直接写入画像"
    )

    agent.invoke(
        {
            "user_input": "你好",
            "user_id": "user-a",
            "session_id": "session-1",
        }
    )

    system_prompt = llm.calls[-1][0]
    assert isinstance(system_prompt, SystemMessage)
    assert "希望的称呼：队长" in system_prompt.content


def _nickname_proposal(user_id: str, value: str, quote: str) -> ProfileCandidateProposal:
    """构造用户明确提供昵称的画像候选（模拟 Ollama 抽取结果）。"""
    candidate = ProfileCandidate(
        user_id=user_id,
        field="nickname",
        value=value,
        reason="用户明确修改昵称",
        needs_user_confirmation=False,
        confidence=0.98,
        evidence=MemoryEvidence(
            session_id="session-1",
            message_ids=[],
            excerpt=quote,
        ),
    )
    return ProfileCandidateProposal(
        candidate=candidate,
        source_actor=ProfileActor.USER,
        explicit_quote=quote,
    )


def test_explicit_profile_statement_creates_and_applies_candidate(tmp_path):
    """用户明确说出画像信息时，回复后应自动产生并应用画像候选。"""
    llm = RecordingLLM()
    agent, _, _ = _agent(tmp_path, llm)
    agent.user_profile_manager.candidate_extractor = StubCandidateExtractor(
        [_nickname_proposal("user-a", "阿辰", "以后叫我阿辰")]
    )

    state = agent.invoke(
        {
            "user_input": "以后叫我阿辰",
            "user_id": "user-a",
            "session_id": "session-1",
        }
    )

    # 候选已产生并自动应用（用户明确表达 + 置信度达标）
    assert len(state["profile_candidate_ids"]) == 1
    assert agent.user_profile_manager.get_profile("user-a").nickname == "阿辰"

    # 下一轮 system prompt 应注入已生效的画像
    agent.user_profile_manager.candidate_extractor = StubCandidateExtractor()
    agent.invoke({"user_input": "你好", "user_id": "user-a", "session_id": "session-1"})
    assert "昵称：阿辰" in llm.calls[-1][0].content


def test_profile_extraction_failure_does_not_break_chat(tmp_path):
    """画像候选提取失败（如 Ollama 不可用）时，聊天主流程不应中断。"""
    llm = RecordingLLM()
    agent, _, _ = _agent(tmp_path, llm)
    agent.user_profile_manager.candidate_extractor = StubCandidateExtractor(
        error=ProfileCandidateExtractionError("无法连接画像候选模型")
    )

    state = agent.invoke(
        {"user_input": "以后叫我阿辰", "user_id": "user-a", "session_id": "session-1"}
    )

    assert state["messages"][-1].content == "好的，我会温柔地陪着你。"
    assert state["profile_candidate_ids"] == []
    assert agent.user_profile_manager.get_profile("user-a").nickname is None


def test_non_profile_message_skips_profile_extraction(tmp_path):
    """未命中画像触发词的消息不应调用候选模型（性能优化）。"""
    llm = RecordingLLM()
    agent, _, _ = _agent(tmp_path, llm)
    extractor = StubCandidateExtractor([_nickname_proposal("user-a", "阿辰", "以后叫我阿辰")])
    agent.user_profile_manager.candidate_extractor = extractor

    # 普通闲聊：不命中触发词 → 提取器不被调用，画像不变
    state = agent.invoke(
        {"user_input": "今天天气真不错", "user_id": "user-a", "session_id": "session-1"}
    )
    assert extractor.call_count == 0
    assert state["profile_candidate_ids"] == []
    assert agent.user_profile_manager.get_profile("user-a").nickname is None

    # 命中触发词：提取器被调用，候选产生
    state = agent.invoke(
        {"user_input": "以后叫我阿辰", "user_id": "user-a", "session_id": "session-1"}
    )
    assert extractor.call_count == 1
    assert len(state["profile_candidate_ids"]) == 1
    assert agent.user_profile_manager.get_profile("user-a").nickname == "阿辰"


def test_short_term_messages_are_trimmed_by_complete_turn(tmp_path):
    llm = RecordingLLM()
    extractor = StubMemoryExtractor(summary="摘要：用户进行了三轮日常对话测试。")
    agent, _, _ = _agent(tmp_path, llm, extractor=extractor, max_messages=4, keep_recent_turns=1)
    session = {"configurable": {"thread_id": "trim-session", "user_id": "user-a"}}

    agent.invoke({"user_input": "第一轮"}, session=session)
    agent.invoke({"user_input": "第二轮"}, session=session)
    state = agent.invoke({"user_input": "第三轮"}, session=session)

    assert [message.content for message in state["messages"]] == [
        "第三轮",
        "好的，我会温柔地陪着你。",
    ]
    # 摘要来自 LLM 合并提取（而非原文截断拼接）
    assert state["session_summary"] == "摘要：用户进行了三轮日常对话测试。"
    # 提取器收到了被裁剪的对话片段（含发生时间）
    fragments, previous_summary = extractor.calls[-1]
    assert any("第一轮" in content for _, content, _ in fragments)
    assert previous_summary == ""


def test_extraction_failure_falls_back_to_rule_based_excerpt(tmp_path):
    """LLM 合并提取失败时降级为规则提取 + 规则摘要，聊天与裁剪不受影响。"""
    llm = RecordingLLM()
    extractor = StubMemoryExtractor(error=MemoryExtractionError("LLM 不可用"))
    agent, long_term, _ = _agent(
        tmp_path, llm, extractor=extractor, max_messages=4, keep_recent_turns=1
    )
    session = {"configurable": {"thread_id": "trim-session", "user_id": "user-a"}}

    agent.invoke({"user_input": "第一轮"}, session=session)
    agent.invoke({"user_input": "第二轮"}, session=session)
    state = agent.invoke({"user_input": "第三轮"}, session=session)

    # 聊天主流程未中断，窗口裁剪正常
    assert [message.content for message in state["messages"]] == [
        "第三轮",
        "好的，我会温柔地陪着你。",
    ]
    # 降级摘要保留被裁剪消息的原文摘录，且长度受限
    assert "第一轮" in state["session_summary"]
    assert "第二轮" in state["session_summary"]
    assert len(state["session_summary"]) <= 1000
    # 降级路径未写入任何长期记忆（普通寒暄不命中规则），memory_write_ids 为空
    assert state["memory_write_ids"] == []
    assert long_term.list("user-a") == []


def test_affinity_starts_at_zero_and_increases_each_round(tmp_path):
    """好感度初始为 0，回复带 {{+N}} 标记时按增量累加，且用户间隔离。"""
    llm = AffectionateLLM(delta=5)
    agent, _, affinity = _agent(tmp_path, llm)

    # 初始为 0（第一轮 system prompt 里也是 0）
    first = agent.invoke({"user_input": "你好呀", "user_id": "user-a", "session_id": "session-1"})
    # 返回的 love_snapshot 是轮末值（judge 节点已解析标记并累加写回 state）
    assert first["love_snapshot"] == 5
    assert affinity.get_love_snapshot("user-a") == 5

    # 第二轮：快照读到上轮累加值，且本轮继续累加
    second = agent.invoke(
        {"user_input": "今天过得开心吗", "user_id": "user-a", "session_id": "session-1"}
    )
    assert second["love_snapshot"] == 10
    assert affinity.get_love_snapshot("user-a") == 10

    # 用户隔离：b 仍是 0
    assert affinity.get_love_snapshot("user-b") == 0


def test_affinity_delta_is_clamped_and_capped_at_100(tmp_path):
    """单轮超限增量被钳制到 ±10，累计好感度上限为 100。"""
    llm = AffectionateLLM(delta=15)  # 超过单轮上限，应被钳制到 +10
    agent, _, affinity = _agent(tmp_path, llm)
    # 预置好感度到 95，验证上限收敛
    affinity.apply_delta("user-a", 95)
    session = {"configurable": {"thread_id": "cap", "user_id": "user-a"}}

    first = agent.invoke({"user_input": "第一轮"}, session=session)  # 95+10 -> 100
    assert first["love_snapshot"] == 100
    assert affinity.get_love_snapshot("user-a") == 100

    # 已到上限，继续累加不再增长
    second = agent.invoke({"user_input": "第二轮"}, session=session)
    assert second["love_snapshot"] == 100
    assert affinity.get_love_snapshot("user-a") == 100


def test_negative_delta_clamped_at_zero(tmp_path):
    """回复带 <<-N>> 负增量标记时扣减好感度，但下限为 0。"""
    llm = AffectionateLLM(delta=-3)
    agent, _, affinity = _agent(tmp_path, llm)
    state = agent.invoke({"user_input": "你好", "user_id": "user-a", "session_id": "session-1"})

    # 0 + (-3) 被钳制到下限 0，聊天不中断
    assert state["messages"][-1].content == "好的，我会温柔地陪着你。<<Happy>>{{-3}}"
    assert affinity.get_love_snapshot("user-a") == 0


def test_reply_without_delta_mark_keeps_love_unchanged(tmp_path):
    """回复未带增量标记（模型忘记输出）时，好感度不变，聊天不中断。"""
    llm = RecordingLLM()
    agent, _, affinity = _agent(tmp_path, llm)
    state = agent.invoke({"user_input": "你好", "user_id": "user-a", "session_id": "session-1"})

    assert state["messages"][-1].content == "好的，我会温柔地陪着你。"
    assert affinity.get_love_snapshot("user-a") == 0


def test_existing_tool_loop_still_returns_tool_result_to_llm(tmp_path):
    llm = ToolCallingLLM()
    agent, _, _ = _agent(tmp_path, llm)

    state = agent.invoke({"user_input": "计算 3**2+5", "user_id": "user-a", "session_id": "tools"})

    assert state["messages"][-1].content == "计算结果是 14。"
    assert any(
        isinstance(message, ToolMessage) and message.content == "14" for message in llm.calls[1]
    )
