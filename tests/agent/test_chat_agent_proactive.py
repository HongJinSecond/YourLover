"""ChatAgent 主动消息轮（proactive_context）集成测试。

验证隐藏指令链路：
    - 指令不出现在任何消息里（只进 system prompt，且本轮后清空）；
    - 主动轮不产生用户 HumanMessage / MemoryItem；
    - 主动回复照常进入 history 与 SQLite（她记得自己说过什么）；
    - 主动轮跳过好感度判定与画像候选提取；
    - 主动轮之后正常轮不受影响。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from agent.affinity.affinity_store import AffinityStore
from agent.affinity.service import AffinityService
from agent.agent.ChatAgent import ChatAgent
from agent.memory.long_term_memory import LongTermMemory
from agent.memory.profile_service import UserProfileManager
from agent.memory.profile_store import ProfileStore
from agent.memory.session_store import SessionStore
from agent.memory.short_term_memory import ShortTermMemory
from agent.memory.store import FileMemoryStore
from agent.prompts import GLOBAL_SYSTEM_RULES
from utils.config_manager import ConfigManager


class RecordingLLM:
    """测试 LLM：记录每次收到的完整消息，回复固定带增量标记。"""

    def __init__(self, reply: str = "在忙吗？团子又蹲在画室门口了<<Happy>>{{+5}}") -> None:
        self.calls: list[list] = []
        self.reply = reply

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return AIMessage(content=self.reply)


class ScriptedLLM:
    """测试 LLM：按脚本依次返回回复（模拟 LLM 偶发空回复）。"""

    def __init__(self, replies: list[str]) -> None:
        self.call_count = 0
        self.replies = list(replies)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        reply = self.replies[min(self.call_count, len(self.replies) - 1)]
        self.call_count += 1
        # 支持脚本里直接放 AIMessage（携带 response_metadata 等自定义信息）
        return reply if isinstance(reply, AIMessage) else AIMessage(content=reply)


class CountingProfileManager(UserProfileManager):
    """画像管理器桩：统计 process_user_message 被调用的次数。"""

    def __init__(self, store):  # type: ignore[no-untyped-def]
        super().__init__(store=store)
        self.call_count = 0

    def process_user_message(self, *args, **kwargs):  # type: ignore[override]
        self.call_count += 1
        return []


def _agent(tmp_path, llm=None):
    """构造使用临时存储的 ChatAgent（好感度初始 0）。"""
    affinity = AffinityService(store=AffinityStore(tmp_path / "affinity.sqlite3"))
    profile = CountingProfileManager(ProfileStore(tmp_path / "profiles.sqlite3"))
    agent = ChatAgent(
        llm or RecordingLLM(),
        ConfigManager(),
        {"configurable": {"thread_id": "default"}},
        short_term_memory=ShortTermMemory(
            session_store=SessionStore(tmp_path / "sessions.sqlite3"),
        ),
        long_term_memory=LongTermMemory(store=FileMemoryStore(tmp_path / "users")),
        user_profile_manager=profile,
        affinity_service=affinity,
        memory_extractor=_FailingExtractor(),
    )
    return agent, affinity, profile


class _FailingExtractor:
    """合并提取器桩：始终失败，强制走规则降级路径（不依赖 LLM）。"""

    def extract(self, fragments, previous_summary=""):
        from agent.memory.llm_extractor import MemoryExtractionError

        raise MemoryExtractionError("测试环境禁用 LLM 提取")


INSTRUCTION = "距离用户上次回复已经过去了约 2.0 小时，这是你第一次主动找他……"


def test_custom_persona_and_global_rules_are_applied_to_proactive_messages(tmp_path):
    llm = RecordingLLM()
    agent, _, _ = _agent(tmp_path, llm)
    agent.config.update(
        {"PERSONA_PROMPT_MODE": "custom", "CUSTOM_PERSONA_PROMPT": "你是爱摄影的阿林。"}, save=False
    )
    agent.invoke({"user_id": "u1", "session_id": "s1", "proactive_context": INSTRUCTION})
    prompt = llm.calls[-1][0].content
    assert "你是爱摄影的阿林。" in prompt
    assert INSTRUCTION in prompt
    assert prompt.index(INSTRUCTION) < prompt.index(GLOBAL_SYSTEM_RULES)
    assert prompt.endswith(GLOBAL_SYSTEM_RULES)


def test_proactive_turn_hides_instruction_and_skips_user_message(tmp_path):
    """主动轮：指令只进 system prompt，不产生用户消息；回复带指令情境。"""
    llm = RecordingLLM()
    agent, _, _ = _agent(tmp_path, llm)
    session = {"configurable": {"thread_id": "s1", "user_id": "u1"}}

    # 先正常聊一轮，会话里已有历史
    agent.invoke({"user_input": "在吗", "user_id": "u1", "session_id": "s1"}, session=session)
    assert isinstance(llm.calls[-1][-1], HumanMessage)  # 正常轮有用户消息

    state = agent.invoke(
        {
            "user_input": "",
            "user_id": "u1",
            "session_id": "s1",
            "proactive_context": INSTRUCTION,
        },
        session=session,
    )

    # 指令进了 system prompt（第一条消息），但没有进任何对话消息
    system_prompt = llm.calls[-1][0].content
    assert INSTRUCTION in system_prompt
    assert "【主动消息指令" in system_prompt
    assert all(INSTRUCTION not in str(m.content) for m in llm.calls[-1][1:])

    # 主动轮没有新增用户 HumanMessage（最后一条是 AI 主动消息）
    messages = state["messages"]
    assert isinstance(messages[-1], AIMessage)
    assert not isinstance(messages[-1], HumanMessage)
    # 主动轮补的占位 HumanMessage 写回 messages，保证 AI 之间交替合法
    # （否则后续正常轮 LLM 输入出现 [..., ai, ai, human] 连续 AI，真实模型会返回空）
    assert any(isinstance(m, HumanMessage) and "他没有回复" in str(m.content) for m in messages)

    # 指令用完即清空，不残留到 state
    assert state.get("proactive_context") == ""


def test_messages_have_no_consecutive_ai_after_proactive(tmp_path):
    """主动轮后 messages 中任何 AI 消息之间都有 Human 分隔（回归 bug）。"""
    llm = RecordingLLM()
    agent, _, _ = _agent(tmp_path, llm)
    session = {"configurable": {"thread_id": "s1", "user_id": "u1"}}
    agent.invoke({"user_input": "在吗", "user_id": "u1", "session_id": "s1"}, session=session)
    state = agent.invoke(
        {
            "user_input": "",
            "user_id": "u1",
            "session_id": "s1",
            "proactive_context": INSTRUCTION,
        },
        session=session,
    )
    state = agent.invoke(
        {"user_input": "我回来啦", "user_id": "u1", "session_id": "s1", "proactive_context": ""},
        session=session,
    )
    types = [type(m) for m in state["messages"]]
    assert not any(
        types[i] is AIMessage and types[i + 1] is AIMessage for i in range(len(types) - 1)
    ), f"LLM 输入出现连续 AI 消息：{types}"

    # 占位 HumanMessage 不进入 history（不污染 SQLite/UI 恢复）
    history_speakers = [str(getattr(item, "speaker", "")).lower() for item in state["history"]]
    assert history_speakers.count("user") == 2  # 两轮真实用户消息
    assert all(
        "他没有回复" not in str(getattr(item.message, "content", "")) for item in state["history"]
    )


def test_messages_from_history_inserts_placeholder_for_consecutive_ai(tmp_path):
    """从 SQLite 恢复历史时，连续 AI 之间补占位（重启后同样不会空回复）。"""
    agent, _, _ = _agent(tmp_path)
    session = {"configurable": {"thread_id": "s1", "user_id": "u1"}}
    agent.invoke({"user_input": "在吗", "user_id": "u1", "session_id": "s1"}, session=session)
    agent.invoke(
        {
            "user_input": "",
            "user_id": "u1",
            "session_id": "s1",
            "proactive_context": INSTRUCTION,
        },
        session=session,
    )

    # 模拟重启：用同一个临时库新建一个 agent（不注入 RecordingLLM 历史，从 SQLite 恢复）
    recovered = ChatAgent(
        RecordingLLM(),
        ConfigManager(),
        {"configurable": {"thread_id": "s1", "user_id": "u1"}},
        short_term_memory=agent.short_term_memory,
        long_term_memory=agent.long_term_memory,
        user_profile_manager=agent.user_profile_manager,
        affinity_service=agent.affinity_service,
        memory_extractor=_FailingExtractor(),
    )
    history = recovered.short_term_memory.load_messages("u1", "s1")
    messages = ChatAgent._messages_from_history(history)
    types = [type(m) for m in messages]
    assert not any(
        types[i] is AIMessage and types[i + 1] is AIMessage for i in range(len(types) - 1)
    ), f"恢复后 messages 出现连续 AI：{types}"
    # 恢复后的占位同样不污染 SQLite
    items = recovered.short_term_memory.load_messages("u1", "s1")
    assert all("他没有回复" not in str(getattr(i.message, "content", "")) for i in items)


def test_proactive_turn_skips_affinity_and_profile(tmp_path):
    """主动轮：回复带 {{+5}} 也不加分；画像提取不被调用。"""
    llm = RecordingLLM()  # 回复带 {{+5}}
    agent, affinity, profile = _agent(tmp_path, llm)
    session = {"configurable": {"thread_id": "s1", "user_id": "u1"}}
    agent.invoke({"user_input": "在吗", "user_id": "u1", "session_id": "s1"}, session=session)
    baseline_love = affinity.get_love_snapshot("u1")  # 第一轮正常对话已 +5
    baseline_profile_calls = profile.call_count

    agent.invoke(
        {
            "user_input": "",
            "user_id": "u1",
            "session_id": "s1",
            "proactive_context": INSTRUCTION,
        },
        session=session,
    )

    assert affinity.get_love_snapshot("u1") == baseline_love  # {{+5}} 被忽略
    assert profile.call_count == baseline_profile_calls


def test_proactive_reply_persisted_and_restored(tmp_path):
    """主动消息保存进 SQLite，恢复会话时可见（她记得自己主动说过什么）。"""
    agent, _, _ = _agent(tmp_path)
    session = {"configurable": {"thread_id": "s1", "user_id": "u1"}}
    agent.invoke({"user_input": "在吗", "user_id": "u1", "session_id": "s1"}, session=session)
    agent.invoke(
        {
            "user_input": "",
            "user_id": "u1",
            "session_id": "s1",
            "proactive_context": INSTRUCTION,
        },
        session=session,
    )

    items = agent.short_term_memory.load_messages("u1", "s1")
    ai_items = [i for i in items if i.speaker == "AI"]
    assert len(ai_items) == 2  # 正常轮回复 + 主动消息
    user_items = [i for i in items if i.speaker == "User"]
    assert len(user_items) == 1  # 主动轮没有新增用户消息


def test_normal_turn_after_proactive_works(tmp_path):
    """主动轮之后的正常轮：用户消息正常入库，好感度标记恢复判定。"""
    llm = RecordingLLM()
    agent, affinity, _ = _agent(tmp_path, llm)
    session = {"configurable": {"thread_id": "s1", "user_id": "u1"}}
    agent.invoke({"user_input": "在吗", "user_id": "u1", "session_id": "s1"}, session=session)
    agent.invoke(
        {
            "user_input": "",
            "user_id": "u1",
            "session_id": "s1",
            "proactive_context": INSTRUCTION,
        },
        session=session,
    )

    # 正常轮：显式传 proactive_context=""（AgentController.chat 的行为）
    state = agent.invoke(
        {
            "user_input": "我回来啦",
            "user_id": "u1",
            "session_id": "s1",
            "proactive_context": "",
        },
        session=session,
    )
    assert isinstance(state["messages"][-2], HumanMessage)
    assert state["messages"][-2].content == "我回来啦"
    # 好感度 = 第一轮 +5，主动轮 +0，本轮正常判定再 +5
    assert affinity.get_love_snapshot("u1") == 10
    # 正常轮 system prompt 不再包含主动指令注入块
    # （基础提示词【主动消息模式】节本身会提到"主动消息指令"字样，须用注入块专属标题判断）
    assert "【主动消息指令（系统内部指令" not in llm.calls[-1][0].content


def test_system_prompt_contains_last_reply_time(tmp_path):
    """system prompt 注入"上次用户回复时间"行（时间感知）。"""
    llm = RecordingLLM()
    agent, _, _ = _agent(tmp_path, llm)
    session = {"configurable": {"thread_id": "s1", "user_id": "u1"}}
    agent.invoke({"user_input": "在吗", "user_id": "u1", "session_id": "s1"}, session=session)

    system_prompt = llm.calls[-1][0].content
    assert "上次用户回复时间" in system_prompt
    # 第一次对话时还没有更早的用户消息 → 跨会话查库也为空 → 首次对话说明
    assert "第一次对话" in system_prompt


def test_last_user_activity_time_across_sessions(tmp_path):
    """get_last_user_activity_time：跨会话返回最后一条用户消息时间。"""
    agent, _, _ = _agent(tmp_path)
    store = agent.short_term_memory.session_store
    assert store.get_last_user_activity_time("u1") is None  # 从未发言

    session = {"configurable": {"thread_id": "s1", "user_id": "u1"}}
    agent.invoke({"user_input": "你好", "user_id": "u1", "session_id": "s1"}, session=session)
    first = store.get_last_user_activity_time("u1")
    assert first is not None

    session2 = {"configurable": {"thread_id": "s2", "user_id": "u1"}}
    agent.invoke({"user_input": "又在吗", "user_id": "u1", "session_id": "s2"}, session=session2)
    second = store.get_last_user_activity_time("u1")
    assert second is not None and second >= first


def test_is_blank_ai_reply_ignores_tool_calls():
    """带 tool_calls 的空 content 是合法工具调用响应，不算空回复。"""
    from agent.agent.ChatAgent import _is_blank_ai_reply

    assert _is_blank_ai_reply(AIMessage(content="  "))  # 纯空白视为空
    assert not _is_blank_ai_reply(
        AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}])
    )
    assert not _is_blank_ai_reply(AIMessage(content="你好"))


def test_blank_reply_retried_and_recovered(tmp_path):
    """LLM 首次返回空回复时自动重试一次，第二次成功则使用正常回复。"""
    llm = ScriptedLLM(["", "在忙呀？"])
    agent, _, _ = _agent(tmp_path, llm)
    session = {"configurable": {"thread_id": "s1", "user_id": "u1"}}
    state = agent.invoke(
        {"user_input": "在吗", "user_id": "u1", "session_id": "s1"}, session=session
    )
    assert llm.call_count == 2  # 空回复触发了一次重试
    assert state["messages"][-1].content == "在忙呀？"


def test_blank_reply_falls_back_after_retry(tmp_path):
    """LLM 连续两次返回空回复时用省略号兜底，入库 AI 消息永不为空。"""
    llm = ScriptedLLM(["", ""])
    agent, _, _ = _agent(tmp_path, llm)
    session = {"configurable": {"thread_id": "s1", "user_id": "u1"}}
    state = agent.invoke(
        {"user_input": "在吗", "user_id": "u1", "session_id": "s1"}, session=session
    )
    assert llm.call_count == 2  # 只重试一次，不无限重试
    assert state["messages"][-1].content == "……"
    # history 里的 AI MemoryItem 同样是兜底文案，SQLite 不落空消息
    ai_items = [i for i in state["history"] if i.speaker == "AI"]
    assert ai_items and str(ai_items[-1].message.content) == "……"


def test_length_truncated_blank_reply_falls_back_without_retry(tmp_path):
    """done_reason=length（窗口截断）的空回复不重试（重试必再撞顶），直接兜底。"""
    truncated = AIMessage(content="", response_metadata={"done_reason": "length"})
    llm = ScriptedLLM([truncated])
    agent, _, _ = _agent(tmp_path, llm)
    session = {"configurable": {"thread_id": "s1", "user_id": "u1"}}
    state = agent.invoke(
        {"user_input": "在吗", "user_id": "u1", "session_id": "s1"}, session=session
    )
    assert llm.call_count == 1  # 未重试
    assert state["messages"][-1].content == "……"


def test_messages_from_history_cleans_blank_ai_messages():
    """恢复历史时把空 content 的 AI 消息替换为省略号（清洗 SQLite 遗留数据）。"""
    from agent.memory.MemoryItem import MemoryItem

    history = [
        MemoryItem(type="Chat", speaker="User", message=HumanMessage(content="在吗"), love=0),
        MemoryItem(type="Chat", speaker="AI", message=AIMessage(content=""), love=0),
        MemoryItem(type="Chat", speaker="User", message=HumanMessage(content="睡了吗"), love=0),
        MemoryItem(type="Chat", speaker="AI", message=AIMessage(content="   "), love=0),
        MemoryItem(type="Chat", speaker="AI", message=AIMessage(content="晚安"), love=0),
    ]
    messages = ChatAgent._messages_from_history(history)
    contents = [str(m.content) for m in messages]
    assert "" not in contents and "   " not in contents  # 空内容全部被替换
    assert contents.count("……") == 2
    # 清洗后 human→ai 交替仍然合法（连续 AI 之间补了主动占位）
    types = [type(m) for m in messages]
    assert not any(
        types[i] is AIMessage and types[i + 1] is AIMessage for i in range(len(types) - 1)
    ), f"清洗后出现连续 AI：{types}"
