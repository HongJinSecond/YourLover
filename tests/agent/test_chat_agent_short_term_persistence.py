from __future__ import annotations

from langchain_core.messages import AIMessage

from agent.agent.ChatAgent import ChatAgent
from agent.memory.long_term_memory import LongTermMemory
from agent.memory.profile_service import (
    ProfileCandidateExtractor,
    ProfileCandidateProposal,
    UserProfileManager,
)
from agent.memory.profile_store import ProfileStore
from agent.memory.session_store import SessionStore
from agent.memory.short_term_memory import ShortTermMemory
from agent.memory.store import FileMemoryStore
from utils.config_manager import ConfigManager


class NoopCandidateExtractor(ProfileCandidateExtractor):
    """测试提取器：不产生画像候选，避免测试连接真实 Ollama。"""

    def extract(
        self,
        user_id: str,
        user_message: str,
        session_id: str | None = None,
        message_id: str | None = None,
    ) -> list[ProfileCandidateProposal]:
        return []


class RecordingLLM:
    """测试 LLM：记录每次收到的消息，并返回固定回复。"""

    def __init__(self) -> None:
        self.calls: list[list] = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return AIMessage(content="固定回复")


def _build_agent(tmp_path, llm, session_store):
    """创建使用临时长短期存储的 ChatAgent（三大管理器注入临时路径）。"""
    long_term = LongTermMemory(store=FileMemoryStore(tmp_path / "users"))
    short_term = ShortTermMemory(session_store=session_store)
    profile = UserProfileManager(
        store=ProfileStore(tmp_path / "profiles.sqlite3"),
        candidate_extractor=NoopCandidateExtractor(),
    )
    return ChatAgent(
        llm,
        ConfigManager(),
        {"configurable": {"thread_id": "default"}},
        short_term_memory=short_term,
        long_term_memory=long_term,
        user_profile_manager=profile,
    )


def test_recreated_agent_restores_short_term_session(tmp_path):
    """重建 Agent 后，同一 Session 应恢复之前的消息并继续对话。"""
    session_store = SessionStore(tmp_path / "sessions.sqlite3")
    first_llm = RecordingLLM()
    first_agent = _build_agent(tmp_path, first_llm, session_store)

    first_agent.invoke({"user_input": "第一轮", "user_id": "u1", "session_id": "s1"})
    first_agent.invoke({"user_input": "第二轮", "user_id": "u1", "session_id": "s1"})

    second_llm = RecordingLLM()
    second_agent = _build_agent(tmp_path, second_llm, session_store)
    second_agent.invoke({"user_input": "第三轮", "user_id": "u1", "session_id": "s1"})

    received_contents = [getattr(message, "content", "") for message in second_llm.calls[0][1:]]
    assert received_contents == [
        "第一轮",
        "固定回复",
        "第二轮",
        "固定回复",
        "第三轮",
    ]
    assert len(session_store.load_memory_items("u1", "s1")) == 6
