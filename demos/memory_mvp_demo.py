"""M8 end-to-end demo for durable memory across chat sessions.

Run from the repository root with::

    conda run -n YourLoverDesk python demos/memory_mvp_demo.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_ollama import ChatOllama

# Support the documented ``python demos/memory_mvp_demo.py`` invocation.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agent.agent.ChatAgent import ChatAgent
from agent.memory import LongTermMemory, ShortTermMemory, UserProfileManager
from agent.memory.profile_store import ProfileStore
from agent.memory.schema import SemanticMemory
from agent.memory.store import FileMemoryStore
from utils.config_manager import config

DEFAULT_MODEL = "qwen3:8b"
DEFAULT_USER_ID = "memory-mvp-demo-user"
DEFAULT_MEMORY_DIR = REPOSITORY_ROOT / "data" / "memory" / "users"

# 会话 1 用较小的短期窗口，几轮内即触发裁剪，从而驱动"裁剪时合并提取"
DEMO_MAX_MESSAGES = 4
DEMO_KEEP_RECENT_TURNS = 1


@dataclass(frozen=True)
class MemoryDemoResult:
    """Observable evidence produced by the two-session demo."""

    memory_file: Path
    first_response: str
    second_response: str
    memory_context: str
    semantic_memories: tuple[str, ...]


def run_memory_mvp_demo(
    llm: Any,
    memory_dir: str | Path = DEFAULT_MEMORY_DIR,
    user_id: str = DEFAULT_USER_ID,
    *,
    session_database: str | Path | None = None,
    profile_database: str | Path | None = None,
    profile_extractor: Any = None,
    memory_extractor: Any = None,
) -> MemoryDemoResult:
    """Write a preference in one session and recall it in a fresh session.

    The second ``ChatAgent`` and ``FileMemoryStore`` are deliberately recreated.
    This makes the demo verify durable JSON memory rather than LangGraph's
    in-process short-term checkpointer.

    短期记忆 SQLite 与用户画像 SQLite 默认放在 memory_dir 的上级目录
    （与全局默认 data/memory/ 布局一致），测试可显式传入隔离路径。
    profile_extractor 为画像候选提取器；缺省时 UserProfileManager 内部使用
    Ollama（真实运行），测试可注入桩提取器避免网络请求。
    memory_extractor 为合并提取器；缺省基于 llm 构建（真实 LLM 提炼），
    测试可注入桩提取器或让提取失败触发规则降级路径。
    会话 1 会连续发送多轮以触发短期窗口裁剪（裁剪时才做长期记忆提取）。
    """

    memory_dir = Path(memory_dir)
    session_database = (
        Path(session_database) if session_database else memory_dir.parent / "sessions.sqlite3"
    )
    profile_database = (
        Path(profile_database) if profile_database else memory_dir.parent / "profiles.sqlite3"
    )

    first_store = FileMemoryStore(memory_dir)
    first_agent = _build_agent(
        llm,
        first_store,
        thread_id="memory-demo-session-1",
        session_database=session_database,
        profile_database=profile_database,
        profile_extractor=profile_extractor,
        memory_extractor=memory_extractor,
    )
    # 连续发送多轮以触发裁剪：首轮的"记住..."随窗口溢出被提取为长期记忆
    for turn in ["记住我喜欢回答短一点", "我在准备考试", "今天天气不错", "嗯嗯好的"]:
        first_agent.invoke(
            {
                "user_input": turn,
                "user_id": user_id,
                "session_id": "memory-demo-session-1",
            }
        )
    first_state = first_agent.invoke(
        {
            "user_input": "好的，明白了",
            "user_id": user_id,
            "session_id": "memory-demo-session-1",
        }
    )

    # Simulate a new process/session by rebuilding all memory-facing services.
    second_store = FileMemoryStore(memory_dir)
    second_agent = _build_agent(
        llm,
        second_store,
        thread_id="memory-demo-session-2",
        session_database=session_database,
        profile_database=profile_database,
        profile_extractor=profile_extractor,
        memory_extractor=memory_extractor,
    )
    second_state = second_agent.invoke(
        {
            "user_input": "我今天有点累，给我一句建议。",
            "user_id": user_id,
            "session_id": "memory-demo-session-2",
        }
    )

    semantic_memories = tuple(
        memory.content
        for memory in second_store.list_memories(user_id, "semantic")
        if isinstance(memory, SemanticMemory)
    )
    memory_context = str(second_state.get("memory_context") or "")
    if not semantic_memories:
        raise RuntimeError("M8 demo failed: session 1 did not persist semantic memory")
    if "简短回答" not in memory_context:
        raise RuntimeError("M8 demo failed: session 2 did not recall the stored preference")

    return MemoryDemoResult(
        memory_file=memory_dir / f"{user_id}.json",
        first_response=_last_message_content(first_state),
        second_response=_last_message_content(second_state),
        memory_context=memory_context,
        semantic_memories=semantic_memories,
    )


def _build_agent(
    llm: Any,
    store: FileMemoryStore,
    thread_id: str,
    *,
    session_database: Path,
    profile_database: Path,
    profile_extractor: Any = None,
    memory_extractor: Any = None,
) -> ChatAgent:
    """用三大记忆管理器组装 ChatAgent（长短期存储路径由调用方指定）。

    profile_extractor 为 None 时画像管理器内部使用默认 Ollama 提取器。
    memory_extractor 为 None 时 ChatAgent 内部基于 llm 构建合并提取器。
    """
    long_term = LongTermMemory(store=store)
    short_term = ShortTermMemory(
        database_path=session_database,
        max_messages=DEMO_MAX_MESSAGES,
        keep_recent_turns=DEMO_KEEP_RECENT_TURNS,
    )
    profile = UserProfileManager(
        store=ProfileStore(profile_database),
        candidate_extractor=profile_extractor,
    )
    return ChatAgent(
        llm,
        config,
        {"configurable": {"thread_id": thread_id}},
        short_term_memory=short_term,
        long_term_memory=long_term,
        user_profile_manager=profile,
        memory_extractor=memory_extractor,
    )


def _last_message_content(state: dict[str, Any]) -> str:
    messages = state.get("messages") or []
    return str(getattr(messages[-1], "content", "")) if messages else ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstrate persistent long-term memory across two ChatAgent sessions."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama chat model name")
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=DEFAULT_MEMORY_DIR,
        help="Directory containing per-user memory JSON files",
    )
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"使用 Ollama 模型：{args.model}")
    result = run_memory_mvp_demo(
        ChatOllama(model=args.model, temperature=0),
        memory_dir=args.memory_dir,
        user_id=args.user_id,
    )
    print(f"会话 1 回复：{result.first_response}")
    print(f"长期记忆文件：{result.memory_file}")
    print("会话 2 召回上下文：")
    print(result.memory_context)
    print(f"会话 2 回复：{result.second_response}")
    print("M8 验收通过：已在新 session 中召回持久化偏好。")


if __name__ == "__main__":
    main()
