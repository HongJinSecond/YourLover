from datetime import datetime, timedelta, timezone

from agent.memory.retriever import MemoryRetriever
from agent.memory.schema import EpisodicMemory, MemoryEvidence, SemanticMemory
from agent.memory.store import FileMemoryStore


def test_retrieve_exam_pressure_finds_state_and_episodic_memories(tmp_path):
    store = FileMemoryStore(tmp_path)
    retriever = MemoryRetriever(store)

    store.add_memory(
        "u_001",
        SemanticMemory(
            user_id="u_001",
            scope="state",
            content="用户最近因为期末考试感到压力大。",
            tags=["考试", "压力"],
            importance=0.7,
        ),
    )
    store.add_memory(
        "u_001",
        EpisodicMemory(
            user_id="u_001",
            title="期末压力陪伴对话",
            summary="用户表达考试焦虑，Agent 先共情再拆分复习计划。",
            recall_triggers=["考试压力", "焦虑", "复习计划"],
            importance=0.9,
        ),
    )
    store.add_memory(
        "u_001",
        SemanticMemory(user_id="u_001", content="用户喜欢回答短一点。", tags=["style"]),
    )

    memories = retriever.retrieve_memories("u_001", "考试压力")

    assert [memory.type.value for memory in memories] == ["episodic", "semantic"]
    assert "考试" in memories[0].summary
    assert memories[1].scope == "state"


def test_retrieve_supports_memory_type_filter_and_top_k(tmp_path):
    store = FileMemoryStore(tmp_path)
    retriever = MemoryRetriever(store)

    store.add_memory(
        "u_001",
        SemanticMemory(user_id="u_001", content="用户考试压力很大。", importance=0.1),
    )
    store.add_memory(
        "u_001",
        SemanticMemory(user_id="u_001", content="用户考试压力需要拆计划。", importance=0.9),
    )
    store.add_memory(
        "u_001",
        EpisodicMemory(user_id="u_001", summary="考试压力对话。", importance=1.0),
    )

    memories = retriever.retrieve_memories(
        "u_001",
        "考试压力",
        memory_type="semantic",
        top_k=1,
    )

    assert len(memories) == 1
    assert memories[0].type.value == "semantic"
    assert memories[0].content == "用户考试压力需要拆计划。"


def test_retrieve_ignores_deleted_and_expired_memories(tmp_path):
    store = FileMemoryStore(tmp_path)
    retriever = MemoryRetriever(store)
    expired_at = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

    deleted_id = store.add_memory(
        "u_001",
        SemanticMemory(user_id="u_001", content="用户考试压力很大。"),
    )
    store.delete_memory("u_001", deleted_id)
    store.add_memory(
        "u_001",
        SemanticMemory(
            user_id="u_001",
            content="用户上个月考试压力很大。",
            expires_at=expired_at,
            importance=1.0,
        ),
    )
    store.add_memory(
        "u_001",
        SemanticMemory(user_id="u_001", content="用户今天考试压力很大。"),
    )

    memories = retriever.retrieve_memories("u_001", "考试压力")

    assert len(memories) == 1
    assert memories[0].content == "用户今天考试压力很大。"


def test_format_memory_context_groups_memories_without_internal_evidence(tmp_path):
    store = FileMemoryStore(tmp_path)
    retriever = MemoryRetriever(store)
    semantic = SemanticMemory(
        user_id="u_001",
        content="用户喜欢简短回答。",
        evidence=MemoryEvidence(message_ids=["msg_001"], excerpt="内部证据"),
    )
    episodic = EpisodicMemory(
        user_id="u_001",
        title="复习计划",
        summary="用户焦虑后接受了复习拆分计划。",
        love_snapshot=77,
        evidence=MemoryEvidence(message_ids=["msg_002"], excerpt="不要直接暴露"),
    )

    context = retriever.format_memory_context([semantic, episodic])

    assert "Semantic memories:" in context
    # 每条记忆带 [YYYY-MM-DD] 日期前缀（取 created_at 前 10 位）
    assert f"- [{semantic.created_at[:10]}] 用户喜欢简短回答。" in context
    assert "Episodic memories:" in context
    assert f"- [{episodic.created_at[:10]}] 复习计划: 用户焦虑后接受了复习拆分计划。" in context
    assert "msg_001" not in context
    assert "不要直接暴露" not in context
    assert "77" not in context


def test_format_memory_context_returns_empty_string_for_no_memories():
    retriever = MemoryRetriever(FileMemoryStore())

    assert retriever.format_memory_context([]) == ""
