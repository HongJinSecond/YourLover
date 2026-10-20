from datetime import datetime

import pytest

from agent.affinity import (
    DEFAULT_LOVE_SNAPSHOT,
    AffinityEvent,
    AffinityService,
    AffinityStore,
)


def _service(tmp_path) -> AffinityService:
    """构造使用临时 SQLite 的好感度服务。"""
    return AffinityService(store=AffinityStore(tmp_path / "affinity.sqlite3"))


def test_new_user_starts_at_zero(tmp_path):
    service = AffinityService(store=AffinityStore(tmp_path / "affinity.sqlite3"))

    assert service.get_love_snapshot("u1") == 0
    assert service.get_love_snapshot("u2") == 0


def test_initial_default_is_zero():
    assert DEFAULT_LOVE_SNAPSHOT == 0


def test_apply_delta_increases_and_persists(tmp_path):
    service = _service(tmp_path)

    assert service.apply_delta("u1", 5) == 5
    assert service.get_love_snapshot("u1") == 5

    assert service.apply_delta("u1", 3) == 8
    # 新实例（模拟重启）仍能读到持久化值
    reloaded = AffinityService(store=AffinityStore(tmp_path / "affinity.sqlite3"))
    assert reloaded.get_love_snapshot("u1") == 8


def test_apply_delta_clamps_to_100(tmp_path):
    service = _service(tmp_path)
    service.apply_delta("u1", 98)
    assert service.apply_delta("u1", 10) == 100
    assert service.get_love_snapshot("u1") == 100


def test_apply_delta_clamps_at_zero(tmp_path):
    service = _service(tmp_path)
    assert service.apply_delta("u1", -5) == 0


def test_apply_delta_validates_input(tmp_path):
    service = _service(tmp_path)
    with pytest.raises(TypeError, match="delta"):
        service.apply_delta("u1", True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="user_id"):
        service.apply_delta(" ", 1)


def test_service_accepts_event_and_keeps_users_isolated(tmp_path):
    service = _service(tmp_path)

    assert service.emit_affinity_event("u1", {"type": "user_thanked"}) is None
    service.emit_affinity_event("u2", {"type": "user_complained"})

    assert [item.event for item in service.get_events("u1")] == [{"type": "user_thanked"}]
    assert [item.event for item in service.get_events("u2")] == [{"type": "user_complained"}]


def test_recorded_event_is_a_defensive_copy(tmp_path):
    service = _service(tmp_path)
    payload = {"type": "memory_written", "memory_ids": ["m1"]}

    service.emit_affinity_event("u1", payload)
    payload["memory_ids"].append("m2")

    recorded = service.get_events("u1")[0]
    assert recorded.event == {"type": "memory_written", "memory_ids": ["m1"]}
    assert recorded.to_dict()["event"] == recorded.event
    assert datetime.fromisoformat(recorded.created_at).tzinfo is not None

    recorded.event["memory_ids"].append("m3")
    assert service.get_events("u1")[0].event["memory_ids"] == ["m1"]


def test_affinity_event_schema_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="user_id"):
        AffinityEvent(user_id="", event={"type": "feedback"})
    with pytest.raises(TypeError, match="event"):
        AffinityEvent(user_id="u1", event="feedback")  # type: ignore[arg-type]


def test_service_validates_snapshot_and_user_id(tmp_path):
    with pytest.raises(ValueError, match="between 0 and 100"):
        AffinityService(default_love_snapshot=101)
    with pytest.raises(TypeError, match="must be an int"):
        AffinityService(default_love_snapshot=True)

    service = _service(tmp_path)
    with pytest.raises(ValueError, match="user_id"):
        service.get_love_snapshot(" ")
    with pytest.raises(ValueError, match="user_id"):
        service.emit_affinity_event("", {"type": "feedback"})


def test_affinity_service_can_be_replaced_without_memory_dependency():
    class DatabaseAffinityService(AffinityService):
        def get_love_snapshot(self, user_id: str) -> int:
            return 88

    replacement = DatabaseAffinityService()

    assert replacement.get_love_snapshot("u1") == 88
