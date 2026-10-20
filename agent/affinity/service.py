from __future__ import annotations

from threading import Lock

from agent.affinity.affinity_store import AffinityStore
from agent.affinity.schema import AffinityEvent

# 新用户初始好感度（0-100）
DEFAULT_LOVE_SNAPSHOT = 0


class AffinityService:
    """好感度系统：按用户持久化好感度分数，并提供增量更新。

    核心逻辑：
        1. get_love_snapshot 读取用户当前好感度（SQLite 持久化，新用户为 0）；
        2. apply_delta 应用一轮对话的增量（由 LLM 判定器给出），
           结果钳制在 0-100 并持久化；
        3. emit_affinity_event / get_events 保留交互事件记录（观测与审计）。
    不直接依赖 memory 模块；LLM 判定器由调用方（ChatAgent）注入。
    """

    def __init__(
        self,
        store: AffinityStore | None = None,
        default_love_snapshot: int = DEFAULT_LOVE_SNAPSHOT,
    ) -> None:
        """初始化好感度服务。

        参数：
            store: 好感度存储，默认 SQLite（data/memory/affinity.sqlite3）。
            default_love_snapshot: 保留兼容参数，仅作为新用户初始值兜底。
        """
        # 拒绝 bool（Python 中 bool 是 int 的子类，需单独拦截）
        if isinstance(default_love_snapshot, bool) or not isinstance(default_love_snapshot, int):
            raise TypeError("default_love_snapshot must be an int")
        if not 0 <= default_love_snapshot <= 100:
            raise ValueError("default_love_snapshot must be between 0 and 100")
        self._store = store if store is not None else AffinityStore()
        self._events: list[AffinityEvent] = []
        # 事件列表的并发锁，保证多线程 emit/get_events 安全
        self._events_lock = Lock()

    def get_love_snapshot(self, user_id: str) -> int:
        """返回指定用户当前好感度（0-100）。新用户返回初始值 0。"""
        _require_user_id(user_id)
        return self._store.get_love(user_id)

    def apply_delta(self, user_id: str, delta: int) -> int:
        """应用一轮对话的好感度增量，返回更新后的好感度（钳制在 0-100）。

        参数：
            user_id: 用户标识
            delta: 好感度增量（由 LLM 判定器给出）
        """
        _require_user_id(user_id)
        if isinstance(delta, bool) or not isinstance(delta, int):
            raise TypeError("delta must be an int")
        current = self._store.get_love(user_id)
        new_value = max(0, min(100, current + delta))
        self._store.update_love(user_id, new_value)
        return new_value

    def emit_affinity_event(self, user_id: str, event: dict) -> None:
        """记录一次交互事件到内存列表。

        参数：
            user_id: 触发事件的用户
            event: 事件负载字典
        事件被包装为不可变的 AffinityEvent 后追加到内存列表，线程安全。
        """
        recorded_event = AffinityEvent(user_id=user_id, event=event)
        with self._events_lock:
            self._events.append(recorded_event)

    def get_events(self, user_id: str | None = None) -> tuple[AffinityEvent, ...]:
        """返回已记录事件的不可变快照。

        参数：
            user_id: 若提供则只返回该用户的事件；若为 None 则返回全部事件。
        返回值为新建的 AffinityEvent 元组，与内部列表解耦，调用方可安全持有。
        """
        if user_id is not None:
            _require_user_id(user_id)
        with self._events_lock:
            # 拷贝出新的 AffinityEvent 对象，避免调用方通过引用修改内部数据
            events = tuple(
                AffinityEvent(
                    user_id=item.user_id,
                    event=item.event,
                    created_at=item.created_at,
                )
                for item in self._events
            )
        if user_id is None:
            return events
        return tuple(item for item in events if item.user_id == user_id)


def _require_user_id(user_id: str) -> None:
    """校验 user_id：必须是非空字符串，否则抛出 ValueError。"""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must not be empty")
