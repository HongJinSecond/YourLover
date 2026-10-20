"""ProactiveEngine 主动对话引擎单元测试。

用注入的定值随机函数（rng/randint）消除随机性，直接调用 evaluate(now)
驱动状态机，无需 Qt 事件循环（QTimer 不会被启动）。
"""

from __future__ import annotations

import time

from agent.affinity.affinity_store import AffinityStore
from agent.affinity.service import AffinityService
from agent.proactive.engine import ProactiveEngine, ProactiveStage
from utils.config_manager import config


class _FakeSettings:
    """最小配置桩：只提供主动对话相关字段。"""

    PROACTIVE_ENABLED = True
    PROACTIVE_MIN_IDLE_SECONDS = 900  # 15 分钟
    PROACTIVE_MAX_IDLE_SECONDS = 43200  # 12 小时
    PROACTIVE_MAX_IGNORED_NUDGES = 3
    PROACTIVE_TICK_SECONDS = 60
    PROACTIVE_ANGRY_PENALTY_MIN = 5
    PROACTIVE_ANGRY_PENALTY_MAX = 15
    PROACTIVE_COOLDOWN_SECONDS = 0  # 默认测试禁用冷却，由冷却用例单独覆盖


def _make_engine(
    affinity: AffinityService | None = None,
    *,
    initial_last_activity: float | None = 1000.0,
    enabled: bool = True,
    cooldown: int = 0,
):
    """构造确定性引擎：静默阈值固定 100s，无视上限固定 2 次，惩罚固定 10。"""
    settings = _FakeSettings()
    settings.PROACTIVE_ENABLED = enabled
    settings.PROACTIVE_COOLDOWN_SECONDS = cooldown
    engine = ProactiveEngine(
        settings,
        affinity_service=affinity,
        user_id="user-a",
        initial_last_activity=initial_last_activity,
        rng=lambda a, b: 100.0,  # 静默阈值恒为 100 秒
        randint=lambda a, b: a if a == 2 else 10,  # 无视上限=2；惩罚=10
    )
    return engine


def _collect(engine: ProactiveEngine):
    """收集两个信号发射的指令文本。"""
    events = {"check_in": [], "angry": []}
    engine.checkInRequested.connect(lambda text: events["check_in"].append(text))
    engine.angryRequested.connect(lambda text: events["angry"].append(text))
    return events


def test_first_check_in_fires_after_idle_threshold(tmp_path):
    """静默达到阈值后发射一次问候，状态转为 NUDGING。"""
    engine = _make_engine()
    events = _collect(engine)

    engine.evaluate(1050.0)  # 静默 50 秒：未到期
    assert events["check_in"] == []
    assert engine.stage is ProactiveStage.IDLE

    engine.evaluate(1101.0)  # 静默 101 秒：到期
    assert len(events["check_in"]) == 1
    assert engine.stage is ProactiveStage.NUDGING
    assert engine.nudges_sent == 1
    assert "第一次主动找用户" in events["check_in"][0]


def test_repeated_ignoring_leads_to_angry_and_silent(tmp_path):
    """问候被无视达到上限后：扣好感度、发射生气消息、进入沉默。"""
    store = AffinityStore(tmp_path / "affinity.sqlite3")
    affinity = AffinityService(store=store)
    store.update_love("user-a", 50)

    engine = _make_engine(affinity)
    events = _collect(engine)

    engine.evaluate(1101.0)  # 第 1 次问候
    engine.evaluate(1202.0)  # 又静默 101 秒：第 2 次问候（达到上限 2）
    assert len(events["check_in"]) == 2
    assert events["angry"] == []

    engine.evaluate(1303.0)  # 第 3 次到期：该生气了
    assert len(events["angry"]) == 1
    assert engine.stage is ProactiveStage.SILENT
    # 好感度被扣 10（50 → 40）
    assert store.get_love("user-a") == 40
    assert "生气" in events["angry"][0] or "委屈" in events["angry"][0]


def test_silent_engine_never_fires_again(tmp_path):
    """生气后的沉默阶段：继续 evaluate 不再发任何主动消息。"""
    engine = _make_engine()
    events = _collect(engine)
    engine.evaluate(1101.0)
    engine.evaluate(1202.0)
    engine.evaluate(1303.0)  # 生气
    assert engine.stage is ProactiveStage.SILENT

    engine.evaluate(99999.0)  # 无论多久都不再发
    assert len(events["check_in"]) == 2  # 生气前的两次问候
    assert len(events["angry"]) == 1


def test_user_activity_resets_state_machine(tmp_path):
    """用户任何消息都会重置状态机（哄她机制），包括生气沉默阶段。"""
    engine = _make_engine()
    events = _collect(engine)
    engine.evaluate(1101.0)
    engine.evaluate(1202.0)
    engine.evaluate(1303.0)  # 生气 → SILENT
    assert engine.stage is ProactiveStage.SILENT

    now = time.time()
    engine.notify_user_activity(now)  # 用户主动开口哄她
    assert engine.stage is ProactiveStage.IDLE
    assert engine.nudges_sent == 0

    engine.evaluate(now + 101.0)  # 重新静默到期 → 又会问候
    assert len(events["check_in"]) == 3  # 生气前 2 次 + 重置后 1 次


def test_disabled_engine_never_fires():
    """PROACTIVE_ENABLED=False 时 evaluate 直接返回。"""
    engine = _make_engine(enabled=False)
    events = _collect(engine)
    engine.evaluate(1101.0)
    engine.evaluate(1303.0)
    assert events["check_in"] == []
    assert events["angry"] == []


def test_instructions_mention_idle_duration():
    """指令文本包含静默时长描述，供 LLM 感知"他多久没理我"。"""
    engine = _make_engine()
    events = _collect(engine)
    engine.evaluate(1101.0)  # 静默 101 秒 ≈ 1.7 分钟
    assert "分钟" in events["check_in"][0]
    assert "{{0}}" in events["check_in"][0]


def test_current_json_settings_used_when_attrs_missing():
    """部分配置对象缺键时使用统一 JSON 配置中的值。"""

    class Empty:
        pass

    engine = ProactiveEngine(Empty(), initial_last_activity=0.0)
    assert engine._min_idle == config.PROACTIVE_MIN_IDLE_SECONDS
    assert engine._max_idle == config.PROACTIVE_MAX_IDLE_SECONDS
    assert engine._cooldown == config.PROACTIVE_COOLDOWN_SECONDS


def test_cooldown_blocks_frequent_check_ins():
    """发完一条主动消息后，冷却期内不再触发下一条（避免频繁打扰）。"""
    engine = _make_engine(cooldown=1800)
    events = _collect(engine)

    engine.evaluate(1101.0)  # 第 1 次问候（静默 101 秒到期）
    assert len(events["check_in"]) == 1

    # 冷却期（30 分钟）内即使静默再次到期也不触发
    engine.evaluate(1202.0)
    engine.evaluate(2101.0)
    assert len(events["check_in"]) == 1
    assert engine.nudges_sent == 1

    # 冷却结束（1800 秒后）且静默再次到期 → 触发第 2 次问候
    engine.evaluate(3001.0)
    assert len(events["check_in"]) == 2


def test_user_activity_clears_cooldown():
    """用户发消息会重置状态机并清空冷却（有互动就不算打扰）。"""
    engine = _make_engine(cooldown=1800)
    events = _collect(engine)
    engine.evaluate(1101.0)  # 第 1 次问候
    assert len(events["check_in"]) == 1

    engine.notify_user_activity(1200.0)  # 用户回复 → 冷却清零、重置计时
    engine.evaluate(1301.0)  # 重置后静默 101 秒到期
    assert len(events["check_in"]) == 2
