"""主动对话引擎：静默检测、随机调度、渐进式问候与生气惩罚。

设计要点（模拟真人的关键）：
    1. 随机静默阈值：每次调度都在 [min_idle, max_idle] 内随机取值，
       像真人一样"有时二十分钟就想找你，有时半天才想起来"；
    2. 渐进式情绪：先温和问候（可以聊新话题或接着旧话题），被无视
       次数达到随机上限后生气（扣好感度），生气后保持沉默；
    3. 哄她重置：用户任意一条消息都会把状态机重置回 IDLE 并重新计时
       （"必须要我们主动发消息哄她"）；
    4. 隐藏指令：引擎只产出指令文本，经 ChatAgent 的 proactive_context
       通道注入 system prompt，用户与 UI 均不可见。

可测试性：核心判定逻辑集中在 evaluate(now)，时间与随机数均可注入，
单元测试无需 Qt 事件循环。
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from enum import Enum, auto
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

from utils.config_manager import config as app_config

logger = logging.getLogger(__name__)


class ProactiveStage(Enum):
    """主动对话状态机的当前阶段。"""

    IDLE = auto()  # 静默计时中，还没发过主动消息
    NUDGING = auto()  # 已发过至少一次问候，继续等待用户回复
    SILENT = auto()  # 生气之后：不再主动发消息，等用户先开口


def _humanize_gap(seconds: float) -> str:
    """把秒数间隔转成中文时长描述（用于指令文本，让 LLM 感知静默时长）。"""
    if seconds < 3600:
        return f"{int(seconds // 60)} 分钟"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} 小时"
    return f"{seconds / 86400:.1f} 天"


class ProactiveEngine(QObject):
    """主动对话引擎：检测用户静默并按阶段发起主动消息。

    信号：
        - checkInRequested(instruction): 该发一条温和的主动问候了；
        - angryRequested(instruction): 无视次数已达上限，该发一条生气
          委屈的消息了（好感度惩罚在发信号前已由引擎执行）。

    参数：
        - settings: 配置对象（ConfigManager 或支持属性访问的配置桩）；
        - affinity_service: 好感度服务，生气时扣好感度；None 则只发消息不扣分；
        - user_id: 好感度归属用户；
        - initial_last_activity: 启动时的最后用户活跃时间戳（如从 SQLite
          恢复），None 则从当前时间开始计时；
        - rng: 随机数函数（默认 random.uniform），测试可注入定值函数。

    计时模型：QTimer 每 tick_seconds 秒轮询一次，比较
    "当前时间 - 最后用户活跃时间" 是否达到随机阈值；用户任何消息都会
    调用 notify_user_activity() 刷新计时起点并重置状态机。
    """

    checkInRequested = Signal(str)
    angryRequested = Signal(str)

    def __init__(
        self,
        settings: Any,
        affinity_service: Any = None,
        user_id: str = "default-user",
        *,
        initial_last_activity: float | None = None,
        rng: Callable[[float, float], float] | None = None,
        randint: Callable[[int, int], int] | None = None,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._affinity_service = affinity_service
        self._user_id = user_id
        self._rng = rng or random.uniform
        self._randint = randint or random.randint

        # 部分配置桩缺键时回落到当前 config.json，不再维护 Python 默认配置。
        self._enabled = bool(getattr(settings, "PROACTIVE_ENABLED", app_config.PROACTIVE_ENABLED))
        self._min_idle = int(
            getattr(
                settings,
                "PROACTIVE_MIN_IDLE_SECONDS",
                app_config.PROACTIVE_MIN_IDLE_SECONDS,
            )
        )
        self._max_idle = int(
            getattr(
                settings,
                "PROACTIVE_MAX_IDLE_SECONDS",
                app_config.PROACTIVE_MAX_IDLE_SECONDS,
            )
        )
        self._max_nudges = max(
            2,
            int(
                getattr(
                    settings,
                    "PROACTIVE_MAX_IGNORED_NUDGES",
                    app_config.PROACTIVE_MAX_IGNORED_NUDGES,
                )
            ),
        )
        tick_seconds = int(
            getattr(settings, "PROACTIVE_TICK_SECONDS", app_config.PROACTIVE_TICK_SECONDS)
        )
        self._penalty_min = int(
            getattr(
                settings,
                "PROACTIVE_ANGRY_PENALTY_MIN",
                app_config.PROACTIVE_ANGRY_PENALTY_MIN,
            )
        )
        self._penalty_max = int(
            getattr(
                settings,
                "PROACTIVE_ANGRY_PENALTY_MAX",
                app_config.PROACTIVE_ANGRY_PENALTY_MAX,
            )
        )
        self._cooldown = int(
            getattr(
                settings,
                "PROACTIVE_COOLDOWN_SECONDS",
                app_config.PROACTIVE_COOLDOWN_SECONDS,
            )
        )

        # 状态机初始值：从上次活跃时间开始静默计时（支持跨启动的"你没理我"）
        self._last_activity = float(
            initial_last_activity if initial_last_activity is not None else time.time()
        )
        self._stage = ProactiveStage.IDLE
        self._nudges_sent = 0
        self._next_delay = self._roll_delay()
        self._nudge_limit = self._roll_nudge_limit()
        self._cooldown_until = 0.0  # 冷却截止时间戳：发完主动消息后的一段时间内不再打扰

        self._timer = QTimer(self)
        self._timer.setInterval(max(1, tick_seconds) * 1000)
        self._timer.timeout.connect(lambda: self.evaluate(time.time()))

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def start(self) -> None:
        """启动轮询定时器（应用启动且 Agent 就绪后调用）。"""
        if self._enabled:
            self._timer.start()
            logger.info(
                "主动对话引擎已启动：静默阈值 %d~%ds，无视上限 %d 次",
                self._min_idle,
                self._max_idle,
                self._nudge_limit,
            )

    def stop(self) -> None:
        """停止轮询（应用退出或 Agent 重连时调用）。"""
        self._timer.stop()

    def notify_user_activity(self, now: float | None = None) -> None:
        """用户发来任何消息（大聊天/迷你聊天）时调用：重置整个状态机。

        "哄她"机制的入口：即此前处于生气沉默阶段，用户主动开口也会
        让她恢复主动发消息的意愿，并重新开始静默计时。
        """
        self._last_activity = float(now if now is not None else time.time())
        self._stage = ProactiveStage.IDLE
        self._nudges_sent = 0
        self._next_delay = self._roll_delay()
        self._nudge_limit = self._roll_nudge_limit()
        self._cooldown_until = 0.0  # 用户活跃后冷却清零（有互动就不算打扰）

    # ------------------------------------------------------------------
    # 核心判定（纯逻辑，可测试）
    # ------------------------------------------------------------------

    def evaluate(self, now: float) -> None:
        """检查静默是否到期并按阶段发射信号（QTimer 每个周期调用一次）。

        逻辑：
            1. 未启用 / 沉默阶段：直接返回（生气后不再主动发消息）；
            2. 冷却期内（发完主动消息后的一段时间）：直接返回，避免频繁打扰；
            3. 静默未到期：返回；
            4. 到期且问候次数未达上限：发射 checkInRequested，
               状态转为 NUDGING 并重新掷下一次静默阈值；
            5. 到期且问候已被无视达上限：先扣好感度（随机惩罚），
               再发射 angryRequested，状态转为 SILENT。
        """
        if not self._enabled or self._stage is ProactiveStage.SILENT:
            return
        if now < self._cooldown_until:  # 主动消息冷却期：刚发过就不重复打扰
            return
        elapsed = now - self._last_activity
        if elapsed < self._next_delay:
            return

        idle_text = _humanize_gap(elapsed)
        if self._nudges_sent < self._nudge_limit:
            self._nudges_sent += 1
            self._stage = ProactiveStage.NUDGING
            self._next_delay = self._roll_delay()
            self._cooldown_until = now + self._cooldown
            self.checkInRequested.emit(
                self._build_check_in_instruction(elapsed, idle_text, self._nudges_sent)
            )
            logger.info(
                "主动问候已触发（第 %d/%d 次，静默 %s）",
                self._nudges_sent,
                self._nudge_limit,
                idle_text,
            )
            return

        # 问候被无视达到上限：生气 + 扣好感度 + 沉默
        penalty = self._randint(self._penalty_min, self._penalty_max)
        if self._affinity_service is not None:
            try:
                new_love = self._affinity_service.apply_delta(self._user_id, -penalty)
                logger.warning(
                    "主动消息被无视 %d 次，好感度 -%d（现为 %d）",
                    self._nudges_sent,
                    penalty,
                    new_love,
                )
            except Exception as exc:  # 好感度扣减失败不阻塞生气消息
                logger.warning("生气惩罚扣好感度失败：%s", exc)
        self._stage = ProactiveStage.SILENT
        self.angryRequested.emit(
            self._build_angry_instruction(elapsed, idle_text, self._nudges_sent, penalty)
        )

    # ------------------------------------------------------------------
    # 状态观测（供测试与调试）
    # ------------------------------------------------------------------

    @property
    def stage(self) -> ProactiveStage:
        """当前状态机阶段。"""
        return self._stage

    @property
    def nudges_sent(self) -> int:
        """本周期已发送（并被无视）的问候次数。"""
        return self._nudges_sent

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _roll_delay(self) -> float:
        """在 [min_idle, max_idle] 内随机掷一次静默阈值（模拟真人节奏）。"""
        return float(self._rng(self._min_idle, self._max_idle))

    def _roll_nudge_limit(self) -> int:
        """随机决定本周期生气前最多被无视的问候次数（2 ~ max_nudges）。"""
        return int(self._randint(2, self._max_nudges))

    def _build_check_in_instruction(self, elapsed: float, idle_text: str, nudge_index: int) -> str:
        """构造温和问候的隐藏指令（注入 system prompt，用户不可见）。"""
        tone_hint = (
            "这是你第一次主动找用户，语气轻松自然就好"
            if nudge_index == 1
            else f"这是你第 {nudge_index} 次主动找用户，对方暂时没回，可以轻轻表达关心，不催促"
        )
        return (
            f"距离用户上次回复已经过去了约 {idle_text}，用户暂时没有回复。{tone_hint}。"
            "请你以当前人物设定的身份主动给用户发一条消息，保持自然的日常聊天口吻："
            "可以关心对方在忙什么、分享一点角色日常，"
            "或者接着你们之前聊到一半的话题继续说。"
            "只输出这条消息本身，不要解释你在做什么，末尾照常带情绪标记和 {{0}} 增量标记。"
        )

    def _build_angry_instruction(
        self, elapsed: float, idle_text: str, nudge_count: int, penalty: int
    ) -> str:
        """构造生气委屈消息的隐藏指令。"""
        return (
            f"你已经主动找了用户 {nudge_count} 次，隔了约 {idle_text} 对方暂时没有回复。"
            "这是本周期最后一次主动消息，可以表达一点失落或委屈。"
            "请以当前人物设定的身份发一条简短消息，可以表达轻微失落或不满，"
            "但尊重用户可能在忙，不质问、查岗、威胁或要求对方证明在乎你。"
            "只输出这条消息本身，不要解释机制，末尾照常带情绪标记和 {{0}} 增量标记。"
        )
