"""Live2D actions used by the desktop character and chat replies."""

import logging

import live2d.v3 as live2d
from PySide6.QtWidgets import QWidget

from avatar.live2d import Live2D
from utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class Live2DController:
    def __init__(self, settings: ConfigManager) -> None:
        # Model loading must wait for the widget's OpenGL context.
        self._render_widget = Live2D(settings)

    @property
    def model(self) -> live2d.LAppModel | None:
        return self._render_widget.model

    def get_render_widget(self) -> QWidget:
        return self._render_widget

    def on_hit_area(self, area_name: str) -> None:
        self.play_motion("Tap")

    def on_shake(self) -> None:
        self.play_motion("Shake", live2d.MotionPriority.FORCE)

    def on_drag_start(self, direction: str) -> None:
        self.play_motion("Move")

    def on_drag_end(self) -> None:
        self.play_motion("Idle")

    def play_motion(self, motion_name: str, priority=None) -> None:
        if self.model is None:
            return
        try:
            self.model.StartRandomMotion(
                motion_name, live2d.MotionPriority.NORMAL if priority is None else priority
            )
        except Exception:
            logger.warning("Live2D 动作播放失败：%s", motion_name)

    def speak(self, emotion: str | None = None) -> None:
        motion_index = {"Shy": 0, "Sad": 1, "Happy": 2}.get(emotion)
        if self.model is None or motion_index is None:
            return
        try:
            self.model.StartMotion("Speak", motion_index)
        except Exception:
            logger.warning("Live2D 说话动作播放失败：%s", emotion)

    def change_expression(self, expression_name: str) -> None:
        if self.model is not None:
            try:
                self.model.SetExpression(expression_name)
            except Exception:
                logger.warning("Live2D 表情切换失败：%s", expression_name)

    def clear_expression(self) -> None:
        self._render_widget.clear_expression()

    def set_state(self, state_name: str) -> None:
        self._render_widget.state = state_name
        if state_name == "Happy":
            self.clear_expression()
        self.change_expression(state_name)
