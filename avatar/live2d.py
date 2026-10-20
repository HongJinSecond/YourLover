"""OpenGL-backed Live2D rendering, pointer tracking and idle animation."""

import logging
from contextlib import suppress

import live2d.v3 as live2d
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCursor, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget

from utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class Live2D(QOpenGLWidget):
    def __init__(self, settings: ConfigManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._model_path = settings.DEFAULT_MODEL_PATH
        self.model = None
        self.state = "Happy"
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._on_frame)
        self._idle_timer = QTimer(self)
        self._idle_timer.setInterval(10000)
        self._idle_timer.timeout.connect(self._on_idle_check)
        surface = QSurfaceFormat()
        surface.setAlphaBufferSize(8)
        surface.setSwapInterval(1)
        self.setFormat(surface)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self.setFixedSize(*settings.CHARACTER_WINDOW_SIZE)

    def initializeGL(self) -> None:
        # All Live2D initialization requires this widget's OpenGL context.
        try:
            live2d.init()
            live2d.glInit()
            live2d.enableLog(False)
            self.model = live2d.LAppModel()
            self.model.LoadModelJson(self._model_path)
            self.model.Resize(*self._settings.CHARACTER_WINDOW_SIZE)
            self.model.SetAutoBreathEnable(True)
            self.model.SetAutoBlinkEnable(True)
            with suppress(Exception):
                self.model.StartRandomMotion("Idle", live2d.MotionPriority.IDLE)
        except Exception:
            self.model = None
            logger.exception("Live2D 模型加载失败")
            return
        self._timer.start()
        self._idle_timer.start()

    def resizeGL(self, width: int, height: int) -> None:
        if self.model:
            self.model.Resize(width, height)

    def paintGL(self) -> None:
        live2d.clearBuffer(0, 0, 0, 0)
        if self.model:
            with suppress(Exception):
                self.model.Draw()

    def _on_frame(self) -> None:
        if self.model is None:
            return
        position = self.mapFromGlobal(QCursor.pos())
        self.model.Drag(position.x(), position.y())
        self.model.Update()
        self.update()

    def _on_idle_check(self) -> None:
        if self.model is None:
            return
        try:
            if self.model.IsMotionFinished() and self.state == "Happy":
                self.model.StartRandomMotion("Idle", live2d.MotionPriority.IDLE)
        except Exception:
            logger.warning("Live2D 待机动作检查失败")

    def clear_expression(self) -> None:
        if self.model:
            self.model.ResetExpressions()
