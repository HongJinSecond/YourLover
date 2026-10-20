"""Conversation title, live emotion and affinity, and settings entry."""

from PySide6.QtCore import QEvent, QPoint, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from core.event_bus import event_bus
from utils.config_manager import config
from widgets.avatar import avatar_path, set_avatar
from widgets.base.base_widget import BaseWidget
from widgets.icons import icon
from widgets.theme import ACCENT, ACCENT_SOFT, BORDER, BUTTON_STYLE, MUTED, SURFACE, TEXT, ui_font


class ChatHeader(BaseWidget):
    settings_clicked = Signal()
    _EMOTIONS = {"Happy": "开心", "Sad": "低落", "Angry": "生气", "Shy": "害羞"}

    def setup_ui(self) -> None:
        self._drag_position: QPoint | None = None
        self.setFixedHeight(90)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(28, 18, 24, 18)
        layout.setSpacing(12)
        self._avatar_label = QLabel("Y", self)
        self._avatar_label.setObjectName("header_avatar")
        self._avatar_label.setFixedSize(42, 42)
        self._avatar_label.setFont(ui_font(16, True))
        self._avatar_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info = QVBoxLayout()
        info.setSpacing(4)
        self._title_text = "YourLover"
        self._title_label = QLabel("YourLover", self)
        self._title_label.setTextFormat(Qt.TextFormat.PlainText)
        self._title_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._title_label.installEventFilter(self)
        self._title_label.setFont(ui_font(12, True))
        self._title_label.setStyleSheet(f"color: {TEXT};")
        self._status_label = QLabel("陪你聊聊今天", self)
        self._status_label.setFont(ui_font(9))
        self._status_label.setStyleSheet(f"color: {MUTED};")
        info.addWidget(self._title_label)
        info.addWidget(self._status_label)
        affinity = QVBoxLayout()
        affinity.setSpacing(6)
        affinity.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self._affection_label = QLabel("好感度  0", self)
        self._affection_label.setFont(ui_font(8))
        self._affection_label.setStyleSheet(f"color: {MUTED};")
        self._affection_progress = QProgressBar(self)
        self._affection_progress.setObjectName("affection_progress")
        self._affection_progress.setRange(0, 100)
        self._affection_progress.setValue(0)
        self._affection_progress.setTextVisible(False)
        self._affection_progress.setFixedSize(86, 5)
        affinity.addWidget(self._affection_label)
        affinity.addWidget(self._affection_progress)
        self._emotional_label = QLabel("开心", self)
        self._emotional_label.setObjectName("emotion_chip")
        self._emotional_label.setFont(ui_font(9))
        self._emotional_label.setFixedHeight(30)
        self._emotional_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._settings_button = QPushButton(self)
        self._settings_button.setProperty("iconOnly", True)
        self._settings_button.setIcon(icon("settings"))
        self._settings_button.setFixedSize(38, 38)
        self._settings_button.setToolTip("设置")
        self._settings_button.setAccessibleName("打开设置")
        self._settings_button.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self._avatar_label)
        layout.addLayout(info, 1)
        layout.addLayout(affinity)
        layout.addSpacing(8)
        layout.addWidget(self._emotional_label)
        layout.addSpacing(4)
        layout.addWidget(self._settings_button)
        window_controls = QHBoxLayout()
        window_controls.setSpacing(2)
        self._minimize_button = QPushButton(self)
        self._maximize_button = QPushButton(self)
        self._close_button = QPushButton(self)
        for button, name, tooltip in (
            (self._minimize_button, "minimize", "最小化"),
            (self._maximize_button, "maximize", "最大化 / 还原"),
            (self._close_button, "close", "关闭聊天窗口"),
        ):
            button.setProperty("iconOnly", True)
            button.setIcon(icon(name, size=16))
            button.setFixedSize(28, 30)
            button.setToolTip(tooltip)
            button.setAccessibleName(tooltip)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            window_controls.addWidget(button)
        layout.addLayout(window_controls)
        self._refresh_identity()

    def connect_signals(self) -> None:
        self._settings_button.clicked.connect(self.settings_clicked.emit)
        self._minimize_button.clicked.connect(lambda: self.window().showMinimized())
        self._maximize_button.clicked.connect(self._toggle_maximized)
        self._close_button.clicked.connect(lambda: self.window().close())
        event_bus.settings_batch_changed.connect(self._on_settings_changed)

    def _toggle_maximized(self) -> None:
        window = self.window()
        if window.isMaximized():
            window.showNormal()
        else:
            window.showMaximized()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self.window().windowHandle()
            if handle is None or not handle.startSystemMove():
                self._drag_position = event.globalPosition().toPoint() - self.window().pos()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_position is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.window().move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_position = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = None
            self._toggle_maximized()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def _on_settings_changed(self, values: dict) -> None:
        if {"CHAT_DISPLAY_NAME", "CHAT_AVATAR"}.intersection(values):
            self._refresh_identity()

    def _refresh_identity(self) -> None:
        self.set_title(getattr(config, "CHAT_DISPLAY_NAME", "YourLover") or "YourLover")
        set_avatar(
            self._avatar_label, avatar_path(getattr(config, "CHAT_AVATAR", ""), config.config_dir)
        )

    def apply_style(self) -> None:
        self.setStyleSheet(
            BUTTON_STYLE
            + f"""
            ChatHeader {{ background: {SURFACE}; border-bottom: 1px solid {BORDER}; color: {TEXT}; }}
            QLabel#header_avatar {{ background: {ACCENT_SOFT}; color: {ACCENT}; border-radius: 13px; }}
            QLabel#emotion_chip {{ background: {ACCENT_SOFT}; color: {ACCENT}; border-radius: 12px; padding: 5px 12px; }}
            QProgressBar#affection_progress {{ background: {ACCENT_SOFT}; border: none; border-radius: 2px; }}
            QProgressBar#affection_progress::chunk {{ background: {ACCENT}; border-radius: 2px; }}
        """
        )

    def set_affection(self, value: int) -> None:
        value = max(0, min(100, value))
        self._affection_progress.setValue(value)
        self._affection_label.setText(f"好感度  {value}")

    def set_emotional(self, state: str) -> None:
        self._emotional_label.setText(self._EMOTIONS.get(state, state))

    def set_status(self, text: str, color: str = MUTED) -> None:
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color: {color};")

    def set_title(self, text: str) -> None:
        self._title_text = text
        self._title_label.setText(text)
        self._title_label.setToolTip(text)
        self._elide_title()

    def _elide_title(self) -> None:
        if self._title_label.isVisible():
            self._title_label.setText(
                self._title_label.fontMetrics().elidedText(
                    self._title_text, Qt.TextElideMode.ElideRight, self._title_label.width()
                )
            )

    def eventFilter(self, obj, event) -> bool:
        if obj is self._title_label and event.type() == QEvent.Type.Resize:
            self._elide_title()
        return super().eventFilter(obj, event)
