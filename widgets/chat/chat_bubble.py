"""Selectable, plain-text conversation messages."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from core.event_bus import event_bus
from utils.config_manager import config
from widgets.base.base_widget import BaseWidget
from widgets.theme import ACCENT, ACCENT_SOFT, BORDER, MUTED, SURFACE, TEXT, ui_font


class MessageRole(Enum):
    USER = auto()
    AGENT = auto()
    SYSTEM = auto()


@dataclass
class MessageData:
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


class ChatBubble(BaseWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        self._message: MessageData | None = None
        super().__init__(parent)

    def setup_ui(self) -> None:
        self._main_layout = QHBoxLayout(self)
        self._main_layout.setContentsMargins(0, 5, 0, 5)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._bubble_container = QWidget(self)
        self._bubble_container.setObjectName("message_card")
        self._bubble_container.setMaximumWidth(520)
        self._bubble_container.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )
        layout = QVBoxLayout(self._bubble_container)
        layout.setContentsMargins(16, 12, 16, 11)
        layout.setSpacing(7)
        self._content_label = QLabel(self)
        self._content_label.setWordWrap(True)
        self._content_label.setTextFormat(Qt.TextFormat.PlainText)
        self._content_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._content_label.setFont(ui_font())
        self._content_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self._time_label = QLabel(self)
        self._time_label.setTextFormat(Qt.TextFormat.PlainText)
        self._time_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._time_label.setFont(ui_font(8))
        layout.addWidget(self._content_label)
        layout.addWidget(self._time_label)
        self._main_layout.addWidget(self._bubble_container)

    def connect_signals(self) -> None:
        event_bus.settings_batch_changed.connect(self._on_settings_changed)

    def _on_settings_changed(self, values: dict) -> None:
        if "CHAT_DISPLAY_NAME" in values and self._message is not None:
            self._update_sender()
            self._update_size()

    def _update_sender(self) -> None:
        sender = (
            "你"
            if self._message.role == MessageRole.USER
            else getattr(config, "CHAT_DISPLAY_NAME", "YourLover") or "YourLover"
        )
        self._sender_text = f"{sender}  ·  {self._message.timestamp:%H:%M}"
        self._time_label.setText(self._sender_text)
        self._time_label.setToolTip(self._sender_text)

    def apply_style(self) -> None:
        if self._message is None:
            return
        user = self._message.role == MessageRole.USER
        system = self._message.role == MessageRole.SYSTEM
        alignment = Qt.AlignmentFlag.AlignRight if user else Qt.AlignmentFlag.AlignLeft
        if system:
            alignment = Qt.AlignmentFlag.AlignHCenter
        self._main_layout.setAlignment(self._bubble_container, alignment)
        background = ACCENT if user else (ACCENT_SOFT if system else SURFACE)
        foreground = "#FFFFFF" if user else TEXT
        border = ACCENT if user else BORDER
        self._bubble_container.setStyleSheet(f"""
            QWidget#message_card {{ background: {background}; border: 1px solid {border}; border-radius: 14px; }}
            QLabel {{ background: transparent; border: none; color: {foreground}; }}
        """)
        self._time_label.setStyleSheet(
            f"color: {'#DDD5F5' if user else MUTED}; background: transparent;"
        )

    def refresh(self, data: MessageData) -> None:
        self._message = data
        self._content_label.setText(data.content)
        self._update_sender()
        self.apply_style()
        self._update_size()

    def set_content(self, text: str) -> None:
        if self._message is not None:
            self._message.content = text
        self._content_label.setText(text)
        self._update_size()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_size()

    def minimumSizeHint(self) -> QSize:
        # The current card width must not become the minimum width of its parent.
        return QSize(120, self.height())

    def _update_size(self) -> None:
        if self._message is None:
            return
        metrics = QFontMetrics(self._content_label.font())
        available = max(120, min(520, self.width() - 36))
        natural = max(
            (metrics.horizontalAdvance(line) for line in self._message.content.splitlines()),
            default=0,
        )
        sender_metrics = QFontMetrics(self._time_label.font())
        sender_width = sender_metrics.horizontalAdvance(self._sender_text)
        width = min(available, max(110, natural + 34, sender_width + 34))
        text_width = width - 34
        self._time_label.setText(
            sender_metrics.elidedText(self._sender_text, Qt.TextElideMode.ElideRight, text_width)
        )
        rect = metrics.boundingRect(
            QRect(0, 0, text_width, 100000), int(Qt.TextFlag.TextWordWrap), self._message.content
        )
        height = max(metrics.height(), rect.height()) + 4
        self._content_label.setFixedSize(text_width, height)
        self._bubble_container.setFixedSize(
            width, height + self._time_label.sizeHint().height() + 30
        )
        self.setFixedHeight(self._bubble_container.height() + 10)

    def message(self) -> MessageData | None:
        return self._message
