"""A selectable, scrollable reply card floating above the character."""

from PySide6.QtCore import QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from widgets.base.base_widget import BaseWidget
from widgets.icons import icon
from widgets.theme import BORDER, BUTTON_STYLE, MUTED, SCROLL_STYLE, TEXT, ui_font


class SpeechBubble(BaseWidget):
    _TAIL_HEIGHT = 10
    _MAX_BUBBLE_WIDTH = 360
    _MAX_TEXT_HEIGHT = 220

    def __init__(self, parent: QWidget | None = None) -> None:
        self._thinking = False
        self._auto_hide_seconds = 8
        self._thinking_dot_count = 0
        super().__init__(parent)
        self._auto_hide_timer = QTimer(self)
        self._auto_hide_timer.setSingleShot(True)
        self._auto_hide_timer.timeout.connect(self.hide_bubble)
        self._thinking_timer = QTimer(self)
        self._thinking_timer.setInterval(400)
        self._thinking_timer.timeout.connect(self._update_thinking_text)

    def setup_ui(self) -> None:
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(20, 14, 20, 18 + self._TAIL_HEIGHT)
        self._layout.setSpacing(8)
        header = QHBoxLayout()
        name = QLabel("YourLover", self)
        name.setFont(ui_font(9, True))
        name.setStyleSheet(f"color: {MUTED}; background: transparent;")
        self._close_button = QPushButton(self)
        self._close_button.setProperty("iconOnly", True)
        self._close_button.setIcon(icon("close", MUTED, 14))
        self._close_button.setFixedSize(22, 22)
        self._close_button.setToolTip("收起回复")
        self._close_button.setAccessibleName("收起回复")
        header.addWidget(name)
        header.addStretch()
        header.addWidget(self._close_button)
        self._scroll_area = QScrollArea(self)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._label = QLabel(self)
        self._label.setFont(ui_font())
        self._label.setTextFormat(Qt.TextFormat.PlainText)
        self._label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._label.setWordWrap(True)
        self._label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        self._label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._label.setStyleSheet(f"color: {TEXT}; background: transparent; padding: 0;")
        self._scroll_area.setWidget(self._label)
        self._layout.addLayout(header)
        self._layout.addWidget(self._scroll_area, 1)

    def connect_signals(self) -> None:
        self._close_button.clicked.connect(self.hide_bubble)

    def apply_style(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setStyleSheet(BUTTON_STYLE + SCROLL_STYLE)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        body = QRectF(3, 2, self.width() - 6, self.height() - self._TAIL_HEIGHT - 5)
        path = QPainterPath()
        path.addRoundedRect(body, 16, 16)
        tail = QPainterPath()
        center = body.center().x()
        tail.moveTo(center - 9, body.bottom() - 1)
        tail.lineTo(center, body.bottom() + self._TAIL_HEIGHT)
        tail.lineTo(center + 9, body.bottom() - 1)
        tail.closeSubpath()
        path = path.united(tail)
        painter.setPen(QPen(QColor(BORDER), 1))
        painter.setBrush(QColor("#FFFFFF"))
        painter.drawPath(path)

    def show_message(self, text: str) -> None:
        self._thinking = False
        self._thinking_timer.stop()
        self._auto_hide_timer.stop()
        self._label.setText(text)
        self._update_size()
        self._scroll_area.verticalScrollBar().setValue(0)
        self.show()
        self.raise_()
        self._start_auto_hide()

    def show_thinking(self) -> None:
        self._auto_hide_timer.stop()
        self._thinking = True
        self._thinking_dot_count = 0
        self._update_thinking_text()
        self.show()
        self.raise_()
        self._thinking_timer.start()

    def hide_bubble(self) -> None:
        self._thinking = False
        self._thinking_timer.stop()
        self._auto_hide_timer.stop()
        self.hide()

    def set_auto_hide(self, seconds: int) -> None:
        self._auto_hide_seconds = max(0, seconds)
        self._auto_hide_timer.stop()

    def _start_auto_hide(self) -> None:
        if not self._thinking and self._auto_hide_seconds > 0:
            self._auto_hide_timer.start(self._auto_hide_seconds * 1000)

    def enterEvent(self, event) -> None:
        self._auto_hide_timer.stop()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._start_auto_hide()
        super().leaveEvent(event)

    def _update_thinking_text(self) -> None:
        self._label.setText("正在想怎么回复你" + "." * self._thinking_dot_count)
        self._thinking_dot_count = (self._thinking_dot_count + 1) % 4
        self._update_size()

    def _update_size(self) -> None:
        metrics = QFontMetrics(self._label.font())
        text = self._label.text()
        max_width = self._MAX_BUBBLE_WIDTH - 48
        rect = metrics.boundingRect(0, 0, max_width, 100000, int(Qt.TextFlag.TextWordWrap), text)
        width = max(240, min(self._MAX_BUBBLE_WIDTH, rect.width() + 48))
        content_width = width - 48
        rect = metrics.boundingRect(
            0, 0, content_width, 100000, int(Qt.TextFlag.TextWordWrap), text
        )
        self._label.setMinimumHeight(rect.height() + 4)
        self.setFixedSize(width, min(self._MAX_TEXT_HEIGHT, rect.height() + 8) + 72)

    def sizeHint(self) -> QSize:
        return QSize(300, 116)
