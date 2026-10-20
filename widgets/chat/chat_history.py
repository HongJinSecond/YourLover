"""Scrollable conversation with an empty state and deferred scrolling."""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

from widgets.base.base_widget import BaseWidget
from widgets.chat.chat_bubble import ChatBubble, MessageData, MessageRole
from widgets.theme import ACCENT, ACCENT_SOFT, BACKGROUND, MUTED, SCROLL_STYLE, TEXT, ui_font


class ChatHistory(BaseWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        self._bubbles: list[ChatBubble] = []
        super().__init__(parent)

    def setup_ui(self) -> None:
        self._scroll_area = QScrollArea(self)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self._container = QWidget()
        self._container.setObjectName("chat_container")
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(28, 20, 28, 18)
        self._layout.setSpacing(7)
        self._empty_state = QWidget(self._container)
        empty = QVBoxLayout(self._empty_state)
        empty.setContentsMargins(20, 44, 20, 44)
        empty.setSpacing(14)
        mark = QLabel("Y", self._empty_state)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(64, 64)
        mark.setFont(ui_font(24, True))
        mark.setStyleSheet(f"background: {ACCENT_SOFT}; color: {ACCENT}; border-radius: 20px;")
        title = QLabel("今天，有什么想和我说的？", self._empty_state)
        title.setFont(ui_font(16, True))
        title.setStyleSheet(f"color: {TEXT};")
        subtitle = QLabel("分享一件小事，或只是安静地聊一会儿。", self._empty_state)
        subtitle.setFont(ui_font(10))
        subtitle.setStyleSheet(f"color: {MUTED};")
        for widget in (mark, title, subtitle):
            empty.addWidget(widget, 0, Qt.AlignmentFlag.AlignHCenter)
        self._layout.addWidget(self._empty_state)
        self._layout.addStretch()
        self._scroll_area.setWidget(self._container)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._scroll_area)

    def apply_style(self) -> None:
        self.setStyleSheet(SCROLL_STYLE + f"QWidget#chat_container {{ background: {BACKGROUND}; }}")

    def connect_signals(self) -> None:
        self._scroll_area.verticalScrollBar().rangeChanged.connect(self.scroll_to_bottom)

    def append_message(self, role: MessageRole, content: str) -> ChatBubble:
        self._empty_state.hide()
        bubble = ChatBubble(self._container)
        bubble.refresh(MessageData(role, content))
        self._layout.insertWidget(self._layout.count() - 1, bubble)
        self._bubbles.append(bubble)
        self.scroll_to_bottom()
        return bubble

    def clear_history(self) -> None:
        for bubble in self._bubbles:
            self._layout.removeWidget(bubble)
            bubble.hide()
            bubble.deleteLater()
        self._bubbles.clear()
        self._empty_state.show()

    def scroll_to_bottom(self) -> None:
        # Wait until Qt has recalculated wrapped-message heights.
        QTimer.singleShot(0, self._scroll_to_bottom)

    def _scroll_to_bottom(self) -> None:
        scrollbar = self._scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def bubbles(self) -> list[ChatBubble]:
        return list(self._bubbles)
