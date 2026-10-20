"""Compact conversation entry that follows the desktop character."""

from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget

from core.event_bus import event_bus
from widgets.icons import icon
from widgets.mini_chat.speech_bubble import SpeechBubble
from widgets.theme import (
    ACCENT,
    ACCENT_SOFT,
    BORDER,
    BUTTON_STYLE,
    MUTED,
    TEXT,
    add_shadow,
    keep_on_screen,
    ui_font,
)
from windows.base_window import BaseWindow


class MiniChatOverlay(BaseWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        self._target_window: QWidget | None = None
        self._expanded = False
        self._bubble: SpeechBubble | None = None
        super().__init__(parent)
        self._bubble = SpeechBubble()

    def _setup_window_flags(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

    def setup_ui(self) -> None:
        shell = QWidget(self)
        outer = QVBoxLayout(shell)
        outer.setContentsMargins(8, 8, 8, 12)
        self._card = QWidget(shell)
        self._card.setObjectName("mini_card")
        add_shadow(self._card, 16, 3)
        row = QHBoxLayout(self._card)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(6)
        self._toggle_button = QPushButton("聊一聊", self)
        self._toggle_button.setObjectName("mini_toggle")
        self._toggle_button.setIcon(icon("chat", ACCENT, 18))
        self._toggle_button.setFont(ui_font(9, True))
        self._toggle_button.setToolTip("展开快速聊天")
        self._input = QLineEdit(self)
        self._input.setObjectName("mini_input")
        self._input.setPlaceholderText("和我说说…")
        self._input.setAccessibleName("快速聊天消息")
        self._input.setFont(ui_font())
        self._input.installEventFilter(self)
        self._send_button = QPushButton(self)
        self._send_button.setProperty("primary", True)
        self._send_button.setIcon(icon("send", "#FFFFFF", 17))
        self._send_button.setFixedSize(36, 36)
        self._send_button.setToolTip("发送 · Enter")
        self._send_button.setAccessibleName("发送快速聊天消息")
        self._close_button = QPushButton(self)
        self._close_button.setProperty("iconOnly", True)
        self._close_button.setIcon(icon("close", MUTED, 16))
        self._close_button.setFixedSize(28, 34)
        self._close_button.setToolTip("收起 · Esc")
        self._close_button.setAccessibleName("收起快速聊天")
        for button in (self._toggle_button, self._send_button, self._close_button):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
        row.addWidget(self._toggle_button)
        row.addWidget(self._input, 1)
        row.addWidget(self._send_button)
        row.addWidget(self._close_button)
        outer.addWidget(self._card)
        self.setCentralWidget(shell)
        self._set_expanded(False)

    def connect_signals(self) -> None:
        self._toggle_button.clicked.connect(self.expand)
        self._close_button.clicked.connect(self.collapse)
        self._send_button.clicked.connect(self._on_send)
        self._input.returnPressed.connect(self._on_send)
        self._input.textChanged.connect(self._sync_send_button)
        event_bus.mini_chat_thinking_started.connect(self.show_thinking)
        event_bus.mini_chat_reply_received.connect(self.show_reply)
        self._sync_send_button()

    def apply_style(self) -> None:
        self.setStyleSheet(
            BUTTON_STYLE
            + f"""
            QWidget#mini_card {{ background: #FFFFFF; border: 1px solid {BORDER}; border-radius: 17px; }}
            QPushButton#mini_toggle {{ background: transparent; color: {ACCENT}; border: none; padding: 3px 6px; }}
            QPushButton#mini_toggle:hover {{ background: {ACCENT_SOFT}; }}
            QLineEdit#mini_input {{ background: transparent; border: none; color: {TEXT}; padding: 5px; selection-background-color: {ACCENT_SOFT}; }}
        """
        )

    def set_target_window(self, window: QWidget) -> None:
        if self._target_window is not None:
            self._target_window.position_changed.disconnect(self._on_target_moved)
        self._target_window = window
        window.position_changed.connect(self._on_target_moved)
        self._reposition()
        self.show_window()

    def expand(self) -> None:
        self._set_expanded(True)

    def collapse(self) -> None:
        self._set_expanded(False)

    def show_reply(self, text: str) -> None:
        self._bubble.show_message(text)
        self._reposition_bubble()
        self._input.setEnabled(True)
        self._sync_send_button()

    def show_thinking(self) -> None:
        self._bubble.show_thinking()
        self._reposition_bubble()
        self._input.setEnabled(False)
        self._sync_send_button()

    def hide_bubble(self) -> None:
        self._bubble.hide_bubble()

    def hide_window(self) -> None:
        if self._bubble is not None:
            self.hide_bubble()
        super().hide_window()

    def _set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self._toggle_button.setVisible(not expanded)
        for widget in (self._input, self._send_button, self._close_button):
            widget.setVisible(expanded)
        self.setFixedSize(420 if expanded else 128, 72 if expanded else 64)
        self._reposition()
        if expanded:
            self.show()
            self.raise_()
            self.activateWindow()
            self._input.setFocus()

    def _sync_send_button(self) -> None:
        self._send_button.setEnabled(self._input.isEnabled() and bool(self._input.text().strip()))

    def _on_send(self) -> None:
        text = self._input.text().strip()
        if not text or not self._input.isEnabled():
            return
        self._input.clear()
        self._input.setEnabled(False)
        self._sync_send_button()
        event_bus.mini_chat_input_sent.emit(text)

    def eventFilter(self, obj, event) -> bool:
        if (
            obj is self._input
            and event.type() == QEvent.Type.KeyPress
            and event.key() == Qt.Key.Key_Escape
        ):
            self.collapse()
            return True
        return super().eventFilter(obj, event)

    def _on_target_moved(self, pos) -> None:
        self._reposition()
        self._reposition_bubble()
        self.raise_()

    def _reposition(self) -> None:
        if self._target_window is None:
            return
        geometry = self._target_window.geometry()
        y = geometry.bottom() + 4 if self._expanded else geometry.bottom() - self.height() + 4
        point = QPoint(geometry.right() - self.width() + 8, y)
        self.move(keep_on_screen(self, point, self._target_window))

    def _reposition_bubble(self) -> None:
        if self._target_window is None or self._bubble is None or not self._bubble.isVisible():
            return
        geometry = self._target_window.geometry()
        point = QPoint(
            geometry.center().x() - self._bubble.width() // 2,
            geometry.top() - self._bubble.height() - 8,
        )
        self._bubble.move(keep_on_screen(self._bubble, point, self._target_window))
        self._bubble.raise_()
