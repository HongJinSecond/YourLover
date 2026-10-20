"""Multiline composer with Ctrl+Enter to send."""

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget

from widgets.base.base_widget import BaseWidget
from widgets.icons import icon
from widgets.theme import BACKGROUND, BORDER, BUTTON_STYLE, MUTED, SURFACE, TEXT, ui_font


class _SendShortcutFilter(QObject):
    send_triggered = Signal()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress:
            if (
                event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier
            ):
                self.send_triggered.emit()
                return True
        return super().eventFilter(obj, event)


class ChatInput(BaseWidget):
    send_clicked = Signal(str)

    def setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 12, 28, 18)
        layout.setSpacing(9)
        self._composer = QWidget(self)
        self._composer.setObjectName("composer")
        row = QHBoxLayout(self._composer)
        row.setContentsMargins(14, 10, 10, 10)
        row.setSpacing(12)
        self._text_edit = QTextEdit(self)
        self._text_edit.setObjectName("message_input")
        self._text_edit.setAcceptRichText(False)
        self._text_edit.setPlaceholderText("想聊些什么？")
        self._text_edit.setAccessibleName("聊天消息")
        self._text_edit.setFixedHeight(70)
        self._text_edit.setFont(ui_font())
        self._send_button = QPushButton("发送", self)
        self._send_button.setProperty("primary", True)
        self._send_button.setIcon(icon("send", "#FFFFFF", 16))
        self._send_button.setFixedSize(88, 40)
        self._send_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._send_button.setToolTip("发送消息 · Ctrl+Enter")
        row.addWidget(self._text_edit, 1)
        row.addWidget(self._send_button, 0, Qt.AlignmentFlag.AlignBottom)
        hint = QLabel("Enter 换行   ·   Ctrl+Enter 发送", self)
        hint.setFont(ui_font(8))
        hint.setStyleSheet(f"color: {MUTED};")
        layout.addWidget(self._composer)
        layout.addWidget(hint, 0, Qt.AlignmentFlag.AlignRight)

    def connect_signals(self) -> None:
        self._send_button.clicked.connect(self._on_send)
        self._text_edit.textChanged.connect(self._sync_send_button)
        self._shortcut_filter = _SendShortcutFilter(self)
        self._shortcut_filter.send_triggered.connect(self._on_send)
        self._text_edit.installEventFilter(self._shortcut_filter)
        self._sync_send_button()

    def apply_style(self) -> None:
        self.setStyleSheet(
            BUTTON_STYLE
            + f"""
            ChatInput {{ background: {BACKGROUND}; }}
            QWidget#composer {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px; }}
            QTextEdit#message_input {{ background: transparent; color: {TEXT}; border: none; padding: 3px; selection-background-color: #EDE9F9; }}
            QTextEdit#message_input:focus {{ border: none; }}
        """
        )

    def _sync_send_button(self) -> None:
        self._send_button.setEnabled(self._text_edit.isEnabled() and bool(self.get_text().strip()))

    def _on_send(self) -> None:
        text = self.get_text().strip()
        if not text or not self._text_edit.isEnabled():
            return
        self.clear()
        self.send_clicked.emit(text)

    def get_text(self) -> str:
        return self._text_edit.toPlainText()

    def set_text(self, text: str) -> None:
        self._text_edit.setPlainText(text)

    def clear(self) -> None:
        self._text_edit.clear()

    def set_enabled(self, enabled: bool) -> None:
        self._text_edit.setEnabled(enabled)
        self._sync_send_button()

    def focus(self) -> None:
        self._text_edit.setFocus()
