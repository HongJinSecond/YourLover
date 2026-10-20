"""The character's context menu with the application's three supported actions."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMenu, QWidget

from core.event_bus import event_bus
from widgets.icons import icon
from widgets.theme import ACCENT_SOFT, BORDER, MUTED, TEXT, ui_font


class ContextMenu(QMenu):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFont(ui_font())
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        self.setStyleSheet(f"""
            QMenu {{ background: #FFFFFF; border: 1px solid {BORDER}; border-radius: 12px; padding: 6px; }}
            QMenu::item {{ color: {TEXT}; padding: 10px 32px 10px 12px; margin: 2px; border-radius: 7px; }}
            QMenu::item:selected {{ background: {ACCENT_SOFT}; }}
            QMenu::separator {{ height: 1px; background: {BORDER}; margin: 5px 10px; }}
        """)
        self.addAction(icon("chat"), "打开聊天", event_bus.show_chat_window_requested.emit)
        self.addAction(icon("settings"), "设置", event_bus.show_setting_window_requested.emit)
        self.addSeparator()
        self.addAction(icon("close", MUTED), "退出", event_bus.application_quit_requested.emit)
