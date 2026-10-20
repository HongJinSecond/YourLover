"""Construct desktop windows and route navigation and character interactions."""

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication

from avatar.avatar_controller_protocol import AvatarControllerProtocol
from core.agent_event import AgentEvent
from core.event_bus import event_bus
from widgets.theme import keep_on_screen
from windows.chat_window import ChatWindow
from windows.lover_window import LoverWindow
from windows.mini_chat_overlay import MiniChatOverlay
from windows.setting_window import SettingWindow


class WindowManager:
    def __init__(self, app: QApplication) -> None:
        self._app = app
        self._avatar_controller: AvatarControllerProtocol | None = None
        self._lover_window = LoverWindow()
        self._chat_window = ChatWindow()
        self._setting_window = SettingWindow()
        self._mini_chat_overlay = MiniChatOverlay()
        self._setting_window.add_default_settings()
        self._setup_event_routing()

    def _setup_event_routing(self) -> None:
        event_bus.show_chat_window_requested.connect(self.show_chat)
        event_bus.show_setting_window_requested.connect(self.show_settings)
        event_bus.user_message_sent.connect(self._chat_window.append_user_message)
        event_bus.avatar_hit_area_clicked.connect(self._on_avatar_hit_area_clicked)
        event_bus.avatar_shaken.connect(self._on_avatar_shaken)
        event_bus.avatar_state_changed.connect(self._on_avatar_state_changed)
        event_bus.avatar_drag_started.connect(self._on_avatar_drag_started)
        event_bus.avatar_drag_finished.connect(self._on_avatar_drag_finished)
        event_bus.agent_reply_received.connect(self._on_agent_reply)

    def run(self) -> None:
        self._lover_window.show_window()
        self._mini_chat_overlay.set_target_window(self._lover_window)

    def quit(self) -> None:
        self._mini_chat_overlay.hide_window()
        self._app.quit()

    def set_avatar_controller(self, controller: AvatarControllerProtocol) -> None:
        self._avatar_controller = controller
        self._lover_window.set_avatar_widget(controller.get_render_widget())

    def show_chat(self) -> None:
        geometry = self._lover_window.geometry()
        self._chat_window.move(
            keep_on_screen(
                self._chat_window, QPoint(geometry.right() + 20, geometry.y()), self._lover_window
            )
        )
        self._chat_window.show_window()

    def show_settings(self) -> None:
        anchor = self._chat_window if self._chat_window.isVisible() else None
        self._setting_window.slide_in(anchor)

    def _on_avatar_hit_area_clicked(self, area_name: str) -> None:
        if self._avatar_controller is not None:
            self._avatar_controller.on_hit_area(area_name)

    def _on_avatar_shaken(self) -> None:
        if self._avatar_controller is not None:
            self._avatar_controller.on_shake()

    def _on_avatar_drag_started(self, direction: str) -> None:
        if self._avatar_controller is not None:
            self._avatar_controller.on_drag_start(direction)

    def _on_avatar_drag_finished(self, x: int, y: int) -> None:
        if self._avatar_controller is not None:
            self._avatar_controller.on_drag_end()

    def _on_avatar_state_changed(self, state_name: str) -> None:
        if self._avatar_controller is not None:
            self._avatar_controller.set_state(state_name)

    def _on_agent_reply(self, event: AgentEvent) -> None:
        if not self._chat_window.isVisible():
            self.show_chat()
        self._lover_window.shake_avatar()

    @property
    def lover_window(self) -> LoverWindow:
        return self._lover_window

    @property
    def chat_window(self) -> ChatWindow:
        return self._chat_window

    @property
    def setting_window(self) -> SettingWindow:
        return self._setting_window

    @property
    def mini_chat_overlay(self) -> MiniChatOverlay:
        return self._mini_chat_overlay
