"""Conversation window and persisted session navigation."""

import logging
import uuid
from datetime import datetime
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout, QSizeGrip, QVBoxLayout, QWidget

from agent.memory.session_store import SessionRecord, SessionStore, SessionStoreError
from core.agent_event import AgentEvent
from core.event_bus import event_bus
from utils.message_filter import extract_emotion
from widgets.chat.chat_header import ChatHeader
from widgets.chat.chat_history import ChatHistory, MessageRole
from widgets.chat.chat_input import ChatInput
from widgets.chat.chat_session_list import ChatSessionList, SessionData
from widgets.theme import BACKGROUND, SUCCESS, ui_font
from windows.base_window import BaseWindow

logger = logging.getLogger(__name__)


class ChatWindow(BaseWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        self._user_id = "default-user"
        self._session_store = SessionStore()
        self._sessions: dict[str, dict] = {}
        super().__init__(parent)
        # App installs its routes after constructing the windows.
        QTimer.singleShot(0, self._activate_current_session)

    def _setup_window_flags(self) -> None:
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("YourLover · 聊天")
        self.setFont(ui_font())
        self.setMinimumSize(900, 560)
        self.resize(1040, 720)

    def setup_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self._session_list = ChatSessionList(self)
        area = QWidget(self)
        area.setObjectName("chat_area")
        layout = QVBoxLayout(area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._header = ChatHeader(self)
        self._history = ChatHistory(self)
        self._input = ChatInput(self)
        layout.addWidget(self._header)
        layout.addWidget(self._history, 1)
        layout.addWidget(self._input)
        root.addWidget(self._session_list)
        root.addWidget(area, 1)
        self._resize_grip = QSizeGrip(self)
        self._resize_grip.setFixedSize(16, 16)
        self._resize_grip.setToolTip("调整窗口大小")
        if not self._restore_persisted_sessions():
            self.create_new_session()

    def connect_signals(self) -> None:
        # WindowManager appends user messages once, before App starts a worker.
        self._input.send_clicked.connect(event_bus.user_message_sent.emit)
        self._header.settings_clicked.connect(event_bus.show_setting_window_requested.emit)
        self._session_list.new_session_requested.connect(self.create_new_session)
        self._session_list.session_clicked.connect(self._on_session_clicked)
        self._session_list.session_deleted.connect(self._on_session_deleted)
        event_bus.agent_thinking_started.connect(self._on_agent_thinking_started)
        event_bus.agent_thinking_finished.connect(self._on_agent_thinking_finished)
        event_bus.agent_reply_received.connect(self._on_agent_reply_received)
        event_bus.state_emotion_change.connect(self._header.set_emotional)
        event_bus.state_love_change.connect(self._header.set_affection)

    def apply_style(self) -> None:
        self.setStyleSheet(f"QMainWindow, QWidget#chat_area {{ background: {BACKGROUND}; }}")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_resize_grip"):
            self._resize_grip.move(self.width() - 18, self.height() - 18)
            self._resize_grip.setVisible(not self.isMaximized())
            self._resize_grip.raise_()

    def _restore_persisted_sessions(self) -> bool:
        try:
            records = [
                record
                for record in self._session_store.list_sessions(self._user_id)
                if record.session_id != "mini-chat"
            ]
            # Load all records before changing the UI, so a failed read is atomic.
            restored = [
                (record, *self._load_display_messages(record.session_id)) for record in records
            ]
            for record, messages, emotion, love in restored:
                self._sessions[record.session_id] = {
                    "messages": messages,
                    "emotion": emotion,
                    "love": love,
                }
                self._session_list.add_session(
                    SessionData(
                        session_id=record.session_id,
                        title=self._session_title(record, messages),
                        preview=self._session_preview(record, messages),
                        updated_at=self._parse_datetime(record.updated_at),
                    )
                )
        except SessionStoreError:
            logger.exception("读取历史会话失败")
            return False
        if records:
            self._session_list.select_session(records[0].session_id)
        return bool(records)

    def _load_display_messages(
        self, session_id: str
    ) -> tuple[list[tuple[MessageRole, str]], str | None, int | None]:
        messages: list[tuple[MessageRole, str]] = []
        emotion = None
        love = None
        for item in self._session_store.load_memory_items(self._user_id, session_id):
            if item.type != "Chat":
                continue
            role = self._display_role(item.speaker)
            content = self._message_content(item.message.content)
            if role is None or not content:
                continue
            if role == MessageRole.AGENT:
                content, emotions = extract_emotion(content)
                content = content.strip()
                if emotions:
                    emotion = emotions[0]
                if isinstance(item.love, int):
                    love = item.love
            if content:
                messages.append((role, content))
        return messages, emotion, love

    @staticmethod
    def _display_role(speaker: str) -> MessageRole | None:
        normalized = speaker.strip().lower()
        if normalized in {"user", "human"}:
            return MessageRole.USER
        if normalized in {"ai", "assistant"}:
            return MessageRole.AGENT
        return None

    @staticmethod
    def _message_content(content: Any) -> str:
        return str(content).strip() if content is not None else ""

    @staticmethod
    def _session_title(record: SessionRecord, messages: list[tuple[MessageRole, str]]) -> str:
        first_user = next((text for role, text in messages if role == MessageRole.USER), "")
        return record.title.strip() or ChatWindow._short_text(first_user, 12) or "新对话"

    @staticmethod
    def _session_preview(record: SessionRecord, messages: list[tuple[MessageRole, str]]) -> str:
        return (
            ChatWindow._short_text(messages[-1][1] if messages else record.summary, 20)
            or "点击开始聊天"
        )

    @staticmethod
    def _short_text(content: str, limit: int) -> str:
        return content[:limit] + ("…" if len(content) > limit else "")

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.now()

    def _activate_current_session(self) -> None:
        session_id = self._session_list.current_session_id()
        if session_id is not None:
            self._on_session_clicked(session_id)

    def create_new_session(self) -> None:
        session_id = uuid.uuid4().hex[:8]
        data = SessionData(session_id, "新对话", "点击开始聊天")
        self._session_store.create_session(self._user_id, session_id, data.title)
        self._sessions[session_id] = {"messages": []}
        self._session_list.add_session(data)

    def switch_session(self, session_id: str) -> None:
        self._session_list.select_session(session_id)

    def _on_session_clicked(self, session_id: str) -> None:
        event_bus.chat_session_changed.emit(session_id)
        self._history.clear_history()
        session = self._sessions.get(session_id, {"messages": []})
        for role, content in session["messages"]:
            self._history.append_message(role, content)
        emotion = session.get("emotion")
        if emotion in {"Happy", "Sad", "Angry", "Shy"}:
            event_bus.state_emotion_change.emit(emotion)
        if session.get("love") is not None:
            event_bus.state_love_change.emit(int(session["love"]))

    def _on_session_deleted(self, session_id: str) -> None:
        try:
            self._session_store.delete_session(self._user_id, session_id)
        except SessionStoreError:
            logger.exception("删除历史会话失败")
            return
        self._sessions.pop(session_id, None)
        self._session_list.remove_session(session_id)
        if not self._sessions:
            self._history.clear_history()

    def _save_message_to_session(self, role: MessageRole, content: str) -> None:
        session_id = self._session_list.current_session_id()
        if session_id is None:
            return
        # Mini chat has its own fixed session; restore the main session before sending.
        if role == MessageRole.USER:
            event_bus.chat_session_changed.emit(session_id)
        session = self._sessions.setdefault(session_id, {"messages": []})
        session["messages"].append((role, content))
        title_source = content if role == MessageRole.USER else session["messages"][0][1]
        data = SessionData(
            session_id,
            self._short_text(title_source, 12) or "新对话",
            self._short_text(content, 20),
        )
        self._session_list.update_session(data)
        try:
            # ChatAgent owns message persistence; the UI only maintains the title.
            self._session_store.update_session_title(self._user_id, session_id, data.title)
        except SessionStoreError:
            logger.exception("更新会话标题失败")

    def append_user_message(self, text: str) -> None:
        self._history.append_message(MessageRole.USER, text)
        self._save_message_to_session(MessageRole.USER, text)

    def append_agent_message(self, text: str) -> None:
        self._history.append_message(MessageRole.AGENT, text)
        self._save_message_to_session(MessageRole.AGENT, text)

    def clear_chat(self) -> None:
        self._history.clear_history()

    def set_status(self, text: str, color: str = SUCCESS) -> None:
        self._header.set_status(text, color)

    def set_input_enabled(self, enabled: bool) -> None:
        self._input.set_enabled(enabled)

    def _on_agent_thinking_started(self) -> None:
        self.set_status("正在思考…", "#AC8546")
        self.set_input_enabled(False)

    def _on_agent_thinking_finished(self) -> None:
        self.set_status("正在回复…")

    def _on_agent_reply_received(self, event: AgentEvent) -> None:
        self.append_agent_message(event.payload.get("content", ""))
        session_id = self._session_list.current_session_id()
        if session_id in self._sessions:
            session = self._sessions[session_id]
            if event.payload.get("emotion") is not None:
                session["emotion"] = event.payload["emotion"]
            if event.payload.get("love") is not None:
                session["love"] = event.payload["love"]
        self.set_status("在线")
        self.set_input_enabled(True)
