"""Exercise real Qt interactions without starting a model or a desktop window."""

from types import SimpleNamespace
from unittest.mock import Mock

from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLineEdit, QWidget

from core.agent_event import AgentEvent
from core.event_bus import event_bus
from widgets.chat.chat_input import ChatInput
from widgets.mini_chat.speech_bubble import SpeechBubble
from widgets.settings.setting_item import TextSettingItem
from widgets.theme import keep_on_screen, screen_bounds


def test_composer_shortcut_respects_busy_and_blank(qt_app):
    composer = ChatInput()
    messages = []
    composer.send_clicked.connect(messages.append)
    composer.set_text("  hello  ")
    QTest.keyClick(composer._text_edit, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    assert messages == ["hello"]
    assert composer.get_text() == ""
    composer._on_send()
    composer.set_text("draft")
    composer.set_enabled(False)
    composer._on_send()
    assert messages == ["hello"]
    assert composer.get_text() == "draft"
    composer.deleteLater()
    qt_app.processEvents()


def test_password_reveal_is_reset_when_restoring_a_form(qt_app):
    row = TextSettingItem("key", "Key", "secret", extra={"password": True})
    widget = row.create_widget()
    row._reveal_action.trigger()
    assert row._line_edit.echoMode() == QLineEdit.EchoMode.Normal
    row.set_value("replacement")
    assert row._line_edit.echoMode() == QLineEdit.EchoMode.Password
    widget.deleteLater()
    qt_app.processEvents()


def test_mini_chat_send_once_busy_reply_and_escape(qt_app):
    from windows.mini_chat_overlay import MiniChatOverlay

    mini = MiniChatOverlay()
    received = []
    handler = received.append
    event_bus.mini_chat_input_sent.connect(handler)
    try:
        mini.expand()
        mini._input.setText(" hello ")
        mini._on_send()
        mini._on_send()
        assert received == ["hello"]
        assert not mini._input.isEnabled()
        mini.show_thinking()
        assert mini._bubble._thinking_timer.isActive()
        mini.show_reply("reply")
        assert mini._input.isEnabled()
        assert not mini._bubble._thinking_timer.isActive()
        event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
        assert mini.eventFilter(mini._input, event)
        assert not mini._expanded
        mini.hide_window()
        assert not mini._bubble.isVisible()
        assert not mini._bubble._auto_hide_timer.isActive()
    finally:
        event_bus.mini_chat_input_sent.disconnect(handler)
        mini._bubble.deleteLater()
        mini.deleteLater()
        qt_app.processEvents()


def test_long_reply_remains_readable_with_scroll_and_literal_markup(qt_app):
    bubble = SpeechBubble()
    bubble.set_auto_hide(0)
    content = "<b>literal text</b>\n" + "This is a long reply. " * 200
    bubble.show_message(content)
    qt_app.processEvents()
    assert bubble.width() <= 360
    assert bubble.height() <= 292
    assert bubble._label.text() == content
    assert bubble._label.textFormat() == Qt.TextFormat.PlainText
    assert bubble._scroll_area.verticalScrollBar().maximum() > 0
    assert not bubble._auto_hide_timer.isActive()
    bubble.hide_bubble()
    bubble.deleteLater()
    qt_app.processEvents()


def test_floating_geometry_is_clamped_to_available_screen(qt_app):
    widget = QWidget()
    widget.resize(420, 72)
    position = keep_on_screen(widget, QPoint(100000, -100000))
    bounds = screen_bounds()
    widget.move(position)
    assert bounds.contains(widget.geometry())
    widget.deleteLater()


def test_main_chat_bubble_reflows_without_truncating_text(qt_app):
    from PySide6.QtWidgets import QVBoxLayout

    from widgets.chat.chat_bubble import ChatBubble, MessageData, MessageRole

    parent = QWidget()
    layout = QVBoxLayout(parent)
    bubble = ChatBubble(parent)
    content = "A longer reply with paragraphs. " * 25 + "\n\nLast paragraph."
    bubble.refresh(MessageData(MessageRole.AGENT, content))
    layout.addWidget(bubble)
    parent.resize(700, 700)
    parent.show()
    qt_app.processEvents()
    wide_height = bubble.height()
    assert bubble._content_label.text() == content
    assert bubble._content_label.height() >= bubble._content_label.heightForWidth(
        bubble._content_label.width()
    )
    parent.resize(400, 700)
    qt_app.processEvents()
    assert bubble.height() > wide_height
    parent.close()
    parent.deleteLater()
    qt_app.processEvents()


def test_chat_turns_and_session_switch_do_not_duplicate_or_replace_messages(
    qt_app, tmp_path, monkeypatch
):
    import windows.chat_window as module
    from agent.memory.session_store import SessionStore

    monkeypatch.setattr(module, "SessionStore", lambda: SessionStore(tmp_path / "sessions.sqlite3"))
    window = module.ChatWindow()
    qt_app.processEvents()
    first = window._session_list.current_session_id()
    for user, reply in [("first", "reply one"), ("second", "reply two")]:
        window.append_user_message(user)
        window._on_agent_reply_received(
            AgentEvent(type="reply", payload={"content": reply, "emotion": "Happy", "love": 55})
        )
    expected = ["first", "reply one", "second", "reply two"]
    assert [bubble.message().content for bubble in window._history.bubbles()] == expected
    window.create_new_session()
    assert not window._history.bubbles()
    window.switch_session(first)
    assert [bubble.message().content for bubble in window._history.bubbles()] == expected
    window._on_session_deleted(first)
    assert not window._history.bubbles()
    window.deleteLater()
    qt_app.processEvents()


def test_exit_waits_for_workers_and_skips_model_reload(qt_app):
    from app import App

    app = App.__new__(App)
    worker = Mock()
    app._active_workers = {worker}
    app._proactive_engine = Mock()
    app._window_manager = SimpleNamespace(
        quit=Mock(),
        lover_window=Mock(),
        chat_window=Mock(),
        setting_window=Mock(),
        mini_chat_overlay=Mock(),
    )
    app._pending_model_reload = True
    app._apply_model_settings = Mock()
    app.quit()
    app._proactive_engine.stop.assert_called_once()
    app._window_manager.quit.assert_not_called()
    app._cleanup_worker(worker)
    app._window_manager.quit.assert_called_once()
    app._apply_model_settings.assert_not_called()
