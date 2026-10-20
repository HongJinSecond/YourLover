"""Events used by the desktop, conversation workers and affinity service."""

from PySide6.QtCore import QObject, Signal

from core.agent_event import AgentEvent


class EventBus(QObject):
    # Character interactions.
    avatar_hit_area_clicked = Signal(str)
    avatar_drag_started = Signal(str)
    avatar_drag_finished = Signal(int, int)
    avatar_shaken = Signal()
    avatar_state_changed = Signal(str)
    avatar_expression_clear = Signal()

    # Conversation and its visible status.
    user_message_sent = Signal(str)
    chat_session_changed = Signal(str)
    agent_reply_received = Signal(AgentEvent)
    agent_thinking_started = Signal()
    agent_thinking_finished = Signal()
    mini_chat_input_sent = Signal(str)
    mini_chat_thinking_started = Signal()
    mini_chat_reply_received = Signal(str)
    state_emotion_change = Signal(str)
    state_love_change = Signal(int)

    # Changed settings after successful persistence, and navigation.
    settings_batch_changed = Signal(dict)
    show_chat_window_requested = Signal()
    show_setting_window_requested = Signal()
    application_quit_requested = Signal()


event_bus = EventBus()
