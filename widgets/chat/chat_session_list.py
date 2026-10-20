"""Conversation navigation; persistence belongs to ChatWindow."""

from dataclasses import dataclass, field
from datetime import datetime

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.event_bus import event_bus
from utils.config_manager import config
from widgets.base.base_widget import BaseWidget
from widgets.icons import icon
from widgets.theme import (
    ACCENT,
    BORDER,
    BUTTON_STYLE,
    MUTED,
    SCROLL_STYLE,
    SIDEBAR,
    TEXT,
    ui_font,
)


@dataclass
class SessionData:
    session_id: str
    title: str
    preview: str = ""
    updated_at: datetime = field(default_factory=datetime.now)


class SessionItem(QFrame):
    clicked = Signal(str)
    delete_clicked = Signal(str)

    def __init__(self, data: SessionData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._data = data
        self.setObjectName("session_item")
        self.setFixedHeight(82)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(data.title)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 8, 10)
        layout.setSpacing(8)
        text = QVBoxLayout()
        text.setSpacing(5)
        self._title_label = QLabel(self)
        self._title_label.setObjectName("session_title")
        self._title_label.setFont(ui_font(10, True))
        self._preview_label = QLabel(self)
        self._preview_label.setObjectName("session_preview")
        self._preview_label.setFont(ui_font(9))
        for label in (self._title_label, self._preview_label):
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            label.setTextFormat(Qt.TextFormat.PlainText)
            text.addWidget(label)
        self._delete_button = QPushButton(self)
        self._delete_button.setProperty("iconOnly", True)
        self._delete_button.setIcon(icon("trash", MUTED, 15))
        self._delete_button.setFixedSize(26, 28)
        self._delete_button.setToolTip("删除对话")
        self._delete_button.setAccessibleName(f"删除对话：{data.title}")
        self._delete_button.clicked.connect(lambda: self.delete_clicked.emit(self._data.session_id))
        layout.addLayout(text, 1)
        layout.addWidget(self._delete_button)
        self.setStyleSheet(
            BUTTON_STYLE
            + f"""
            QFrame#session_item {{ background: transparent; border: 1px solid transparent; border-radius: 12px; }}
            QFrame#session_item:hover {{ background: #E9EBF4; }}
            QFrame#session_item[selected="true"] {{ background: #FFFFFF; border-color: {BORDER}; }}
            QFrame#session_item:focus {{ border-color: {ACCENT}; }}
            QLabel {{ background: transparent; border: none; }}
            QLabel#session_title {{ color: {TEXT}; }}
            QLabel#session_preview {{ color: {MUTED}; }}
        """
        )
        self.update_data(data)

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", selected)
        self.style().unpolish(self)
        self.style().polish(self)

    def update_data(self, data: SessionData) -> None:
        self._data = data
        self.setToolTip(f"{data.title}\n{data.updated_at:%m/%d %H:%M}")
        self.setAccessibleName(data.title)
        self._update_text()

    def _update_text(self) -> None:
        for label, value in (
            (self._title_label, self._data.title),
            (self._preview_label, self._data.preview),
        ):
            label.setText(
                label.fontMetrics().elidedText(
                    value.replace("\n", " "), Qt.TextElideMode.ElideRight, max(60, label.width())
                )
            )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_text()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._data.session_id)
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.clicked.emit(self._data.session_id)
            event.accept()
            return
        super().keyPressEvent(event)


class ChatSessionList(BaseWidget):
    session_clicked = Signal(str)
    session_deleted = Signal(str)
    new_session_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        self._items: dict[str, SessionItem] = {}
        self._current_session_id: str | None = None
        super().__init__(parent)

    def setup_ui(self) -> None:
        self.setObjectName("chat_session_list")
        self.setFixedWidth(248)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 26, 16, 20)
        layout.setSpacing(20)
        brand = QVBoxLayout()
        brand.setContentsMargins(8, 0, 0, 0)
        brand.setSpacing(5)
        self._brand_title = QLabel(self)
        self._brand_title.setFont(ui_font(16, True))
        self._brand_title.setStyleSheet(f"color: {TEXT};")
        self._brand_title.setTextFormat(Qt.TextFormat.PlainText)
        self._brand_title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._brand_title.installEventFilter(self)
        subtitle = QLabel("你的桌面陪伴助手", self)
        subtitle.setFont(ui_font(9))
        subtitle.setStyleSheet(f"color: {MUTED};")
        brand.addWidget(self._brand_title)
        brand.addWidget(subtitle)
        self._new_button = QPushButton("新建对话", self)
        self._new_button.setProperty("primary", True)
        self._new_button.setIcon(icon("plus", "#FFFFFF", 17))
        self._new_button.setFixedHeight(42)
        self._new_button.setFont(ui_font(10, True))
        self._new_button.setCursor(Qt.CursorShape.PointingHandCursor)
        caption = QHBoxLayout()
        label = QLabel("最近的对话", self)
        label.setFont(ui_font(9))
        label.setStyleSheet(f"color: {MUTED};")
        self._count_label = QLabel("0", self)
        self._count_label.setStyleSheet(f"color: {MUTED};")
        caption.addWidget(label)
        caption.addStretch()
        caption.addWidget(self._count_label)
        self._scroll_area = QScrollArea(self)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self._container = QWidget()
        self._container.setObjectName("session_container")
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(0, 0, 0, 0)
        self._container_layout.setSpacing(6)
        self._container_layout.addStretch()
        self._scroll_area.setWidget(self._container)
        footer = QLabel("留一点时间，给自己。", self)
        footer.setFont(ui_font(8))
        footer.setStyleSheet(f"color: {MUTED};")
        layout.addLayout(brand)
        layout.addWidget(self._new_button)
        layout.addLayout(caption)
        layout.addWidget(self._scroll_area, 1)
        layout.addWidget(footer, 0, Qt.AlignmentFlag.AlignHCenter)
        self._refresh_brand_title()

    def connect_signals(self) -> None:
        self._new_button.clicked.connect(self.new_session_requested.emit)
        event_bus.settings_batch_changed.connect(self._on_settings_changed)

    def _on_settings_changed(self, values: dict) -> None:
        if "CHAT_DISPLAY_NAME" in values:
            self._refresh_brand_title()

    def _refresh_brand_title(self) -> None:
        name = getattr(config, "CHAT_DISPLAY_NAME", "YourLover") or "YourLover"
        self._brand_title.setToolTip(name)
        if self._brand_title.isVisible():
            name = self._brand_title.fontMetrics().elidedText(
                name, Qt.TextElideMode.ElideRight, self._brand_title.width()
            )
        self._brand_title.setText(name)

    def eventFilter(self, obj, event) -> bool:
        if obj is self._brand_title and event.type() == QEvent.Type.Resize:
            self._refresh_brand_title()
        return super().eventFilter(obj, event)

    def apply_style(self) -> None:
        self.setStyleSheet(
            BUTTON_STYLE
            + SCROLL_STYLE
            + f"""
            QWidget#chat_session_list {{ background: {SIDEBAR}; border-right: 1px solid {BORDER}; }}
            QWidget#session_container {{ background: transparent; }}
        """
        )

    def add_session(self, data: SessionData) -> None:
        if data.session_id in self._items:
            self.remove_session(data.session_id)
        item = SessionItem(data, self._container)
        item.clicked.connect(self.select_session)
        item.delete_clicked.connect(self.session_deleted.emit)
        self._container_layout.insertWidget(len(self._items), item)
        self._items[data.session_id] = item
        self._count_label.setText(str(len(self._items)))
        self.select_session(data.session_id)

    def remove_session(self, session_id: str) -> None:
        item = self._items.pop(session_id, None)
        if item is not None:
            self._container_layout.removeWidget(item)
            item.hide()
            item.deleteLater()
        self._count_label.setText(str(len(self._items)))
        if self._current_session_id == session_id:
            self._current_session_id = None
            if self._items:
                self.select_session(next(iter(self._items)))

    def update_session(self, data: SessionData) -> None:
        if data.session_id in self._items:
            self._items[data.session_id].update_data(data)

    def select_session(self, session_id: str) -> None:
        if session_id not in self._items:
            return
        for key, item in self._items.items():
            item.set_selected(key == session_id)
        self._current_session_id = session_id
        self.session_clicked.emit(session_id)

    def current_session_id(self) -> str | None:
        return self._current_session_id
