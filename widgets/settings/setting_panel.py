"""Scrollable form with local field signals; drafts never reach the event bus."""

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from widgets.base.base_widget import BaseWidget
from widgets.settings.setting_item import SettingItem
from widgets.theme import SCROLL_STYLE


class SettingPanel(BaseWidget):
    item_value_changed = Signal(str, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        self._items: dict[str, SettingItem] = {}
        super().__init__(parent)

    def setup_ui(self) -> None:
        self._scroll_area = QScrollArea(self)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self._container = QWidget()
        self._container.setStyleSheet("background: transparent;")
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(28, 8, 28, 16)
        self._layout.setSpacing(20)
        self._layout.addStretch()
        self._scroll_area.setWidget(self._container)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._scroll_area)

    def apply_style(self) -> None:
        self.setStyleSheet(SCROLL_STYLE)

    def add_item(self, item: SettingItem) -> None:
        self.remove_item(item.key)
        widget = item.create_widget(self._container)
        item.value_changed.connect(self.item_value_changed.emit)
        self._layout.insertWidget(len(self._items), widget)
        self._items[item.key] = item

    def remove_item(self, key: str) -> None:
        item = self._items.pop(key, None)
        if item is not None:
            self._layout.removeWidget(item._widget)
            item.on_removed()

    def get_value(self, key: str) -> Any:
        item = self._items.get(key)
        return item.get_value() if item is not None else None

    def set_item_enabled(self, key: str, enabled: bool) -> None:
        if key in self._items:
            self._items[key].set_enabled(enabled)

    def set_item_visible(self, key: str, visible: bool) -> None:
        if key in self._items:
            self._items[key].set_visible(visible)

    def get_values(self) -> dict[str, Any]:
        return {key: item.get_value() for key, item in self._items.items()}

    def set_values(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            if key in self._items:
                self._items[key].set_value(value)

    def items(self) -> list[SettingItem]:
        return list(self._items.values())
