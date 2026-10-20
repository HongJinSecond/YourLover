"""Typed form rows for application settings."""

from typing import Any

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QComboBox, QLabel, QLineEdit, QVBoxLayout, QWidget

from widgets.icons import icon
from widgets.theme import FIELD_STYLE, MUTED, TEXT, ui_font


class SettingItem(QObject):
    value_changed = Signal(str, object)

    def __init__(
        self, key: str, label: str, default_value: Any, extra: dict[str, Any] | None = None
    ) -> None:
        super().__init__()
        self._key = key
        self._label = label
        self._value = default_value
        self._extra = extra or {}
        self._widget: QWidget | None = None

    @property
    def key(self) -> str:
        return self._key

    def get_value(self) -> Any:
        return self._value

    def set_value(self, value: Any) -> None:
        self._value = value
        self._update_editor(value)

    def on_value_changed(self, value: Any) -> None:
        self._value = value
        self.value_changed.emit(self._key, value)

    def set_enabled(self, enabled: bool) -> None:
        if self._widget is not None:
            self._widget.setEnabled(enabled)

    def set_visible(self, visible: bool) -> None:
        if self._widget is not None:
            self._widget.setVisible(visible)

    def create_widget(self, parent: QWidget | None = None) -> QWidget:
        container = QWidget(parent)
        container.setObjectName(f"setting_item_{self._key}")
        self.setParent(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        label = QLabel(self._label, container)
        label.setFont(ui_font(9, True))
        label.setStyleSheet(f"color: {TEXT};")
        editor = self.create_editor(container)
        editor.setFont(ui_font(10))
        editor.setAccessibleName(self._label)
        editor.setStyleSheet(FIELD_STYLE)
        label.setBuddy(editor)
        layout.addWidget(label)
        layout.addWidget(editor)
        help_text = self._extra.get("help_text")
        if help_text:
            help_label = QLabel(help_text, container)
            help_label.setWordWrap(True)
            help_label.setFont(ui_font(8))
            help_label.setStyleSheet(f"color: {MUTED};")
            layout.addWidget(help_label)
        self._widget = container
        return container

    def create_editor(self, parent: QWidget) -> QWidget:
        raise NotImplementedError

    def _update_editor(self, value: Any) -> None:
        raise NotImplementedError

    def on_removed(self) -> None:
        if self._widget is not None:
            self._widget.hide()
            self._widget.deleteLater()


class TextSettingItem(SettingItem):
    def create_editor(self, parent: QWidget) -> QWidget:
        self._line_edit = QLineEdit(str(self._value), parent)
        if "max_length" in self._extra:
            self._line_edit.setMaxLength(self._extra["max_length"])
        self._line_edit.setPlaceholderText(self._extra.get("placeholder", ""))
        self._line_edit.setMinimumHeight(44)
        self._line_edit.textChanged.connect(self.on_value_changed)
        if self._extra.get("password"):
            self._line_edit.setEchoMode(QLineEdit.EchoMode.Password)
            self._reveal_action = self._line_edit.addAction(
                icon("eye"), QLineEdit.ActionPosition.TrailingPosition
            )
            self._reveal_action.setToolTip("显示密钥")
            self._reveal_action.setCheckable(True)
            self._reveal_action.toggled.connect(self._set_password_visible)
        return self._line_edit

    def _set_password_visible(self, visible: bool) -> None:
        self._line_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if visible else QLineEdit.EchoMode.Password
        )
        self._reveal_action.setIcon(icon("eye_off" if visible else "eye"))
        self._reveal_action.setToolTip("隐藏密钥" if visible else "显示密钥")

    def _update_editor(self, value: Any) -> None:
        self._line_edit.setText(str(value))
        if self._extra.get("password"):
            self._reveal_action.setChecked(False)


class OptionSettingItem(SettingItem):
    def create_editor(self, parent: QWidget) -> QWidget:
        self._combo_box = QComboBox(parent)
        self._combo_box.setMinimumHeight(44)
        labels = self._extra.get("option_labels", {})
        for option in self._extra.get("options", []):
            self._combo_box.addItem(labels.get(option, str(option)), option)
        self._update_editor(self._value)
        self._combo_box.currentIndexChanged.connect(
            lambda: self.on_value_changed(self._combo_box.currentData())
        )
        return self._combo_box

    def _update_editor(self, value: Any) -> None:
        index = self._combo_box.findData(value)
        if index >= 0:
            self._combo_box.setCurrentIndex(index)
