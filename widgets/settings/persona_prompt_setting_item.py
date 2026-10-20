"""Read-only default persona preview and a separately retained custom draft."""

from typing import Any

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from agent.prompts import DEFAULT_PERSONA_PROMPT
from widgets.settings.setting_item import SettingItem


class PersonaPromptSettingItem(SettingItem):
    custom_mode_requested = Signal()

    def __init__(self, value: str) -> None:
        super().__init__(
            "CUSTOM_PERSONA_PROMPT",
            "人物描述",
            value,
            extra={
                "help_text": "只需描述身份、性格、背景和说话习惯。权限、好感度、工具及输出规则由系统自动追加。"
            },
        )
        self._use_default = True

    def create_editor(self, parent: QWidget) -> QWidget:
        editor = QWidget(parent)
        layout = QVBoxLayout(editor)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self._text_edit = QPlainTextEdit(editor)
        self._text_edit.setFixedHeight(260)
        self._text_edit.setAccessibleName("人物描述")
        self._text_edit.setPlaceholderText("例如：你叫小林，是一位爱摄影、说话简洁的朋友……")
        self._text_edit.textChanged.connect(self._on_text_changed)
        row = QHBoxLayout()
        self._copy_button = QPushButton("以默认人设为模板", editor)
        self._copy_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._copy_button.setToolTip("用默认人物描述替换自定义草稿，并切换为可编辑模式；保存后才生效")
        self._copy_button.clicked.connect(self._copy_default)
        row.addWidget(self._copy_button)
        row.addStretch()
        layout.addWidget(self._text_edit)
        layout.addLayout(row)
        self._update_editor(self._value)
        return editor

    def set_use_default(self, use_default: bool) -> None:
        self._use_default = use_default
        self._update_editor(self._value)

    def _update_editor(self, value: Any) -> None:
        with QSignalBlocker(self._text_edit):
            self._text_edit.setReadOnly(self._use_default)
            self._text_edit.setPlainText(
                DEFAULT_PERSONA_PROMPT if self._use_default else str(value or "")
            )
        self._text_edit.setToolTip(
            "系统默认人物描述（只读）；切换到自定义或复制为模板后编辑。"
            if self._use_default
            else "自定义只影响人物描述，系统规则始终由程序追加。"
        )

    def _on_text_changed(self) -> None:
        if not self._use_default:
            self.on_value_changed(self._text_edit.toPlainText())

    def _copy_default(self) -> None:
        self.set_value(DEFAULT_PERSONA_PROMPT)
        self.on_value_changed(DEFAULT_PERSONA_PROMPT)
        self.custom_mode_requested.emit()
