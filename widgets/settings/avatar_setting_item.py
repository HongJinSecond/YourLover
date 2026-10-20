"""Avatar selection is a draft until the settings window successfully saves it."""

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QImageReader
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from widgets.avatar import avatar_path, read_avatar, set_avatar
from widgets.settings.setting_item import SettingItem
from widgets.theme import ACCENT, ACCENT_SOFT, ui_font


class AvatarSettingItem(SettingItem):
    def __init__(self, value: str, root: Path) -> None:
        super().__init__(
            "CHAT_AVATAR",
            "头像",
            value,
            extra={"help_text": "选择 icon 目录中的图片，或上传本地图片。保存后生效。"},
        )
        self._root = root

    def create_editor(self, parent: QWidget) -> QWidget:
        editor = QWidget(parent)
        layout = QHBoxLayout(editor)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        self._preview = QLabel("Y", editor)
        self._preview.setFixedSize(56, 56)
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setFont(ui_font(18, True))
        self._preview.setStyleSheet(
            f"background: {ACCENT_SOFT}; color: {ACCENT}; border-radius: 13px;"
        )
        controls = QVBoxLayout()
        controls.setSpacing(8)
        self._combo_box = QComboBox(editor)
        self._combo_box.setMinimumHeight(44)
        self._combo_box.setMinimumWidth(0)
        self._combo_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._combo_box.setAccessibleName("选择聊天头像")
        buttons = QHBoxLayout()
        self._upload_button = QPushButton("上传图片", editor)
        self._reset_button = QPushButton("恢复默认", editor)
        for button in (self._upload_button, self._reset_button):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumHeight(36)
            buttons.addWidget(button)
        buttons.addStretch()
        controls.addWidget(self._combo_box)
        controls.addLayout(buttons)
        layout.addWidget(self._preview, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(controls, 1)
        self._update_editor(self._value)
        self._combo_box.currentIndexChanged.connect(self._on_selection_changed)
        self._upload_button.clicked.connect(self._upload)
        self._reset_button.clicked.connect(lambda: self._select_value(""))
        return editor

    def _update_editor(self, value: Any) -> None:
        value = str(value or "")
        self._combo_box.blockSignals(True)
        try:
            self._combo_box.clear()
            self._combo_box.addItem("默认字母头像", "")
            directory = self._root / "icon"
            if directory.is_dir():
                for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
                    if path.is_file() and QImageReader(str(path)).canRead():
                        self._combo_box.addItem(path.name, f"icon/{path.name}")
            index = self._combo_box.findData(value)
            if index < 0:
                path = avatar_path(value, self._root)
                suffix = "（不可用）" if read_avatar(path).isNull() else "（待保存）"
                self._combo_box.addItem(f"{path.name}{suffix}", value)
                index = self._combo_box.count() - 1
            self._combo_box.setCurrentIndex(index)
            self._combo_box.setToolTip(self._combo_box.currentText())
        finally:
            self._combo_box.blockSignals(False)
        set_avatar(self._preview, avatar_path(value, self._root))
        self._reset_button.setEnabled(bool(value))

    def _on_selection_changed(self) -> None:
        value = self._combo_box.currentData() or ""
        set_avatar(self._preview, avatar_path(value, self._root))
        self._combo_box.setToolTip(self._combo_box.currentText())
        self._reset_button.setEnabled(bool(value))
        self.on_value_changed(value)

    def _select_value(self, value: str) -> None:
        self.set_value(value)
        self.on_value_changed(value)

    def _upload(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self._widget,
            "选择聊天头像",
            str(self._root / "icon"),
            "图片 (*.png *.jpg *.jpeg *.webp *.bmp *.ico *.svg)",
        )
        if not filename:
            return
        path = Path(filename).resolve()
        if read_avatar(path).isNull():
            QMessageBox.warning(self._widget, "无法读取头像", "请选择有效的图片文件。")
            return
        self._select_value(str(path))
