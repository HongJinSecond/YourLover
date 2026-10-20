"""Chat appearance and model settings; validate, persist, then notify the application."""

from typing import Any

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from agent.models.OpenAIModels import (
    DEFAULT_OPENAI_BASE_URL,
    normalize_base_url,
    validate_openai_settings,
)
from core.event_bus import event_bus
from utils.config_manager import config
from widgets.avatar import import_avatar
from widgets.icons import icon
from widgets.settings.avatar_setting_item import AvatarSettingItem
from widgets.settings.persona_prompt_setting_item import PersonaPromptSettingItem
from widgets.settings.setting_item import OptionSettingItem, SettingItem, TextSettingItem
from widgets.settings.setting_panel import SettingPanel
from widgets.theme import (
    ACCENT,
    ACCENT_SOFT,
    BORDER,
    BUTTON_STYLE,
    MUTED,
    SURFACE,
    TEXT,
    add_shadow,
    keep_on_screen,
    screen_bounds,
    ui_font,
)
from windows.base_window import BaseWindow


class SettingWindow(BaseWindow):
    saved = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        self._saved_values: dict[str, Any] = {}
        self._drag_position: QPoint | None = None
        super().__init__(parent)

    def _setup_window_flags(self) -> None:
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowTitle("YourLover · 设置")
        self.resize(520, 690)

    def setup_ui(self) -> None:
        shell = QWidget(self)
        outer = QVBoxLayout(shell)
        outer.setContentsMargins(12, 12, 12, 16)
        self._central_widget = QWidget(shell)
        self._central_widget.setObjectName("settings_card")
        add_shadow(self._central_widget)
        layout = QVBoxLayout(self._central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QWidget(self)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(28, 26, 22, 22)
        header_layout.setSpacing(14)
        mark = QLabel(self)
        mark.setObjectName("settings_mark")
        mark.setPixmap(icon("settings", ACCENT, 22).pixmap(22, 22))
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(44, 44)
        title_layout = QVBoxLayout()
        title_layout.setSpacing(5)
        self._title_label = QLabel("设置", self)
        self._title_label.setFont(ui_font(16, True))
        subtitle = QLabel("定制形象、人物与模型服务", self)
        subtitle.setFont(ui_font(9))
        subtitle.setStyleSheet(f"color: {MUTED};")
        title_layout.addWidget(self._title_label)
        title_layout.addWidget(subtitle)
        self._close_button = QPushButton(self)
        self._close_button.setProperty("iconOnly", True)
        self._close_button.setIcon(icon("close"))
        self._close_button.setFixedSize(32, 32)
        self._close_button.setToolTip("关闭设置")
        self._close_button.setAccessibleName("关闭设置")
        header_layout.addWidget(mark)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        header_layout.addWidget(self._close_button, 0, Qt.AlignmentFlag.AlignTop)
        self._setting_panel = SettingPanel(self)
        self._provider_note = QLabel(self)
        self._provider_note.setFont(ui_font(9))
        self._provider_note.setWordWrap(True)
        self._provider_note.setObjectName("provider_note")
        note_layout = QVBoxLayout()
        note_layout.setContentsMargins(28, 6, 28, 18)
        note_layout.addWidget(self._provider_note)
        footer = QWidget(self)
        footer.setObjectName("settings_footer")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(28, 18, 28, 20)
        self._save_button = QPushButton("保存设置", self)
        self._save_button.setProperty("primary", True)
        self._save_button.setMinimumHeight(40)
        self._cancel_button = QPushButton("取消", self)
        self._cancel_button.setMinimumHeight(40)
        footer_layout.addStretch()
        footer_layout.addWidget(self._cancel_button)
        footer_layout.addSpacing(6)
        footer_layout.addWidget(self._save_button)
        for button in (self._save_button, self._cancel_button, self._close_button):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(header)
        layout.addWidget(self._setting_panel, 1)
        layout.addLayout(note_layout)
        layout.addWidget(footer)
        outer.addWidget(self._central_widget)
        self.setCentralWidget(shell)
        self.setFont(ui_font())

    def connect_signals(self) -> None:
        self._close_button.clicked.connect(self.hide_window)
        self._cancel_button.clicked.connect(self.hide_window)
        self._save_button.clicked.connect(self.save_settings)
        self._setting_panel.item_value_changed.connect(self._on_setting_changed)

    def apply_style(self) -> None:
        self.setStyleSheet(
            BUTTON_STYLE
            + f"""
            QWidget#settings_card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 20px; color: {TEXT}; }}
            QLabel#settings_mark {{ background: {ACCENT_SOFT}; border-radius: 13px; }}
            QLabel#provider_note {{ background: #F5F3FB; color: #777087; border-radius: 10px; padding: 12px; }}
            QWidget#settings_footer {{ border-top: 1px solid {BORDER}; background: transparent; }}
        """
        )

    def _on_setting_changed(self, key: str, value: Any) -> None:
        if key == "LLM_PROVIDER":
            self._sync_model_fields()
        elif key == "PERSONA_PROMPT_MODE":
            self._sync_prompt_fields()

    def _sync_prompt_fields(self) -> None:
        self._persona_item.set_use_default(self.get_setting_value("PERSONA_PROMPT_MODE") == "default")

    def _use_custom_persona(self) -> None:
        self.set_values({"PERSONA_PROMPT_MODE": "custom"})

    def _sync_model_fields(self) -> None:
        remote = self.get_setting_value("LLM_PROVIDER") == "openai"
        for key in ("OPENAI_BASE_URL", "OPENAI_MODEL_ID", "OPENAI_API_KEY"):
            self._setting_panel.set_item_enabled(key, remote)
            self._setting_panel.set_item_visible(key, remote)
        for key in ("api_base", "CHAT_MODEL_NAME"):
            self._setting_panel.set_item_enabled(key, not remote)
            self._setting_panel.set_item_visible(key, not remote)
        self._provider_note.setText(
            "保存后在下一次聊天中验证连接。所选模型需支持工具调用。"
            if remote
            else "请先启动 Ollama，并确保所填模型已下载。保存时将检查本地连接。"
        )

    def add_setting_item(self, item: SettingItem) -> None:
        self._setting_panel.add_item(item)

    def get_setting_value(self, key: str) -> Any:
        return self._setting_panel.get_value(key)

    def get_all_values(self) -> dict[str, Any]:
        return self._setting_panel.get_values()

    def set_values(self, values: dict[str, Any]) -> None:
        self._setting_panel.set_values(values)

    def save_settings(self) -> None:
        values = self.get_all_values()
        try:
            values["CHAT_DISPLAY_NAME"] = str(values["CHAT_DISPLAY_NAME"]).strip() or "YourLover"
            if values["PERSONA_PROMPT_MODE"] not in {"default", "custom"}:
                raise ValueError("请选择系统默认或自定义人物提示词。")
            if (
                values["PERSONA_PROMPT_MODE"] == "custom"
                and not str(values["CUSTOM_PERSONA_PROMPT"]).strip()
            ):
                raise ValueError("请填写自定义人物描述，或切换为系统默认提示词。")
            model_changed = any(
                value != self._saved_values.get(key)
                for key, value in values.items()
                if key in {
                    "LLM_PROVIDER", "api_base", "CHAT_MODEL_NAME",
                    "OPENAI_MODEL_ID", "OPENAI_BASE_URL", "OPENAI_API_KEY",
                }
            )
            if model_changed and values.get("LLM_PROVIDER") == "openai":
                model, url, key = validate_openai_settings(
                    values["OPENAI_MODEL_ID"], values["OPENAI_BASE_URL"], values["OPENAI_API_KEY"]
                )
                values.update(OPENAI_MODEL_ID=model, OPENAI_BASE_URL=url, OPENAI_API_KEY=key)
            elif model_changed:
                values["api_base"] = normalize_base_url(values["api_base"])
                values["CHAT_MODEL_NAME"] = str(values["CHAT_MODEL_NAME"]).strip()
                if not values["CHAT_MODEL_NAME"]:
                    raise ValueError("请填写 Ollama 模型名称")
            if values["CHAT_AVATAR"] != self._saved_values.get("CHAT_AVATAR"):
                values["CHAT_AVATAR"] = import_avatar(values["CHAT_AVATAR"], config.config_dir)
            config.update(values)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "设置未保存", str(exc))
            return
        changes = {key: value for key, value in values.items() if value != self._saved_values.get(key)}
        self._saved_values = dict(values)
        self.set_values(values)
        self.saved.emit()
        event_bus.settings_batch_changed.emit(changes)
        self.hide_window()

    def slide_in(self, anchor: QWidget | None = None) -> None:
        self._saved_values = {
            key: getattr(config, key, value) for key, value in self._saved_values.items()
        }
        self.set_values(self._saved_values)
        self._sync_model_fields()
        self._sync_prompt_fields()
        bounds = screen_bounds(anchor)
        self.resize(520, min(690, bounds.height() - 24))
        point = (
            QPoint(anchor.geometry().right() + 12, anchor.y())
            if anchor is not None
            else bounds.center() - self.rect().center()
        )
        self.move(keep_on_screen(self, point, anchor))
        self.show_window()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() < 110:
            self._drag_position = event.globalPosition().toPoint() - self.pos()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_position is not None:
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_position = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.hide_window()
        else:
            super().keyPressEvent(event)

    def add_default_settings(self) -> None:
        self.add_setting_item(
            TextSettingItem(
                "CHAT_DISPLAY_NAME",
                "显示名称",
                getattr(config, "CHAT_DISPLAY_NAME", "YourLover"),
                extra={"placeholder": "YourLover", "max_length": 64},
            )
        )
        self.add_setting_item(
            AvatarSettingItem(getattr(config, "CHAT_AVATAR", ""), config.config_dir)
        )
        self.add_setting_item(
            OptionSettingItem(
                "PERSONA_PROMPT_MODE",
                "人物提示词",
                getattr(config, "PERSONA_PROMPT_MODE", "default"),
                extra={
                    "options": ["default", "custom"],
                    "option_labels": {"default": "系统默认（只读）", "custom": "自定义人物"},
                },
            )
        )
        self._persona_item = PersonaPromptSettingItem(getattr(config, "CUSTOM_PERSONA_PROMPT", ""))
        self._persona_item.custom_mode_requested.connect(self._use_custom_persona)
        self.add_setting_item(self._persona_item)
        self.add_setting_item(
            OptionSettingItem(
                "LLM_PROVIDER",
                "模型服务",
                getattr(config, "LLM_PROVIDER", "ollama"),
                extra={
                    "options": ["ollama", "openai"],
                    "option_labels": {"ollama": "本地 Ollama", "openai": "OpenAI 兼容接口"},
                },
            )
        )
        self.add_setting_item(
            TextSettingItem(
                "api_base", "Ollama URL", getattr(config, "api_base", "http://localhost:11434")
            )
        )
        self.add_setting_item(
            TextSettingItem(
                "CHAT_MODEL_NAME",
                "Ollama 模型",
                config.CHAT_MODEL_NAME,
                extra={"placeholder": "已下载的模型名称"},
            )
        )
        self.add_setting_item(
            TextSettingItem(
                "OPENAI_BASE_URL",
                "API URL",
                getattr(config, "OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
                extra={
                    "placeholder": "https://服务商地址/v1",
                    "help_text": "填写 Base URL，保留 /v1 等版本路径。",
                },
            )
        )
        self.add_setting_item(
            TextSettingItem(
                "OPENAI_API_KEY",
                "API Key",
                getattr(config, "OPENAI_API_KEY", ""),
                extra={"password": True, "placeholder": "输入服务商提供的密钥"},
            )
        )
        self.add_setting_item(
            TextSettingItem(
                "OPENAI_MODEL_ID",
                "Model ID",
                getattr(config, "OPENAI_MODEL_ID", ""),
                extra={"placeholder": "服务商提供的模型 ID"},
            )
        )
        self._saved_values = self.get_all_values()
        self._sync_model_fields()
        self._sync_prompt_fields()
