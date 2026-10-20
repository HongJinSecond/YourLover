"""Default prompt preservation, custom drafts and save-only activation."""

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from agent.prompts import DEFAULT_PERSONA_PROMPT, GLOBAL_SYSTEM_RULES, get_persona_prompt
from core.event_bus import event_bus
from utils.config_manager import ConfigManager


@pytest.fixture
def persona_settings(tmp_path, monkeypatch, qt_app):
    import windows.setting_window as module

    path = tmp_path / "config.json"
    path.write_text('{"CHAT_MODEL_NAME": "local-model"}', encoding="utf-8")
    config = ConfigManager(path)
    monkeypatch.setattr(module, "config", config)
    warning = Mock()
    monkeypatch.setattr(module.QMessageBox, "warning", warning)
    window = module.SettingWindow()
    window.add_default_settings()
    yield window, config, path, warning
    window.deleteLater()
    qt_app.processEvents()


def test_default_is_read_only_and_uses_bundled_copy(persona_settings, qt_app):
    window, config, path, warning = persona_settings
    editor = window._persona_item._text_edit
    template = Path(__file__).resolve().parents[2] / "agent/prompts/default_persona.md"
    before = hashlib.sha256(template.read_bytes()).hexdigest()
    assert editor.isReadOnly()
    assert editor.toPlainText() == DEFAULT_PERSONA_PROMPT
    assert "【输出格式规约】" not in editor.toPlainText()
    assert GLOBAL_SYSTEM_RULES not in editor.toPlainText()
    QTest.keyClicks(editor, "cannot edit default")
    assert editor.toPlainText() == DEFAULT_PERSONA_PROMPT
    window.set_values({"CUSTOM_PERSONA_PROMPT": "a saved custom draft"})
    window.save_settings()
    restored = ConfigManager(path)
    assert restored.PERSONA_PROMPT_MODE == "default"
    assert get_persona_prompt(restored) == DEFAULT_PERSONA_PROMPT
    assert restored.CUSTOM_PERSONA_PROMPT == "a saved custom draft"
    assert hashlib.sha256(template.read_bytes()).hexdigest() == before
    warning.assert_not_called()


def test_custom_draft_survives_mode_switches_and_restart(persona_settings):
    window, config, path, warning = persona_settings
    draft = "你叫阿林，喜欢摄影。\n说话简洁，保留自己的判断。"
    window.set_values({"PERSONA_PROMPT_MODE": "custom"})
    editor = window._persona_item._text_edit
    assert not editor.isReadOnly()
    editor.setPlainText(draft)
    assert get_persona_prompt(config) == DEFAULT_PERSONA_PROMPT
    window.set_values({"PERSONA_PROMPT_MODE": "default"})
    assert editor.isReadOnly()
    assert editor.toPlainText() == DEFAULT_PERSONA_PROMPT
    window.set_values({"PERSONA_PROMPT_MODE": "custom"})
    assert editor.toPlainText() == draft
    window.save_settings()
    restored = ConfigManager(path)
    assert get_persona_prompt(restored) == draft
    window.set_values({"PERSONA_PROMPT_MODE": "default"})
    window.save_settings()
    restored = ConfigManager(path)
    assert get_persona_prompt(restored) == DEFAULT_PERSONA_PROMPT
    assert restored.CUSTOM_PERSONA_PROMPT == draft
    warning.assert_not_called()


def test_copy_default_creates_editable_custom_without_changing_default(persona_settings):
    window, config, path, warning = persona_settings
    QTest.mouseClick(window._persona_item._copy_button, Qt.MouseButton.LeftButton)
    editor = window._persona_item._text_edit
    assert window.get_setting_value("PERSONA_PROMPT_MODE") == "custom"
    assert not editor.isReadOnly()
    assert editor.toPlainText() == DEFAULT_PERSONA_PROMPT
    editor.appendPlainText("额外习惯：爱用短句。")
    window.save_settings()
    assert get_persona_prompt(ConfigManager(path)).endswith("额外习惯：爱用短句。")
    window.set_values({"PERSONA_PROMPT_MODE": "default"})
    assert editor.isReadOnly()
    assert editor.toPlainText() == DEFAULT_PERSONA_PROMPT
    warning.assert_not_called()


def test_cancel_restores_saved_prompt_and_mode(persona_settings):
    window, config, path, warning = persona_settings
    window.set_values({"PERSONA_PROMPT_MODE": "custom", "CUSTOM_PERSONA_PROMPT": "已保存的人物"})
    window.save_settings()
    original = path.read_bytes()
    window._persona_item._text_edit.setPlainText("未保存的新人物")
    window.set_values({"PERSONA_PROMPT_MODE": "default"})
    QTest.mouseClick(window._cancel_button, Qt.MouseButton.LeftButton)
    window.slide_in()
    assert window.get_setting_value("PERSONA_PROMPT_MODE") == "custom"
    assert window._persona_item._text_edit.toPlainText() == "已保存的人物"
    assert get_persona_prompt(config) == "已保存的人物"
    assert path.read_bytes() == original
    warning.assert_not_called()
    window.hide_window()


def test_empty_custom_cannot_save_but_default_allows_empty_draft(persona_settings):
    window, config, path, warning = persona_settings
    original = path.read_bytes()
    window.set_values({"PERSONA_PROMPT_MODE": "custom", "CUSTOM_PERSONA_PROMPT": " \n "})
    events = []
    handler = events.append
    event_bus.settings_batch_changed.connect(handler)
    try:
        window.save_settings()
    finally:
        event_bus.settings_batch_changed.disconnect(handler)
    warning.assert_called_once()
    assert path.read_bytes() == original
    assert events == []
    warning.reset_mock()
    window.set_values({"PERSONA_PROMPT_MODE": "default"})
    window.save_settings()
    assert get_persona_prompt(ConfigManager(path)) == DEFAULT_PERSONA_PROMPT
    warning.assert_not_called()


def test_prompt_only_save_neither_validates_nor_reconnects_model(persona_settings, monkeypatch):
    import windows.setting_window as module
    from app import App

    window, config, path, warning = persona_settings
    config.update({"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "", "OPENAI_MODEL_ID": ""})
    window.slide_in()
    validate = Mock(side_effect=AssertionError("prompt edits must not validate model settings"))
    monkeypatch.setattr(module, "validate_openai_settings", validate)
    application = App.__new__(App)
    application._pending_model_reload = False
    application._apply_model_settings = Mock()
    event_bus.settings_batch_changed.connect(application._on_config_update)
    try:
        window.set_values({"PERSONA_PROMPT_MODE": "custom", "CUSTOM_PERSONA_PROMPT": "新的角色"})
        window.save_settings()
    finally:
        event_bus.settings_batch_changed.disconnect(application._on_config_update)
    validate.assert_not_called()
    application._apply_model_settings.assert_not_called()
    assert not application._pending_model_reload
    assert json.loads(path.read_text(encoding="utf-8"))["CUSTOM_PERSONA_PROMPT"] == "新的角色"
    warning.assert_not_called()


def test_failed_save_does_not_activate_prompt_or_emit_changes(persona_settings, monkeypatch):
    window, config, path, warning = persona_settings
    original = path.read_bytes()
    monkeypatch.setattr(config, "_save", Mock(side_effect=OSError("write failure")))
    window.set_values({"PERSONA_PROMPT_MODE": "custom", "CUSTOM_PERSONA_PROMPT": "不能生效的角色"})
    events = []
    handler = events.append
    event_bus.settings_batch_changed.connect(handler)
    try:
        window.save_settings()
    finally:
        event_bus.settings_batch_changed.disconnect(handler)
    assert get_persona_prompt(config) == DEFAULT_PERSONA_PROMPT
    assert path.read_bytes() == original
    assert events == []
    warning.assert_called_once()
