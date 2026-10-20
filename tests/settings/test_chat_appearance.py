"""Appearance drafts, managed avatar persistence and live chat updates."""

import json
from unittest.mock import Mock

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel

from core.event_bus import event_bus
from utils.config_manager import ConfigManager
from widgets.avatar import avatar_path, set_avatar


@pytest.fixture
def appearance_settings(tmp_path, monkeypatch, qt_app):
    import widgets.chat.chat_bubble as bubble_module
    import widgets.chat.chat_header as header_module
    import widgets.chat.chat_session_list as sidebar_module
    import windows.setting_window as settings_module

    path = tmp_path / "config.json"
    path.write_text('{"CHAT_MODEL_NAME": "local-model"}', encoding="utf-8")
    config = ConfigManager(path)
    for module in (settings_module, header_module, bubble_module, sidebar_module):
        monkeypatch.setattr(module, "config", config)
    window = settings_module.SettingWindow()
    window.add_default_settings()
    header = header_module.ChatHeader()
    bubble = bubble_module.ChatBubble()
    bubble.refresh(bubble_module.MessageData(bubble_module.MessageRole.AGENT, "Hello"))
    row = next(item for item in window._setting_panel.items() if item.key == "CHAT_AVATAR")
    warning = Mock()
    monkeypatch.setattr(settings_module.QMessageBox, "warning", warning)
    yield window, row, header, bubble, config, path, warning
    for widget in (window, header, bubble):
        widget.deleteLater()
    qt_app.processEvents()


def make_image(path):
    image = QImage(240, 120, QImage.Format.Format_RGB32)
    image.fill(QColor("#7762C5"))
    assert image.save(str(path))
    return path


def choose_upload(row, path, monkeypatch):
    from widgets.settings.avatar_setting_item import QFileDialog

    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(path), ""))
    QTest.mouseClick(row._upload_button, Qt.MouseButton.LeftButton)


def test_upload_save_live_update_and_restart_from_another_directory(
    appearance_settings, tmp_path, monkeypatch
):
    window, row, header, bubble, config, path, warning = appearance_settings
    original = make_image(tmp_path / "照片.png")
    choose_upload(row, original, monkeypatch)
    window.set_values({"CHAT_DISPLAY_NAME": "  小爱  "})
    assert not (tmp_path / "icon").exists()
    assert header._title_label.text() == "YourLover"
    assert header._avatar_label.text() == "Y"
    assert not row._preview.pixmap().isNull()
    events = []
    handler = events.append
    event_bus.settings_batch_changed.connect(handler)
    try:
        window.save_settings()
    finally:
        event_bus.settings_batch_changed.disconnect(handler)
    warning.assert_not_called()
    assert len(events) == 1
    assert set(events[0]) == {"CHAT_DISPLAY_NAME", "CHAT_AVATAR"}
    assert header._title_label.text() == "小爱"
    assert bubble._time_label.text().startswith("小爱  ·")
    assert not header._avatar_label.pixmap().isNull()
    assert config.CHAT_AVATAR.startswith("icon/avatar-")
    assert (tmp_path / config.CHAT_AVATAR).is_file()
    original.unlink()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    restored = ConfigManager(path)
    assert restored.CHAT_DISPLAY_NAME == "小爱"
    label = QLabel()
    label.resize(42, 42)
    set_avatar(label, avatar_path(restored.CHAT_AVATAR, restored.config_dir))
    assert not label.pixmap().isNull()
    label.deleteLater()


def test_cancel_discards_upload_and_name_without_copying_files(
    appearance_settings, tmp_path, monkeypatch
):
    window, row, header, bubble, config, path, warning = appearance_settings
    original_config = path.read_bytes()
    original = make_image(tmp_path / "draft.png")
    choose_upload(row, original, monkeypatch)
    window.set_values({"CHAT_DISPLAY_NAME": "未保存的名字"})
    QTest.mouseClick(window._cancel_button, Qt.MouseButton.LeftButton)
    window.slide_in()
    assert row.get_value() == ""
    assert row._preview.text() == "Y"
    assert window.get_setting_value("CHAT_DISPLAY_NAME") == "YourLover"
    assert header._title_label.text() == "YourLover"
    assert path.read_bytes() == original_config
    assert not (tmp_path / "icon").exists()
    window.hide_window()


def test_select_existing_icon_and_restore_default(appearance_settings, tmp_path):
    window, row, header, bubble, config, path, warning = appearance_settings
    directory = tmp_path / "icon"
    directory.mkdir()
    image = make_image(directory / "我的头像.png")
    (directory / "not-an-image.txt").write_text("text", encoding="utf-8")
    window.slide_in()
    assert row._combo_box.count() == 2
    row._combo_box.setCurrentIndex(row._combo_box.findData("icon/我的头像.png"))
    window.save_settings()
    assert config.CHAT_AVATAR == "icon/我的头像.png"
    assert not header._avatar_label.pixmap().isNull()
    assert set(directory.iterdir()) == {image, directory / "not-an-image.txt"}
    QTest.mouseClick(row._reset_button, Qt.MouseButton.LeftButton)
    window.set_values({"CHAT_DISPLAY_NAME": "  "})
    window.save_settings()
    assert config.CHAT_AVATAR == ""
    assert config.CHAT_DISPLAY_NAME == "YourLover"
    assert header._avatar_label.text() == "Y"
    assert image.exists()
    warning.assert_not_called()


@pytest.mark.parametrize("kind", ["missing", "corrupt"])
def test_unavailable_avatar_falls_back_without_changing_saved_path(
    appearance_settings, tmp_path, kind
):
    window, row, header, bubble, config, path, warning = appearance_settings
    value = "icon/unavailable.png"
    if kind == "corrupt":
        (tmp_path / "icon").mkdir()
        (tmp_path / value).write_bytes(b"not an image")
    config.update({"CHAT_AVATAR": value})
    header._refresh_identity()
    window.slide_in()
    assert header._avatar_label.text() == "Y"
    assert row._preview.text() == "Y"
    window.set_values({"CHAT_DISPLAY_NAME": "新名字"})
    window.save_settings()
    assert config.CHAT_AVATAR == value
    assert header._title_label.text() == "新名字"
    warning.assert_not_called()


def test_invalid_upload_keeps_previous_selection(appearance_settings, tmp_path, monkeypatch):
    window, row, header, bubble, config, path, warning = appearance_settings
    original = path.read_bytes()
    invalid = tmp_path / "invalid.png"
    invalid.write_bytes(b"invalid image")
    choose_upload(row, invalid, monkeypatch)
    warning.assert_called_once()
    assert row.get_value() == ""
    assert path.read_bytes() == original
    assert not (tmp_path / "icon").exists()


def test_avatar_write_failure_does_not_change_config_or_chat(
    appearance_settings, tmp_path, monkeypatch
):
    window, row, header, bubble, config, path, warning = appearance_settings
    original_config = path.read_bytes()
    original = make_image(tmp_path / "photo.png")
    choose_upload(row, original, monkeypatch)
    (tmp_path / "icon").write_text("blocks directory creation", encoding="utf-8")
    window.set_values({"CHAT_DISPLAY_NAME": "新名字"})
    window.save_settings()
    warning.assert_called_once()
    assert path.read_bytes() == original_config
    assert getattr(config, "CHAT_DISPLAY_NAME", "YourLover") == "YourLover"
    assert header._title_label.text() == "YourLover"


def test_appearance_only_save_does_not_validate_or_reload_model(
    appearance_settings, monkeypatch
):
    from app import App

    window, row, header, bubble, config, path, warning = appearance_settings
    # Even an incomplete model connection must allow editing appearance.
    config.update({"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "", "OPENAI_MODEL_ID": ""})
    window.slide_in()
    application = App.__new__(App)
    application._pending_model_reload = False
    application._apply_model_settings = Mock()
    event_bus.settings_batch_changed.connect(application._on_config_update)
    try:
        window.set_values({"CHAT_DISPLAY_NAME": "小爱"})
        window.save_settings()
    finally:
        event_bus.settings_batch_changed.disconnect(application._on_config_update)
    warning.assert_not_called()
    application._apply_model_settings.assert_not_called()
    assert not application._pending_model_reload
    assert json.loads(path.read_text(encoding="utf-8"))["CHAT_DISPLAY_NAME"] == "小爱"
    assert header._title_label.text() == "小爱"


def test_avatar_dialog_cancel_leaves_draft_unchanged(appearance_settings, monkeypatch):
    from widgets.settings.avatar_setting_item import QFileDialog

    window, row, header, bubble, config, path, warning = appearance_settings
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("", ""))
    QTest.mouseClick(row._upload_button, Qt.MouseButton.LeftButton)
    assert row.get_value() == ""
    assert row._preview.text() == "Y"
    warning.assert_not_called()


def test_long_name_keeps_header_and_message_inside_their_widths(appearance_settings, qt_app):
    window, row, header, bubble, config, path, warning = appearance_settings
    name = "A long companion name " * 3
    header.resize(680, 90)
    header.show()
    bubble.resize(680, bubble.height())
    bubble.show()
    window.set_values({"CHAT_DISPLAY_NAME": name})
    window.save_settings()
    qt_app.processEvents()
    assert header.width() == 680
    assert header._title_label.toolTip() == config.CHAT_DISPLAY_NAME
    assert header._title_label.text().endswith("\u2026")
    assert bubble._bubble_container.width() <= 520
    assert bubble._time_label.fontMetrics().horizontalAdvance(bubble._time_label.text()) <= (
        bubble._bubble_container.width() - 34
    )
    warning.assert_not_called()


def test_sidebar_name_updates_on_save_and_restores_after_restart(
    appearance_settings, monkeypatch, qt_app
):
    import widgets.chat.chat_session_list as module

    window, row, header, bubble, config, path, warning = appearance_settings
    sidebar = module.ChatSessionList()
    try:
        assert sidebar._brand_title.text() == "YourLover"
        window.set_values({"CHAT_DISPLAY_NAME": "小爱"})
        assert sidebar._brand_title.text() == "YourLover"
        window.save_settings()
        assert sidebar._brand_title.text() == header._title_label.text() == "小爱"
        monkeypatch.setattr(module, "config", ConfigManager(path))
        restored = module.ChatSessionList()
        assert restored._brand_title.text() == "小爱"
        restored.deleteLater()
        monkeypatch.setattr(module, "config", config)
        window.set_values({"CHAT_DISPLAY_NAME": " "})
        window.save_settings()
        assert sidebar._brand_title.text() == "YourLover"
        warning.assert_not_called()
    finally:
        sidebar.deleteLater()
        qt_app.processEvents()


def test_sidebar_long_name_is_elided_without_changing_its_width(appearance_settings, qt_app):
    from widgets.chat.chat_session_list import ChatSessionList

    window, row, header, bubble, config, path, warning = appearance_settings
    sidebar = ChatSessionList()
    try:
        sidebar.show()
        qt_app.processEvents()
        window.set_values({"CHAT_DISPLAY_NAME": "A long companion name " * 3})
        window.save_settings()
        qt_app.processEvents()
        assert sidebar.width() == 248
        assert sidebar._brand_title.toolTip() == config.CHAT_DISPLAY_NAME
        assert sidebar._brand_title.text().endswith("\u2026")
        assert sidebar._brand_title.fontMetrics().horizontalAdvance(
            sidebar._brand_title.text()
        ) <= sidebar._brand_title.width()
        warning.assert_not_called()
    finally:
        sidebar.deleteLater()
        qt_app.processEvents()
