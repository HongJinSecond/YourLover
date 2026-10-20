"""设置界面的离屏集成测试：保存、恢复、密钥隔离和模型热加载。"""

import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QLineEdit

from utils.config_manager import ConfigManager


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def settings_window(tmp_path, monkeypatch, qt_app):
    import windows.setting_window as module

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"CHAT_MODEL_NAME": "local-model"}), encoding="utf-8")
    config = ConfigManager(str(path))
    monkeypatch.setattr(module, "config", config)
    window = module.SettingWindow()
    window.add_default_settings()
    yield window, config, path
    window.deleteLater()
    qt_app.processEvents()


def test_save_remote_settings_masks_key_and_restores_after_restart(settings_window, qt_app, capsys):
    from core.event_bus import event_bus

    window, config, path = settings_window
    window.set_values(
        {
            "LLM_PROVIDER": "openai",
            "OPENAI_BASE_URL": " https://example.test/v1/chat/completions ",
            "OPENAI_MODEL_ID": " provider-model ",
            "OPENAI_API_KEY": " test-secret-key ",
        }
    )
    key_row = next(item for item in window._setting_panel.items() if item.key == "OPENAI_API_KEY")
    assert key_row._line_edit.echoMode() == QLineEdit.EchoMode.Password
    assert key_row._widget.isEnabled()
    assert not key_row._widget.isHidden()
    local_row = next(
        item for item in window._setting_panel.items() if item.key == "CHAT_MODEL_NAME"
    )
    assert not local_row._widget.isEnabled()
    assert local_row._widget.isHidden()
    events = []
    handler = lambda values: events.append(values)
    event_bus.settings_batch_changed.connect(handler)
    try:
        window.save_settings()
    finally:
        event_bus.settings_batch_changed.disconnect(handler)
    assert len(events) == 1
    restored = ConfigManager(str(path))
    assert restored.LLM_PROVIDER == "openai"
    assert restored.OPENAI_API_KEY == "test-secret-key"
    assert restored.OPENAI_MODEL_ID == "provider-model"
    assert restored.OPENAI_BASE_URL == "https://example.test/v1"
    assert restored.CHAT_MODEL_NAME == "local-model"
    assert "test-secret-key" not in path.read_text(encoding="utf-8")
    assert "test-secret-key" not in capsys.readouterr().out
    window.set_values({"OPENAI_MODEL_ID": "unsaved-draft"})
    window.slide_in()
    assert window.get_setting_value("OPENAI_MODEL_ID") == "provider-model"
    window.hide()


def test_invalid_settings_do_not_persist_or_reload(settings_window, monkeypatch):
    import windows.setting_window as module

    window, config, path = settings_window
    original = path.read_text(encoding="utf-8")
    warning = Mock()
    monkeypatch.setattr(module.QMessageBox, "warning", warning)
    window.set_values({"LLM_PROVIDER": "openai", "OPENAI_MODEL_ID": "model", "OPENAI_API_KEY": ""})
    window.save_settings()
    warning.assert_called_once()
    assert path.read_text(encoding="utf-8") == original
    assert getattr(config, "LLM_PROVIDER", "ollama") == "ollama"


def test_legacy_config_defaults_to_ollama_and_keeps_remote_fields(settings_window):
    window, config, path = settings_window
    assert window.get_setting_value("LLM_PROVIDER") == "ollama"
    window.set_values({"LLM_PROVIDER": "openai", "OPENAI_MODEL_ID": "remote-model"})
    window.set_values({"LLM_PROVIDER": "ollama"})
    assert window.get_setting_value("OPENAI_MODEL_ID") == "remote-model"
    assert window.get_setting_value("CHAT_MODEL_NAME") == "local-model"


def test_model_reload_waits_for_workers_and_keeps_session(qt_app, monkeypatch):
    import app as module

    application = module.App.__new__(module.App)
    application._active_workers = {object()}
    application._pending_model_reload = False
    application._startup_message = None
    application._agent_controller = SimpleNamespace(
        configurable={"user_id": "u", "session_id": "selected-session"}
    )
    reloads = []
    application._init_agent_controller = lambda: reloads.append(
        dict(application._agent_controller.configurable)
    )
    application._on_config_update({"LLM_PROVIDER": "openai"})
    assert reloads == []
    assert application._pending_model_reload
    application._active_workers.clear()
    application._apply_model_settings()
    assert reloads == [{"user_id": "u", "session_id": "selected-session"}]
    assert not application._pending_model_reload


def test_app_uses_saved_remote_values_and_retains_session(qt_app, monkeypatch):
    import app as module

    application = module.App.__new__(module.App)
    application._agent_controller = SimpleNamespace(
        configurable={"user_id": "u", "session_id": "selected-session"}
    )
    application._init_proactive_engine = Mock()
    controller = SimpleNamespace(
        APIConnect=Mock(return_value=SimpleNamespace(message="configured"))
    )
    factory = Mock(return_value=controller)
    monkeypatch.setattr(module, "AgentController", factory)
    monkeypatch.setattr(
        module,
        "config",
        SimpleNamespace(
            LLM_PROVIDER="openai",
            OPENAI_MODEL_ID="remote-model",
            OPENAI_BASE_URL="https://example.test/v1",
            OPENAI_API_KEY="test-key",
        ),
    )
    application._init_agent_controller()
    assert factory.call_args.args[1] == {"user_id": "u", "session_id": "selected-session"}
    controller.APIConnect.assert_called_once_with(
        LLM_type="openai",
        LLM_name="remote-model",
        url="https://example.test/v1",
        api_key="test-key",
    )


@pytest.mark.parametrize("contents", ["broken-secret-json", "[]", "{}"])
def test_invalid_credentials_allow_settings_recovery(tmp_path, contents, capsys):
    path = tmp_path / "config.json"
    path.write_text('{"CHAT_MODEL_NAME": "local-model"}', encoding="utf-8")
    credentials = tmp_path / "data" / "model_credentials.json"
    credentials.parent.mkdir()
    credentials.write_text(contents, encoding="utf-8")
    config = ConfigManager(str(path))
    assert config.OPENAI_API_KEY == ""
    assert contents not in capsys.readouterr().out
    config.update({"OPENAI_API_KEY": "replacement-key"})
    assert ConfigManager(str(path)).OPENAI_API_KEY == "replacement-key"
    config.update({"OPENAI_API_KEY": ""})
    assert ConfigManager(str(path)).OPENAI_API_KEY == ""


def test_worker_authentication_failure_does_not_display_or_log_key(qt_app, capsys):
    import app as module

    class AuthenticationFailure(Exception):
        status_code = 401

    controller = SimpleNamespace(
        provider="openai",
        configurable={"user_id": "u", "session_id": "s"},
        chat=Mock(side_effect=AuthenticationFailure("sensitive-response-and-key")),
    )
    worker = module.AgentWorker(controller, "hello")
    replies = []
    worker.reply_ready.connect(replies.append)
    worker.run()
    assert len(replies) == 1
    assert "鉴权失败" in replies[0].message
    assert "sensitive-response-and-key" not in replies[0].message
    captured = capsys.readouterr()
    assert "sensitive-response-and-key" not in captured.out + captured.err
    worker.deleteLater()


def test_character_window_reads_size_from_json(qt_app, tmp_path, monkeypatch):
    import windows.lover_window as module

    path = tmp_path / "config.json"
    path.write_text('{"CHARACTER_WINDOW_SIZE": [320, 480]}', encoding="utf-8")
    monkeypatch.setattr(module, "config", ConfigManager(path))
    window = module.LoverWindow()
    assert (window.width(), window.height()) == (320, 480)
    window.deleteLater()
    qt_app.processEvents()
