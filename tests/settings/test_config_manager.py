"""统一 JSON 配置的路径、加载失败和隔离写入行为。"""

import json

import pytest

from utils.config_manager import DEFAULT_CONFIG_PATH, ConfigManager


def test_default_config_uses_repository_json_from_another_working_directory(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"CHAT_MODEL_NAME": "wrong-file"}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    loaded = ConfigManager()
    expected = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert loaded.CHAT_MODEL_NAME == expected["CHAT_MODEL_NAME"]
    assert loaded.PERSONA_PROMPT_MODE == expected["PERSONA_PROMPT_MODE"]
    assert loaded.CUSTOM_PERSONA_PROMPT == expected["CUSTOM_PERSONA_PROMPT"]


def test_explicit_relative_config_keeps_save_location_after_chdir(tmp_path, monkeypatch):
    original_dir = tmp_path / "original"
    original_dir.mkdir()
    path = original_dir / "config.json"
    path.write_text('{"CHAT_MODEL_NAME": "before"}', encoding="utf-8")
    monkeypatch.chdir(original_dir)
    loaded = ConfigManager("config.json")
    monkeypatch.chdir(tmp_path)
    loaded.update({"CHAT_MODEL_NAME": "after", "OPENAI_API_KEY": "test-key"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"CHAT_MODEL_NAME": "after"}
    assert ConfigManager(path).OPENAI_API_KEY == "test-key"
    assert not (tmp_path / "config.json").exists()
    assert not (tmp_path / "data").exists()


def test_missing_json_fails_with_config_path(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="missing.json"):
        ConfigManager(path)


@pytest.mark.parametrize("content", ["not-json", "[]", "null", "123"])
def test_invalid_config_fails_without_silently_using_defaults(tmp_path, content):
    path = tmp_path / "config.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="config.json"):
        ConfigManager(path)


def test_missing_attribute_has_clear_error_and_supports_explicit_fallback(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")
    loaded = ConfigManager(path)
    with pytest.raises(AttributeError, match="RAG_N_RESULTS"):
        _ = loaded.RAG_N_RESULTS
    assert getattr(loaded, "RAG_N_RESULTS", 2) == 2


def test_in_memory_override_does_not_write_or_change_another_instance(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"RAG_N_RESULTS": 20}', encoding="utf-8")
    loaded = ConfigManager(path)
    loaded.update({"RAG_N_RESULTS": 2}, save=False)
    assert loaded.RAG_N_RESULTS == 2
    assert ConfigManager(path).RAG_N_RESULTS == 20
    assert not (tmp_path / "data").exists()
