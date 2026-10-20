import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from agent.prompts import extract_legacy_persona

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
logger = logging.getLogger(__name__)


class ConfigManager:
    """统一从 config.json 加载应用配置，不维护另一套 Python 默认配置。

    默认路径相对项目根目录，避免工作目录不同导致读错配置。
    显式路径用于隔离配置（例如测试）；API Key 单独保存在同级 data 目录。
    """

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH):
        self._config_path = Path(config_path).resolve()
        self._credentials_path = self._config_path.parent / "data" / "model_credentials.json"
        self._api_key = ""
        self._data: Dict[str, Any] = {}
        self._load()

    @property
    def config_dir(self) -> Path:
        """配置文件所在目录，也是用户上传头像等资源的相对路径基准。"""
        return self._config_path.parent

    def _load(self):
        """从唯一配置文件加载；文件缺失或无效时明确报错。"""
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"找不到应用配置文件：{self._config_path}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"应用配置文件不是有效的 UTF-8 JSON：{self._config_path}") from exc
        if not isinstance(self._data, dict):
            raise ValueError(f"应用配置文件必须是 JSON 对象：{self._config_path}")
        legacy_prompt = self._data.pop("TEST_SYSTEM_PROMPT", None)
        if isinstance(legacy_prompt, str) and legacy_prompt.strip():
            # Preserve existing personas when loading an older configuration.
            # New configurations explicitly choose the improved bundled default.
            self._data.setdefault("CUSTOM_PERSONA_PROMPT", extract_legacy_persona(legacy_prompt))
            self._data.setdefault("PERSONA_PROMPT_MODE", "custom")
        # 兼容曾手动写入配置的密钥；下次保存时从 config.json 中移除。
        self._api_key = self._data.pop("OPENAI_API_KEY", "")
        if self._credentials_path.exists():
            try:
                credentials = json.loads(self._credentials_path.read_text(encoding="utf-8"))
                self._api_key = str(credentials["OPENAI_API_KEY"] or "")
            except (OSError, ValueError, KeyError, TypeError):
                # 密钥文件损坏时允许用户进入设置重新填写，不输出文件正文。
                self._api_key = ""
                logger.warning("无法读取本地模型密钥，请在设置中重新填写 API Key。")

    def _save(self):
        """分别原子替换密钥与普通配置文件，避免密钥进入版本控制。"""
        self._credentials_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json(self._credentials_path, {"OPENAI_API_KEY": self._api_key})
        self._write_json(self._config_path, self._data)

    @staticmethod
    def _write_json(path: Path, values: dict) -> None:
        """先写临时文件再替换，避免写入中断导致原文件内容被截断。"""
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(values, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    # -- 属性访问，支持 config.CHAT_MODEL_NAME 用法 --
    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if name == "OPENAI_API_KEY":
            return self._api_key
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"配置文件 {self._config_path} 缺少配置项 '{name}'") from None

    # -- 批量修改与保存 --
    def update(self, new_values: Dict[str, Any], save: bool = True):
        """
        批量更新配置项。
        """
        values = dict(new_values)
        previous_data, previous_key = dict(self._data), self._api_key
        if "OPENAI_API_KEY" in values:
            self._api_key = str(values.pop("OPENAI_API_KEY") or "").strip()
        self._data.update(values)
        try:
            if save:
                self._save()
        except OSError:
            self._data, self._api_key = previous_data, previous_key
            raise


# 所有运行时模块共享此实例；更新配置必须使用 update()。
config = ConfigManager()
