"""Create the configured model and serialize access to its conversation graph."""

import logging
import threading
from dataclasses import dataclass
from typing import Any

import requests

from agent.agent.ChatAgent import ChatAgent
from agent.memory.long_term_memory import LongTermMemory
from agent.memory.MemoryItem import MemoryItem
from agent.memory.profile_service import LLMProfileCandidateExtractor, UserProfileManager
from agent.memory.vector_memory_store import VectorMemoryStore
from agent.models.OllamaModels import get_ollama_llm
from agent.models.OpenAIModels import get_openai_llm, normalize_base_url
from utils.config_manager import ConfigManager
from utils.message_filter import extract_emotion

logger = logging.getLogger(__name__)


class AgentController:
    def __init__(self, settings: ConfigManager, configurable: dict) -> None:
        self.is_connected = False
        self.settings = settings
        self.configurable = dict(configurable)
        self.provider = "ollama"
        # One graph invocation at a time, including mini-chat and proactive turns.
        self._invoke_lock = threading.Lock()

    def chat(self, user_input: str, configurable: dict | None = None) -> "AgentMessage":
        if not self.is_connected:
            return AgentMessage("还没有连接上大模型，不能进行聊天~", "Sad")
        return self._invoke(user_input, configurable)

    def chat_proactive(self, instruction: str, configurable: dict | None = None) -> "AgentMessage":
        if not self.is_connected:
            return AgentMessage("（主动消息跳过：还没有连接上大模型）", "Sad")
        return self._invoke("", configurable, instruction)

    def _invoke(
        self, user_input: str, configurable: dict | None, instruction: str = ""
    ) -> "AgentMessage":
        with self._invoke_lock:
            runtime = dict(configurable if configurable is not None else self.configurable)
            state = self.agent.invoke(
                {
                    "user_input": user_input,
                    "user_id": runtime["user_id"],
                    "session_id": runtime["session_id"],
                    # Always reset this field so hidden instructions cannot leak into the next turn.
                    "proactive_context": instruction,
                }
            )
        message = AgentMessage(state["history"][-1])
        # MemoryItem is created before the affinity node updates this round's value.
        message.love = int(state.get("love_snapshot", message.love))
        return message

    def APIConnect(
        self, LLM_type: str, LLM_name: str, url: str | None = None, api_key: str | None = None
    ) -> "AgentMessage":
        """Load a provider; compatible APIs are validated on the first chat request."""
        self.is_connected = False
        self.provider = LLM_type.strip().lower()
        try:
            self._agent_init(LLM_name, url, api_key)
        except (ValueError, RuntimeError, ImportError) as exc:
            return AgentMessage(str(exc), "Sad")
        self.is_connected = True
        if self.provider == "openai":
            return AgentMessage("OpenAI 兼容接口配置已加载，将在首次聊天时验证连接。", "Happy")
        return AgentMessage(f"{LLM_name} 连接成功了！现在可以开始聊天了！", "Happy")

    def _agent_init(self, model_name: str, url: str | None, api_key: str | None) -> None:
        dependencies: dict[str, Any] = {}
        if self.provider == "ollama":
            base_url = normalize_base_url(url or "http://localhost:11434")
            model_name = model_name.strip()
            if not model_name:
                raise ValueError("请填写 Ollama 模型名称")
            try:
                response = requests.get(base_url + "/api/tags", timeout=5)
                response.raise_for_status()
                models = [model["name"] for model in response.json().get("models", [])]
            except (requests.RequestException, ValueError) as exc:
                raise ValueError("无法连接 Ollama，请检查服务地址并确认服务已启动。") from exc
            if model_name not in models:
                raise ValueError(f"Ollama 模型 {model_name} 不存在")
            llm = get_ollama_llm(model_name=model_name, base_url=base_url)
        elif self.provider == "openai":
            llm = get_openai_llm(model_name, url, api_key)
            dependencies["user_profile_manager"] = UserProfileManager(
                candidate_extractor=LLMProfileCandidateExtractor(llm)
            )
        else:
            raise ValueError(f"不支持的模型服务：{self.provider}")
        # 向量长期记忆只取决于 EMBEDDING_PROVIDER，与聊天服务解耦：
        # none 时 VectorMemoryStore 抛错，降级为 JSON+关键词检索（Plan B）。
        try:
            dependencies["long_term_memory"] = LongTermMemory(vector_store=VectorMemoryStore())
        except Exception as exc:
            logger.warning("向量长期记忆不可用，降级为 JSON+关键词检索：%s", exc)
            dependencies["long_term_memory"] = LongTermMemory()
        self.agent = ChatAgent(
            llm, self.settings, {"configurable": dict(self.configurable)}, **dependencies
        )


@dataclass(init=False)
class AgentMessage:
    message: str
    ai_emo: str
    love: int | None
    type: str
    speaker: str
    user_emo: dict | None

    def __init__(self, source: MemoryItem | str, emotion: str | None = None) -> None:
        if isinstance(source, MemoryItem):
            self.type = source.type
            self.speaker = source.speaker
            self.love = source.love
            self.user_emo = source.user_emo
            self.message, emotions = extract_emotion(source.message.content)
            self.ai_emo = emotions[0] if emotions else "Happy"
        else:
            self.message = str(source)
            self.ai_emo = emotion or "Happy"
            self.type = "Chat"
            self.speaker = "AI"
            self.love = None
            self.user_emo = None
