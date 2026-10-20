"""使用内存 HTTP transport 验证兼容协议，不调用真实服务或消耗额度。"""

import json
from functools import partial
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import HumanMessage

from agent.models.OpenAIModels import get_openai_llm, validate_openai_settings


@pytest.mark.parametrize(
    "url,expected",
    [
        (" https://example.test/custom/v1/ ", "https://example.test/custom/v1"),
        ("http://localhost:8080/v1/chat/completions", "http://localhost:8080/v1"),
        ("https://example.test/api/v3", "https://example.test/api/v3"),
    ],
)
def test_custom_urls_preserve_provider_paths(url, expected):
    assert validate_openai_settings(" model-id ", url, " test-key ") == (
        "model-id",
        expected,
        "test-key",
    )


@pytest.mark.parametrize(
    "model,url,key",
    [
        ("", "https://example.test/v1", "test-key"),
        ("model", "https://example.test/v1", ""),
        ("model", "example.test/v1", "test-key"),
        ("model", "file:///tmp/model", "test-key"),
        ("model", "https://name:password@example.test/v1", "test-key"),
        ("model", "https://example.test/v1?api_key=secret", "test-key"),
        ("model", "https://example.test/bad path", "test-key"),
        ("model", "https://example.test/v1", "test\nkey"),
    ],
)
def test_invalid_configuration_is_rejected_locally(model, url, key):
    with pytest.raises(ValueError):
        validate_openai_settings(model, url, key)


def _completion(content=None, tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "provider-model",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
    }


def test_chat_completions_sends_auth_model_and_tools():
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=_completion("你好<<Happy>>{{+2}}"))

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        llm = get_openai_llm(
            "provider-model", "https://example.test/custom/v1", "test-key", http_client=client
        )
        assert requests == []  # 配置保存/客户端创建不访问 /models 或付费推理。
        reply = llm.bind_tools(
            [
                {
                    "name": "calculator",
                    "description": "计算",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                    },
                }
            ]
        ).invoke([HumanMessage(content="你好")])
    assert reply.content.startswith("你好")
    assert str(requests[0].url) == "https://example.test/custom/v1/chat/completions"
    assert requests[0].headers["Authorization"] == "Bearer test-key"
    payload = json.loads(requests[0].content)
    assert payload["model"] == "provider-model"
    assert payload["tools"][0]["function"]["name"] == "calculator"
    assert not {"num_ctx", "think", "reasoning"}.intersection(payload)


def test_remote_controller_tool_loop_and_profile_without_ollama(tmp_path, monkeypatch):
    """完整走控制器/图/真实 OpenAI 适配器，验证工具循环与画像均走同一接口。"""
    import agent.agent_controller as module
    from agent.affinity.affinity_store import AffinityStore
    from agent.affinity.service import AffinityService
    from agent.memory.long_term_memory import LongTermMemory
    from agent.memory.profile_service import UserProfileManager
    from agent.memory.profile_store import ProfileStore
    from agent.memory.short_term_memory import ShortTermMemory
    from utils.config_manager import ConfigManager

    payloads = []

    def handle(request):
        body = json.loads(request.content)
        payloads.append(body)
        if not body.get("tools"):
            content = json.dumps(
                {
                    "candidates": [
                        {
                            "field": "preferred_address",
                            "value": "队长",
                            "reason": "用户要求",
                            "source": "user_explicit",
                            "confidence": 0.99,
                            "needs_confirmation": False,
                            "evidence": "以后叫我队长",
                        }
                    ]
                },
                ensure_ascii=False,
            )
            return httpx.Response(200, json=_completion(content))
        if len(payloads) == 1:
            return httpx.Response(
                200,
                json=_completion(
                    tool_calls=[
                        {
                            "id": "call-calc",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression":"3**2+5"}',
                            },
                        }
                    ]
                ),
            )
        return httpx.Response(200, json=_completion("结果是14<<Happy>>{{+2}}"))

    def forbid_ollama(*args, **kwargs):
        pytest.fail("OpenAI 模式不能调用 Ollama")

    monkeypatch.setattr(module.requests, "get", forbid_ollama)
    monkeypatch.setattr(module, "get_ollama_llm", forbid_ollama)
    monkeypatch.setattr(
        module, "LongTermMemory", partial(LongTermMemory, root_dir=tmp_path / "users")
    )
    monkeypatch.setattr(
        module,
        "UserProfileManager",
        partial(UserProfileManager, store=ProfileStore(tmp_path / "profiles.sqlite3")),
    )
    monkeypatch.setattr(
        module,
        "ChatAgent",
        partial(
            module.ChatAgent,
            short_term_memory=ShortTermMemory(database_path=tmp_path / "sessions.sqlite3"),
            affinity_service=AffinityService(store=AffinityStore(tmp_path / "affinity.sqlite3")),
        ),
    )
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(module, "get_openai_llm", partial(get_openai_llm, http_client=client))
        controller = module.AgentController(
            ConfigManager(), {"user_id": "user-a", "session_id": "main"}
        )
        controller.APIConnect("openai", "provider-model", "https://example.test/v1", "test-key")
        assert controller.is_connected
        assert payloads == []
        result = controller.chat("帮我计算3**2+5")
        assert result.message == "结果是14"
        assert result.love == 2
        assert payloads[1]["messages"][-1]["role"] == "tool"
        assert payloads[1]["messages"][-1]["content"] == "14"
        controller.chat("以后叫我队长", {"user_id": "user-a", "session_id": "mini-chat"})
        assert controller.configurable["session_id"] == "main"
        assert (
            controller.agent.user_profile_manager.get_profile("user-a").preferred_address == "队长"
        )
        assert controller.agent.long_term_memory.vector_store is None


def test_ollama_uses_configured_url_and_requested_model(monkeypatch):
    import agent.agent_controller as module
    from utils.config_manager import ConfigManager

    calls = []
    monkeypatch.setattr(
        module.requests,
        "get",
        lambda url, **kw: (
            calls.append(url)
            or SimpleNamespace(
                raise_for_status=lambda: None, json=lambda: {"models": [{"name": "chosen-model"}]}
            )
        ),
    )
    monkeypatch.setattr(module, "get_ollama_llm", lambda **kw: calls.append(kw) or object())
    monkeypatch.setattr(module, "ChatAgent", lambda *a, **kw: object())
    controller = module.AgentController(ConfigManager(), {"user_id": "u", "session_id": "s"})
    controller.APIConnect("ollama", "chosen-model", "http://localhost:12345/")
    assert controller.is_connected
    assert calls == [
        "http://localhost:12345/api/tags",
        {"model_name": "chosen-model", "base_url": "http://localhost:12345"},
    ]
    controller.APIConnect("openai", "", "https://example.test/v1", "test-key")
    assert not controller.is_connected


def test_remote_rag_queries_existing_documents_without_embeddings(monkeypatch):
    from importlib import import_module
    from unittest.mock import Mock

    from langchain_core.documents import Document

    module = import_module("agent.tools.RAGTool")
    monkeypatch.setattr(module, "app_config", SimpleNamespace(LLM_PROVIDER="openai"))
    rag = module.RAG.__new__(module.RAG)
    rag.config = module.app_config
    rag.chroma_db = Mock()
    rag.chroma_db.similarity_search_with_relevance_scores.side_effect = AssertionError(
        "不应调用本地 embedding"
    )
    rag.bm25_retriever = module.BM25Retriever()
    rag.bm25_retriever.build_from_documents(
        [
            Document(page_content="校园助手提供课程提醒功能", metadata={"source": "guide.txt"}),
        ]
    )
    result = rag.query("课程提醒")
    assert "BM25" in result
    assert "课程提醒功能" in result
    assert "guide.txt" in result
    rag.chroma_db.similarity_search_with_relevance_scores.assert_not_called()
