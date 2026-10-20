"""OpenAI Chat Completions 兼容接口：配置校验与 LangChain 模型构造。"""

from typing import Any
from urllib.parse import urlsplit, urlunsplit

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def normalize_base_url(url: str) -> str:
    """校验 HTTP(S) 服务地址，保留服务商自定义路径并去掉尾部斜杠。"""
    value = str(url or "").strip()
    if any(char.isspace() for char in value):
        raise ValueError("API URL 不能包含空格或换行")
    try:
        parsed = urlsplit(value)
        _ = parsed.port  # 触发非法端口校验，不发送网络请求。
    except ValueError as exc:
        raise ValueError("API URL 格式不正确") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("API URL 必须是完整的 http:// 或 https:// 地址")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("API URL 请填写服务根地址，不要包含账号、查询参数或片段")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def validate_openai_settings(model_name: str, base_url: str, api_key: str) -> tuple[str, str, str]:
    """返回清理后的 (Model ID, Base URL, API Key)，缺项时给出可展示的错误。"""
    model_name = str(model_name or "").strip()
    api_key = str(api_key or "").strip()
    if not model_name:
        raise ValueError("请填写 OpenAI 兼容接口的 Model ID")
    if not api_key:
        raise ValueError("请填写 API Key（无鉴权的本地兼容服务可填写任意非空值）")
    if any(char.isspace() for char in api_key):
        raise ValueError("API Key 不能包含空格或换行")
    base_url = normalize_base_url(base_url)
    # 兼容用户直接粘贴完整接口地址；不假定所有服务都使用 /v1 前缀。
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")]
    return model_name, base_url, api_key


def get_openai_llm(model_name: str, base_url: str, api_key: str, **kwargs: Any) -> Any:
    """构造支持 invoke/bind_tools 的兼容模型；初始化不发起远程推理。

    参数为用户填写的 Model ID、API 根地址和密钥。使用 Chat Completions，
    不发送 Ollama 的 num_ctx/think 参数，也不要求服务商实现 /models。
    kwargs 供测试注入 HTTP transport，或由调用方调整超时等客户端选项。
    """
    model_name, base_url, api_key = validate_openai_settings(model_name, base_url, api_key)
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "缺少 langchain-openai，请在 Mypy 环境安装 environment.yml 中的依赖"
        ) from exc
    options = {"timeout": 60.0, "max_retries": 1, "use_responses_api": False}
    options.update(kwargs)
    return ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key, **options)


def describe_openai_error(exc: Exception) -> str:
    """把远程错误转换为用户可读提示，不展示响应正文或鉴权信息。"""
    status = getattr(exc, "status_code", None)
    if status == 401:
        return "API 鉴权失败，请检查 API Key。"
    if status == 403:
        return "API 拒绝访问，请检查密钥权限及模型访问权限。"
    if status == 404:
        return "未找到接口或模型，请检查 API URL 和 Model ID。"
    if status == 429:
        return "API 请求受限或额度不足，请检查服务商额度后重试。"
    if status == 400:
        return "API 不接受当前请求，请确认模型支持 Chat Completions 和工具调用。"
    if status:
        return f"API 请求失败（HTTP {status}），请稍后重试或检查服务商状态。"
    if "Timeout" in type(exc).__name__:
        return "API 请求超时，请检查网络和 API URL 后重试。"
    return "API 调用失败，请检查网络、API URL 和模型配置。"
