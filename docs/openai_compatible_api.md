# 模型服务配置：Ollama 与 OpenAI 兼容接口

在桌面应用的设置界面，将「模型服务」切换为「OpenAI 兼容接口」，填写以下三项并保存：

| 设置项 | 含义 | 示例 |
| --- | --- | --- |
| API URL | 服务商提供的 Base URL，保留其版本或其他路径 | `https://api.openai.com/v1` |
| API Key | 服务商生成的密钥 | 在密码输入框中填写 |
| Model ID | 服务商支持的模型标识，使用实际 ID | 服务商模型列表中的 ID |

URL 支持自定义网关路径；如果粘贴的是完整的 `/chat/completions` 地址，保存时会自动去掉这一后缀。不会自动追加 `/v1`。URL 不能包含账号、查询参数或片段；鉴权通过 Bearer API Key 发送。无鉴权的本地兼容服务可以填写任意非空 Key。

保存后立即加载配置。如果正在生成回复，会等当前请求完成后切换，保留当前选中的会话。启动及保存配置不会访问 `/models` 或发送测试推理；首次聊天才验证实际连接与模型权限。鉴权、地址、额度或网络错误会显示在回复中。取消设置会放弃未保存的修改。

Ollama 的模型名、URL 与兼容接口配置分别保留，可随时切回「本地 Ollama」。缺少服务类型的旧配置仍使用 Ollama。

## 兼容范围与数据路径

- 使用 Chat Completions 协议和 function tools，不使用 Responses API。所选模型需要支持工具调用；只支持文本生成或 Responses 的端点不能完整运行当前 Agent。协议参考：[OpenAI Chat Completions API](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)。
- 普通聊天、主动对话、会话摘要、长期记忆提取和用户画像提取复用所配置的聊天模型。远程服务会收到这些功能所需的对话与上下文；画像提取等辅助请求也会消耗服务商额度。
- OpenAI 模式不假定服务商同时提供 embedding 接口。动态长期记忆使用已有的本地 JSON 存储和关键词检索，知识库查询使用 BM25，可在不启动 Ollama 的情况下聊天及查询已有知识。Ollama 向量库中的历史动态记忆不会自动迁移至 JSON；聊天记录和画像仍使用原 SQLite 存储。
- 知识库新增文档的向量化上传仍沿用原 Ollama embedding 流程；本次没有增加远程 embedding 配置。

## 配置持久化

普通设置写入 `config.json`：`LLM_PROVIDER`（`ollama` / `openai`）、`OPENAI_BASE_URL`、`OPENAI_MODEL_ID`，以及原有的 `api_base`、`CHAT_MODEL_NAME`。

API Key 以明文单独保存在本机 `data/model_credentials.json`，该目录已被 Git 忽略；没有使用系统凭据保险库或加密存储。界面遮蔽密钥，设置变更和远程错误不会打印密钥或服务商错误正文。不要分享本机密钥文件。兼容手动在旧 `config.json` 中配置 `OPENAI_API_KEY` 的情况，下次保存时将其移至独立文件。

## 开发与验证

新增依赖记录在 `environment.yml`：`langchain-openai==1.4.1`、`openai==2.54.0`。

```bat
conda activate Mypy
python -m pip install langchain-openai==1.4.1 openai==2.54.0
python -m pytest tests -q
```

`agent/models/OpenAIModels.py` 负责配置校验、URL 处理、模型构造及受控错误提示。`AgentController` 选择模型与辅助存储依赖；设置界面校验并持久化，`App` 等工作线程空闲后重建 Agent。测试使用内存 HTTP transport 和离屏 Qt，不需要真实密钥或外部推理请求。
