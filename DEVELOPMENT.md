# 开发入口

项目使用 Python 3.12、PySide6、live2d-py、LangGraph、LangChain、SQLite 和 Chroma。

## 架构

```text
main.py
  └─ App
      ├─ WindowManager → 角色窗口 / 主聊天 / 迷你聊天 / 设置
      ├─ Live2DController → Live2DWidget
      ├─ AgentWorker → AgentController → ChatAgent (LangGraph)
      └─ ProactiveEngine → AgentWorker
```

窗口通过 Qt 信号与 `core/event_bus.py` 协调。组件管理展示和输入，App 管理线程与生命周期，AgentController 构造模型并串行执行对话图。

ChatAgent 的工作流为：恢复短期会话 → 加载画像与好感度 → 检索长期记忆 → 注入用户输入 → 模型/工具循环 → 提取画像候选 → 裁剪并提取摘要和长期记忆 → 更新好感度 → 保存短期记忆。

实际工具为计算器、文件读写和 RAG。工具继承 `BaseTool`，由 `agent.tools` 自动发现，依据函数签名生成 LangChain 参数 schema。

## 文档

| 文档 | 内容 |
| --- | --- |
| [项目报告与面试准备](docs/project_report.md) | 整体架构、实现细节、设计取舍、能力边界与面试追问 |
| [桌面 UI](docs/desktop_ui_development_guide.md) | 窗口、组件、事件流和交互约定 |
| [配置管理](docs/configuration.md) | config.json、ConfigManager、密钥存储 |
| [人物提示词](docs/persona_prompts.md) | 默认人设、自定义草稿、固定全局规则与下一轮生效 |
| [模型接口](docs/openai_compatible_api.md) | Ollama / OpenAI 兼容接口与设置热加载 |
| [记忆系统](docs/chatbot_memory_module_development.md) | 短期会话、长期记忆、画像与存储 |
| [主动对话](docs/proactive_dialogue_mechanism.md) | 静默计时、主动消息及好感度 |
| [本轮重构](docs/refactoring_and_ui.md) | 清理范围、界面调整及验证方法 |

## 本地开发

```cmd
conda activate Mypy
python main.py
python -m pytest -q
```

格式与静态检查使用 `ruff`（开发工具，需要时运行 `python -m pip install ruff`），规则在 `pyproject.toml`。业务测试使用替身模型，不访问真实模型接口；需要 Ollama 的手工演示保留在 `demos/memory_mvp_demo.py`。知识库上传入口为 `python -m streamlit run agent/rag_upload_ui.py`。

修改记忆逻辑时保持 SQLite 表结构和 JSON schema 兼容；界面操作不应重复保存 Agent 已持久化的消息。新增功能应具备实际实现和调用链，再加入界面与事件总线。
