# YourLover

基于 PySide6 和 Live2D 的桌面陪伴助手，支持主聊天、角色旁的迷你聊天、多会话记忆、画像和好感度，以及静默后的主动对话。

模型服务可在设置中选择本地 Ollama 或 OpenAI Chat Completions 兼容接口。远程接口填写 API URL、API Key 和 Model ID；API Key 独立保存在本机 `data/model_credentials.json`，不写入 `config.json`。

## 运行

已有 Mypy 环境：

```cmd
conda activate Mypy
python main.py
```

新环境可使用 `conda env create -f environment.yml` 安装依赖。

右键角色打开聊天或设置；角色旁“聊一聊”可以展开快速输入。主聊天 Ctrl+Enter 发送、Enter 换行，迷你聊天 Enter 发送、Esc 收起。

## 开发

架构和文档索引见 [DEVELOPMENT.md](DEVELOPMENT.md)。回归测试运行 `python -m pytest -q`；开发格式检查运行 `python -m ruff check .` 和 `python -m ruff format --check .`。
