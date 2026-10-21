# 配置管理约定

应用只有一份可变普通配置：项目根目录的 `config.json`。模型、人物提示词选择与自定义描述、角色窗口、RAG、记忆窗口和主动对话参数均由 `utils/config_manager.py` 的共享实例 `config` 加载，不再存在单独的 Python 默认配置类。默认人物副本与全局规则是 `agent/prompts/` 中的固定应用资源，不参与设置写回。

```python
from utils.config_manager import ConfigManager, config

model_id = config.OPENAI_MODEL_ID
config.update({"RAG_N_RESULTS": 10})  # 更新内存并写回 JSON
```

新增配置项时在 `config.json` 中维护其值；业务代码通过 `config` 或构造函数传入的 `ConfigManager` 读取。不要重新建立一套 Python 配置常量。业务组件需要隔离配置时，可注入独立实例：

```python
isolated_config = ConfigManager("/path/to/test/config.json")
isolated_config.update({"RAG_N_RESULTS": 2}, save=False)
```

默认配置路径根据模块所在位置确定，不受启动工作目录影响。显式传入的相对路径会在构造时解析为绝对路径，后续切换目录不会改变保存位置。配置值中的资源路径（例如 `CHROMA_PATH`、`DEFAULT_MODEL_PATH`）保持原有语义，未在本次重构中统一改写。

配置文件缺失、JSON 无效或顶层不是对象时明确报错；普通属性缺失时抛出说明配置路径及键名的 `AttributeError`。可选功能仍可显式使用 `getattr(config, key, fallback)` 兼容旧配置。主动对话引擎接收部分配置对象时，缺项从共享 JSON 配置读取；JSON 自身需要提供完整的主动对话参数。

`OPENAI_API_KEY` 继续单独保存在配置文件同级目录下的 `data/model_credentials.json`，不写入普通配置。密钥文件损坏时允许进入设置重新填写。具体协议和存储说明见 [模型服务配置](openai_compatible_api.md)。

## Embedding 服务

Embedding（向量化）与聊天模型服务相互独立，由 `EMBEDDING_PROVIDER` 单独选择：

| 配置项 | 取值 | 说明 |
| --- | --- | --- |
| `EMBEDDING_PROVIDER` | `ollama` / `openai` / `none` | 向量化服务提供方；旧配置缺少该键时默认 `ollama` |
| `EMBEDDING_MODEL_NAME` | 模型名 | 仅 `ollama` 模式使用，例如 `qwen3-embedding:4b` |
| `EMBEDDING_API_BASE` | 服务地址 | 仅 `ollama` 模式使用；留空默认 `http://localhost:11434`，与聊天的 `api_base` 相互独立 |
| `EMBEDDING_OPENAI_MODEL_ID` | 模型 ID | 仅 `openai` 模式使用 |
| `EMBEDDING_OPENAI_BASE_URL` | 服务地址 | 仅 `openai` 模式使用，与聊天的 `OPENAI_BASE_URL` 相互独立 |
| `EMBEDDING_OPENAI_API_KEY` | 密钥 | 仅 `openai` 模式使用，与聊天的 `OPENAI_API_KEY` 相互独立；保存在 `data/model_credentials.json` |

`EMBEDDING_PROVIDER=none` 表示不启用任何 embedding：动态长期记忆自动降级为 JSON 文件 + 关键词检索（Plan B），RAG 知识库整体禁用（工具调用返回"没有设置Embedding，RAG功能已被禁用"，知识库上传页提示后退出）。历史向量数据保留在 Chroma 中，重新启用后可继续使用。

设置窗口的「Embedding 服务」下拉框切换提供方，并按取值显隐对应输入框；Embedding 的服务地址与密钥均为独立配置项，不与聊天服务共用。保存时校验模型名/Model ID 与密钥非空。

设置窗口先调用 `config.update()`，成功后发出 `settings_batch_changed`，事件只包含本次发生变化的字段。模型设置变更时，`App` 等活跃回复结束后重建 Agent；仅修改聊天名称或头像时立即更新 UI，人物提示词变更从下一轮开始生效，这两类修改都不验证或重连模型。

## 人物提示词

`PERSONA_PROMPT_MODE` 选择 `default` 或 `custom`，`CUSTOM_PERSONA_PROMPT` 独立保存自定义草稿。默认人物描述始终保留只读副本；权限、好感度和输出协议等全局规则不提供编辑入口，由程序自动追加。旧版 `TEST_SYSTEM_PROMPT` 由 ConfigManager 按已知章节边界迁移为人物描述。使用方式、组成和兼容边界见 [人物提示词](persona_prompts.md)。

## 聊天名称与头像

在桌面设置顶部修改“显示名称”和“头像”，点击“保存设置”后立即生效并在重启后保留。显示名称同时用于聊天顶部、侧边栏标题和消息气泡署名，不修改系统提示词中的人物设定。侧边栏名称过长时显示省略号，悬停可查看完整名称。

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `CHAT_DISPLAY_NAME` | `YourLover` | 聊天显示名称，空白输入恢复默认名称 |
| `CHAT_AVATAR` | 空字符串 | 相对配置目录的头像路径，例如 `icon/avatar-xxx.png`；空值使用默认 `Y` 字母头像 |

头像下拉框读取配置文件同级的 `icon/` 目录中可识别的图片。可以直接将图片放入该目录后重新打开设置，也可以点击“上传图片”选择 PNG、JPEG、WebP、BMP、ICO 或 SVG 图片并预览。上传只暂存选择，保存时才将外部图片转换为最大 512 像素的 PNG 存入 `icon/`，使用内容摘要命名，避免同名图片互相覆盖。已经在 `icon/` 中的图片直接使用。取消设置不会复制图片或更改聊天界面。

头像路径不受启动工作目录影响。目录不存在、未选择头像、图片被删除或损坏时，聊天顶部和设置预览均回退到默认字母头像。点击“恢复默认”并保存可以清空头像配置，已有图片文件会保留。
