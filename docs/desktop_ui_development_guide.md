# 桌面 UI 开发指南

## 界面与职责

桌面端采用 PySide6。主聊天、迷你聊天和设置共用浅色背景、紫色强调色、字体、图标及控件样式。

| 文件 | 职责 |
| --- | --- |
| `main.py` | 创建 App，启动 Qt 事件循环 |
| `app.py` | 模型控制器、工作线程、主动消息、配置热加载和退出生命周期 |
| `windows/window_manager.py` | 创建四个窗口，协调导航与角色交互 |
| `windows/lover_window.py` | 透明角色窗口、点击区域、拖拽和摇晃检测 |
| `windows/chat_window.py` | 主聊天、会话切换与标题维护、历史恢复 |
| `windows/mini_chat_overlay.py` | 跟随角色的快速聊天入口与回复浮层 |
| `windows/setting_window.py` | 模型服务表单、校验、保存与恢复 |
| `widgets/theme.py` | 颜色、字体、通用样式、阴影、屏幕位置限制 |
| `widgets/icons.py` | 本地 SVG 图标，缓存普通和高 DPI 图像 |
| `core/event_bus.py` | 跨窗口和业务层的实际事件 |

窗口关闭时隐藏，便于再次打开；退出通过 `application_quit_requested → App.quit()` 处理。App 停止主动对话计时、隐藏窗口，待已有聊天线程完成后退出 Qt。

## 主聊天

`ChatWindow` 组合四个组件：

- `ChatSessionList`：会话标题、最后一条消息预览、新建、切换、删除。
- `ChatHeader`：角色名称、状态、情绪、实际好感度、设置入口。
- `ChatHistory`：空状态、可选择复制的消息卡片、滚动和历史清理。
- `ChatInput`：多行输入，Enter 换行，Ctrl+Enter 发送，空消息和忙碌状态禁止发送。

消息按纯文本显示，避免用户输入或模型输出中的 HTML 被意外解释。气泡依据内容和可用宽度计算换行高度；历史列表在布局更新后滚动到底部。主窗口不显示系统标题栏，拖动聊天顶部可以移动窗口，双击顶部可最大化或还原；右侧提供最小化、最大化/还原和关闭按钮，关闭仍为隐藏聊天窗口。拖动右下角可以调整尺寸，最小 900 × 560。

主聊天请求流程：

```text
ChatInput.send_clicked
  → event_bus.user_message_sent
  → WindowManager 在当前会话追加用户消息
  → App 创建 AgentWorker，捕获本次 session_id
  → AgentController.chat → ChatAgent.invoke
  → App._on_agent_reply
  → event_bus.agent_reply_received
  → ChatWindow 追加完整回复、恢复输入
```

`AgentWorker` 是 QThread，模型调用不能放进 GUI 主线程。App 持有活跃线程引用，在 `finished` 后回收。`AgentController` 用锁串行执行主聊天、迷你聊天和主动消息的图调用。

`SessionStore` 负责实际 SQLite 存储。ChatWindow 保存展示缓存及会话标题；消息正文由 ChatAgent 保存，避免双写。启动时恢复最近会话，过滤工具消息及内部情绪、好感度标记。

## 迷你聊天

`MiniChatOverlay` 收起时是“聊一聊”胶囊，展开后是单行输入栏。Enter 发送，Esc 收起；发送后立即禁用输入，收到回复后恢复。

请求经过 `mini_chat_input_sent → App → AgentWorker(mini_chat=True)`，使用固定 `session_id="mini-chat"`，回复走 `mini_chat_reply_received`，与主窗口会话隔离。

`SpeechBubble` 是独立透明浮层：

- 支持思考状态、文本选择与长回复滚动。
- 最大宽度 360，高度 292；避免长回复覆盖大块桌面。
- 默认 8 秒后隐藏，悬停暂停，关闭按钮立即收起并停止计时器。
- 输入栏与回复位置随角色移动，并限制在当前屏幕可用区域内。

## 设置

设置界面提供聊天形象、人物提示词和已经接入的模型选项。显示名称用于聊天顶部、侧边栏和消息署名；头像支持选择 `icon/` 中的图片、上传预览和恢复默认。人物提示词支持默认只读预览、自定义编辑和复制默认作为模板，切换模式保留自定义草稿；全局规则不出现在编辑器中，详见 [人物提示词](persona_prompts.md)。

| 服务 | 字段 |
| --- | --- |
| 本地 Ollama | 服务 URL、模型名称 |
| OpenAI 兼容接口 | API URL、API Key、Model ID |

切换服务时隐藏另一组字段并保留其值。API Key 默认遮罩，可临时查看；再次打开设置时恢复遮罩及已保存内容，取消不写入配置。

表单由 `SettingPanel` 和 `SettingItem` 组成。`TextSettingItem` 提供文本及密码输入，`OptionSettingItem` 提供带显示名称的选项，`AvatarSettingItem` 提供头像预览与选择，`PersonaPromptSettingItem` 分离只读默认副本与可编辑草稿。表单的逐项变化只在本地传播；保存通过校验后，才调用 `config.update()` 并发送一次包含变更字段的 `settings_batch_changed`。

App 收到模型变更事件后重建模型控制器；若有请求正在执行，则等所有线程结束后再应用。保留当前会话，不重建 Live2D 控件。仅修改显示名称或头像时立即刷新 UI，人物提示词从下一轮开始使用，均不重建模型控制器。

配置入口与密钥存储见 [配置管理](configuration.md)，服务接入细节见 [OpenAI 兼容接口](openai_compatible_api.md)。

## 角色和右键菜单

`LoverWindow` 仅处理窗口和鼠标交互，实际 Live2D 渲染、动作和表情由 `Live2DController`/`Live2DWidget` 实现。`AvatarControllerProtocol` 描述 WindowManager 使用的接口。

右键菜单使用原生 `QMenu`，提供打开聊天、设置和退出。只有实际生效的动作才进入菜单与事件总线。

## 开发与验证

调整颜色、字体和公共控件优先修改 `widgets/theme.py`；新增图标维护 `widgets/icons.py`。新增设置必须同时具备实际消费者、保存校验与交互验证。

在项目根目录运行：

```cmd
conda activate Mypy
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

Qt 交互测试采用 offscreen 平台，不需要打开桌面。测试使用临时数据目录，避免触碰用户的 RAG 和会话数据。界面验收还应查看主聊天空状态、多轮消息、最小窗口、两种设置表单及迷你聊天长回复截图。真实 Live2D 的 OpenGL 渲染需要在桌面环境运行 `python main.py` 检查。
