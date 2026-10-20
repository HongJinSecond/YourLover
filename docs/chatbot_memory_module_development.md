# 🧠 YourLover 记忆模块开发文档

> 📌 **本文档记录记忆模块当前代码的真实状态，并说明后续长期记忆应如何继续接入。**
>
> 2026-09 交接更新：原零散的记忆实现已重构为三大管理器（`ShortTermMemory` / `LongTermMemory` / `UserProfileManager`），
> 作为 `agent/memory` 包的统一入口；本文同步更新了目录结构、接口速查和待办事项，作为交接依据。
> （原设计稿 `chatbot_memory_module_design.md` 已删除，其设计要点已并入本文。）

---

## 📋 目录

1. [🎯 设计目标与本期范围](#1-设计目标与本期范围)
2. [📍 当前实现现状](#2-当前实现现状)
3. [🧩 记忆分层与职责边界](#3-记忆分层与职责边界)
4. [🗂️ 持久化与数据结构](#4-持久化与数据结构)
5. [⚙️ 运行机制设计](#5-运行机制设计)
6. [🔧 与当前代码的改造映射](#6-与当前代码的改造映射)
7. [✅ 实施阶段与验收标准](#7-实施阶段与验收标准)
8. [🚀 MVP 之后的优化方向](#8-mvp-之后的优化方向)
9. [⚠️ 风险、已知问题与开发约定](#9-风险已知问题与开发约定)
10. [📚 关键接口速查](#10-关键接口速查)

---

## 1. 🎯 设计目标与本期范围

### 1.1 设计目标

记忆系统的目标是让 ChatAgent 具有连续性，而不是把每一句聊天都永久保存。它需要解决三个层次的问题：

1. **短期记忆**：同一个 Session 内能理解最近对话，不会每轮都像第一次见面。
2. **长期记忆**：跨 Session 保存仍有价值的信息，例如稳定偏好、近期状态和重要互动经历。
3. **可治理记忆**：长期信息要有来源、状态、更新时间和遗忘策略；模型不能把猜测直接写成用户事实。

最终在一轮对话中，应向模型提供下列“工作记忆”：

```text
基础系统提示词
+ 当前好感度快照
+ 用户画像摘要
+ 与当前问题相关的长期记忆
+ 当前 Session 的摘要
+ 当前 Session 最近若干轮消息
```

### 1.2 本期范围

本期目标是跑通可验证的 MVP，而不是一次性建设完整的向量记忆平台。

| 范围 | 本期目标 |
| :--- | :--- |
| 短期记忆 | Session 级持久化、恢复、窗口裁剪、会话摘要。 |
| 动态长期记忆 | semantic / episodic 的候选、JSON 存储、关键词检索和 Prompt 格式化。 |
| 用户画像 | 固定字段、权限校验、SQLite、候选和审计日志。 |
| 模型抽取 | 先使用本地 Ollama `qwen2.5:3b` 产生画像候选；后续可替换模型适配器。 |
| Agent 接入 | 先完善“每轮加载—回复—候选写入”的完整闭环。 |

本期不把以下能力作为完成前提：

- 向量数据库、embedding 和重排序。
- 高质量 LLM 会话摘要。
- procedural memory 的直接 Prompt 注入。
- 记忆编辑、删除、确认的完整桌面 UI。

### 1.3 好感度与记忆的边界

`love` 是关系状态，不是记忆模块的权威数据。记忆模块只读取当前 `love_snapshot`，并可向 Affinity 模块发出“感谢、拒绝、纠正、情绪转折”等互动事件。

| 模块 | 负责 | 不负责 |
| :--- | :--- | :--- |
| Memory | 保存会话、筛选与检索长期记忆、维护画像候选。 | 计算和持久化权威好感度。 |
| Affinity | 按 `user_id` 维护好感度及其变化原因。 | 直接写入用户画像或长期记忆。 |
| ChatAgent | 编排上下文、调用模型、触发记忆与好感度流程。 | 成为所有数据的直接存储层。 |

---

## 2. 📍 当前实现现状

### 2.1 当前目录与模块

```text
agent/
  agent/
    ChatAgent.py                     # LangGraph 对话流程和 Prompt 组装（注入三大管理器）
  memory/
    __init__.py                      # 统一入口：导出三大管理器（推荐只从这里 import）
    short_term_memory.py             # ★ 短期记忆管理器（组合 session_store + window）
    long_term_memory.py             # ★ 动态长期记忆管理器（组合 store + classifier + retriever）
    profile_service.py               # ★ 用户画像管理器（原 long_memory.py；含 Ollama 候选抽取）
    session_store.py                 # 短期记忆 SQLite 存储（被 ShortTermMemory 组合）
    window.py                        # 短期消息窗口裁剪（被 ShortTermMemory 组合）
    schema.py                        # 动态长期记忆数据结构（被 LongTermMemory 组合）
    store.py                         # 动态长期记忆 JSON Store（被 LongTermMemory 组合）
    classifier.py                    # 动态记忆候选分类（被 LongTermMemory 组合）
    retriever.py                     # 动态记忆关键词检索（被 LongTermMemory 组合）
    profile_candidate.py             # 画像候选传递对象
    user_profile.py                  # 画像字段、类型和权限策略
    profile_store.py                 # 画像 SQLite Store 与审计（被 UserProfileManager 组合）
    MemoryItem.py                    # 短期记忆项（原始会话事件的元数据包装）
    EmotionMemory.py                 # LangGraph 运行时状态 EmotionState 定义
  proactive/
    engine.py                        # 主动对话引擎（状态机+随机调度+生气惩罚，QTimer 轮询）
    __init__.py                      # 导出 ProactiveEngine
windows/
  chat_window.py                     # 普通会话的创建、切换、恢复和显示
tests/
  agent/test_chat_agent_short_term_persistence.py
  agent/test_chat_agent_memory_integration.py
  agent/test_memory_mvp_demo.py
  memory/...（schema / store / classifier / retriever / window / profile 共 7 个测试文件）
```

**推荐使用方式**：`from agent.memory import ShortTermMemory, LongTermMemory, UserProfileManager`，
不要直接调用底层的 `SessionStore` / `FileMemoryStore` / `ProfileStore`（除非写存储层单元测试）。

### 2.2 功能状态

| 能力 | 状态 | 当前情况 |
| :--- | :---: | :--- |
| Session 创建、列出、重命名、删除 | ✅ | `SessionStore` 已使用 `user_id + session_id` 隔离，由 `ShortTermMemory` 转发。 |
| 会话原始消息持久化 | ✅ | 当前轮 `MemoryItem` 写入 SQLite，可恢复为 `BaseMessage`（交接时修复了用户/工具消息未落库的 bug，见 2.5）。 |
| 短期窗口裁剪 | ✅ | 保留最近轮次，删除旧状态消息；裁剪时触发"LLM 合并提取"（一次产出摘要+长期记忆，失败降级规则摘录）。 |
| 会话摘要持久化 | ✅ | 摘要和已摘要进度写入 Session 表。 |
| ChatWindow 历史恢复 | ✅ | 普通会话可在应用启动后恢复显示。 |
| 动态记忆 JSON 读写 | ✅ | 每用户一个 JSON 文件，支持语义/经历/规则候选（现为降级路径）。 |
| 动态记忆关键词检索 | ✅ | 结合关键词、重要度、新近程度和过期过滤（现为降级路径）。 |
| 动态记忆向量存储与检索 | ✅ | Chroma 向量库主路径（`vector_memory_store.py`），embedding 用 `qwen3-embedding:4b`，每条带日期元数据。 |
| 动态记忆 LLM 合并提取 | ✅ | `llm_extractor.py` 一次调用产出摘要+长期记忆条目（分类型、可触发词、不存原文）。 |
| 动态记忆基础 Agent 接入 | ✅ | 检索每轮刷新；写入在裁剪节点经 LLM 合并提取完成（失败降级规则提取）。 |
| 用户画像字段和权限 | ✅ | 模型只能产生候选，不能直接改画像。 |
| 用户画像候选抽取 | ✅ | 默认由 Ollama `qwen2.5:3b` 适配器处理（纯 HTTP，未用向量库）。 |
| 用户画像进入 Prompt | ✅ | `ChatAgent` 每轮经 `UserProfileManager.get_profile_context()` 注入 system prompt。 |
| 画像候选自动写入对话流程 | ✅ | `write_profile_candidate` 节点触发词命中才调用 `process_user_message()`；Ollama 不可用时自动降级（见 5.4）。 |
| 候选确认 UI | ⏳ | 后端有确认/拒绝接口，UI 尚未接入。 |
| system prompt 注入当前时间 | ✅ | `_assemble_system_prompt()` 注入"当前时间"，长期记忆每条带 `[日期]` 前缀。 |
| 好感度机制 | ✅ | 初始 0（惰性创建），回复末尾 `{{+N}}` 增量标记（主 LLM 顺带输出，零额外调用），单轮钳 ±10、上限 100，SQLite 持久化（见 5.6）。 |
| 时间感知 | ✅ | system prompt 注入"上次用户回复时间"及距今间隔（会话 history → 跨会话 SQLite → 首次对话说明三级来源），配合人设生活节奏表按时间推断"她那边"在干什么（见 [主动对话机制文档](./proactive_dialogue_mechanism.md) 第 2 节）。 |
| 主动对话 | ✅ | `ProactiveEngine` 状态机（IDLE→NUDGING→SILENT）：静默随机 15 分钟~12 小时触发问候，被无视 2~3 次生气（随机扣 5~15 好感度）后沉默，用户主动发消息哄她即全量重置；隐藏指令经 `proactive_context` 通道只进 system prompt，不入消息/历史/SQLite（见 [主动对话机制文档](./proactive_dialogue_mechanism.md)）。 |

### 2.3 当前 ChatAgent 图

当前 `agent/agent/ChatAgent.py` 的主链路如下：

```text
START
  ↓（条件路由 _route_on_session）
  ├── "load"（新 Session / 切换 / 重启）→ load_short_term_memory（从 SQLite 恢复）
  └── "skip"（同 Session 后续轮次，checkpoint 已含短期消息）
  ↓
load_runtime_context        ← 每轮：好感度快照 + 用户画像上下文
  ↓
retrieve_long_term_memory   ← 每轮：向量检索（失败降级关键词）+ 稳定偏好召回
  ↓
user_input → llm ⇄ tools（LLM 发起工具调用时）
  ↓
write_profile_candidate     ← 画像候选写入（触发词命中才调，Ollama 失败自动降级）
  ↓
trim_short_term_window      ← 裁剪 + LLM 合并提取（一次产出摘要+长期记忆）
  ↓
judge_affinity              ← 每轮：正则解析回复中 {{+N}} 好感度标记并累加（钳 ±10，上限 100）
  ↓
emit_affinity_event
  ↓
save_short_term_memory
  ↓
END
```

同一个 `session_id` 的后续轮次走 `"skip"` 分支直接进入 `load_runtime_context`：
短期消息由 LangGraph checkpoint 维护（不重复读 SQLite），但好感度快照、
用户画像和动态记忆检索**每轮都刷新**——这是长期记忆闭环的关键约定（交接时已修复）。

**主动消息轮**（`proactive_context` 非空，由 ProactiveEngine 触发）对上述图有两处改动：
`user_input` 节点不创建用户消息（返回空串）、`write_profile_candidate` 与
`judge_affinity` 直接跳过；`_assemble_system_prompt` 追加【主动消息指令】隐藏指令块，
`save_short_term_memory`（最后一个节点）只保存主动消息并清空指令。详见
[主动对话机制文档](./proactive_dialogue_mechanism.md)。

### 2.4 待接入节点（交接 TODO）

`write_memory_candidates` 之后应新增 `write_profile_candidate` 节点，调用
`user_profile_manager.process_user_message(user_id, user_input, session_id)`
产生画像候选；必须 try/except 捕获 `ProfileCandidateExtractionError` 并降级
（Ollama 不可用时不能中断聊天主流程）。代码中的 TODO 注释位于
`ChatAgent._write_memory_candidates()` 的文档字符串内。

### 2.5 交接时修复的问题（重要，便于理解旧测试失败原因）

| 问题 | 根因 | 修复 |
| :--- | :--- | :--- |
| 短期记忆从未真正写入 SQLite | `_user_input` / `_tool_call` 把 `message.content` 字符串存进 `MemoryItem.message`，`_save_short_term_memory` 用 `getattr(message, "content")` 匹配不到用户轮次，导致每轮都走 `create_session` 而不是 `save_turn`；`SessionStore` 序列化也要求 `BaseMessage`。 | 统一存消息对象（`HumanMessage` / `ToolMessage` 本体）。约定：**`MemoryItem.message` 必须是 `BaseMessage`**。 |
| 短期记忆恢复节点未生效 | HEAD 版本图中 `"load"` 分支直接路由到 `load_runtime_context`，`load_short_term_memory` 节点没有被接入。 | 重构时已接入：`"load" → load_short_term_memory → load_runtime_context`。 |
| 同 Session 后续轮次不刷新上下文 | 旧图 `"skip"` 直接进入 `user_input`，跳过画像与检索。 | `"skip" → load_runtime_context → retrieve_long_term_memory → user_input`，每轮刷新。 |

测试基线：`python -m pytest tests`（conda 环境 `YourLoverDesk`）当前 109 个用例全部通过。（2026-10 空回复根因实证：qwen3 系列思考 token 撞 Ollama 默认 num_ctx=4096 被截断（done_reason=length）→ `get_ollama_llm` 默认 `num_ctx=8192` + `reasoning=False`（config.json 可覆盖），`_chat` 对空回复分类兜底，详见 `docs/proactive_dialogue_mechanism.md` 6.4 节。）

### 2.6 当前短期记忆已经完成到什么程度

短期记忆的基本链路已经打通：

```text
创建/切换 Session
    ↓
从 SQLite 恢复该 Session 的 summary + 最近消息
    ↓
本轮用户/助手消息进入 LangGraph state
    ↓
窗口过长时按轮次裁剪，并更新 summary
    ↓
本轮消息和 summary 写回 SQLite
    ↓
应用重启后，ChatWindow 与 ChatAgent 都能分别恢复所需内容
```

这里要区分两类恢复：

- **ChatWindow** 需要恢复完整聊天历史，供用户查看。
- **ChatAgent** 只恢复 Session 摘要和最近窗口，供本轮模型推理。

系统提示词属于运行时配置：`ChatAgent._chat()` 每轮临时创建 `SystemMessage`，不写入 `MemoryItem`，也不会存到会话数据库。

---

## 3. 🧩 记忆分层与职责边界

### 3.1 身份与会话

| 标识符 | 作用 | 生命周期 |
| :--- | :--- | :--- |
| `user_id` | 用户数据隔离键；画像、动态记忆、好感度都以它为归属。 | 跨 Session。 |
| `session_id` | 一段独立聊天的标识。 | 一个会话。 |
| `thread_id` | LangGraph checkpoint 的标识；当前约定等同于 `session_id`。 | 一个会话。 |
| `event_id` / `message_id` | 一条原始会话消息的稳定标识。 | 一条消息。 |
| `memory_id` / `candidate_id` | 长期记忆或画像候选的稳定标识。 | 一个长期对象。 |

一个用户可同时拥有多个 Session。当前普通聊天窗口管理多个会话；Mini Chat 使用固定 `session_id="mini-chat"`，会独立持久化，但不出现在普通聊天历史列表中。

```text
default-user
├── session-a：日常聊天
├── session-b：学习计划
└── mini-chat：悬浮窗小聊天
```

### 3.2 四层数据

| 层级 | 用途 | 内容 | 权威存储 | 进入 Prompt 的方式 |
| :--- | :--- | :--- | :--- | :--- |
| 原始会话记录 | 界面恢复、审计、重建短期状态 | 用户/助手/工具消息 | `sessions.sqlite3` | 不全量注入。 |
| 短期记忆 | 保持当前会话连续性 | summary + 最近若干轮消息 | SQLite + LangGraph checkpoint | 每轮注入。 |
| 动态长期记忆 | 召回跨会话的相关信息 | 偏好、状态、事实、经历 | `{user_id}.json` | 仅检索 Top K 后注入。 |
| 用户画像 | 提供稳定而受控的信息 | 称呼、语言、时区等字段 | `profiles.sqlite3` | 每轮注入简短摘要。 |

### 3.3 `MemoryItem` 的定位

`MemoryItem` 表示一条原始会话事件：用户消息、助手回复或工具消息。它不是长期记忆的通用实体。

```text
MemoryItem
  └── 原始会话消息，保留 BaseMessage、角色、情绪、时间等元数据

SemanticMemory / EpisodicMemory / ProceduralCandidate
  └── 从原始消息筛选、摘要、去重后形成的动态长期记忆

UserProfile / StoredProfileCandidate
  └── 有字段白名单和权限约束的用户画像
```

`MemoryItem.message` 应继续保持为 LangChain 的 `BaseMessage`，不要改为普通字符串。保存时由 `SessionStore` 转换为消息字典 JSON；恢复时再还原 `BaseMessage`，这样不会丢失角色、工具调用和消息 ID。

### 3.4 短期压缩不等于长期记忆

| 机制 | 解决的问题 | 触发 | 结果 |
| :--- | :--- | :--- | :--- |
| 短期窗口裁剪 | Prompt 不能无限增长 | 超过窗口大小 | 旧消息从运行时 state 移除，并写入会话摘要。 |
| 长期记忆写入 | 重要信息不能因跨会话而遗忘 | 明确声明、强反馈、裁剪前筛选等 | 写入可检索的长期候选或画像候选。 |

因此，长期记忆候选不能只在窗口满时生成。例如用户说“以后叫我队长”，即使消息仍在窗口内，也应立即形成候选；否则用户很快结束会话时，这条偏好不会进入长期层。

---

## 4. 🗂️ 持久化与数据结构

### 4.1 当前持久化目录

```text
data/
  memory/
    sessions.sqlite3             # Session、原始会话消息、会话摘要
    profiles.sqlite3             # 用户画像、画像候选、变更审计
    users/
      {user_id}.json             # 动态长期记忆
```

原始会话、动态长期记忆、用户画像分开存储是有意为之：它们更新频率、检索方式、权限要求都不同。

### 4.2 Session 数据库：短期记忆和历史记录

默认路径：`data/memory/sessions.sqlite3`

| 表 | 关键字段 | 用途 |
| :--- | :--- | :--- |
| `chat_sessions` | `user_id`、`session_id`、`title`、`summary`、`summarized_through_sequence` | 保存会话基本信息和已压缩部分。 |
| `session_memory_items` | `event_id`、`user_id`、`session_id`、`sequence_no`、`message_json` | 保存会话原始消息和元数据。 |

`message_json` 使用 LangChain 的消息字典 JSON，不存 Python 对象。删除一个 Session 时，相关消息通过外键级联删除。

### 4.3 动态长期记忆文件

默认路径：`data/memory/users/{user_id}.json`

当前文件承载三类对象：

| 类型 | 作用 | 是否直接注入 |
| :--- | :--- | :---: |
| `semantic` | 偏好、事实、近期状态、长期目标。 | 是，需检索。 |
| `episodic` | 重要互动的摘要、用户目标、行动与结果。 | 是，需检索。 |
| `procedural_candidate` | 关于未来互动方式的建议。 | 否，先作为候选。 |

动态长期记忆至少应保留这些可治理信息：

| 字段类别 | 最低要求 | 用途 |
| :--- | :--- | :--- |
| 身份 | `memory_id`、`user_id`、`memory_type` | 更新、删除和用户隔离。 |
| 内容 | `content` 或 `summary`、`scope` | 给检索器和 Prompt 使用。 |
| 来源 | Session、消息 ID、原文证据 | 解释与纠错。 |
| 生命周期 | 创建/更新时间、`status`、可选 `expires_at` | 更新、遗忘和过期处理。 |
| 质量 | `importance`、`confidence`、敏感等级 | 排序和写入策略。 |

特别规则：

- “最近在赶项目”“今天情绪差”等状态必须带有效期，不能永久保存。
- 同一 `scope` 的冲突事实应更新或标记旧记录失效，不应无限堆积。
- 密码、支付信息、身份证号、精确住址等不进入长期记忆。
- 没有用户证据的模型推断不能作为确定事实写入。

### 4.4 用户画像数据库

默认路径：`data/memory/profiles.sqlite3`

| 表 | 用途 |
| :--- | :--- |
| `user_profiles` | 每个用户当前生效的固定字段和版本。 |
| `profile_update_candidates` | 待确认、已应用、已拒绝或过期的候选更新。 |
| `profile_change_logs` | 画像实际变化的审计日志。 |

当前字段策略：

| 字段 | 用户可改 | 系统可改 | 模型可直接改 | 注入 Prompt |
| :--- | :---: | :---: | :---: | :---: |
| `name` | ✅ | ✅ | ❌ | ✅ |
| `nickname` | ✅ | ✅ | ❌ | ✅ |
| `preferred_address` | ✅ | ✅ | ❌ | ✅ |
| `identity` | ✅ | ✅ | ❌ | ✅ |
| `language` | ✅ | ✅ | ❌ | ✅ |
| `timezone` | ✅ | ✅ | ❌ | ✅ |
| `onboarding_completed` | ❌ | ✅ | ❌ | ❌ |

模型只能提出画像候选。`ProfileStore` 负责字段白名单、值类型、操作者权限和事务性应用；候选应用时，画像更新、候选状态更新、审计日志会在同一事务中写入。

### 4.5 为什么画像不直接放进 `{user_id}.json`

动态记忆适合“可召回、可过期、可多条并存”的信息；画像适合“固定字段、需权限、当前值唯一”的信息。把两者混在同一个 JSON 中会带来三个问题：

1. 模型候选和用户确认难以控制权限。
2. “当前称呼”与“历史称呼偏好”难以处理冲突。
3. 审计和版本更新无法保证一致性。

所以当前采用 JSON 存动态长期记忆、SQLite 存用户画像的混合方案。

---

## 5. ⚙️ 运行机制设计

### 5.1 目标单轮流程

长期记忆接入完成后，每一轮遵循下面的顺序：

```text
1. 接收 user_id、session_id、当前用户输入
2. 如果是切换/重启：恢复该 Session 的 summary 与最近消息
3. 每轮读取最新 love_snapshot 和 profile_context
4. 用当前用户输入检索相关 semantic / episodic memories
5. 组装工作记忆并调用 LLM / tools
6. 产生回复后，生成动态记忆候选和画像候选
7. 裁剪短期窗口并更新 summary
8. 保存本轮会话消息、摘要和候选写入结果
9. 将带 session_id 的回复事件返回 UI
```

这里的核心原则是：

- **短期恢复按 Session 发生**：切换或重启时从 SQLite 恢复。
- **运行时上下文按轮发生**：好感度、画像、动态检索每轮刷新。
- **长期记忆按候选发生**：不是每一句都入库，也不只等窗口溢出才处理。

### 5.2 短期记忆运行机制

当前窗口参数为 `max_messages=20`、`keep_recent_turns=6`。裁剪规则：

1. 未超过 `max_messages` 时不裁剪。
2. 从一条 `HumanMessage` 到下一条 `HumanMessage` 前的消息算一轮。
3. 保留最近 `keep_recent_turns` 个完整轮次；工具消息不会和所在轮次拆散。
4. 仍然超出窗口时，继续移除最早的保留轮次。
5. 将移除的可摘要消息连同旧摘要交给**当前 LLM 做合并提取**：一次调用同时产出
   新 `session_summary` 与长期记忆条目（见 5.3），实现于
   `ChatAgent._trim_short_term_window()` + `LLMMemoryExtractor`。

LLM 合并提取失败（异常或空输出）时自动降级为规则式摘录（每条片段截 80 字拼接，
总长限 1000 字，`ChatAgent._legacy_session_summary()`），保证压缩失败不丢失
会话消息、不中断聊天。摘要边界：

```text
输入：本次被裁剪的消息 + 旧 summary
输出：新的、有限长度的 session_summary
```

摘要服务不应直接读数据库、改 UI 或负责动态长期记忆写入。

### 5.3 动态长期记忆的写入机制

当前长期记忆采用 **LLM 合并提取 + Chroma 向量存储** 主路径：

```text
trim_short_term_window 触发（消息数 > max_messages）
  ↓
被裁剪消息（带 MemoryItem.speak_time 发生时间）
  ↓
LLMMemoryExtractor.extract()  ← 一次 LLM 调用，同时产出两部分（分隔符隔开）
  ├── <<<SESSION_SUMMARY>>>      → session_summary（短期记忆）
  └── <<<LONG_TERM_MEMORIES>>>  → 长期记忆条目（已按 semantic/episodic/procedural 切分）
        ↓
LongTermMemory.write_llm_memories()
  ├── 主路径：VectorMemoryStore → Chroma 向量化，系统盖章日期元数据
  └── 降级：JSON（SemanticMemory + tags），关键词可检索
```

LLM 提取规则（提示词内约定）：
- 值得存：用户明确"记住/记得/请记得"、明确偏好、重要事实、身份、情绪事件、未来计划/约定、"问题-解决"过程；
- 不存：寒暄、玩笑、模型自行推断的偏好、一次性闲聊；
- 每条必须是**提炼后的理解**（非原文），标注 `[semantic]` / `[episodic]` / `[procedural]` 类型；
- 存储日期由系统在写入时盖章（`created_date` 元数据），注入 Prompt 时以 `[日期]` 前缀展示。

主路径失败（LLM 提取异常 / Chroma 不可用 / embedding 模型未启动）时自动降级：
- 提取失败 → 回退 `LongTermMemory.write_candidates()`（规则分类器）+ 规则式摘要；
- 检索失败 → 回退 `_legacy_retrieve_context()`（关键词 + 偏好无条件召回）。

写入判断参考（降级规则路径仍沿用）：

| 情况 | 建议处理 |
| :--- | :--- |
| “记住我喜欢简短回答” | semantic preference 候选。 |
| “我最近在准备答辩” | semantic state，设置 TTL。 |
| 用户和 Agent 完成一次有效复习计划 | episodic 候选，记录目标、行动和结果摘要。 |
| “以后日报按三段式写” | procedural candidate；暂不直接进入系统规则。 |
| 玩笑、模糊猜测、一次性闲聊 | 不写长期记忆。 |
| 模型自行推断的偏好 | 不直接写入；必要时向用户确认。 |

### 5.4 用户画像候选机制

#### 触发预检（性能优化）

画像候选提取**不是每轮都调 LLM**：`ChatAgent._write_profile_candidate` 先用
正则触发词（`matches_profile_trigger`，见 `profile_service.py` 的
`PROFILE_TRIGGER_PATTERNS`）对当轮用户消息做预判，**命中才调用候选模型**，
未命中直接跳过（`profile_candidate_ids` 为空）。普通闲聊不再产生 LLM 请求。

```text
用户消息
  ↓
命中画像触发词？(matches_profile_trigger)
  ├── 否 → 跳过本轮画像候选（不调 LLM）
  └── 是 → Ollama 候选抽取器（qwen2.5:3b）
            ↓
        field + value + evidence + confidence + explicit
            ↓
        ProfileStore 校验权限、字段和值
            ↓
        满足严格自动应用条件？
          ├── 是：更新画像 + 候选标记 APPLIED + 记审计日志
          └── 否：候选标记 PENDING，等待用户确认或系统处理
```

#### 画像字段白名单

字段定义在 `agent/memory/user_profile.py` 的 `PROFILE_FIELD_POLICIES`，
是画像权限的唯一依据（模型返回白名单外字段一律拒绝）：

| 字段 | 中文名 | 类型 | 允许写入者 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `name` | 姓名 | text | USER/SYSTEM | 最长 100 字 |
| `nickname` | 昵称 | text | USER/SYSTEM | 最长 100 字 |
| `preferred_address` | 希望的称呼 | text | USER/SYSTEM | 最长 100 字 |
| `identity` | 身份 | text | USER/SYSTEM | 最长 200 字 |
| `language` | 常用语言 | text | USER/SYSTEM | 最长 50 字 |
| `timezone` | 时区 | text | USER/SYSTEM | 最长 50 字 |
| `onboarding_completed` | 初始化状态 | boolean | 仅 SYSTEM | 不注入上下文、不可清空 |

模型永远不能直接写画像（`MODEL` 角色只产候选）；兴趣/目标/情绪/回答风格
不属于画像，提示词明确禁止提取，它们属于动态长期记忆。

#### 触发词规则（PROFILE_TRIGGER_PATTERNS）

规则宁全勿缺：漏触发会导致画像信息不落库，误触发最多浪费一次 LLM 调用
（相比旧的"每轮都调"仍是大幅裁剪）。当前规则覆盖：

| 画像字段 | 触发词（正则片段） |
| :--- | :--- |
| 姓名/昵称/称呼 | `我叫`、`我是`、`名字是`、`名字叫`、`叫我`、`别叫我`、`请叫我`、`可以叫我`、`以后叫我`、`就叫`、`称呼我`、`你叫我` |
| 身份 | `我是一名`、`我是一位`、`我是做`（`我是` 兜底身份） |
| 语言 | `说中文/英文/英语`、`用中文/英文/英语`、`讲中文/英文/英语`、`语言` |
| 时区 | `时区` |

> 维护说明：新增画像表达方式时，优先补触发词而不是放宽字段白名单——
> 触发词决定"是否值得让 LLM 看一眼"，白名单决定"模型能写什么"。

#### 候选处理与自动应用

严格自动应用至少要求：用户明确表达（`source=user_explicit`）、置信度达到
`0.85`、引用文本真实存在于原始输入、候选值可验证、字段允许用户更新、
候选不要求确认。

例子：

| 用户输入 | 正确归属 |
| :--- | :--- |
| “以后叫我队长。” | `preferred_address` 画像候选。 |
| “我叫林林。” | `name` 或 `nickname` 候选，按抽取结果和证据处理。 |
| “我这周很焦虑。” | 动态状态记忆，带 TTL，不应变成固定画像。 |
| “猜猜我喜欢什么。” | 不生成画像更新。 |

如果 Ollama 未启动、请求超时或返回格式错误，画像候选流程应跳过并记录错误，**不能中断正常聊天**。

### 5.5 长期记忆检索机制

每轮在 LLM 调用前检索：

```text
query = 当前用户输入
    ↓
MemoryRetriever.retrieve_memories(user_id, query, top_k)
    ↓
过滤已过期记录与 procedural candidate
    ↓
按关键词匹配、重要度、新近程度排序
    ↓
MemoryRetriever.format_memory_context(...)
    ↓
写入本轮 memory_context
```

当前是关键词检索，适合验证“写入—召回—注入”闭环。未来使用向量库时，只替换 `MemoryRetriever` 的内部排序实现，不让 `ChatAgent` 直接依赖 Chroma、Milvus 或其他具体组件。

### 5.6 好感度机制

好感度（Affinity）是关系状态，与记忆模块分离。实现位于 `agent/affinity/`：

| 组件 | 职责 |
| :--- | :--- |
| `AffinityStore` | SQLite 持久化：`affinity_state(user_id PK, love_value, updated_at)`，新用户初始 0。 |
| `AffinityService` | 读取快照、应用增量（`apply_delta`，钳制 0-100）、事件记录。 |

增量来源（零额外 LLM 调用）：主 LLM 按 system prompt 约定在每条回复末尾
依次输出情绪标记与好感度增量标记，两者**刻意使用两套符号**，避免正则
解析互相干扰：

| 标记 | 符号 | 示例 | 解析方 |
| :--- | :--- | :--- | :--- |
| 情绪标记 | 双尖括号 `<<...>>` | `<<Happy>>` | UI 侧 `message_filter.extract_emotion` |
| 好感度增量标记 | 双花括号 `{{...}}` | `{{+3}}` / `{{-3}}` | `ChatAgent._judge_affinity` |

`ChatAgent._judge_affinity` 节点仅做正则解析（`_AFFINITY_DELTA_RE` =
`\{\{([+-]?\d{1,3})\}\}`），不再单独调用 LLM 判定（2026-10 重构：删除了
原 `LLMAffinityJudge` 判定器，每轮 LLM 调用从 2 次降回 1 次）。
`message_filter.extract_emotion` 会同时剔除两种标记，避免残留到聊天界面；
`agent_controller` 取 `<<...>>` 列表的第一个作表情，两套标记互不干扰。

规则：
- **初始好感度为 0，惰性初始化**：没有显式的"建号"步骤。
  `AffinityStore.get_love(user_id)`（`affinity_store.py`）首次读取某
  `user_id` 时查无记录，即 `INSERT ... VALUES (?, 0, ...)` 创建 0 分行并
  返回 0。调用链：每轮 `load_runtime_context` →
  `AffinityService.get_love_snapshot` → `AffinityStore.get_love`。
  因此第一次对话（无任何记忆）时好感度即为 0。
- **每轮对话后累加**：`judge_affinity` 节点从最近一条 AI 回复中提取
  增量（取最后一个标记），`apply_delta` 累加。
- **双重钳制**：单轮增量钳制 ±10（`MAX_AFFINITY_DELTA_PER_TURN`）；
  累计结果由 `apply_delta` 钳制到 `[0, 100]`。
- **降级**：回复未带标记（模型忘记输出）时增量 0，好感度不变，聊天不中断。

人设联动：system prompt 固定追加 `agent/prompts/global_rules.md` 中的
【好感度与关系阶段】准则。0~19 保持分寸 → 20~39 熟悉放松 →
40~59 增加信任 → 60~79 在人物与用户意愿允许的范围内亲近 →
80~100 稳定熟悉。关系不因分数增长被强制改成恋人，朋友等自定义
人设仍保留原有边界。人物描述由默认只读副本或自定义草稿提供。

时序约定：`love_snapshot` 在每轮开头由 `load_runtime_context` 读取（本轮
system prompt 用上轮值）；`judge_affinity` 在轮末解析标记并累加写回
state——新值从**下一轮**开始生效，与画像"本轮说、下轮生效"一致。

```text
llm 回复（末尾带 <<Happy>>{{+3}}）
  ↓
judge_affinity（正则解析 {{+N}} 增量标记，钳制 ±10）
  ↓
AffinityService.apply_delta(user_id, delta)  →  钳制 [0,100] + SQLite 持久化
  ↓
emit_affinity_event（事件携带最新 love_snapshot）
```

### 5.7 工作记忆与 Prompt 组装

推荐 Prompt 层次：

```text
[System]
基础人格与安全规则

[Runtime Context]
当前 love_snapshot
用户画像摘要 profile_context
相关动态长期记忆 memory_context
当前会话摘要 session_summary

[Conversation]
最近窗口消息 + 当前用户消息
```

当前 `_assemble_system_prompt()` 已注入基础系统提示词、好感度、`profile_context`、`memory_context` 和 `session_summary`（画像已接入，并在 Prompt 中明确提示模型“画像与长期记忆仅供辅助回复，不应当作用户本轮新说的话”）。

注入预算原则：

1. 基础系统提示词始终保留且长度固定。
2. 画像字段少、值短；未设置字段不输出。
3. 长期记忆只取 Top K，每条限制长度，过期记录不注入。
4. Session summary 有长度上限。
5. 最近对话保留完整轮次，优先级高于很久以前的记忆。

当上下文过长时，优先减少动态召回数量，再压缩 summary，最后才减少最近轮次。

---

## 6. 🔧 与当前代码的改造映射

### 6.1 目标图与当前差距

原目标图（区分“只在切换时恢复会话”和“每轮刷新运行时上下文”）**已基本实现**，
当前实际图见 2.3：

```text
llm
  ↓
write_profile_candidate     ← 画像候选（触发词命中才调）
  ↓
trim_short_term_window      ← 裁剪 + LLM 合并提取（摘要+长期记忆）
  ↓
judge_affinity              ← 好感度增量标记解析与累加（钳 ±10，上限 100）
  ↓
emit_affinity_event
  ↓
save_short_term_memory
```

画像候选节点要点：

1. 触发词预判（`matches_profile_trigger`）命中才调用
   `process_user_message(user_id, user_input, session_id, message_id)`；
2. `try/except ProfileCandidateExtractionError`，失败时记录日志并返回 `{}`（降级不中断）；
3. 候选结果（`ProfileProcessingResult` 列表）写入 state，供 UI 事件使用。

`load_short_term_memory` 已保留“同 Session 不重复加载”的优化，同时不再跳过画像、
动态检索和好感度快照（`"skip"` 分支仍会经过 `load_runtime_context` 和
`retrieve_long_term_memory`）。

### 6.2 节点职责与状态字段

| 节点 | 作用 | 关键输出 |
| :--- | :--- | :--- |
| `load_short_term_memory` | 仅在需要时恢复 Session 摘要和最近消息。 | `messages`、`session_summary`。 |
| `load_runtime_context` | 每轮读取好感度快照并格式化用户画像。 | `love_snapshot`、`profile_context`。 |
| `retrieve_long_term_memory` | 每轮按当前输入召回动态记忆 + 稳定偏好。 | `memory_context`。 |
| `write_profile_candidate` | 触发词命中才调用 `UserProfileManager` 处理本轮用户消息。 | `profile_candidate_ids`。 |
| `trim_short_term_window` | 裁剪运行时消息，LLM 合并提取摘要+长期记忆。 | 删除指令、`session_summary`、`memory_write_ids`。 |
| `judge_affinity` | 正则解析最近 AI 回复中的 `{{+N}}` 增量标记并累加（无标记则不变）。 | `love_snapshot`（更新后）。 |
| `emit_affinity_event` | 发射记忆写入完成事件（含最新好感度）。 | 事件记录。 |
| `save_short_term_memory` | 保存当前轮会话消息及摘要进度。 | SQLite 持久化结果。 |

建议在 Agent state 中只保留本轮所需的轻量视图：

```text
user_id / session_id / current_message_id
session_summary
love_snapshot
profile_context
memory_context
retrieved_memory_ids
```

不要把整个用户画像对象、完整 JSON 记忆文件或全部数据库查询记录放进 LangGraph state。

### 6.3 需要修改的既有文件

> ⚠️ 下列文件是后续接入长期记忆时需要改动的现有架构文件。按照当前协作约定，真正开始编辑前要先说明本次改动范围。

| 文件 | 后续改动 | 原因 |
| :--- | :--- | :--- |
| `agent/agent/ChatAgent.py` | ~~注入画像管理器，调整路由为每轮刷新上下文~~（已完成）；剩余：增加 `write_profile_candidate` 节点。 | 完成画像候选运行时闭环。 |
| `app.py` | 完整转发回复的 `session_id`；必要时增加候选展示事件。 | 防止切换 Session 后回复显示串台。 |
| `core/agent_event.py` | 仅在候选交互需要更多事件字段时扩展。 | 让 UI 与服务通过事件传递数据。 |
| `windows/chat_window.py` | 将候选确认/拒绝、回复显示严格限制在匹配的 Session。 | UI 不直接操作记忆数据库。 |

新增的独立逻辑优先放入 `agent/memory/`，例如未来的 `memory_policy.py`、`summary_service.py`；存储层不能依赖 PySide 或直接发 UI 信号。

### 6.4 Session 与 UI 的正确性要求

`AgentEvent` 已有 `session_id` 字段，但当前部分回复事件构造与显示链路没有完整保留它。用户在模型回复期间切换会话时，旧会话回复可能显示到新会话。

正确链路必须是：

```text
发送消息时捕获 session_id
  ↓
AgentWorker / AgentController 原样透传
  ↓
回复事件携带同一个 session_id
  ↓
ChatWindow 仅更新匹配的会话
```

这个问题与长期记忆无直接数据依赖，但它影响多 Session 的来源正确性，应在候选 UI 之前修复或至少加入回归测试。

---

## 7. ✅ 实施阶段与验收标准

### Phase 1：短期记忆 MVP（基本完成）

- [x] Session SQLite 建表与 CRUD。
- [x] 当前轮消息持久化、重启恢复。
- [x] 窗口裁剪与会话摘要持久化。
- [x] ChatWindow 会话历史恢复。
- [ ] 补充 GUI 回归测试。
- [ ] 从消息数窗口演进到 token 预算窗口。

验收：创建两个普通会话并来回切换；重启应用后每个会话历史独立恢复；模型只使用摘要和最近窗口，不串 Session。

### Phase 2：动态长期记忆闭环（当前阶段）

- [x] semantic / episodic / procedural candidate 数据模型。
- [x] 每用户 JSON Store、基础分类器、关键词检索器。
- [x] 基础检索与候选写入节点。
- [x] 将动态检索调整为每轮运行（交接时完成）。
- [ ] 补齐候选去重、冲突更新和 TTL 的端到端验证。
- [ ] 在调试日志中记录本轮召回 ID 和写入结果（注意脱敏）。

验收：用户明确说出一个偏好，关闭当前 Session 后在新 Session 问相关问题，可以只召回该用户的相关偏好；临时状态过期后不再被召回。

### Phase 3：用户画像接入（紧随其后）

- [x] 画像字段策略、权限校验、SQLite 和审计。
- [x] Ollama `qwen2.5:3b` 画像候选适配器。
- [x] 在 ChatAgent 中每轮调用 `get_profile_context()` 并注入 Prompt（交接时完成）。
- [ ] 在回复后调用 `process_user_message()` 产生画像候选（即 6.1 的 `write_profile_candidate` 节点）。
- [ ] 确保 Ollama 不可用时对话正常降级（try/except + 日志）。
- [ ] 提供候选确认/拒绝的 UI 或临时系统入口。

验收：用户说“以后叫我队长”后，候选有证据、状态与审计；符合严格规则时应用，不符合时停留在待确认；下一轮 Prompt 可得到已生效画像。

### Phase 4：质量与扩展

- [x] 引入高质量摘要（LLM 合并提取已接入 `trim_short_term_window`，一次产出摘要+长期记忆，失败自动降级）。
- [ ] 引入 token 预算（按 token 而非消息条数控制上下文）。
- [x] 接入向量检索（Chroma 主路径，`vector_memory_store.py`；关键词作为降级兜底）。
- [ ] 混合检索 + 重排序（向量 + BM25 + RRF，参考 `RAGTool.py`）。
- [ ] 支持用户查看、修改、删除、导出自己的记忆。
- [ ] 增加敏感信息策略、评估集、监控和遗忘任务。

---

## 8. 🚀 MVP 之后的优化方向

MVP 的目标是先验证“能存、能取、能影响回复”。跑通之后，优化重点不应是立刻增加更多记忆类型，而是逐步提高**上下文质量、检索准确度、可控性和可验证性**。

### 8.1 短期记忆：从规则窗口升级为 token 预算与语义压缩

当前短期记忆使用“消息条数 + 保留最近轮次 + LLM 合并提取（摘要+长期记忆一次产出，失败降级规则摘录）”的方案。它简单、稳定，适合 MVP，但有明显限制：

| 当前规则 | 问题 |
| :--- | :--- |
| `max_messages=20` | 一条很长的消息和一句“好的”都只算一条，不能真实控制模型上下文。 |
| `keep_recent_turns=6` | 固定保留最近轮次，不理解哪一轮更重要。 |
| LLM 滚动摘要（自由散文） | 已解决截断丢信息问题，但未按小节组织，token 精度仍不可控。 |
| 所有消息同等处理 | 工具输出、闲聊、用户纠正和未完成任务的重要性不同。 |

建议按三个阶段演进，而不是一次重写：

#### 阶段 A：Token 预算窗口

先把“消息数量”改成“上下文预算”。每轮先为各部分预留上限，再用剩余 token 放最近对话。

```text
模型总上下文预算
├── 基础 System Prompt：固定预算
├── 用户画像 + 好感度：小且固定的预算
├── 动态长期记忆 Top K：可缩减预算
├── Session Summary：可缩减预算
└── 最近会话消息：使用剩余预算，按完整轮次保留
```

实现原则：

1. 计数逻辑要封装在独立的 `token_counter` 或摘要服务中，不散落在 `ChatAgent`。
2. 优先裁剪无关的动态记忆，再缩短 summary，最后才丢弃最近完整轮次。
3. 工具返回过长时保存原始记录，但向模型注入短摘要或关键字段。
4. 不同模型 tokenizer 不同；接口应接收“模型对应的计数器”，不能把某个模型的 token 规则写死。

验收不是“正好不超过 token”，还要验证：用户最近的约束、问题和 Agent 未完成任务在裁剪后仍然存在。

#### 阶段 B：增量语义摘要（基础版已实现）

已实现增量摘要：`ChatAgent._trim_short_term_window()` 只总结本次离开窗口的轮次，
并和旧摘要合并交给当前 LLM（`LLMMemoryExtractor`）——**同一调用同时产出
会话摘要与长期记忆条目**，不再重读整个会话，也不再是 80 字截断拼接。
降级与进度要求均已满足：LLM 失败时回退规则式摘录（`_legacy_session_summary()`），
`summarized_through_sequence` 防止重复合并。

后续增强方向——把摘要组织为固定小节，而不是自由散文：

```text
[已确认事实]
- 用户希望回答简洁。

[当前主题与进度]
- 正在排查长期记忆接入；已完成短期持久化。

[未完成事项]
- 下一步需要将 profile_context 注入 ChatAgent。

[本会话约束]
- 修改既有文件前先说明影响范围。
```

这样做的好处是：模型更容易利用摘要；测试可以逐项断言“约束是否仍在”；旧摘要也可按字段合并、替换或删除。

摘要模型失败时必须降级为当前规则式摘录，不能因为压缩失败而丢失会话消息。摘要本身应记录覆盖到哪个 `sequence_no`，防止重复合并。

#### 阶段 C：重要度与固定内容

窗口裁剪前可先给“对话轮次”而不是单条消息评估重要度：

| 高优先级，应尽量保留 | 低优先级，可优先压缩 |
| :--- | :--- |
| 用户纠正、明确偏好、任务约束、未完成计划、工具关键结果 | 寒暄、重复确认、无结果的工具中间输出、重复情绪描述 |

初版可以继续使用透明的规则：包含“记住”“不要”“以后”“下一步”、任务 ID、失败原因的轮次加分。之后再增加模型辅助分类，但模型结果只能影响“保留/摘要策略”，不应直接删除原始记录。

### 8.2 动态长期记忆：从关键词召回升级为混合检索

当前 `MemoryRetriever` 已能跑通关键词、重要度、时效性排序。后续建议按下面的顺序增强：

1. **先完善元数据过滤**：始终按 `user_id` 过滤；过滤已删除、已过期、未确认、类型不匹配的记录。
2. **加入主题和时间信号**：例如“最近”“上次”“考试”应提高近期状态和相关经历的权重。
3. **接入向量召回**：为可检索内容生成 embedding，解决“喜欢爵士”与“爱听音乐”这类同义表达。
4. **混合检索和重排序**：关键词保证精确匹配，向量保证语义匹配，最后根据重要度、时间和 query 的相关性重排 Top K。
5. **检索反馈**：记录“本轮实际召回了哪些 ID”，人工或评测集据此判断是否召回正确，而不是只看模型回复表面上是否自然。

向量库应仍藏在 `MemoryRetriever` 后面。`ChatAgent` 只需要“输入 query，得到有限条已格式化记忆”，不应关心底层是 JSON、SQLite 还是 Chroma。

### 8.3 长期记忆写入：从规则候选升级为可合并、可遗忘的数据

当前规则分类器适合作为安全的第一层。后续优化应围绕“写得更少，但更准”：

| 优化点 | 方向 |
| :--- | :--- |
| 规则 + 模型协作 | 规则先识别明确触发，模型把复杂对话提炼为结构化候选；模型仍不能直接写库。 |
| 去重与合并 | 同一偏好反复出现时更新原记录的证据、重要度和更新时间，而不是新增多条。 |
| 冲突处理 | 用户明确纠正时，新信息优先，旧记录标记 `superseded` 或失效。 |
| TTL 与遗忘 | 近期状态、临时计划自动过期；低重要度且长期未使用的记录可归档。 |
| 敏感策略 | 在生成候选前进行敏感信息检测；高风险内容默认不自动保存。 |
| 用户反馈闭环 | 用户说“这不是我”“别记这个”时能定位来源并删除/修正。 |

长期记忆的质量指标不是“库里有多少条”，而是“被召回的信息是否真实、相关、不过期、不会让用户感到被误解”。

### 8.4 用户画像：从后端候选扩展到用户可控的配置

画像后端完成后，下一步不只是注入 Prompt，还要把控制权交给用户：

1. 在聊天窗口或设置页展示 `PENDING` 候选，让用户确认或拒绝。
2. 支持用户直接修改昵称、称呼、语言等允许字段，直接修改要写审计日志。
3. 每个画像字段展示来源和更新时间，避免“系统为什么这样叫我”无法解释。
4. 允许用户查看、导出、清空画像；清空应保留必要的业务边界，但删除用户可控数据。
5. 当多个画像字段互相冲突时，优先用户手动修改，其次用户明确表述，最后才是模型候选。

不要把所有动态事实都塞进画像。画像应保持少、稳定、可解释；经常变化的信息仍放在带 TTL 的动态记忆中。

### 8.5 会话与摘要的长期管理

当 Session 数量变多后，需要把“聊天历史”与“工作记忆”继续区分：

| 方向 | 目的 |
| :--- | :--- |
| 会话标题自动建议 | 根据最初几轮对话生成候选标题，但用户标题优先。 |
| 会话归档 | 老会话不必默认加载到 UI，但不立即删除。 |
| 会话级摘要 | 结束/长时间未活动时生成最终摘要，成为动态长期记忆候选的来源之一。 |
| 并发与幂等 | 为每个事件使用稳定 ID；重复回调不能重复写消息或候选。 |
| Session 事件关联 | 回复、工具结果、候选通知都必须携带 `session_id`，避免 UI 串台。 |

### 8.6 评估、观测与迁移

记忆系统如果只凭聊天感觉，很难判断是“记住了”还是“模型猜对了”。MVP 后应逐步补齐以下能力：

| 能力 | 最小做法 |
| :--- | :--- |
| 回归评测集 | 准备 20～50 个固定场景：偏好、纠正、临时状态、跨 Session 召回、拒绝候选、过期记忆。 |
| 可观测日志 | 记录脱敏后的检索 query、召回 ID、写入/拒绝原因、summary 覆盖序号和耗时。 |
| Prompt 调试视图 | 仅开发模式下显示各上下文部分的长度与条数，不暴露敏感原文。 |
| 数据迁移 | 为 JSON 和 SQLite schema 保留版本号、备份、迁移脚本和回滚策略。 |
| 压力测试 | 验证多 Session、长会话、大量记忆、Ollama 不可用、数据库占用时的降级行为。 |

建议每完成一项优化，都增加“写入 → 新 Session 召回 → 用户纠正/删除 → 再次验证”的端到端测试，而不是只测试单个 Store 方法。

---

## 9. ⚠️ 风险、已知问题与开发约定

### 9.1 当前已知问题

| 问题 | 影响 | 处理方向 |
| :--- | :--- | :--- |
| 画像候选已接入对话流程 | `write_profile_candidate` 节点每轮运行，Ollama 失败自动降级；候选确认 UI 未接。 | 见 2.4 / 6.1。 |
| 摘要为 LLM 自由散文压缩 | 已解决规则截断丢信息问题；token 精度与小节化组织仍待改进（见 8.1 阶段 B）。 |
| 向量检索仅按 query 召回 | 未做向量+BM25 混合与重排，同义召回靠 embedding 质量。 | 参考 `RAGTool.py` 增加 BM25 + RRF 融合。 |
| 回复 Session ID 透传不完整 | 切换会话时可能显示串台。 | 补齐 `app.py` 到 ChatWindow 的事件链路。 |
| 候选确认无 UI | 低置信候选无法由用户处理。 | 通过事件总线接入设置页或聊天窗口。 |
| Ollama 候选抽取依赖本地服务 | 画像候选流程依赖 Ollama 运行。 | 接入时必须 try/except 降级（见 2.4）。 |

已解决（交接时修复，保留记录便于理解历史）：

| 已解决问题 | 说明 |
| :--- | :--- |
| 同 Session 后续轮次跳过加载链路 | `"skip"` 分支现在仍经过 `load_runtime_context` 和 `retrieve_long_term_memory`，每轮刷新。 |
| 用户画像尚未注入 Prompt | `_load_runtime_context` 每轮调用 `get_profile_context()`，`_assemble_system_prompt` 已注入。 |
| 短期记忆未真正写入 SQLite | `MemoryItem.message` 统一存 `BaseMessage`（见 2.5）。 |

### 9.2 记忆污染与冲突处理

长期记忆不是只追加，还要可更新、可纠错、可遗忘：

| 场景 | 处理规则 |
| :--- | :--- |
| 用户纠正称呼或偏好 | 新信息优先；旧动态记忆标记失效或更新，画像写审计。 |
| 近期状态变化 | 新状态替代旧状态，旧记录到期或失效。 |
| 同一偏好反复出现 | 提高已有记录的重要度/置信度，不重复创建大量记录。 |
| 玩笑、角色扮演、模型猜测 | 不作为确定长期事实。 |
| 敏感信息 | 默认不写；如产品未来允许，必须有独立确认、加密/脱敏和删除机制。 |

### 9.3 开发约定

> 💡 **约定一：模型输出先是候选。** 特别是画像信息，模型没有直接写库权限。

> 💡 **约定二：数据必须可追溯。** 新长期记忆字段要考虑来源消息、创建/更新时间、状态、过期时间和用户隔离。

> 💡 **约定三：存储层不依赖 UI。** Store 与 Service 不能导入 PySide，也不直接控制窗口。

> 💡 **约定四：失败必须降级。** 检索为空、JSON/SQLite 写入失败或 Ollama 未运行时，聊天主流程不能中断。

> 💡 **约定五：Prompt 不是权威数据源。** `profile_context`、`memory_context` 和 `session_summary` 都是运行时视图，不能反向当作数据库。

> 💡 **约定六：改已有文件先说明。** 新模块优先独立放到 `agent/memory/`；改 `ChatAgent.py`、`app.py`、`chat_window.py` 前先列出影响范围。

> 💡 **约定七：提示词不列具体工具清单。** 工具经 `llm.bind_tools(tool_map)` 绑定，模型按工具 description 自行选择；system prompt 只提醒"具备工具调用能力"与调用时机（日常谈心不调用、结果用自己的口吻转述），避免提示词与工具注册表双份维护。

> 💡 **约定八：结构化标记分符号体系。** 情绪标记用双尖括号 `<<Happy>>`，好感度增量用双花括号 `{{+3}}`，新增输出标记必须使用新符号并同步 `message_filter` 剔除逻辑，防止正则解析互相干扰。

> 💡 **约定九：人物描述与全局规则分层维护。** `PERSONA_PROMPT_MODE` 和 `CUSTOM_PERSONA_PROMPT` 保存在 `config.json`；默认人物副本与通用规则分别维护在 `agent/prompts/default_persona.md` 和 `global_rules.md`。每轮开始快照人设，组装顺序为人物描述 → 运行时资料与主动指令 → 固定全局规则。界面只允许编辑人物描述，不允许写回默认副本或基础协议。
> - **生活与表达连续性**：人物背景、情绪与称呼前后一致，避免每轮重新编造共同经历或强行亲近。
> - **真实能力边界**：角色日常属于虚构背景，不能冒充真实感知或已执行的线下行为；不靠否认技术身份维持人设。
> 详细使用、迁移和测试约定见 [人物提示词](persona_prompts.md)。

> 💡 **约定十：主动消息指令走独立隐藏通道。** 主动对话（ProactiveEngine）产出的指令文本只能经 `EmotionState.proactive_context` 注入 system prompt：不创建用户消息、不写入 history/SQLite，由 `save_short_term_memory`（最后节点）统一清空 + 控制器显式传空串双保险。新增"系统注入的隐藏指令"场景时沿用此通道，禁止伪造用户消息（会污染消息列表与活跃时间统计）。详细机制见 [主动对话机制文档](./proactive_dialogue_mechanism.md)。

### 9.4 最低隐私要求

1. 所有查询必须以 `user_id` 作为强制过滤条件。
2. 不在日志中输出完整画像、完整 Prompt 或未脱敏聊天记录。
3. “忘记用户”功能未来需要同时覆盖 Session、动态 JSON、画像、候选和审计数据，删除范围必须明确。
4. 新增敏感字段前，要先定义保存授权、展示、编辑、删除和过期规则。

---

## 10. 📚 关键接口速查（三大管理器）

> 统一入口：`from agent.memory import ShortTermMemory, LongTermMemory, UserProfileManager`

### 10.0 三大管理器总览

| 管理器 | 文件 | 组合的底层模块 | 权威存储 |
| :--- | :--- | :--- | :--- |
| `ShortTermMemory` | `agent/memory/short_term_memory.py` | `SessionStore`（SQLite）+ `window.py` 裁剪函数 | `data/memory/sessions.sqlite3` |
| `LongTermMemory` | `agent/memory/long_term_memory.py` | `FileMemoryStore`（JSON）+ `MemoryClassifier` + `MemoryRetriever` | `data/memory/users/{user_id}.json` |
| `UserProfileManager` | `agent/memory/profile_service.py`（原 `long_memory.py`） | `ProfileStore`（SQLite）+ `OllamaProfileCandidateExtractor` | `data/memory/profiles.sqlite3` |

构造函数均支持注入底层实现（测试传临时路径，生产用默认值）：

```python
# 短期：可注入 SessionStore 或 database_path，窗口参数 max_messages=20 / keep_recent_turns=6
ShortTermMemory(session_store=SessionStore(tmp / "sessions.sqlite3"))

# 长期：可注入 store/classifier/retriever 或 root_dir，preference_top_k=20
LongTermMemory(store=FileMemoryStore(tmp / "users"))

# 画像：可注入 store 与候选提取器，auto_apply_confidence=0.85
UserProfileManager(store=ProfileStore(tmp / "profiles.sqlite3"))
```

ChatAgent 的注入方式（`agent/agent/ChatAgent.py`）：

```python
agent = ChatAgent(
    llm,
    settings,
    {"configurable": {"thread_id": session_id}},
    short_term_memory=short_term,  # 不传则用默认 SQLite 存储
    long_term_memory=long_term,  # 不传则用默认 JSON 存储 + 规则分类器
    user_profile_manager=profile,  # 不传则用默认画像库 + Ollama 候选抽取
    affinity_service=None,  # 可选，默认 AffinityService()
)
```

### 10.1 ShortTermMemory（短期记忆）

| 接口 | 用途 |
| :--- | :--- |
| `create_session(user_id, session_id, title="")` | 创建 Session（已存在则幂等返回现有记录）。 |
| `get_session(user_id, session_id)` | 读取 Session 记录；不存在返回 `None`。 |
| `list_sessions(user_id)` | 列出用户全部 Session（按更新时间倒序），供 ChatWindow 恢复列表。 |
| `rename_session(user_id, session_id, title)` | 更新会话标题。 |
| `delete_session(user_id, session_id)` | 删除会话及级联消息。 |
| `load_session(user_id, session_id, recent_limit=None)` | 加载快照：摘要 + 最近 `max_messages` 条 `MemoryItem`；会话切换/重启恢复用。 |
| `load_messages(user_id, session_id, limit=None)` | 读取会话 `MemoryItem` 列表。 |
| `load_item_records(user_id, session_id, limit=None)` | 读取带 `event_id`/`sequence_no` 的完整记录（摘要进度同步用）。 |
| `save_turn(user_id, session_id, items, event_ids=None)` | 保存一轮 `MemoryItem`；同一 `event_id` 重复保存为更新。 |
| `update_summary(user_id, session_id, summary, through_seq)` | 更新会话摘要和已摘要进度。 |
| `trim_window(messages, max_messages=None, keep_recent_turns=None)` | 裁剪消息，返回 `(保留, 被裁剪)`。 |
| `select_memory_writes(removed_messages)` | 从被裁剪消息中筛出可写长期记忆的消息。 |

### 10.2 LongTermMemory（动态长期记忆）

| 接口 | 用途 |
| :--- | :--- |
| `retrieve(user_id, query, top_k=5)` | 按当前输入检索相关记忆（语义 + 情节，相关度降序）。 |
| `retrieve_context(user_id, query, top_k=5)` | 检索 + 无条件召回稳定偏好 + 去重 + 格式化，直接注入 system prompt。 |
| `write_candidates(messages, user_id, love_snapshot=None)` | 从消息提取语义/程序性/情节候选并去重持久化，返回记忆 ID 列表。 |
| `add(user_id, memory)` / `update(user_id, memory_id, updates)` | 手动增改一条记忆（维护/调试）。 |
| `delete(user_id, memory_id)` | 软删除（status → DELETED，数据保留）。 |
| `list(user_id, memory_type=None, include_deleted=False)` | 列出用户记忆，可按类型过滤。 |

写入规则（`MemoryClassifier`）：语义记忆按 `(scope, content)` 去重；
程序性候选直接追加（默认需用户确认）；情节记忆仅在消息构成完整
“问题-行动-结果”时生成。

### 10.3 UserProfileManager（用户画像）

| 接口 | 用途 | 调用时机 |
| :--- | :--- | :--- |
| `get_profile(user_id)` | 读取画像（首次自动创建空画像）。 | 按需 |
| `get_profile_context(user_id)` | 非空字段格式化为 Prompt 上下文。 | 每轮（`load_runtime_context` 节点） |
| `process_user_message(user_id, message, session_id=None, message_id=None)` | Ollama 抽取候选并审核（applied/pending/rejected）；**失败抛 `ProfileCandidateExtractionError`，调用方必须捕获降级**。 | 每轮回复后（节点待接入，见 2.4） |
| `get_pending_candidates(user_id)` | 读取待确认候选，供 UI 展示。 | UI 拉取 |
| `confirm_candidate(user_id, candidate_id)` | 用户确认后以 USER 权限应用。 | UI 触发 |
| `reject_candidate(user_id, candidate_id, reason=...)` | 拒绝候选，画像不变。 | UI 触发 |
| `update_user_field(user_id, field, value, reason)` | USER 权限直接改字段（设置界面用；MODEL 权限不可用）。 | UI 触发 |
| `update_system_field(user_id, field, value, reason)` | SYSTEM 权限直接改字段（如 `onboarding_completed`）。 | 应用内部 |
| `list_change_logs(user_id)` | 画像变更审计日志。 | 调试/展示 |

候选自动应用的严格条件：用户明确表达（`explicit_quote` 真实存在于原文）、
置信度 ≥ `auto_apply_confidence`（默认 0.85）、字段允许 USER 修改、
候选未要求确认。否则置为 PENDING 等待用户确认，模型猜测一律不能直接生效。

> 兼容说明：`LongMemoryService` 仍作为 `UserProfileManager` 的别名导出
> （`agent/memory/profile_service.py` 底部），仅为平滑迁移保留，新代码请统一使用 `UserProfileManager`。

---

## 📌 下一次开发前检查清单

1. 本次新数据属于原始会话、短期摘要、动态长期记忆还是用户画像？
2. 是否带有 `user_id`、`session_id`、消息来源和过期策略？
3. 模型输出只是候选，还是已经有明确用户/系统授权？
4. 本轮检索和写入失败时，聊天是否仍可继续？
5. 是否会影响多 Session 隔离、系统提示词边界或用户隐私？
6. 是否至少新增一个“写入 → 重启/新 Session → 召回/验证”的测试？

> ✅ 每次先回答这六个问题，再开始改代码，可以让短期记忆、动态长期记忆和用户画像按同一套边界逐步演进。
