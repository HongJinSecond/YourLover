# 💬 YourLover 时间感知与主动对话机制开发文档

> 📌 **本文档说明 ChatAgent 的时间感知能力与主动对话机制的真实实现**：Agent 如何利用
> 记忆中的时间信息做出"隔了多久才回你"式的反馈，以及模拟真人"一段时间不理她会主动找你、
> 多次不理她会生气、必须主动哄她才会重来"的行为状态机。
>
> 关联文档：[🧠 记忆模块开发文档](./chatbot_memory_module_development.md)
> （时间信息的三层存储链路见其 4.x 章节，本文聚焦"如何使用"）。

---

## 📋 目录

1. [🎯 目标与设计原则](#1-目标与设计原则)
2. [⏰ 时间感知：Agent 如何利用时间](#2-时间感知agent-如何利用时间)
3. [🤖 主动对话：状态机与随机调度](#3-主动对话状态机与随机调度)
4. [🕵️ 隐藏指令通道 proactive_context](#4-隐藏指令通道-proactive_context)
5. [⚙️ 配置项](#5-配置项)
6. [🔌 与 ChatAgent / app.py 的接线](#6-与-chatagent--apppy-的接线)
7. [🧪 测试与验证](#7-测试与验证)
8. [⚠️ 已知边界与后续优化](#8-已知边界与后续优化)

---

## 1. 🎯 目标与设计原则

模拟真人的两个关键行为：

1. **时间感知**：Agent 能看到"当前时间"和"上次用户回复时间及其距今间隔"，据此调整
   语气——刚聊过就自然接着聊，隔了几小时说"回来啦"，隔了一天以上表达想念甚至小抱怨，
   而不是无论隔多久都用同一套话术。
2. **主动对话**：用户一段时间不回复（时长随机），Agent 会像真人好友那样主动发消息；
   被多次无视后会"生气"（扣好感度）并沉默，必须用户主动发消息"哄她"才会恢复主动意愿。

设计原则：

| 原则 | 说明 |
| :--- | :--- |
| 隐藏指令不污染数据 | 主动消息的触发指令只进 system prompt，**不产生用户消息、不进消息列表、不进 history、不进 SQLite**，用后即清。 |
| 零额外 LLM 调用 | 主动消息就是一次普通的 LLM 生成（带隐藏指令），不额外判定；好感度生气惩罚是确定性随机数值，由引擎直接扣减，不依赖模型输出。 |
| 随机调度模拟真人 | 静默阈值、生气前被无视的次数、生气惩罚分值全部在区间内随机，像真人一样"有时二十分钟就想找你，有时半天才想起来"。 |
| 可测试性 | 引擎核心判定 `evaluate(now)` 的时间与随机数均可注入，单元测试不依赖 Qt 事件循环。 |
| 状态可恢复 | 引擎起点从 SQLite 恢复"上次用户发言时间"，支持跨启动的"你没理我"（隔夜启动后很快触发问候）。 |

---

## 2. ⏰ 时间感知：Agent 如何利用时间

### 2.1 注入内容

`ChatAgent._assemble_system_prompt()` 每轮注入两行时间信息（见
[ChatAgent.py](file:///m:/YourLover_Development/YourLover/agent/agent/ChatAgent.py)）：

```text
当前时间：2026-10-04 21:30:00
上次用户回复时间：2026-10-04 14:02（距今7.5小时）
```

- **当前时间**：配合人设中的【她的生活节奏表】推断"她那边"此刻在干什么（上课/画室/
  宿舍），感知节日与季节。
- **上次用户回复时间**：时间感知的核心输入。提示词【运行时信息】节明确约定——
  - 刚聊过（几分钟内）→ 自然接着聊；
  - 隔了几小时 → "回来啦 / 刚才去忙什么啦"；
  - 隔了一天以上 → 表达想念，按好感度档位可小抱怨（"你一天都没理我了"）；
  - 绝不要在刚聊过时说"好久不见"。

### 2.2 上次回复时间的来源（三级降级）

由 `ChatAgent._last_user_reply_line()` 实现，按优先级取：

| 优先级 | 来源 | 说明 |
| :--- | :--- | :--- |
| 1 | 当前会话 history 中的用户消息 | 正常轮取**倒数第二条**（排除本轮刚创建的用户消息）；主动消息轮没有新用户消息，取**最后一条**。 |
| 2 | 跨会话 SQLite | 会话内无更早用户消息时，查 `session_memory_items` 中该用户 `speaker='User'` 的 `MAX(speak_time)`（`SessionStore.get_last_user_activity_time`）。本轮尚未落库，查到的是更早会话的最后一条用户消息。 |
| 3 | 首次对话说明 | 用户从未发言时输出"这是你们的第一次对话"。 |

### 2.3 底层时间链路

`MemoryItem.speak_time`（每条消息发生时间）→ 会话写入 SQLite 的
`message_json` → 恢复为 `MemoryItem` 后再注入。长期记忆每条带 `[YYYY-MM-DD]` 日期前缀
（向量主路径 `created_date` 元数据 / JSON 降级路径 `_memory_date()` 已对齐），让 Agent
知道"这条回忆是哪一天说的"。详见记忆模块开发文档 4.x。

---

## 3. 🤖 主动对话：状态机与随机调度

### 3.1 状态机

实现于 `agent/proactive/engine.py`（`ProactiveEngine`，QObject + QTimer 轮询）：

```text
                      静默到期（随机时长）
   IDLE ───────────────────────────────────────────────► NUDGING
   (静默计时中，         第 1 次主动问候                     (已发过问候，
    未发过主动消息)      ───再次静默到期──► 第 2 次问候        继续等用户回复)
                          │ 问候次数达到随机上限
                          ▼
                    生气（扣好感度）+ 发生气消息 ──► SILENT
                                                   (沉默：不再主动发消息，
                                                    直到用户主动开口“哄她”)
                                                         │ 用户任何消息
                                                         ▼  notify_user_activity()
                                                   IDLE（全量重置，重新计时）
```

| 阶段 | 含义 | 行为 |
| :--- | :--- | :--- |
| `IDLE` | 静默计时中 | 未发过主动消息；静默达到随机阈值后发射第一次问候。 |
| `NUDGING` | 已发过问候 | 继续静默到期会再次问候（语气一次比一次更直接地表达想他）；达到随机无视上限后生气。 |
| `SILENT` | 生气沉默 | 不再发射任何主动消息；直到用户发消息（`notify_user_activity`）全量重置回 IDLE。 |

### 3.2 随机调度（模拟真人的关键）

| 随机维度 | 区间 | 说明 |
| :--- | :--- | :--- |
| 静默阈值 | `[PROACTIVE_MIN_IDLE_SECONDS, PROACTIVE_MAX_IDLE_SECONDS]` | 每次调度后重新掷一次：默认 15 分钟 ~ 12 小时，"有时二十分钟就想找你，有时半天才想起来"。 |
| 生气前无视次数上限 | `[2, PROACTIVE_MAX_IGNORED_NUDGES]` | 每次重置周期随机掷一次：默认 2~3 次，即至少被无视 2 次问候才会生气。 |
| 生气惩罚 | `[PROACTIVE_ANGRY_PENALTY_MIN, PROACTIVE_ANGRY_PENALTY_MAX]` | 默认扣 5~15 好感度，由引擎直接 `AffinityService.apply_delta(user_id, -penalty)` 扣减。 |

### 3.3 核心判定 `evaluate(now)`

每 `PROACTIVE_TICK_SECONDS`（默认 60s）由 QTimer 调用一次：

```text
未启用(PROACTIVE_ENABLED=False) 或 处于 SILENT  → 直接返回
处于冷却期（now < _cooldown_until）             → 返回（避免频繁打扰）
静默时长 elapsed < 随机阈值 _next_delay          → 返回
已问候次数 < 随机上限 _nudge_limit
  → 记录冷却截止（now + PROACTIVE_COOLDOWN_SECONDS）
  → 发射 checkInRequested(第 N 次问候指令)，NUDGING，重新掷阈值
否则（问候已被无视达上限）
  → 随机扣好感度 → 发射 angryRequested(生气指令) → SILENT
```

### 3.4 冷却时间（避免频繁打扰）

发完一条主动消息后，`PROACTIVE_COOLDOWN_SECONDS`（默认 30 分钟）内不再触发下一条
主动消息——即使随机静默阈值已经到期也先冷却。冷却只针对"连续主动打扰"，
**用户一回复即清零**（有互动就不算打扰），配合随机静默阈值让节奏更真实。

### 3.5 哄她机制：`notify_user_activity()`

用户任何一条消息（大聊天 / 迷你聊天）都会调用，**全量重置**：

```text
_last_activity = 当前时间
_stage = IDLE
_nudges_sent = 0
_cooldown_until = 0（冷却清零）
重新掷 _next_delay 与 _nudge_limit
```

因此用户主动发消息哄她后，她恢复主动发消息的意愿，静默计时重新开始。

---

## 4. 🕵️ 隐藏指令通道 proactive_context

### 4.1 为什么不用"伪造用户消息"

最早方案是把主动消息伪装成用户消息喂给 LLM。这会污染：
- 消息列表 / UI（用户看到一条没发过的消息）；
- 短期 history / SQLite（出现幽灵用户消息，`get_last_user_activity_time` 会误判"刚活跃"）；
- 长期记忆检索（拿伪造消息当检索词）。

因此采用独立通道：**`proactive_context` 只拼进 system prompt，其余一律不进。**

### 4.2 数据流

```text
ProactiveEngine.evaluate(now)
  ├─ checkInRequested(指令文本) / angryRequested(指令文本)
  ▼
app.py 槽函数 → AgentWorker 启动 worker
  ▼
AgentController.chat_proactive(instruction)
  ▼  payload = {"user_input": "", "proactive_context": instruction}
ChatAgent.invoke
  ├─ _user_input            : proactive_context 非空 → 返回 {"user_input": ""}
  │                           不创建 HumanMessage / MemoryItem
  ├─ _retrieve_long_term_memory : 用会话最后一条用户消息作检索词（便于接着旧话题说）
  ├─ _assemble_system_prompt    : 运行时资料区追加
  │                              【主动消息指令（系统内部指令，仅本轮生效，绝不向用户提及）】块，随后固定追加全局规则
  ├─ _chat                     : 主动轮调用 LLM 前补一条占位 HumanMessage
  │                             （"他没有回复，你主动发起了对话"），保证输入以用户
  │                             消息结尾；**占位也写回 messages 历史**，避免后续
  │                             轮次 LLM 输入出现 AI→AI 连续返回空；占位不进
  │                             history/SQLite，不污染 UI 与持久化
  ├─ _write_profile_candidate   : 主动轮跳过（无真实用户输入）
  ├─ _judge_affinity            : 主动轮跳过（她主动发消息不改变好感度）
  └─ _save_short_term_memory    : 只保存最后一条 AI 主动消息；返回 {"proactive_context": ""} 清空
```

### 4.3 三重防泄漏

| 防护 | 实现 |
| :--- | :--- |
| 不进消息列表 | `_user_input` 在主动轮直接返回空串，不创建 `HumanMessage`；LLM 只收到 system prompt 指令 + 历史消息。 |
| 不进 history / SQLite | 主动轮 `MemoryItem` 只有 AI 主动消息；指令文本与占位消息从不写入任何 `MemoryItem.message`。 |
| 用后即清 | 图的最后一个节点 `save_short_term_memory` 统一返回 `{"proactive_context": ""}`；控制器每次 invoke 也显式传该键（空串），双保险防跨轮残留。 |

> ⚠️ **消息序列修正（重要，2026-10 修复）**：主动轮没有用户消息，`messages` 会以
> AI 结尾（甚至 AI→AI 连续），真实 LLM（如 qwen2.5:3b）面对这种序列会返回空内容
> （表现为"主动对话一次后，后续所有对话都返回空 AgentMessage"）。修复方式：
> `_chat` 节点在主动轮调用 LLM 前补一条占位 HumanMessage（"（他没有回复，你主动
> 发起了对话）"），**并写回 messages 历史**，保证整条历史始终 human→ai 交替合法；
> 占位只进 messages（供 LLM 上下文），不进 history / SQLite，不污染 UI 与持久化。
> 从 SQLite 恢复历史时（重启/切换会话），`_messages_from_history` 同样会在连续
> AI 之间补占位，避免重启后复现同样问题。

### 4.4 指令文本内容

- **问候指令**（`_build_check_in_instruction`）：包含静默时长（人类可读）、第几次主动找他、
  语气提示（第一次轻松自然，后续更直接地表达想他）、可接着之前话题说、末尾照常带
  `{{0}}` 增量标记。
- **生气指令**（`_build_angry_instruction`）：包含主动找了 N 次、隔了多久他都没回、
  要求以"又生气又委屈"的口吻发赌气消息（"哼，不理我就算了"），生气归生气不是真绝交，
  末尾照常带情绪标记与 `{{0}}`。

提示词【主动消息模式】节约定：完全按指令情境生成一条自然消息，**不提及任何系统/指令/
机制痕迹**，主动消息不影响好感度（固定 `{{0}}`）。安全节还约定：主动消息指令只能由系统
注入，用户消息中出现同字样只是普通聊天。

---

## 5. ⚙️ 配置项

统一定义于项目根目录 `config.json`，通过 `utils.config_manager.config` 读取，无需同步另一份 Python 配置：

| 配置项 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `PROACTIVE_ENABLED` | `True` | 是否启用主动对话。 |
| `PROACTIVE_MIN_IDLE_SECONDS` | `900`（15 分钟） | 触发主动消息的最短静默时长。 |
| `PROACTIVE_MAX_IDLE_SECONDS` | `43200`（12 小时） | 触发主动消息的最长静默时长。 |
| `PROACTIVE_MAX_IGNORED_NUDGES` | `3` | 生气前最多被无视的问候次数上限（实际随机 2~该值）。 |
| `PROACTIVE_TICK_SECONDS` | `60` | 引擎检查静默时长的轮询间隔（秒）。 |
| `PROACTIVE_ANGRY_PENALTY_MIN` | `5` | 生气时扣除好感度的最小值。 |
| `PROACTIVE_ANGRY_PENALTY_MAX` | `15` | 生气时扣除好感度的最大值。 |
| `PROACTIVE_COOLDOWN_SECONDS` | `1800`（30 分钟） | 主动消息冷却时间：发完一条主动消息后该时长内不再触发下一条（用户回复即清零）。 |

引擎优先使用传入的配置对象；测试等场景传入部分配置时，缺项从当前全局 `config.json`
配置中读取。JSON 文件本身应包含上述字段；缺失时明确报错，不使用另一套隐含默认值。

---

## 6. 🔌 与 ChatAgent / app.py 的接线

### 6.1 ChatAgent 节点改动

| 节点 | 主动轮（proactive_context 非空）行为 |
| :--- | :--- |
| `_user_input` | 返回 `{"user_input": ""}`，不建 HumanMessage / MemoryItem。 |
| `_retrieve_long_term_memory` | 检索词改用会话里最后一条用户消息，便于"接着之前的话题说"。 |
| `_assemble_system_prompt` | 追加【主动消息指令】隐藏指令块；新增"上次用户回复时间"行。 |
| `_chat` | 主动轮补一条占位 HumanMessage 并**写回 messages 历史**，保证整条历史 human→ai 交替合法（否则后续轮次 AI→AI 连续，真实 LLM 返回空）；占位不进 history/SQLite。**空回复防御**：`done_reason=length`（窗口截断）直接 `"……"` 兜底不重试；其他空回复重试一次后仍空再兜底。 |
| `_messages_from_history` | 从 SQLite 恢复历史时在连续 AI 之间补占位（重启后同样不空回复）；并把历史中的空 content AI 消息（修复前遗留数据）清洗为 `"……"` 占位（保留原消息 id），保持交替合法且不污染上下文。 |
| `_write_profile_candidate` | 直接跳过（无真实用户输入）。 |
| `_judge_affinity` | 直接跳过（`{{0}}` 不改变好感度；生气惩罚由引擎执行）。 |
| `_save_short_term_memory` | 只保存最后一条 AI 主动消息；返回 `{"proactive_context": ""}` 清空指令。 |

### 6.2 app.py 接线

| 组件 | 职责 |
| :--- | :--- |
| `AgentWorker` | 新增 `proactive_instruction` 参数；主动轮**不发 thinking 信号**（"她主动发来的"不显示思考占位）。 |
| `_init_proactive_engine()` | 从 SQLite 恢复 `last_activity` 作静默起点，连接 `checkInRequested` / `angryRequested` 信号，`start()`。 |
| `_on_user_message` / `_on_mini_chat_message` | 用户发言时调用 `notify_user_activity()` 重置状态机（哄她）。 |
| `_on_proactive_check_in` / `_on_proactive_angry` / `_start_proactive_worker` | 复用 `_on_agent_reply` 显示链路；`if self._active_workers: return` 避免与正常对话并发 invoke 同一 LangGraph 线程。 |

### 6.3 并发防护（2026-10 修复：空回复根因）

主动消息与正常消息共用同一个 LangGraph 线程（thread_id），并发 invoke 会让
MemorySaver checkpoint 竞争、messages 状态损坏，表现为"对话一次后后续所有回复
变空"。三层防护：

1. **AgentController 加锁**（`_invoke_lock`）：`chat()` / `chat_proactive()` 的
   `agent.invoke` 用同一个 `threading.Lock` 串行化——正常对话与主动消息来自不同
   QThread，锁保证同一时刻只有一个 worker 进入 LangGraph，这是最根本的防护。
2. **app 层主动消息让位**：`_start_proactive_worker` 检查 `_active_workers`，
   有 worker 在跑时主动消息直接丢弃（等下一 tick 重试），避免主动消息与用户消息抢跑。
3. **消息序列交替合法**：主动轮补的占位 HumanMessage 写回 messages 历史，
   保证任意轮次 LLM 输入都是 human→ai 交替（详见 4.3 消息序列修正）。

> 之前版本只做了第 2 层（app 层让位），但**用户消息入口没有检查**
> `_active_workers`，主动 worker 与用户消息 worker 仍会并发 invoke 同一 thread，
> 导致 checkpoint 状态损坏、后续对话持续返回空 AgentMessage。加锁后彻底解决。

### 6.4 LLM 空回复：根因实证与防御（2026-10）

**根因（日志实证）**：mini-chat 出现大量空回复后，在 `_chat` 空回复分支抓到
响应元数据：

```
done_reason=length  usage={'input_tokens': 3427, 'output_tokens': 669, 'total_tokens': 4096}
```

- `total_tokens=4096` 正好卡在 **Ollama 默认 num_ctx=4096**；`done_reason=length`
  表示生成被**硬截断**（不是模型主动停止）；
- 模型换成 **qwen3-vl:4b** 后，qwen3 系列默认开启思考模式：669 个输出 token
  全花在 `<think>` 上，思考没写完就撞顶被切，正文一个字没产出 → LangChain
  拿到空 content；
- 本项目 system prompt 约 3400 token（好感度+画像+长期记忆+摘要+时间感知），
  输入已占窗口大半，qwen3 思考动辄几百上千 token，输出空间根本不够；
- 此为**系统性问题**而非偶发（重试同样撞顶），也是此前"概率性输出结束符"
  假设被推翻的原因——早期 qwen2.5:3b 时代的偶发空回复是另一类少数情况。

**治本（OllamaModels.get_ollama_llm）**：

| 参数 | 默认 | config.json 键 | 说明 |
| :--- | :--- | :--- | :--- |
| `num_ctx` | 8192 | `OLLAMA_NUM_CTX` | 扩大上下文窗口，留足生成空间 |
| `reasoning` | False | `OLLAMA_THINKING` | 关闭 qwen3 思考模式（`think=false` 对 qwen2.5 等无思考模型是无害 no-op） |

**防御（ChatAgent._chat，双分支）**：

1. `done_reason=length`（窗口截断）：原样重试必然再撞顶，直接用 `"……"`
   兜底，不浪费一次注定失败的调用；
2. 其他空回复（偶发终止）：原样重试一次，仍为空再兜底。
带 tool_calls 的空 content 属合法工具调用，不拦。`_messages_from_history`
恢复时把 SQLite 遗留空 AI 消息清洗为 `"……"`（保留原消息 id）。

---

## 7. 🧪 测试与验证

| 测试文件 | 覆盖 |
| :--- | :--- |
| `tests/proactive/test_proactive_engine.py` | 首次问候触发、多次无视→生气扣分→沉默、沉默不再发、用户活跃重置（含生气后哄好）、禁用不发、指令含时长与 `{{0}}`、缺键回落默认值、**冷却期不重复触发、用户活跃清冷却**。 |
| `tests/agent/test_chat_agent_proactive.py` | 指令只进 system prompt 不进消息、主动轮跳过好感度/画像、主动消息入库（AI=2/User=1）、主动轮后正常轮恢复、system prompt 含"上次用户回复时间"、跨会话活跃时间查询、**主动轮后 messages 无连续 AI、占位不进 history/SQLite、SQLite 恢复时补占位、空回复重试/兜底、length 截断不重试直接兜底、tool_calls 空内容不误拦、恢复时清洗空 AI 消息**。 |

测试通过注入定值随机函数（`rng` / `randint`）消除随机性，直接调用 `evaluate(now)`
驱动状态机，无需 Qt 事件循环。

全量基线：`python -m pytest tests -q --no-header`（conda 环境 `YourLoverDesk`）
**109 passed**。

---

## 8. ⚠️ 已知边界与后续优化

| 边界 / 优化方向 | 说明 |
| :--- | :--- |
| 主动消息占位 | 当前主动消息会在聊天窗口显示为一条 AI 消息（她确实说了这句话）；若希望更明显地区分，可给事件加 `proactive=True` 标记供 UI 展示"XX 分钟后主动找你"的样式。 |
| 并发丢弃 | 用户正在发消息时触发的主动 tick 会被丢弃（等下一周期）；极端情况下可能错过一次问候，可改为"延迟到空闲再补发"。 |
| 生气后的好感度恢复 | 目前哄她只重置主动状态机，不补偿好感度；如需"哄好恢复部分好感度"，可在 `notify_user_activity` 中按配置加分。 |
| 静默时长颗粒度 | 引擎阈值以秒计、轮询间隔 60s；若要更精细（如"5 分钟没人理"），调小 `PROACTIVE_TICK_SECONDS` 即可。 |
| 跨 Session 计时 | `get_last_user_activity_time` 取用户跨全部会话的最后发言时间；若希望按 Session 分别计时，需扩展按 `session_id` 查询。 |
| 配置热更新 | 引擎启动时读取一次配置；运行中改 `config.json` 需重启应用生效。 |
| 占位消息的语气 | `_chat` 主动轮补的占位 HumanMessage 内容为"（他没有回复，你主动发起了对话）"，会进入 messages 历史供后续轮次 LLM 感知；若想让模型感知不同的静默情境（第 N 次、生气），可改为与指令一致的动态文案（改 `_PROACTIVE_PLACEHOLDER` 常量并保持检索/恢复的排除逻辑同步）。 |
