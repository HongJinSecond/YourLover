"""基于 LangGraph 的情感陪伴聊天 Agent，集成 M7 长期记忆能力。

该模块定义 ChatAgent，通过 LangGraph 编排一个多节点工作流，在每一轮
对话中完成：

    加载短期记忆（仅切换/重启时）→ 加载运行时上下文（好感度+画像，每轮）
    → 检索长期记忆（每轮）→ 注入用户输入 → LLM/工具调用
    → 写入画像候选 → 合并提取并裁剪短期窗口（LLM 一次产出摘要+长期记忆）
    → 解析回复中的好感度增量标记并持久化 → 发射好感度事件 → 保存短期记忆

记忆依赖全部通过三个管理器注入（agent/memory 包的统一入口）：
    - ShortTermMemory    短期记忆：Session 存取、消息持久化、窗口裁剪
    - LongTermMemory     动态长期记忆：向量检索/写入（主）+ JSON/关键词（降级）
    - UserProfileManager 用户画像：每轮读取画像上下文注入 Prompt；
                         回复后抽取画像候选（Ollama 不可用时自动降级跳过）
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.affinity.service import AffinityService
from agent.memory.EmotionMemory import EmotionState
from agent.memory.llm_extractor import (
    LLMMemoryExtractor,
    MemoryExtractionError,
)
from agent.memory.long_term_memory import LongTermMemory, VectorMemoryStore
from agent.memory.MemoryItem import MemoryItem
from agent.memory.profile_service import (
    ProfileCandidateExtractionError,
    UserProfileManager,
    matches_profile_trigger,
)
from agent.memory.short_term_memory import ShortTermMemory
from agent.memory.window import Message
from agent.prompts import compose_system_prompt, get_persona_prompt
from agent.tools import tool_map
from utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


def _humanize_gap(seconds: float) -> str:
    """把秒数间隔转成人类可读的中文时长描述（用于时间感知提示行）。"""
    if seconds < 60:
        return "刚刚"
    if seconds < 3600:
        return f"{int(seconds // 60)} 分钟"
    if seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f} 小时"
    return f"{seconds / 86400:.1f} 天"


# 主动轮补的占位用户消息内容：只进 messages（保证 LLM 消息交替合法），
# 不进 history/SQLite，不展示给用户；检索与恢复时据此识别并跳过。
_PROACTIVE_PLACEHOLDER = "（他没有回复，你主动发起了对话）"

# LLM 空回复的兜底文案。实证两类成因（2026-10 日志取证）：
#   ① 上下文窗口截断（大头）：done_reason=length，qwen3 系列默认思考模式
#      消耗几百 token 的 <think>，撞上 num_ctx 上限（Ollama 默认 4096，
#      本项目 system prompt 就占约 3400）被硬切，正文一个字没产出；
#   ② 小模型偶发立即终止（少数）。
# 空消息一旦写回 messages/SQLite，会随上下文反复进入 LLM 输入，污染上下文
# 并诱发复读；窗口参数与思考开关在 OllamaModels.get_ollama_llm 修治本。
_EMPTY_REPLY_FALLBACK = "……"


def _is_blank_ai_reply(message: BaseMessage) -> bool:
    """判断是否为无效空回复：文本内容为空白且未发起工具调用。

    注意：带 tool_calls 的空 content 是合法的工具调用响应，不算空回复。
    """
    if getattr(message, "tool_calls", None):
        return False
    return not str(getattr(message, "content", "") or "").strip()


class ChatAgent:
    """情感陪伴 Agent：由 memory 模块三大管理器组合而成的 LangGraph 代理。

    核心职责：
        1. 组装短期记忆/长期记忆/用户画像/好感度服务等依赖；
        2. 构建 LangGraph 工作流并暴露 invoke 入口；
        3. 在每轮对话中完成长期记忆的检索、写入与短期窗口裁剪。

    依赖注入：short_term_memory / long_term_memory / user_profile_manager /
    affinity_service / memory_extractor 均可由调用方传入，便于测试与替换。
    短期窗口参数（max_messages / keep_recent_turns）由 ShortTermMemory 持有。
    """

    def __init__(
        self,
        llm: Any,
        config: ConfigManager,
        session: dict[str, Any],
        *,
        short_term_memory: ShortTermMemory | None = None,
        long_term_memory: LongTermMemory | None = None,
        user_profile_manager: UserProfileManager | None = None,
        affinity_service: AffinityService | None = None,
        memory_extractor: LLMMemoryExtractor | None = None,
    ) -> None:
        """初始化 Agent 并构建工作流。

        参数：
            llm: LangChain LLM 实例（需支持 bind_tools 与 invoke）
            config: 全局配置（含 system prompt 等）
            session: 运行时会话配置（含 configurable 等）
            short_term_memory: 短期记忆管理器，默认 SQLite 存储 +
                窗口参数 max_messages=20 / keep_recent_turns=6
            long_term_memory: 动态长期记忆管理器，默认向量存储主路径
                + JSON 降级；未注入时自动尝试构建 Chroma 向量存储
            user_profile_manager: 用户画像管理器，默认 SQLite 画像库；
                候选抽取用 Ollama qwen2.5:3b（仅 process_user_message 时才联网）
            affinity_service: 好感度服务，默认 SQLite 存储（初始 0）
            memory_extractor: LLM 合并提取器（摘要+长期记忆），默认基于
                self.llm 构建 LLMMemoryExtractor
        """
        self.config = config
        self.session = session
        self.session_id = None
        self.llm = llm
        self.tools = tool_map
        self.persona_prompt = get_persona_prompt(config)
        # 三大记忆管理器：窗口参数（max_messages/keep_recent_turns）由 ShortTermMemory 持有
        self.short_term_memory = short_term_memory or ShortTermMemory(
            max_messages=config.SHORT_TERM_MEMORY_MAX_MESSAGES,
            keep_recent_turns=config.SHORT_TERM_MEMORY_KEEP_RECENT_TURNS,
        )
        self.long_term_memory = long_term_memory or self._default_long_term_memory()
        self.user_profile_manager = user_profile_manager or UserProfileManager()
        self.affinity_service = affinity_service or AffinityService()
        self.memory_extractor = memory_extractor or LLMMemoryExtractor(self.llm)
        self.max_messages = self.short_term_memory.max_messages  # 兼容属性：短期窗口上限
        self.keep_recent_turns = self.short_term_memory.keep_recent_turns  # 兼容属性：保留轮次数
        # 若 LLM 支持 bind_tools 则绑定工具，否则直接使用原 LLM
        self.llm_with_tools = (
            self.llm.bind_tools(list(self.tools.values()))
            if hasattr(self.llm, "bind_tools")
            else self.llm
        )
        self.__build_workflow()

    def _default_long_term_memory(self) -> LongTermMemory:
        """构造默认长期记忆管理器：尝试启用 Chroma 向量主路径，失败降级 JSON。"""
        try:
            vector_store = VectorMemoryStore()
            return LongTermMemory(vector_store=vector_store)
        except Exception as exc:  # Chroma 依赖/embedding 模型不可用时不阻塞启动
            logger.warning("长期记忆向量存储不可用，降级为 JSON+关键词：%s", exc)
            return LongTermMemory()

    def invoke(
        self, user_input: dict[str, Any], session: dict[str, Any] | None = None
    ) -> EmotionState:
        """执行一轮对话。

        参数：
            user_input: 用户输入负载，可含 user_id/session_id 与文本内容
            session: 可选的运行时会话配置，覆盖初始化时的 session

        user_id 标识长期记忆归属；session_id（或 LangGraph 的 thread_id）
        标识短期会话状态。两者缺省时分别回退到 "default-user"/"default-session"。
        """
        # Snapshot once per turn: settings take effect on the next invocation,
        # never halfway through a tool-calling loop already in progress.
        self.persona_prompt = get_persona_prompt(self.config)
        runtime_session = session or self.session
        configurable = dict(runtime_session.get("configurable") or {})
        payload = dict(user_input)
        session_id = str(
            payload.get("session_id") or configurable.get("thread_id") or "default-session"
        )
        user_id = str(payload.get("user_id") or configurable.get("user_id") or "default-user")
        configurable["thread_id"] = session_id
        runtime_session = {**runtime_session, "configurable": configurable}
        payload.update(user_id=user_id, session_id=session_id)
        return self.workflow.invoke(payload, config=runtime_session)

    def save_graph_figure(self, filename: str) -> None:
        """将工作流图导出为 PNG 图片，便于可视化调试。"""
        graph_png = self.workflow.get_graph().draw_mermaid_png()
        with open(filename, "wb") as file:
            file.write(graph_png)

    def __build_workflow(self) -> None:
        """构建 LangGraph 工作流。

        节点流转（短期恢复按 Session 发生，运行时上下文按轮刷新）：
            START --(条件)-->
                "load"（新 Session/切换/重启）: load_short_term_memory
                        -> load_runtime_context -> retrieve_long_term_memory -> user_input
                "skip"（同 Session 后续轮次）: load_runtime_context
                        -> retrieve_long_term_memory -> user_input
            公共链路: user_input -> llm
                      -> (条件) tools 或 write_profile_candidate
            tools -> llm（工具执行后回到 LLM 继续生成）
            write_profile_candidate -> trim_short_term_window
                 -> judge_affinity（解析回复中的 {{+N}} 好感度标记）
                 -> emit_affinity_event
                 -> save_short_term_memory -> END

        同一 session_id 的后续轮次由 MemorySaver checkpoint 恢复短期消息
        （跳过 SQLite 重复加载），但好感度快照、用户画像和动态记忆检索
        每轮都会刷新——这是长期记忆闭环的关键约定。
        """
        graph = StateGraph(EmotionState)
        graph.add_node("load_short_term_memory", self._load_short_term_memory)
        graph.add_node("load_runtime_context", self._load_runtime_context)
        graph.add_node("retrieve_long_term_memory", self._retrieve_long_term_memory)
        graph.add_node("user_input", self._user_input)
        graph.add_node("llm", self._chat)
        graph.add_node("tools", self._tool_call)
        graph.add_node("write_profile_candidate", self._write_profile_candidate)
        graph.add_node("trim_short_term_window", self._trim_short_term_window)
        graph.add_node("judge_affinity", self._judge_affinity)
        graph.add_node("emit_affinity_event", self._emit_affinity_event)
        graph.add_node("save_short_term_memory", self._save_short_term_memory)

        graph.add_conditional_edges(
            START,
            self._route_on_session,
            {"load": "load_short_term_memory", "skip": "load_runtime_context"},
        )
        graph.add_edge("load_short_term_memory", "load_runtime_context")
        graph.add_edge("load_runtime_context", "retrieve_long_term_memory")
        graph.add_edge("retrieve_long_term_memory", "user_input")
        graph.add_edge("user_input", "llm")
        graph.add_conditional_edges(
            "llm", self._should_call_tool, {"tools": "tools", "write": "write_profile_candidate"}
        )
        graph.add_edge("tools", "llm")
        graph.add_edge("write_profile_candidate", "trim_short_term_window")
        graph.add_edge("trim_short_term_window", "judge_affinity")
        graph.add_edge("judge_affinity", "emit_affinity_event")
        graph.add_edge("emit_affinity_event", "save_short_term_memory")
        graph.add_edge("save_short_term_memory", END)
        self.workflow = graph.compile(checkpointer=MemorySaver())

    def _route_on_session(self, state: EmotionState) -> Literal["load", "skip"]:
        """START 条件路由：判断本轮是否需要从 SQLite 恢复短期记忆。

        判断依据：self.session_id 是否等于当前 session_id。
            - 相等：短期消息已在本进程加载过，由 MemorySaver checkpoint
              维护后续轮次，走 "skip"（但仍每轮刷新运行时上下文和记忆检索）。
            - 不等（含首次进入 self.session_id 为 None、session_id 切换）：
              走 "load"，从 SQLite 恢复摘要和最近消息。
        """
        if self.session_id == state.get("session_id"):
            return "skip"
        return "load"

    def _should_call_tool(self, state: EmotionState) -> Literal["tools", "write"]:
        """条件路由：LLM 最后一条消息含 tool_calls 则进入工具节点，否则进入记忆写入。"""
        messages = state.get("messages") or []
        if messages and getattr(messages[-1], "tool_calls", None):
            return "tools"
        return "write"

    def _load_short_term_memory(self, state: EmotionState) -> dict[str, Any]:
        """加载短期记忆：恢复 Session 摘要和最近消息（仅切换/重启时执行）。

        逻辑：若 LangGraph 已恢复当前线程状态则不重复加载；否则从 SQLite
        读取最近消息，恢复为 messages 和 MemoryItem history。
        """
        user_id = state["user_id"]
        session_id = state["session_id"]
        self.session_id = session_id

        # 切回进程内已有的 LangGraph 线程时，checkpoint 已经包含短期状态。
        if state.get("messages") or state.get("session_summary"):
            return {}

        snapshot = self.short_term_memory.load_session(user_id, session_id)
        if snapshot is None:
            self.short_term_memory.create_session(user_id, session_id)
            return {}

        history = [record.item for record in snapshot.items]
        return {
            "messages": self._messages_from_history(history),
            "history": history,
            "session_summary": snapshot.session.summary,
        }

    @staticmethod
    def _messages_from_history(history: Sequence[MemoryItem]) -> list[BaseMessage]:
        """把 MemoryItem 历史还原为 messages，并做两处清洗。

        1. 清洗无效空回复：修复前的 SQLite 遗留数据中可能存有空 content
           的 AI 消息，恢复时替换为省略号占位（保留原消息 id），既保持
           human→ai 交替合法，又避免空内容污染 LLM 上下文。
        2. 补主动轮占位：SQLite 里主动轮的 AI 消息按原样保存（占位不进
           库），恢复后可能出现 AI→AI 连续（如 [H1, A1, A_proactive]），
           在连续 AI 之间插入 _PROACTIVE_PLACEHOLDER 占位。
        两类占位都只进 messages，不写回 history。
        """
        restored: list[BaseMessage] = []
        for item in history:
            message = item.message
            if isinstance(message, AIMessage) and _is_blank_ai_reply(message):
                message = AIMessage(content=_EMPTY_REPLY_FALLBACK, id=message.id)
            if isinstance(message, AIMessage) and restored and isinstance(restored[-1], AIMessage):
                restored.append(HumanMessage(content=_PROACTIVE_PLACEHOLDER))
            restored.append(message)
        return restored

    def _load_runtime_context(self, state: EmotionState) -> dict[str, Any]:
        """加载运行时上下文（每轮执行）：好感度快照 + 用户画像上下文。

        画像上下文来自用户明确提供或确认的字段（称呼、语言等），
        注入 system prompt 后模型可以据此称呼用户。
        同时重置本轮的写入结果字段（memory_write_ids / profile_candidate_ids），
        避免上一轮的结果残留到本轮。
        """
        user_id = state["user_id"]
        return {
            "love_snapshot": self.affinity_service.get_love_snapshot(user_id),
            "profile_context": self.user_profile_manager.get_profile_context(user_id),
            "memory_context": "",
            "memory_write_ids": [],
            "profile_candidate_ids": [],
        }

    def _retrieve_long_term_memory(self, state: EmotionState) -> dict[str, str]:
        """检索长期记忆并格式化为上下文字符串（每轮执行）。

        委托给 LongTermMemory.retrieve_context：
            1. 按当前 user_input 检索 top_k 相关记忆；
            2. 额外召回用户稳定偏好（scope=preference），保证
               "回答要简短"这类偏好即使与本轮主题无关也能影响回复；
            3. 合并去重后格式化为 memory_context 注入 system prompt。

        主动消息轮没有用户输入：改用会话里最后一条**真实用户消息**作为
        检索词（便于"接着之前的话题说"），占位消息（_PROACTIVE_PLACEHOLDER）
        不参与检索；没有历史用户消息时用空串（退化为偏好召回）。
        """
        user_id = state["user_id"]
        query = state.get("user_input", "")
        if state.get("proactive_context"):
            messages = state.get("messages") or []
            last_human = next(
                (
                    message.content
                    for message in reversed(messages)
                    if isinstance(message, HumanMessage)
                    and str(getattr(message, "content", "") or "").strip()
                    and str(message.content).strip() != _PROACTIVE_PLACEHOLDER
                ),
                None,
            )
            query = str(last_human or "")
        return {"memory_context": self.long_term_memory.retrieve_context(user_id, query)}

    def _save_short_term_memory(self, state: EmotionState) -> dict[str, Any]:
        """保存短期记忆：持久化本轮新增事件并同步 Session 摘要。

        正常轮：从 history 中找到本轮最后一条用户输入，只保存它及其后的
        AI/工具事件；消息 ID 缺失时用时间戳生成稳定事件 ID，避免重复写入。

        主动消息轮：history 里没有本轮用户消息，直接保存最后一条 AI 事件
        （即主动消息本体）——让她记得自己主动说过什么，历史恢复时也能看到。
        """
        user_id = state["user_id"]
        session_id = state["session_id"]
        history = list(state.get("history") or [])
        user_input = state.get("user_input", "")

        if state.get("proactive_context"):
            # 主动轮只保存最后一条 AI 事件（_chat 节点刚创建的主动消息）
            if history and str(getattr(history[-1], "speaker", "")).lower() == "ai":
                item = history[-1]
                event_id = (
                    str(getattr(item.message, "id", None))
                    or f"{session_id}:{item.speak_time!r}:proactive"
                )
                self.short_term_memory.save_turn(user_id, session_id, [item], event_ids=[event_id])
            else:
                self.short_term_memory.create_session(user_id, session_id)
            # 主动轮结束统一清空隐藏指令（save 是图的最后一个节点）；
            # 控制器下次 invoke 也会显式传入空串，双保险防跨轮残留
            return {"proactive_context": ""}

        # 从后向前定位本轮用户事件，避免重复保存之前的 history。
        turn_start = next(
            (
                index
                for index in range(len(history) - 1, -1, -1)
                if history[index].speaker.lower() == "user"
                and getattr(history[index].message, "content", None) == user_input
            ),
            None,
        )
        turn_items = history[turn_start:] if turn_start is not None else []

        if turn_items:
            event_ids = [
                str(getattr(item.message, "id", None))
                if getattr(item.message, "id", None)
                else f"{session_id}:{item.speak_time!r}:{index}"
                for index, item in enumerate(turn_items)
            ]
            self.short_term_memory.save_turn(
                user_id,
                session_id,
                turn_items,
                event_ids=event_ids,
            )
        else:
            self.short_term_memory.create_session(user_id, session_id)

        session = self.short_term_memory.get_session(user_id, session_id)
        summary = state.get("session_summary") or ""
        if session is not None and summary != session.summary:
            records = self.short_term_memory.load_item_records(user_id, session_id)
            retained_count = len(state.get("messages") or [])
            removed_count = max(0, len(records) - retained_count)
            summarized_through = session.summarized_through_sequence
            if removed_count:
                summarized_through = max(
                    summarized_through,
                    records[removed_count - 1].sequence_no,
                )
            self.short_term_memory.update_summary(
                user_id,
                session_id,
                summary,
                summarized_through,
            )
        return {}

    def _user_input(self, state: EmotionState) -> dict[str, Any]:
        """把本轮用户输入包装为 HumanMessage，并写入 history 视图。

        主动消息轮（proactive_context 非空）没有真实用户输入：
        跳过 HumanMessage 与 MemoryItem 写入，保证隐藏指令和空输入
        不进入消息列表、history 与 SQLite。
        """
        if state.get("proactive_context"):
            return {"user_input": ""}  # 清空占位，防止上一轮残留
        user_message = HumanMessage(
            content=state["user_input"],
            additional_kwargs={"session_id": state["session_id"]},
        )
        return {
            "messages": [user_message],
            "history": [
                MemoryItem(
                    type="Chat",
                    speaker="User",
                    message=user_message,  # 必须存消息对象：SQLite 序列化与 .content 访问都依赖它
                    love=state["love_snapshot"],
                )
            ],
        }

    def _chat(self, state: EmotionState) -> dict[str, Any]:
        """调用 LLM 生成回复。

        组装 system prompt（含好感度、用户画像、长期记忆上下文、会话摘要）
        后连同当前消息列表调用 LLM。若 LLM 未发起工具调用，则把 AI 回复
        写入 history 视图。

        主动消息轮：system prompt 追加 proactive_context 中的隐藏指令。
        该字段不在本节点清空（下游 judge/save 节点仍需它识别主动轮），
        而是由图的最后一个节点 save_short_term_memory 统一清空；控制器
        每次 invoke 都显式传入该键，双保险防止指令跨轮残留。

        主动轮没有用户消息，直接以 AI 结尾调用 LLM 会得到空回复；并且
        若主动轮的 AI 消息进入历史，后续正常轮的 LLM 输入会变成
        [..., ai, ai, human] 的连续 AI 序列，真实 LLM 同样会返回空。
        因此主动轮先补一条占位 HumanMessage 再调用 LLM，并**把占位也
        写回 messages 历史**，保证整条历史始终 human→ai 交替合法。
        占位消息只进 messages（供 LLM 上下文），不进 history/SQLite，
        不会污染 UI 显示与持久化数据。
        """
        system_message = SystemMessage(content=self._assemble_system_prompt(state))
        messages = list(state.get("messages") or [])
        placeholder: HumanMessage | None = None
        if state.get("proactive_context"):
            placeholder = HumanMessage(content=_PROACTIVE_PLACEHOLDER)
            messages.append(placeholder)
        response = self.llm_with_tools.invoke([system_message, *messages])
        # 防御：空 content 且无 tool_calls 的回复不写回（空消息入库会反复
        # 进入 LLM 输入，污染上下文诱发复读）。分两类处理：
        #   - done_reason=length（窗口截断，qwen3 思考撞 num_ctx 顶）：原样
        #     重试必然再撞顶，直接用省略号兜底；
        #   - 其他空回复（偶发终止）：原样重试一次，仍为空再兜底。
        if _is_blank_ai_reply(response):
            metadata = getattr(response, "response_metadata", None) or {}
            if metadata.get("done_reason") == "length":
                logger.warning(
                    "LLM 回复被上下文窗口截断（done_reason=length），使用兜底文案"
                    "（建议检查 OLLAMA_NUM_CTX / OLLAMA_THINKING 配置）"
                )
                response = AIMessage(content=_EMPTY_REPLY_FALLBACK)
            else:
                logger.warning(
                    "LLM 返回空回复，重试一次（session=%s）done_reason=%s usage=%s",
                    state.get("session_id"),
                    metadata.get("done_reason"),
                    getattr(response, "usage_metadata", None),
                )
                response = self.llm_with_tools.invoke([system_message, *messages])
                if _is_blank_ai_reply(response):
                    logger.warning("LLM 重试仍为空回复，使用兜底文案占位")
                    response = AIMessage(content=_EMPTY_REPLY_FALLBACK)
        # 主动轮把占位与 AI 回复一并写回，保证后续轮次历史交替合法
        result: dict[str, Any] = {
            "messages": [placeholder, response] if placeholder else [response]
        }
        result["history"] = [
            MemoryItem(
                type="Chat",
                speaker="AI",
                message=response,
                love=state["love_snapshot"],
            )
        ]
        return result

    def _assemble_system_prompt(self, state: EmotionState) -> str:
        """组装 system prompt：本轮人设快照 + 运行时上下文 + 固定全局规则。

        当前时间让模型感知"现在是什么时候"；上次用户回复时间与距今间隔
        赋予模型时间感知（隔了很久要表达想念/小抱怨，刚聊过则自然继续）；
        长期记忆每条带日期，让模型知道"这条回忆是哪一天说的"。
        主动消息轮追加 proactive_context 隐藏指令块（用后即清，不入消息）。
        明确提示 LLM：画像与长期记忆仅供辅助回复，不应当作用户本轮新说的话。
        """
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        last_time_line = self._last_user_reply_line(state, now)
        memory_context = state.get("memory_context") or "（暂无可用长期记忆）"
        profile_context = state.get("profile_context") or "（暂无用户画像）"
        session_summary = state.get("session_summary") or "（暂无短期会话摘要）"
        prompt = (
            f"当前时间：{now_str}\n"
            f"{last_time_line}\n"
            f"当前好感度快照：{state['love_snapshot']}（0-100）。\n"
            f"当前用户画像：\n"
            f"{profile_context}\n\n"
            "以下长期记忆仅用于辅助回复，不要把它当作用户本轮新说的话：\n"
            f"{memory_context}\n\n"
            "已裁剪的本次会话摘要：\n"
            f"{session_summary}"
        )
        proactive_context = state.get("proactive_context") or ""
        if proactive_context:
            prompt += (
                "\n\n【主动消息指令（系统内部指令，仅本轮生效，绝不向用户提及）】\n"
                f"{proactive_context}"
            )
        return compose_system_prompt(self.persona_prompt, prompt)

    def _last_user_reply_line(self, state: EmotionState, now: datetime) -> str:
        """生成"上次用户回复时间"提示行，赋予模型时间感知。

        时间来源（按优先级）：
            1. 当前会话 history 中的用户消息：正常轮排除本轮刚创建的那条
               （取倒数第二条），主动消息轮没有新用户消息（取最后一条）；
            2. 会话内无更早用户消息时，跨会话查 SQLite（本轮尚未落库，
               查到的是更早会话的最后一条用户消息）。
        用户从未发言（全新用户）时返回首次对话说明行。
        """
        user_id = state["user_id"]
        history = list(state.get("history") or [])
        user_times = [
            item.speak_time
            for item in history
            if str(getattr(item, "speaker", "")).lower() == "user"
            and getattr(item, "speak_time", None)
        ]
        if state.get("proactive_context"):
            last_time = user_times[-1] if user_times else None
        else:
            # 正常轮：最后一条用户消息是本轮刚创建的，取倒数第二条
            last_time = user_times[-2] if len(user_times) >= 2 else None
        if last_time is None:
            try:
                last_time = self.short_term_memory.get_last_user_activity_time(user_id)
            except Exception as exc:  # 查询失败不阻塞聊天
                logger.warning("查询上次用户活跃时间失败：%s", exc)
                last_time = None
        if last_time is None:
            return "上次用户回复时间：这是你们的第一次对话。"
        gap_seconds = max(0, now.timestamp() - float(last_time))
        return (
            f"上次用户回复时间：{datetime.fromtimestamp(last_time).strftime('%Y-%m-%d %H:%M')}"
            f"（距今{_humanize_gap(gap_seconds)}）"
        )

    def _tool_call(self, state: EmotionState) -> dict[str, Any]:
        """执行 LLM 发起的工具调用。

        遍历最后一条消息的 tool_calls，逐个调用对应工具：
            - 工具不存在时返回错误字符串；
            - 工具执行异常时把异常信息回传给 LLM（让 LLM 自行处理失败）。
        每个工具调用结果包装为 ToolMessage，并同步写入 history 视图。
        """
        messages = state.get("messages") or []
        if not messages or not getattr(messages[-1], "tool_calls", None):
            return {"messages": []}

        tool_messages: list[ToolMessage] = []
        history: list[MemoryItem] = []
        for tool_call in messages[-1].tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            if tool_name not in self.tools:
                result = f"工具 '{tool_name}' 不存在"
            else:
                try:
                    result = self.tools[tool_name].invoke(tool_args)
                except Exception as exc:  # 工具失败时把错误回传给 LLM
                    result = f"工具执行失败: {exc}"
            tool_message = ToolMessage(content=str(result), tool_call_id=tool_call.get("id", ""))
            tool_messages.append(tool_message)
            history.append(
                MemoryItem(
                    type="ToolCall",
                    speaker="AI",
                    message=tool_message,  # 存消息对象，保持与 SQLite 序列化约定一致
                    love=state["love_snapshot"],
                )
            )
        return {"messages": tool_messages, "history": history}

    def _write_profile_candidate(self, state: EmotionState) -> dict[str, Any]:
        """写入画像候选节点：从本轮用户输入抽取画像候选（委托 UserProfileManager）。

        逻辑：
            1. 触发词预判：本轮用户消息命中画像触发规则（matches_profile_trigger）
               才调用候选模型，未命中直接跳过——避免每轮都发一次 LLM 请求；
            2. 命中后定位本轮用户消息，取其消息 ID 写入候选证据（保证可追溯）；
            3. 调用 process_user_message 由候选模型抽取并逐条审核候选
               （自动应用 / 待用户确认 / 拒绝）；
            4. 抽取失败（Ollama 未启动、超时、返回非法 JSON）时捕获
               ProfileCandidateExtractionError 降级：记录警告日志并跳过，
               聊天主流程不中断。

        返回值：profile_candidate_ids 为本轮产生的画像候选 ID 列表，
        供 UI 事件与调试观测使用（未触发或失败时为空）。
        """
        user_id = state["user_id"]
        session_id = state["session_id"]
        user_input = state.get("user_input", "")

        # 主动消息轮没有真实用户输入：不做画像候选提取
        if state.get("proactive_context"):
            return {"profile_candidate_ids": []}

        # 触发词预判：普通闲聊未命中画像相关词，直接跳过 LLM 提取
        if not matches_profile_trigger(user_input):
            return {"profile_candidate_ids": []}

        # 定位本轮用户消息，把 LangChain 消息 ID 传给候选证据
        messages = state.get("messages") or []
        current_user = next(
            (
                item
                for item in reversed(messages)
                if isinstance(item, HumanMessage) and item.content == user_input
            ),
            None,
        )
        message_id = getattr(current_user, "id", None)

        try:
            results = self.user_profile_manager.process_user_message(
                user_id,
                user_input,
                session_id,
                message_id,
            )
        except ProfileCandidateExtractionError as exc:
            # 候选模型不可用：降级为跳过本轮画像候选，聊天不中断
            logger.warning("画像候选提取失败，已跳过本轮画像候选写入：%s", exc)
            return {"profile_candidate_ids": []}

        return {
            "profile_candidate_ids": [
                result.candidate_id for result in results if result.candidate_id is not None
            ]
        }

    # 好感度增量标记：匹配回复末尾的 {{+N}} / {{-N}}（N 为 1~3 位整数）。
    # 与情绪标记 <<Happy>> 刻意使用两套符号（双花括号 vs 双尖括号），
    # 避免共用 <<>> 时正则解析互相干扰。
    _AFFINITY_DELTA_RE = re.compile(r"\{\{([+-]?\d{1,3})\}\}")
    # 单轮好感度增量绝对值上限（防止模型输出异常大的数值）
    MAX_AFFINITY_DELTA_PER_TURN = 10

    def _judge_affinity(self, state: EmotionState) -> dict[str, Any]:
        """好感度更新节点：解析本轮 AI 回复中的好感度增量标记并持久化。

        机制（零额外 LLM 调用）：
            主 LLM 生成回复时按 system prompt 约定在末尾附加 {{+N}} /
            {{-N}} 增量标记（双花括号，与情绪标记 <<Happy>> 的双尖括号
            分属两套符号），本节点仅做正则解析——不再单独调用 LLM 判定，
            节省每轮一次模型请求。

        逻辑：
            1. 取最近一条 AI 回复，用 _AFFINITY_DELTA_RE 提取增量标记
              （取最后一个，防正文误带多个标记时取错）；
            2. 无标记 → 增量 0（模型忘记输出时不惩罚，好感度不变）；
            3. 增量钳制到 ±MAX_AFFINITY_DELTA_PER_TURN，交给
               AffinityService.apply_delta 累加并钳制到 0-100 持久化。

        更新后的 love_snapshot 写回 state：本轮 system prompt 用的仍是
        轮初快照（上轮值），新值从下一轮开始生效。

        主动消息轮跳过好感度判定：她主动发消息不改变好感度；
        被无视的惩罚由 ProactiveEngine 直接调用 apply_delta 执行
        （确定性数值，不依赖模型输出）。
        """
        if state.get("proactive_context"):
            return {}  # 主动消息不影响好感度
        user_id = state["user_id"]
        messages = state.get("messages") or []
        last_ai = next(
            (message for message in reversed(messages) if isinstance(message, AIMessage)),
            None,
        )
        ai_reply = str(getattr(last_ai, "content", "") or "")
        matches = self._AFFINITY_DELTA_RE.findall(ai_reply)
        if not matches:
            return {}  # 回复未带增量标记：本轮好感度不变
        try:
            delta = int(matches[-1])
        except ValueError:
            return {}
        delta = max(-self.MAX_AFFINITY_DELTA_PER_TURN, min(self.MAX_AFFINITY_DELTA_PER_TURN, delta))
        if delta == 0:
            return {}
        new_love = self.affinity_service.apply_delta(user_id, delta)
        return {"love_snapshot": new_love}

    def _emit_affinity_event(self, state: EmotionState) -> dict[str, Any]:
        """向好感度服务发射"记忆写入完成"事件，供后续好感度计算消费。

        事件携带本轮写入的记忆 ID 与当前好感度快照（已含本轮增量）。
        """
        self.affinity_service.emit_affinity_event(
            state["user_id"],
            {
                "type": "memory_write_completed",
                "session_id": state["session_id"],
                "memory_ids": list(state.get("memory_write_ids") or []),
                "love_snapshot": state.get("love_snapshot", 0),
            },
        )
        return {}

    def _trim_short_term_window(self, state: EmotionState) -> dict[str, Any]:
        """裁剪短期窗口，并对被裁剪消息做 LLM 合并提取（摘要 + 长期记忆）。

        逻辑：
            1. 对当前消息做裁剪，生成 RemoveMessage 列表让 LangGraph 移除旧消息；
            2. 有消息被裁剪时，把【旧摘要 + 当前时间 + 被裁剪对话(含发生时间)】
               交给 memory_extractor 一次调用，同时产出新摘要与长期记忆条目
               （合并提取，避免摘要、记忆各调一次 LLM）；
            3. 新摘要写入 session_summary；长期记忆条目交给 LongTermMemory
               持久化（主走向量库，失败降级 JSON）；
            4. LLM 提取失败时整体降级：长期记忆走规则分类器(write_candidates)、
               摘要走规则式摘录，聊天主流程不中断。
        """
        messages = state.get("messages") or []
        _, removed = self.short_term_memory.trim_window(messages)
        removals = [
            RemoveMessage(id=message.id) for message in removed if getattr(message, "id", None)
        ]
        result: dict[str, Any] = {"messages": removals, "memory_write_ids": []}

        fragments = self._removed_with_times(removed, state.get("history") or [])
        if not fragments:
            return result

        previous_summary = state.get("session_summary") or ""
        user_id = state["user_id"]
        session_id = state["session_id"]

        try:
            extraction = self.memory_extractor.extract(fragments, previous_summary)
            memory_ids = self.long_term_memory.write_llm_memories(
                user_id, extraction.memories, session_id
            )
            result["memory_write_ids"] = memory_ids
            if extraction.summary:
                result["session_summary"] = extraction.summary[: self.config.SUMMARY_MAX_CHARS]
        except MemoryExtractionError as exc:
            # LLM 合并提取失败：降级为规则提取 + 规则摘要，聊天不中断
            logger.warning("LLM 合并提取失败，降级为规则提取+规则摘要：%s", exc)
            memory_ids = self.long_term_memory.write_candidates(
                removed,
                user_id,
                love_snapshot=state["love_snapshot"],
            )
            result["memory_write_ids"] = memory_ids
            legacy = self._legacy_session_summary(
                previous_summary,
                [(speaker, content) for speaker, content, _ in fragments],
            )
            if legacy:
                result["session_summary"] = legacy
        return result

    def _removed_with_times(
        self,
        removed_messages: Sequence[Message],
        history: Sequence[MemoryItem],
    ) -> list[tuple[str, str, str | None]]:
        """把被裁剪消息转成 (speaker, content, time_str) 片段，附上发生时间。

        时间来自 history 中 MemoryItem.speak_time（消息对象按身份匹配），
        缺失时省略，供 LLM 提取器理解对话的时间顺序。
        """
        time_by_message: dict[int, datetime] = {}
        for item in history or []:
            message = getattr(item, "message", None)
            speak_time = getattr(item, "speak_time", None)
            if message is not None and speak_time:
                time_by_message[id(message)] = datetime.fromtimestamp(speak_time)

        speakers = {"human": "用户", "ai": "AI", "tool": "工具"}
        fragments: list[tuple[str, str, str | None]] = []
        for message in self.short_term_memory.select_memory_writes(removed_messages):
            content = str(getattr(message, "content", "") or "").strip()
            if not content:
                continue
            speaker = speakers.get(getattr(message, "type", ""), "用户")
            message_time = time_by_message.get(id(message))
            time_str = message_time.strftime("%Y-%m-%d %H:%M") if message_time is not None else None
            fragments.append((speaker, content[: self.config.SUMMARY_FRAGMENT_MAX_CHARS], time_str))
        return fragments

    def _legacy_session_summary(
        self,
        previous_summary: str,
        fragments: list[tuple[str, str]],
    ) -> str:
        """降级摘要策略：旧摘要 + 新片段，遵循当前配置的摘要长度上限。"""
        new_summary = "；".join(content[:80] for _, content in fragments)
        return "；".join(part for part in (previous_summary, new_summary) if part)[
            -self.config.SUMMARY_MAX_CHARS :
        ]
