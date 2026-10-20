"""情感陪伴 Agent 的运行时状态定义。

定义 EmotionState TypedDict，作为 LangGraph 各节点共享的状态容器；
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from agent.memory.MemoryItem import MemoryItem


class EmotionState(TypedDict, total=False):
    """ChatAgent LangGraph 各节点共享的运行时状态。

    字段：
        messages: 当前会话消息列表（使用 add_messages reducer，支持增量合并）
        history: 兼容历史视图的 MemoryItem 列表（使用 operator.add 合并）
        user_input: 本轮用户输入文本
        user_id: 用户唯一标识（用于长期记忆归属）
        session_id: 会话唯一标识（用于短期状态）
        profile_context: 用户画像上下文字符串（每轮从 UserProfileManager 刷新）
        memory_context: 检索到的长期记忆上下文字符串（每轮刷新）
        session_summary: 本次会话的裁剪摘要
        love_snapshot: 当前好感度快照（0-100，每轮刷新）
        memory_write_ids: 本轮写入的长期记忆 ID 列表
        profile_candidate_ids: 本轮产生的画像候选 ID 列表（提取失败降级时为空）
        proactive_context: 主动消息隐藏指令（仅主动对话轮非空）：
            由 ProactiveEngine 注入、_chat 节点消费后清空，不会出现在
            任何消息里，也不写入历史/SQLite——用户与 UI 均不可见
    """

    messages: Annotated[list[BaseMessage], add_messages]
    history: Annotated[list[MemoryItem], operator.add]
    user_input: str
    user_id: str
    session_id: str
    profile_context: str
    memory_context: str
    session_summary: str
    love_snapshot: int
    memory_write_ids: list[str]
    profile_candidate_ids: list[str]
    proactive_context: str
