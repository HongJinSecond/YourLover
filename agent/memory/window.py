"""短期对话窗口裁剪工具。

负责在消息数量超过短期窗口上限时，按"对话轮次"裁剪消息列表，
保留头部系统提示与最近若干完整轮次，把被裁剪的消息交给长期记忆写入流程。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

# 消息对象类型（兼容 LangChain 各类 Message）
Message: TypeAlias = object
MessageList: TypeAlias = list[Message]
# 带原始索引的消息组，用于裁剪时跟踪消息位置
IndexedGroup: TypeAlias = list[tuple[int, Message]]


def should_trim(messages: Sequence[Message], max_messages: int) -> bool:
    """判断消息列表是否已超过短期窗口上限，需要裁剪。"""
    _validate_max_messages(max_messages)
    return len(messages) > max_messages


def trim_messages(
    messages: Sequence[Message],
    max_messages: int,
    keep_recent_turns: int,
) -> tuple[MessageList, MessageList]:
    """裁剪消息列表，保留头部系统提示与最近若干完整轮次。

    一个"对话轮次"以 HumanMessage 开始，包含其后直到下一个 HumanMessage 之前
    的所有 AI/工具消息。按轮次裁剪可保证工具调用结果与其发起的 assistant
    消息不被拆散。

    返回 (retained, removed)：
        - retained: 保留的消息（按原顺序）
        - removed: 被裁剪的消息（按原顺序），供长期记忆写入使用
    裁剪逻辑：
        1. 若未超限直接返回原列表；
        2. 固定保留头部所有 SystemMessage；
        3. 在剩余消息中按轮次分组，保留最后 keep_recent_turns 个轮次；
        4. 若保留后仍超限，则从最早的保留轮次开始逐个丢弃，直到不超限或只剩一个轮次。
    """
    _validate_max_messages(max_messages)
    if keep_recent_turns < 0:
        raise ValueError("keep_recent_turns must be greater than or equal to 0")

    original = list(messages)
    if not should_trim(original, max_messages):
        return original, []

    pinned_count = _leading_system_count(original)
    pinned_indices = set(range(pinned_count))
    groups = _conversation_groups(original, start_index=pinned_count)
    retained_groups = groups[-keep_recent_turns:] if keep_recent_turns else []

    retained_indices = set(pinned_indices)
    for group in retained_groups:
        retained_indices.update(index for index, _ in group)

    # 保留后仍超限时，从最早的保留轮次开始丢弃
    while len(retained_indices) > max_messages and len(retained_groups) > 1:
        dropped_group = retained_groups.pop(0)
        for index, _ in dropped_group:
            retained_indices.discard(index)

    retained = [message for index, message in enumerate(original) if index in retained_indices]
    removed = [message for index, message in enumerate(original) if index not in retained_indices]
    return retained, removed


def select_messages_for_memory_write(removed_messages: Sequence[Message]) -> MessageList:
    """从被裁剪的消息中筛选可喂给长期记忆分类器的消息。

    排除 SystemMessage（系统提示无需记忆）与 RemoveMessage（裁剪标记本身无内容）。
    """
    return [
        message
        for message in removed_messages
        if not (_is_system_message(message) or _is_remove_message(message))
    ]


def _validate_max_messages(max_messages: int) -> None:
    """校验 max_messages >= 1，否则抛出 ValueError。"""
    if max_messages < 1:
        raise ValueError("max_messages must be greater than or equal to 1")


def _leading_system_count(messages: Sequence[Message]) -> int:
    """统计消息列表头部连续的 SystemMessage 数量，作为裁剪时固定的前缀。"""
    count = 0
    for message in messages:
        if not _is_system_message(message):
            break
        count += 1
    return count


def _conversation_groups(
    messages: Sequence[Message],
    start_index: int,
) -> list[IndexedGroup]:
    """从 start_index 开始把消息按对话轮次分组。

    每遇到一个 HumanMessage 就开启新轮次，同一轮次内包含其后的所有
    AI/工具消息，保证工具调用结果与发起它的 assistant 消息不被拆散。
    """
    groups: list[IndexedGroup] = []
    current_group: IndexedGroup = []

    for index in range(start_index, len(messages)):
        message = messages[index]
        if _is_human_message(message) and current_group:
            groups.append(current_group)
            current_group = []
        current_group.append((index, message))

    if current_group:
        groups.append(current_group)

    return groups


def _is_human_message(message: Message) -> bool:
    """判断是否为 HumanMessage（兼容 LangChain 的 type 属性与类名两种判断方式）。"""
    return _message_kind(message) == "human" or message.__class__.__name__ == "HumanMessage"


def _is_system_message(message: Message) -> bool:
    """判断是否为 SystemMessage。"""
    return _message_kind(message) == "system" or message.__class__.__name__ == "SystemMessage"


def _is_remove_message(message: Message) -> bool:
    """判断是否为 LangGraph 的 RemoveMessage（裁剪标记）。"""
    return _message_kind(message) == "remove" or message.__class__.__name__ == "RemoveMessage"


def _message_kind(message: Message) -> str | None:
    """读取消息的 type 属性，兼容 LangChain Message 的 type 字段。"""
    kind = getattr(message, "type", None)
    if isinstance(kind, str):
        return kind
    return None
