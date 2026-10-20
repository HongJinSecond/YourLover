"""
Descripttion: 兼容历史视图的单条记忆项。
version:
Author: HongJin
email: 756547077@qq.com
Date: 2026-06-25 12:13:25
LastEditors: HongJin
LastEditTime: 2026-06-25 13:31:02
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, Optional

from langchain_core.messages import BaseMessage


@dataclass
class MemoryItem:
    """兼容历史视图的单条记忆项：记录一轮对话的元数据。

    该类作为 EmotionState.history 的元素，用于在不依赖长期记忆存储时
    保留每轮对话的"谁说了什么、当时好感度与情感状态"信息。
    根据 type 字段区分对话/思考/工具调用三种记忆，get_memory 方法会
    按类型输出不同格式的上下文字符串。

    字段：
        type: 记忆类型（Chat/Think/ToolCall），决定 get_memory 的输出格式
        speaker: 发言方（System/AI/User）
        message: LangChain 消息对象
        love: 当时的好感度快照，默认 0
        ai_emo: AI 当时的情感状态（情感词或情感得分字典）
        user_emo: 用户当时的情感状态（情感词或情感得分字典）
        speak_time: 对话发生时间戳，默认取当前时间
        importance: 记忆重要程度，默认 0.5
    """

    # MEMORY_ID = None # 记忆的唯一标识符 (也许需要)
    type: str
    """判断记忆类型是不是对话记忆，如果是思考或者工具调用的记忆需要特殊处理。 (Chat, Think, ToolCall)"""
    speaker: str
    """System, AI 或者 User"""
    message: BaseMessage
    """Langchain的消息对象。"""
    love: int = 0
    """好感度，默认0"""
    ai_emo: Optional[Dict[str, float]] = None
    """AI当前对话的情感状态，可以存储情感词汇比如["快乐", "悲伤"]等，也可以存储情感得分比如{"快乐": 0.8, "悲伤": 0.2}。根据后续开发决策。"""
    user_emo: Optional[Dict[str, float]] = None
    """用户当前对话的情感状态，可以存储情感词汇比如["快乐", "悲伤"]等，也可以存储情感得分比如{"快乐": 0.8, "悲伤": 0.2}。根据后续开发决策。"""
    speak_time: float = None
    """对话发生时的时间戳，默认当前时间"""
    importance: float = 0.5
    """记忆的重要程度，默认0.5"""

    def __post_init__(self):
        """初始化后处理：speak_time 未指定时填充为当前时间戳。"""
        if self.speak_time is None:
            self.speak_time = datetime.now().timestamp()

    # 获取记忆内容添加到上下文中，根据记忆类型进行不同处理
    def get_memory(self) -> str:
        """获取记忆内容用于上下文，根据记忆类型进行不同处理。

        - Chat：返回 "{speaker}: {content}"；
        - Think：返回 "[思考] {content}"；
        - ToolCall：返回 "[工具调用] {content}"；
        - 其它：退化为 "{speaker}: {content}"。
        """
        if self.type == "Chat":
            # 一般情况
            return f"{self.speaker}: {self.message.content}"

        elif self.type == "Think":
            # 思考类型
            return f"[思考] {self.message.content}"

        elif self.type == "ToolCall":
            # 工具调用类型
            return f"[工具调用] {self.message.content}"

        else:
            return f"{self.speaker}: {self.message.content}"

    def to_dict(self) -> dict:
        """转换为字典，用于序列化（注意 message 字段需自行处理可序列化性）。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryItem":
        """从字典创建实例，用于反序列化（要求 data 的键与字段名完全匹配）。"""
        return cls(**data)

    def __str__(self):
        """返回简短描述，便于调试（仅展示前 30 字内容）。"""
        return f"MemoryItem({self.type}, {self.speaker}, {self.message.content[:1000]}...)"
