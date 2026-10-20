import re
from collections import namedtuple

MessageAndEmotion = namedtuple("MessageAndEmotion", ["message", "emotion"])


def extract_emotion(text):
    """
    从字符串中提取所有 <<...>> 内的内容，并移除这些标记。
    同时移除 {{...}} 好感度增量标记（与情绪标记分属两套符号，由
    ChatAgent.judge_affinity 节点解析，这里只负责从展示文本中剔除）。
    返回 (清理后的消息, 特殊字符列表)
    """
    # 查找所有 <<...>> 中的内容（非贪婪匹配），即情绪标记
    emotion = re.findall(r"<<(.*?)>>", text)
    # 移除所有 <<...>> 标记
    message = re.sub(r"<<.*?>>", "", text)
    # 移除 {{...}} 好感度标记（不参与情绪解析，避免残留到界面）
    message = re.sub(r"\{\{.*?\}\}", "", message)
    # 清理多余空格（可选）
    message = message.strip()

    return MessageAndEmotion(message=message, emotion=emotion)
