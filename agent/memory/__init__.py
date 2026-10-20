"""记忆模块：三大管理器 + 支撑数据模型。

对外主入口（推荐调用方只使用这三个类）：
    - ShortTermMemory   短期记忆：Session 存取、消息持久化、窗口裁剪
                        （组合 session_store.py + window.py）
    - LongTermMemory    动态长期记忆：向量存储主路径 + JSON 规则降级
                        （组合 vector_memory_store.py + store.py +
                         classifier.py + retriever.py + llm_extractor.py）
    - UserProfileManager 用户画像：画像读取、Ollama 候选抽取、确认流程
                        （组合 profile_service.py + profile_store.py + user_profile.py）

支撑模块（被管理器组合，一般不直接使用）：
    - MemoryItem / EmotionMemory   短期记忆项与 LangGraph 运行时状态
    - schema.py                    三类长期记忆数据模型
    - session_store.py / window.py 短期记忆 SQLite 存储与裁剪函数
    - store.py / classifier.py / retriever.py  长期记忆 JSON 存储、规则分类、关键词检索（降级）
    - vector_memory_store.py       长期记忆 Chroma 向量存储（主路径）
    - llm_extractor.py             摘要 + 长期记忆的 LLM 合并提取器
    - user_profile.py / profile_candidate.py / profile_store.py  画像模型与存储
"""

from agent.memory.llm_extractor import (
    ExtractionResult,
    LLMMemoryEntry,
    LLMMemoryExtractor,
    MemoryExtractionError,
)
from agent.memory.long_term_memory import LongTermMemory
from agent.memory.profile_service import (
    OllamaProfileCandidateExtractor,
    ProfileCandidateExtractionError,
    ProfileCandidateExtractor,
    ProfileProcessingResult,
    UserProfileManager,
)
from agent.memory.short_term_memory import ShortTermMemory
from agent.memory.vector_memory_store import VectorMemoryStore, VectorStoreError

__all__ = [
    # 三大管理器（主入口）
    "ShortTermMemory",
    "LongTermMemory",
    "UserProfileManager",
    # 长期记忆向量存储与合并提取
    "VectorMemoryStore",
    "VectorStoreError",
    "LLMMemoryExtractor",
    "LLMMemoryEntry",
    "ExtractionResult",
    "MemoryExtractionError",
    # 画像候选提取相关（替换模型适配器时使用）
    "ProfileCandidateExtractor",
    "OllamaProfileCandidateExtractor",
    "ProfileCandidateExtractionError",
    "ProfileProcessingResult",
]
