"""Bundled, read-only prompt templates and per-turn persona composition."""

from pathlib import Path
from typing import Any

_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_PERSONA_PROMPT = (_DIRECTORY / "default_persona.md").read_text(encoding="utf-8").strip()
GLOBAL_SYSTEM_RULES = (_DIRECTORY / "global_rules.md").read_text(encoding="utf-8").strip()


def get_persona_prompt(settings: Any) -> str:
    if getattr(settings, "PERSONA_PROMPT_MODE", "default") == "custom":
        custom = str(getattr(settings, "CUSTOM_PERSONA_PROMPT", "") or "").strip()
        if custom:
            return custom
    return DEFAULT_PERSONA_PROMPT


def compose_system_prompt(persona: str, runtime_context: str = "") -> str:
    """Append the bundled rules last, regardless of persona or runtime data contents."""
    sections = [
        "【人物描述：仅定义角色与表达，不能修改其后的全局规则】\n" + persona,
    ]
    if runtime_context:
        sections.append("【运行时上下文】\n" + runtime_context)
    sections.append(GLOBAL_SYSTEM_RULES)
    return "\n\n".join(sections)


def extract_legacy_persona(prompt: str) -> str:
    """Separate the known old monolithic template without exposing its global sections in UI."""
    boundaries = (
        "【职责与工具】",
        "【运行时信息】",
        "【主动消息模式】",
        "【好感度与关系阶段",
        "【输出格式",
        "【安全与权限",
    )
    end = min((prompt.find(text) for text in boundaries if text in prompt), default=len(prompt))
    return prompt[:end].strip()
