"""Messages passed from the application coordinator to the chat UI."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
