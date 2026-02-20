from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

VALID_ROLES = {"system", "user", "assistant"}

FORMATS = ["auto", "sharegpt", "messages", "alpaca", "csv_custom", "completion"]


@dataclass
class Message:
    role: str
    content: str

    def __post_init__(self):
        assert self.role in VALID_ROLES, f"Invalid role: {self.role}"
        assert isinstance(self.content, str), "Content must be a string"


@dataclass
class CanonicalRecord:
    id: str
    messages: List[Message]
    meta: Dict[str, Any] = field(default_factory=dict)
