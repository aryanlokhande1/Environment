"""Environment result contract."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

@dataclass
class EnvironmentResult:
    events: list[dict[str, Any]]
    state: "EnvironmentState"
    terminal: bool
    pending_events: list[dict[str, Any]]
    explanations: list[dict[str, Any]]
    decisions: list[dict[str, Any]]

if TYPE_CHECKING:
    from environment.core.state import EnvironmentState
