"""Typed boundary for an external Transformer/RL policy."""
from __future__ import annotations
from typing import Any, Protocol
import pandas as pd
from .action import EnvironmentAction

class Transformer(Protocol):
    def encode(self, cumulative_gold_history: pd.DataFrame) -> Any: ...

class RLPolicy(Protocol):
    def choose_action(self, embedding: Any) -> EnvironmentAction: ...
