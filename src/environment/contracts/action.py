"""RL-to-environment action contract."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping

CHANNELS = frozenset({"SMS", "WA", "GRCS"})
TIME_BUCKETS = frozenset({"MORNING", "AFTERNOON", "EVENING", "NIGHT"})
THEME_PLACEHOLDER = "THEME_PLACEHOLDER"

@dataclass(frozen=True)
class EnvironmentAction:
    campaign_sent: bool
    channel_id: str | None = None
    theme: str | None = None
    time_bucket: str | None = None

    def __post_init__(self) -> None:
        if not self.campaign_sent:
            if any(value is not None for value in (self.channel_id, self.theme, self.time_bucket)):
                raise ValueError("NO_ACTION cannot contain campaign fields")
            return
        channel = str(self.channel_id or "").upper()
        bucket = str(self.time_bucket or "").upper()
        theme = str(self.theme or THEME_PLACEHOLDER).upper()
        if channel not in CHANNELS:
            raise ValueError(f"unsupported channel_id: {channel}")
        if bucket not in TIME_BUCKETS:
            raise ValueError(f"unsupported time_bucket: {bucket}")
        if theme != THEME_PLACEHOLDER:
            raise ValueError("theme-specific effects are unsupported; use THEME_PLACEHOLDER")
        object.__setattr__(self, "channel_id", channel)
        object.__setattr__(self, "time_bucket", bucket)
        object.__setattr__(self, "theme", theme)

    @classmethod
    def no_action(cls) -> "EnvironmentAction":
        return cls(False)

    @classmethod
    def campaign(cls, channel_id: str, time_bucket: str,
                 theme: str = THEME_PLACEHOLDER) -> "EnvironmentAction":
        return cls(True, channel_id, theme, time_bucket)

    @classmethod
    def parse(cls, value: "EnvironmentAction | Mapping[str, Any]") -> "EnvironmentAction":
        """Validate an action returned by an external policy.

        Unknown fields are rejected so a misspelled policy output cannot be
        silently interpreted as NO_ACTION.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("policy action must be EnvironmentAction or a mapping")
        allowed = {"campaign_sent", "channel_id", "theme", "time_bucket"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown action fields: {sorted(unknown)}")
        if "campaign_sent" not in value or not isinstance(value["campaign_sent"], bool):
            raise ValueError("campaign_sent must be present and boolean")
        return cls(
            campaign_sent=value["campaign_sent"],
            channel_id=value.get("channel_id"),
            theme=value.get("theme"),
            time_bucket=value.get("time_bucket"),
        )

    @staticmethod
    def mask(*, terminal: bool, before_expiry: bool) -> dict[str, bool]:
        """Return the action-family mask exposed to an external RL policy."""
        send_allowed = not terminal and before_expiry
        return {"NO_ACTION": not terminal, "SEND": send_allowed}
