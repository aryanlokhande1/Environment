"""Map realized V3 world events into the original Gold Events column contract.

Unknown source-system fields remain NULL. Audit-only decisions, start markers,
and lifecycle expiry markers are never disguised as historical Gold telemetry.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping

GOLD_COLUMNS = (
    "application_id", "context_id", "event_datetime", "event_name", "journey_stage",
    "journey_substage", "site_subsection", "channel_id", "platform_type", "product_id",
    "partner_id", "trigger_type", "time_bucket", "visitnum", "utm_source",
    "utm_medium", "utm_campaign", "source_type", "campaign_name", "session_id",
    "channel", "utm_channel", "day_of_week_num", "initiated_by",
)
ENGAGEMENT_LABELS = frozenset({"campaign_open_read", "campaign_clicked"})


def historical_row_key(source_name: str, source_row: int) -> str:
    return "H|" + sha256(f"{source_name}|{source_row}".encode()).hexdigest()


def simulated_row_key(run_id: str, event_key: str) -> str:
    return "S|" + sha256(f"{run_id}|{event_key}".encode()).hexdigest()


def to_gold_row(event: Mapping[str, Any]) -> dict[str, Any] | None:
    kind = str(event["event_type"])
    if kind in {"SIMULATION_START_EVENT", "TERMINAL_EVENT"}:
        return None
    if kind == "CAMPAIGN_SEND_EVENT":
        label, name = "campaign_sent", "campaign_sent"
    elif kind == "CAMPAIGN_RESPONSE_EVENT":
        label = str(event.get("model_level"))
        if label not in ENGAGEMENT_LABELS:
            raise ValueError(f"unsupported observed campaign engagement: {label}")
        name = label
    elif kind in {"JOURNEY_EVENT", "ACTION_CONDITIONED_JOURNEY_EVENT", "TRANSACTION_EVENT"}:
        label = str(event["journey_substage"])
        # Gold event_name is not determined by journey_substage (including PTP).
        name = None
    else:
        raise ValueError(f"unsupported event type for Gold adapter: {kind}")
    row = {column: None for column in GOLD_COLUMNS}
    row.update({
        "application_id": event["application_id"],
        "context_id": event["context_id"],
        "event_datetime": event["event_datetime"],
        "event_name": name,
        "journey_stage": None if kind in {"CAMPAIGN_SEND_EVENT", "CAMPAIGN_RESPONSE_EVENT"}
                         else event.get("journey_stage"),
        "journey_substage": label,
        "trigger_type": event.get("trigger_type"),
        "time_bucket": str(event["time_bucket"]).lower() if event.get("time_bucket") else None,
        "channel": event.get("channel"),
        "day_of_week_num": event["event_datetime"].weekday() + 1,
    })
    return {
        "cumulative_row_key": simulated_row_key(str(event["run_id"]), str(event["event_key"])),
        "run_id": event["run_id"], "is_simulated": True,
        "simulation_event_key": event["event_key"], "source_file": None,
        "source_row": None, **row,
    }
