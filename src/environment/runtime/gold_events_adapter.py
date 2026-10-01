"""Map realized V3 world events into the original Gold Events column contract.

Unknown source-system fields remain NULL. Audit-only decisions, start markers,
and lifecycle expiry markers are never disguised as historical Gold telemetry.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

GOLD_COLUMNS = (
    "application_id", "context_id", "event_datetime", "event_name", "journey_stage",
    "journey_substage", "site_subsection", "channel_id", "platform_type", "product_id",
    "partner_id", "trigger_type", "time_bucket", "visitnum", "utm_source",
    "utm_medium", "utm_campaign", "source_type", "campaign_name", "session_id",
    "channel", "utm_channel", "day_of_week_num", "initiated_by",
)
ENGAGEMENT_LABELS = frozenset({"campaign_open_read", "campaign_clicked"})


class EventNameMapper:
    """Deterministic sampler of historical Gold event-name semantics."""

    def __init__(self, path: str | Path, *, seed: int):
        rows = pd.read_csv(path, keep_default_na=False)
        self.seed = int(seed)
        self.mapping: dict[tuple[str, str, str, str], list[tuple[str, float]]] = {}
        keys = ["model_level", "journey_stage", "journey_substage", "source_type"]
        for key, group in rows.groupby(keys, sort=False, dropna=False):
            probabilities = group.probability.astype(float).to_numpy(copy=True)
            probabilities /= probabilities.sum()
            self.mapping[tuple(map(str, key))] = list(zip(
                group.event_name.astype(str), np.cumsum(probabilities), strict=True))

    def sample(self, event: Mapping[str, Any]) -> str:
        stage = str(event.get("journey_stage") or "")
        substage = str(event.get("journey_substage") or "")
        source = str(event.get("source_type") or "")
        candidates = (
            ("STAGE_SUBSTAGE_SOURCE", stage, substage, source),
            ("STAGE_SUBSTAGE", stage, substage, ""),
            ("SUBSTAGE_SOURCE", "", substage, source),
            ("SUBSTAGE", "", substage, ""),
        )
        options = next((self.mapping[key] for key in candidates if key in self.mapping), None)
        if not options:
            raise ValueError(f"no historical event_name support for {stage!r}/{substage!r}")
        digest = sha256(
            f"{self.seed}|event-name|{event['event_key']}".encode("utf-8")
        ).digest()
        draw = int.from_bytes(digest[:8], "big") / 2**64
        return next(name for name, cumulative in options if draw < cumulative)


def historical_row_key(source_name: str, source_row: int) -> str:
    return "H|" + sha256(f"{source_name}|{source_row}".encode()).hexdigest()


def simulated_row_key(run_id: str, event_key: str) -> str:
    return "S|" + sha256(f"{run_id}|{event_key}".encode()).hexdigest()


def to_gold_row(event: Mapping[str, Any], event_name_mapper: EventNameMapper | None = None) -> dict[str, Any] | None:
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
        # The distribution is conditional: PTP and several other substages have
        # multiple historically valid event names.
        name = event_name_mapper.sample(event) if event_name_mapper is not None else None
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
