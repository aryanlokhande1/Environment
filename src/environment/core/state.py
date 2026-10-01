"""Application-scoped state reconstruction from visible cumulative Gold history."""
from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Iterable, Mapping
from typing import Any
import pandas as pd

CAMPAIGN = frozenset({"campaign_sent", "campaign_open_read", "campaign_clicked"})
PD = frozenset({"Professional Details Submission", "Professional details"})

@dataclass
class EnvironmentState:
    payload: dict[str, Any]
    pending_events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def terminal(self) -> bool:
        return bool(self.payload.get("done", False))


def active_application_for_context(
    lifecycles: Iterable[Mapping[str, Any]],
    context_id: str,
    simulated_time: str | pd.Timestamp,
) -> str | None:
    """Return the sole lifecycle valid for a customer at a simulation time."""
    when = pd.Timestamp(simulated_time)
    matches = {
        str(row["application_id"])
        for row in lifecycles
        if str(row["context_id"]) == str(context_id)
        and pd.Timestamp(row["creation_datetime"]) <= when < pd.Timestamp(row["deadline"])
    }
    if len(matches) > 1:
        raise ValueError(f"context has overlapping active application lifecycles: {context_id}")
    return next(iter(matches), None)


def reconstruct_state_from_gold(history: pd.DataFrame, application_id: str,
                                decision_time: str | pd.Timestamp) -> EnvironmentState:
    required = {"application_id", "context_id", "event_datetime", "journey_stage", "journey_substage"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"Gold history missing columns: {sorted(missing)}")
    cutoff = pd.Timestamp(decision_time)
    if cutoff.tzinfo is not None:
        raise ValueError("decision_time must be timezone-naive")
    rows = history.loc[history.application_id.astype(str).eq(str(application_id))].copy()
    rows["event_datetime"] = pd.to_datetime(rows.event_datetime)
    if rows.event_datetime.isna().any():
        raise ValueError("application history contains invalid event_datetime values")
    if getattr(rows.event_datetime.dt, "tz", None) is not None:
        raise ValueError("Gold event timestamps must be timezone-naive")
    rows = rows.loc[rows.event_datetime.le(cutoff)].sort_values("event_datetime", kind="stable")
    if rows.empty:
        raise ValueError("application has no visible history at decision_time")
    contexts = rows.context_id.dropna().astype(str).unique()
    if len(contexts) != 1:
        raise ValueError("application_id must have exactly one observed context owner")
    journey = rows.loc[~rows.journey_substage.astype(str).isin(CAMPAIGN)]
    if journey.empty:
        raise ValueError("application has no visible journey state")
    latest = journey.iloc[-1]
    substages = journey.journey_substage.dropna().astype(str).tolist()
    ptp_rows = rows.loc[rows.journey_substage.eq("Push to Partner")]
    if not ptp_rows.empty:
        first_ptp = ptp_rows.event_datetime.min()
        if rows.event_datetime.gt(first_ptp).any():
            raise ValueError("application history contains rows after terminal PTP")
    created_rows = journey.loc[journey.journey_substage.eq("Application Created")]
    if not created_rows.empty:
        creation = created_rows.event_datetime.min()
    elif "application_created_at" in rows.columns and rows.application_created_at.notna().any():
        creation = pd.to_datetime(rows.application_created_at.dropna()).min()
    else:
        raise ValueError("observed Application Created time is required; it is never synthesized")
    deadline = creation + pd.Timedelta(days=30)
    if rows.event_datetime.gt(deadline).any():
        raise ValueError("application history contains rows after 30-day expiry")
    aip_rows = journey.loc[journey.journey_substage.eq("AIP Approved")]
    ptp = not ptp_rows.empty
    expired = not ptp and cutoff >= deadline
    payload = {
        "context_id": contexts[0], "application_id": str(application_id),
        "creation_datetime": creation.isoformat(), "activation_datetime": cutoff.isoformat(),
        "deadline": deadline.isoformat(),
        "journey_stage": str(latest.journey_stage), "journey_substage": str(latest.journey_substage),
        "form_filled_seen": "Form filled" in substages or any(value in PD for value in substages),
        "professional_details_seen": any(value in PD for value in substages),
        "aip_approved_seen": not aip_rows.empty,
        "latest_aip_datetime": None if aip_rows.empty else aip_rows.event_datetime.max().isoformat(),
        "journey_close_seen": "Journey Close" in substages,
        "last_journey_datetime": latest.event_datetime.isoformat(),
        "ptp_generation": 0, "done": ptp or expired, "success": ptp,
        "termination_reason": ("PUSH_TO_PARTNER_SUCCESS" if ptp else
                               "APPLICATION_EXPIRED_30_DAYS" if expired else None),
        "terminal_datetime": (first_ptp.isoformat() if ptp else deadline.isoformat() if expired else None),
        "event_sequence": 0, "decision_sequence": 0, "pending_sequence": 0,
        "natural_generation": 0, "random_counter": 0, "audit_sequence": 0,
        "events_today": 0, "events_today_date": None, "recent_journey_times": [],
        "campaign_sends_seen": int(rows.journey_substage.eq("campaign_sent").sum()),
        "last_campaign_send_datetime": (None if not rows.journey_substage.eq("campaign_sent").any()
            else rows.loc[rows.journey_substage.eq("campaign_sent"), "event_datetime"].max().isoformat()),
        "observed_journey_substages": list(dict.fromkeys(substages)),
        "population_origin": "CUMULATIVE_GOLD_RECONSTRUCTION",
    }
    return EnvironmentState(payload)
