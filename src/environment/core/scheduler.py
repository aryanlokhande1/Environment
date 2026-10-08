"""Decision-opportunity boundary with an evidence-safe daily fallback."""
from __future__ import annotations
import pandas as pd

TODO_INTRA_DAY_CAMPAIGN_OPPORTUNITY_MODEL = "TODO_INTRA_DAY_CAMPAIGN_OPPORTUNITY_MODEL"

def get_decision_opportunities(state: object, start_time: str | pd.Timestamp,
                               end_time: str | pd.Timestamp) -> list[pd.Timestamp]:
    start, end = pd.Timestamp(start_time), pd.Timestamp(end_time)
    if start.tzinfo is not None or end.tzinfo is not None:
        raise ValueError("decision opportunity timestamps must be timezone-naive")
    if start >= end or bool(getattr(state, "terminal", False)):
        return []
    payload = getattr(state, "payload", {})
    if payload.get("_runtime_version") == "corrected-v2":
        raise ValueError("corrected-v2 decisions are caller-controlled; supply next_decision_time")
    activation_value = payload.get("activation_datetime", payload.get("creation_datetime"))
    deadline_value = payload.get("deadline")
    active_start = max(start, pd.Timestamp(activation_value)) if activation_value else start
    active_end = min(end, pd.Timestamp(deadline_value)) if deadline_value else end
    terminal_value = payload.get("terminal_datetime")
    if terminal_value:
        active_end = min(active_end, pd.Timestamp(terminal_value))
    if active_start >= active_end:
        return []

    # The bundle supports exactly one opportunity on each active application
    # day. On the activation/partial first day it occurs at the visible window
    # start; later days use midnight. Multiple intra-day points require a
    # separately versioned empirical opportunity artifact.
    points: list[pd.Timestamp] = []
    day = active_start.normalize()
    while day < active_end:
        point = max(day, active_start)
        if point < active_end:
            points.append(point)
        day += pd.Timedelta(days=1)
    return points
