"""Decision-opportunity boundary for future intra-day scheduling."""
from __future__ import annotations
import pandas as pd

TODO_INTRA_DAY_CAMPAIGN_OPPORTUNITY_MODEL = "TODO_INTRA_DAY_CAMPAIGN_OPPORTUNITY_MODEL"

def get_decision_opportunities(state: object, start_time: str | pd.Timestamp,
                               end_time: str | pd.Timestamp) -> list[pd.Timestamp]:
    start, end = pd.Timestamp(start_time), pd.Timestamp(end_time)
    if start >= end or bool(getattr(state, "terminal", False)):
        return []
    # Current validated contract: one opportunity per active application-day.
    first = start.normalize()
    if first < start:
        first = start
    return [first] if first < end else []
