"""Business-approved AIP -> PTP chronology, scoped to an application."""
from __future__ import annotations

from hashlib import sha256
import json

import pandas as pd


RULE = {
    "version": "AIP_THEN_PTP_APPLICATION_V1",
    "authority": "business-owner clarification supplied 2026-09-24",
    "order": ["event_datetime", "source_file_order", "source_row"],
    "prior_aip": "latest preceding AIP Approved in the same uniquely owned application lifecycle",
    "equal_timestamp": "source file order and source row resolve ties",
    "deadline": "observed creation_datetime + 30 calendar days; inclusive for PTP",
    "terminal": "first valid PTP only; application terminal, context reusable",
    "status_fields": "secondary evidence; never eligibility filters",
}
MAPPING_SHA256 = sha256(json.dumps(RULE, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
VALID = "VALID_PROGRESSION_PTP"
CAMPAIGN = {"campaign_sent", "campaign_open_read", "campaign_clicked"}


def classify_ptp(events: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Classify every candidate, retaining duplicates and outliers for audit.

    Missing creation/time or conflicting application ownership cannot establish
    a valid lifecycle and is reported as ambiguous with an explicit reason.
    """
    owners = cohort.set_index("application_id")
    ordered = events.sort_values(
        ["application_id", "event_datetime", "source_file_order", "source_row"],
        kind="stable", na_position="last",
    )
    records = []
    for app, group in ordered.groupby("application_id", sort=False):
        owner = owners.loc[app]
        context = owner.context_id
        creation = pd.Timestamp(owner.creation_datetime)
        unique_owner = pd.notna(context) and group.context_id.dropna().nunique() == 1
        latest_aip = None
        terminal = False
        previous_journey = None
        for row in group.itertuples(index=False):
            timestamp = pd.Timestamp(row.event_datetime)
            state = row.journey_substage
            if state == "AIP Approved" and pd.notna(creation) and pd.notna(timestamp) and timestamp >= creation:
                latest_aip = timestamp
            if state == "Push to Partner":
                reason = None
                if not unique_owner or pd.isna(creation) or pd.isna(timestamp) or timestamp < creation:
                    classification = "PTP_AMBIGUOUS_APPLICATION"
                    reason = ("NON_UNIQUE_APPLICATION_OWNERSHIP" if not unique_owner else
                              "MISSING_OBSERVED_CREATION" if pd.isna(creation) else
                              "MISSING_EVENT_TIME" if pd.isna(timestamp) else "PTP_BEFORE_CREATION")
                elif terminal:
                    classification = "POST_TERMINAL_PTP"
                elif timestamp > creation + pd.Timedelta(days=30):
                    classification = "PTP_AFTER_EXPIRY"
                elif latest_aip is None:
                    classification = "PTP_WITHOUT_PRIOR_AIP"
                else:
                    classification = VALID
                    terminal = True
                records.append({
                    "application_id": app, "context_id": row.context_id,
                    "event_datetime": timestamp, "source_file_order": row.source_file_order,
                    "source_row": row.source_row, "classification": classification, "reason": reason,
                    "latest_prior_aip_datetime": latest_aip,
                    "aip_to_ptp_seconds": None if latest_aip is None else (timestamp - latest_aip).total_seconds(),
                    "directly_after_aip": previous_journey == "AIP Approved",
                    "event_name": row.event_name, "journey_stage": row.journey_stage,
                    "journey_substage": state, "source_type": row.source_type,
                    "initiated_by": row.initiated_by, "trigger_type": row.trigger_type,
                })
            if state not in CAMPAIGN and pd.notna(state):
                previous_journey = state
    return pd.DataFrame(records)


def lifecycle_rows(events: pd.DataFrame, cohort: pd.DataFrame, classified: pd.DataFrame) -> pd.DataFrame:
    """Retain all application-owned historical PTP, regardless of AIP telemetry.

    `classified` supplies evidence quality only; it cannot remove a PTP.
    Observed creation and the 30-day application window remain required for
    constructing a timed lifecycle, while the audit retains every candidate.
    """
    data = events.merge(cohort[["application_id", "creation_datetime", "context_id"]],
                        on="application_id", suffixes=("", "_owner"), validate="many_to_one")
    created = pd.to_datetime(data.creation_datetime)
    first_ptp = (data.loc[data.journey_substage.eq("Push to Partner")]
                 .sort_values(["application_id", "event_datetime", "source_file_order", "source_row"], kind="stable")
                 .drop_duplicates("application_id").set_index("application_id"))
    terminal_time = data.application_id.map(first_ptp.event_datetime)
    terminal_file = data.application_id.map(first_ptp.source_file_order)
    terminal_row = data.application_id.map(first_ptp.source_row)
    before_terminal = (terminal_time.isna() | data.event_datetime.lt(terminal_time) |
        (data.event_datetime.eq(terminal_time) &
         (data.source_file_order.lt(terminal_file) |
          (data.source_file_order.eq(terminal_file) & data.source_row.le(terminal_row)))))
    eligible = (data.context_id_owner.notna() & data.event_datetime.ge(created) &
                data.event_datetime.le(created + pd.Timedelta(days=30)) & before_terminal)
    return data.loc[eligible].drop(columns=["creation_datetime", "context_id_owner"])
