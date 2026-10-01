"""Resumable one-day Personal Loan environment; no database or future visibility.

The runner commits the serializable state/pending queue after each day. Policy
is an optional callback over current visible state, so a later RL policy does
not need to change the environment transition code.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from hashlib import sha256
import heapq
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import (CampaignAction, PersonalLoanSimulator, PolicyChoice,
                        can_approve_aip, can_push_to_partner, resolve_bucket_time,
                        stable_seed, EVENT_PRIORITY, EXPIRY_REASON, SUCCESS_REASON,
                        PRODUCT_SCOPE, TRANSACTION_EVENT)
from .event_intensity import EmpiricalEventIntensity
from .hazard import StageContinuationModel
from .progression import VALID
from .campaign_decision import CampaignDecisionPolicy

DAY_END = pd.Timestamp("2026-06-01")


def _timestamp(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value)


def _iso(value: Any) -> str:
    return _timestamp(value).isoformat()


class DailyClosedLoopEnvironment:
    def __init__(self, artifact_dir: Path, *, seed: int, run_id: str,
                 daily_send_probability: float | None = None,
                 initial_limit: int | None = None, arrival_limit: int | None = None):
        self.artifact_dir = Path(artifact_dir)
        self.seed = int(seed)
        self.run_id = str(run_id)
        if initial_limit is not None and initial_limit < 1:
            raise ValueError("initial_limit must be positive")
        if arrival_limit is not None and arrival_limit < 1:
            raise ValueError("arrival_limit must be positive")
        self.initial_limit = initial_limit
        self.arrival_limit = arrival_limit
        self.sim = PersonalLoanSimulator(self.artifact_dir, seed=self.seed,
                                         start="2026-05-01", end="2026-07-01")
        self.ptp_model = self.sim.ptp_hazard or self.sim.ptp_model
        self.continuation = (StageContinuationModel(self.artifact_dir)
                             if (self.artifact_dir / "stage_continuation_manifest.json").exists()
                             else None)
        self.intensity = EmpiricalEventIntensity(self.artifact_dir)
        self.max_events_per_day = int(self.intensity.manifest["max_observed_events_per_active_day"])
        self.max_events_per_ten_minute_bin = int(self.intensity.manifest["max_observed_events_per_ten_minute_bin"])
        decision_manifest_path = self.artifact_dir / "campaign_decision_policy_manifest.json"
        self.campaign_decision = (CampaignDecisionPolicy(self.artifact_dir)
                                  if decision_manifest_path.exists() else None)
        opportunity = (self.campaign_decision.manifest if self.campaign_decision is not None else
                       json.loads((self.artifact_dir / "empirical_campaign_opportunity.json").read_text(encoding="utf-8")))
        if "no May" not in opportunity["source_period"]:
            raise ValueError("daily campaign decision policy may fit only immutable history")
        # Training-support tables remain in the research archive. Runtime
        # integrity is established by the frozen bundle manifest and each
        # model's own source/model hashes; raw fitting inputs are not required.
        if self.intensity.manifest["source_cohort_sha256"] != opportunity["source_cohort_sha256"]:
            raise ValueError("event intensity and campaign decision policy use different PL cohorts")
        legacy_probability = opportunity.get("daily_send_probability", opportunity.get("global_send_probability"))
        probability = legacy_probability if daily_send_probability is None else daily_send_probability
        if not 0 <= float(probability) <= 1:
            raise ValueError("daily send probability override must be within [0,1]")
        self.daily_send_probability = float(probability)
        self.daily_send_probability_override = daily_send_probability
        self.opportunity_manifest = opportunity
        arrivals_path = self.artifact_dir / "may_application_arrivals.parquet"
        manifest = json.loads((self.artifact_dir / "may_arrival_manifest.json").read_text(encoding="utf-8"))
        if sha256(arrivals_path.read_bytes()).hexdigest() != manifest["schedule_sha256"] or manifest["seed"] != self.seed:
            raise ValueError("empirical May arrival schedule or seed mismatch")
        self.arrivals = pd.read_parquet(arrivals_path)
        if arrival_limit is not None:
            self.arrivals = self.arrivals.assign(_rank=self.arrivals.application_id.map(
                lambda value: sha256(f"{self.seed}|{value}".encode()).hexdigest()))
            self.arrivals = self.arrivals.sort_values("_rank", kind="stable").head(arrival_limit).drop(columns="_rank")

    def arrivals_for_day(self, day: pd.Timestamp) -> list[dict[str, Any]]:
        return self.arrivals.loc[pd.to_datetime(self.arrivals.activation_datetime).dt.normalize().eq(day.normalize())].to_dict("records")

    def initial_population(self) -> list[dict[str, Any]]:
        return self.sim.cohort(self.initial_limit)

    def new_state(self, raw: Mapping[str, Any], *, activation: pd.Timestamp,
                  context_sequence: int = 0) -> dict[str, Any]:
        creation = _timestamp(raw["application_created_at"])
        deadline = creation + pd.Timedelta(days=30)
        if not creation <= activation < deadline:
            raise ValueError("application activation outside observed 30-day lifecycle")
        aip = raw.get("latest_aip_datetime")
        aip = None if aip is None or pd.isna(aip) else _timestamp(aip)
        observed = raw.get("observed_journey_substages", [])
        if observed is None or (isinstance(observed, float) and pd.isna(observed)):
            observed = []
        observed = list(map(str, observed))
        snapshot_substage = str(raw["snapshot_substage"])
        if snapshot_substage not in observed:
            observed.append(snapshot_substage)
        return {
            "context_id": str(raw["snapshot_context_id"]), "application_id": str(raw["application_id"]),
            "creation_datetime": _iso(creation), "activation_datetime": _iso(activation),
            "deadline": _iso(deadline), "journey_stage": str(raw["snapshot_stage"]),
            "journey_substage": snapshot_substage,
            "form_filled_seen": bool(raw.get("form_filled_seen", False))
                or bool(raw.get("professional_details_seen", False)) or aip is not None,
            "professional_details_seen": bool(raw.get("professional_details_seen", False)) or aip is not None,
            "aip_approved_seen": aip is not None,
            "latest_aip_datetime": None if aip is None else _iso(aip),
            "journey_close_seen": bool(raw.get("journey_close_seen", False)),
            "last_journey_datetime": _iso(raw.get("snapshot_event_datetime", activation)),
            "ptp_generation": 0,
            "done": False, "success": False, "termination_reason": None,
            "terminal_datetime": None,
            "event_sequence": int(context_sequence), "decision_sequence": 0,
            "pending_sequence": 0, "natural_generation": 0, "random_counter": 0,
            "audit_sequence": 0,
            "events_today": 0, "events_today_date": None,
            "recent_journey_times": [],
            "campaign_sends_seen": int(raw.get("campaign_sends_seen", 0)),
            "last_campaign_send_datetime": (None if raw.get("last_campaign_send_datetime") is None
                                              or pd.isna(raw.get("last_campaign_send_datetime")) else
                                              _iso(raw.get("last_campaign_send_datetime"))),
            "observed_journey_substages": observed,
            "population_origin": str(raw.get("population_origin", "MAY1_ACTIVE_COHORT")),
        }

    def _rng(self, state: dict, purpose: str) -> np.random.Generator:
        state["random_counter"] += 1
        value = stable_seed(self.seed, state["application_id"], purpose, state["random_counter"])
        state["last_rng_reference"] = {"purpose": purpose, "counter": state["random_counter"],
                                       "stable_seed": value}
        return np.random.default_rng(value)

    def _audit(self, state: dict, when: pd.Timestamp, category: str, explanation: dict[str, Any],
               *, decision_id: str | None = None, event_key: str | None = None) -> None:
        state["audit_sequence"] += 1
        self.audit_records.append({
            "run_id": self.run_id,
            "explanation_id": f"{self.run_id}|{state['application_id']}|{state['audit_sequence']}",
            "application_id": state["application_id"], "context_id": state["context_id"],
            "explanation_time": _timestamp(when), "category": category,
            "decision_id": decision_id, "event_key": event_key,
            "explanation": {
                "state": {key: state.get(key) for key in (
                    "journey_stage", "journey_substage", "creation_datetime", "deadline",
                    "form_filled_seen", "professional_details_seen", "aip_approved_seen",
                    "latest_aip_datetime")},
                **explanation,
                "rng_reference": state.get("last_rng_reference"),
            },
        })

    @staticmethod
    def _schedule(state: dict, pending: list[dict], when: Any, kind: str,
                  payload: dict[str, Any] | None = None) -> None:
        when = _timestamp(when)
        if when > _timestamp(state["deadline"]):
            return
        state["pending_sequence"] += 1
        pending.append({"when": _iso(when), "priority": EVENT_PRIORITY[kind],
                        "order": state["pending_sequence"], "kind": kind,
                        "payload": payload or {}})

    def _schedule_natural(self, state: dict, pending: list[dict], now: pd.Timestamp,
                          horizon: pd.Timestamp) -> None:
        state["natural_generation"] += 1
        lookup = (self.sim.natural_regime.lookup(state)
                  if self.sim.natural_regime is not None else None)
        options = list(lookup.options if lookup is not None else
                       self.sim.natural.get(state["journey_substage"], ()))
        if not options:
            self._audit(state, now, "ORGANIC_CONTINUATION", {
                "pathway": "ORGANIC", "selected_outcome": "NO_EVENT",
                "reason": "NO_BUSINESS_VALID_NATURAL_SUCCESSOR",
                "business_guards": {"can_approve_aip": can_approve_aip(state),
                                    "can_push_to_partner": can_push_to_partner(state)},
            })
            return
        rng = self._rng(state, "natural_occurrence_and_state")
        if self.continuation is not None:
            occurrence = self.continuation.sample_window(state, now, horizon, rng)
            if occurrence.when is None:
                self._audit(state, now, "ORGANIC_CONTINUATION", {
                    "pathway": "ORGANIC", "selected_outcome": "TEMPORARY_INACTIVITY",
                    "model": "stage_continuation", "model_version": self.continuation.manifest["model_version"],
                    "model_hash": self.continuation.manifest["hazard_sha256"],
                    **occurrence.evidence,
                    "business_guards": {"application_remains_active": True},
                })
                return
            when, level = occurrence.when, occurrence.evidence["model_level"]
        else:
            wait_us, level = self.intensity.sample_wait(state["journey_substage"],
                                                         max(0, state["events_today"] - 1), rng)
            if wait_us is None:
                return
            when = now + pd.Timedelta(microseconds=wait_us)
        weights = np.asarray([float(item["probability"]) for item in options])
        weights /= weights.sum()
        transition_draw = float(rng.random())
        selected_index = min(int(np.searchsorted(np.cumsum(weights), transition_draw, side="right")), len(options) - 1)
        selected = options[selected_index]
        blocked_reason = ("ACTIVE_LIFECYCLE_ALREADY_CREATED" if selected["response"] == "Application Created" else
                          "PTP_NATURAL_TRANSITION_FORBIDDEN" if selected["response"] == "Push to Partner" else
                          "AIP_PREREQUISITE_NOT_OBSERVED" if selected["response"] == "AIP Approved"
                          and not can_approve_aip(state) else None)
        if blocked_reason is not None:
            self._audit(state, now, "ORGANIC_CONTINUATION", {
                "pathway": "ORGANIC", "model": "stage_continuation+natural_transition",
                "model_version": (None if self.sim.natural_regime is None else
                                  self.sim.natural_regime.manifest["model_version"]),
                "model_hash": (self.sim.model_manifest["natural_model_sha256"] if self.sim.natural_regime is None else
                               self.sim.natural_regime.manifest["model_sha256"]),
                "conditioning_level": "STATE" if lookup is None else lookup.model_level,
                "regime": None if lookup is None else lookup.regime,
                "backoff_level": None if lookup is None else lookup.backoff_level,
                "support_count": int(selected["support_count"]),
                "candidate_distribution": [{"response": str(item["response"]),
                                            "probability": float(weight)}
                                           for item, weight in zip(options, weights)],
                "transition_rng_draw": transition_draw,
                "selected_outcome": "TEMPORARY_INACTIVITY",
                "blocked_candidate": str(selected["response"]), "reason": blocked_reason,
                "occurrence_evidence": occurrence.evidence if self.continuation else None,
                "business_guards": {"can_approve_aip": can_approve_aip(state),
                                    "can_push_to_partner": can_push_to_partner(state),
                                    "blocked_mass_renormalized": False},
            })
            return
        if when > _timestamp(state["deadline"]) or (when == _timestamp(state["deadline"])
                                                    and selected["response"] != "Push to Partner"):
            return
        kind = TRANSACTION_EVENT if selected["response"] == "Push to Partner" else "JOURNEY_EVENT"
        self._schedule(state, pending, when, kind, {
            "response": str(selected["response"]), "natural_generation": state["natural_generation"],
            "support_count": int(selected["support_count"]), "model_level": str(selected["model_level"]),
            "intensity_level": level,
        })

        self._audit(state, now, "ORGANIC_CONTINUATION", {
            "pathway": "ORGANIC", "model": "stage_continuation+natural_transition",
            "model_version": (self.continuation.manifest["model_version"] if self.continuation else
                              self.intensity.manifest["model_version"]),
            "model_hash": (self.continuation.manifest["hazard_sha256"] if self.continuation else
                           self.intensity.manifest["model_sha256"]),
            "transition_model_version": (self.sim.model_manifest.get("model_version")
                                         if self.sim.natural_regime is None else
                                         self.sim.natural_regime.manifest["model_version"]),
            "transition_model_hash": (self.sim.model_manifest["natural_model_sha256"]
                                      if self.sim.natural_regime is None else
                                      self.sim.natural_regime.manifest["model_sha256"]),
            "conditioning_level": "STATE" if lookup is None else lookup.model_level,
            "regime": None if lookup is None else lookup.regime,
            "backoff_level": None if lookup is None else lookup.backoff_level,
            "support_count": (int(selected["outgoing_transition_count"])
                              if "outgoing_transition_count" in selected else int(selected["support_count"])),
            "candidate_distribution": [{"response": str(item["response"]),
                                        "probability": float(weight)}
                                       for item, weight in zip(options, weights)],
            "transition_rng_draw": transition_draw,
            "selected_outcome": str(selected["response"]),
            "sampled_event_time": _iso(when),
            "occurrence_evidence": occurrence.evidence if self.continuation else None,
            "business_guards": {"can_approve_aip": can_approve_aip(state),
                                "can_push_to_partner": can_push_to_partner(state)},
        })

    def _schedule_ptp_outcome(self, state: dict, pending: list[dict], now: pd.Timestamp,
                              horizon: pd.Timestamp) -> None:
        if self.ptp_model is None or not can_push_to_partner(state):
            return
        if any(item["kind"] == TRANSACTION_EVENT and item["payload"].get("ptp_outcome")
               and item["payload"].get("ptp_generation") == state["ptp_generation"] for item in pending):
            return
        rng = self._rng(state, "ptp_hazard")
        if hasattr(self.ptp_model, "sample_window"):
            result = self.ptp_model.sample_window(state, now, horizon, rng)
            when, evidence = result.when, result.evidence
        else:
            when = self.ptp_model.sample_time(state, now, self.seed)
            evidence = {"reason": "LEGACY_ONE_DRAW", "support_count": self.ptp_model.denominator,
                        "model_level": self.ptp_model.manifest["model_version"]}
        self._audit(state, now, "PTP_HAZARD", {
            "pathway": "TERMINAL_OUTCOME", "model": "ptp_hazard",
            "model_version": self.ptp_model.manifest["model_version"],
            "model_hash": self.ptp_model.manifest.get("hazard_sha256",
                                                       self.ptp_model.manifest.get("gap_pool_sha256")),
            "selected_outcome": "PUSH_TO_PARTNER" if when is not None else "NO_PTP_THIS_WINDOW",
            "sampled_event_time": None if when is None else _iso(when),
            "hazard_evidence": evidence,
            "business_guards": {"can_push_to_partner": True,
                                "same_application_prerequisites": True},
        })
        if when is not None:
            self._schedule(state, pending, when, TRANSACTION_EVENT, {
                "response": "Push to Partner", "ptp_outcome": True,
                "ptp_generation": state["ptp_generation"],
                "support_count": int(evidence.get("support_count", 0)),
                "model_level": str(evidence.get("model_level", self.ptp_model.manifest["model_version"])),
            })

    def _emit(self, state: dict, when: pd.Timestamp, kind: str, **fields: Any) -> dict[str, Any]:
        state["event_sequence"] += 1
        if state["events_today_date"] != when.date().isoformat():
            state["events_today_date"] = when.date().isoformat()
            state["events_today"] = 0
        if kind in {"JOURNEY_EVENT", "ACTION_CONDITIONED_JOURNEY_EVENT", TRANSACTION_EVENT}:
            state["events_today"] += 1
            if state["events_today"] > self.max_events_per_day:
                raise RuntimeError("simulated daily journey intensity exceeds observed PL maximum")
            recent = [value for value in state["recent_journey_times"]
                      if _timestamp(value).floor("10min") == when.floor("10min")]
            recent.append(_iso(when))
            state["recent_journey_times"] = recent
            if len(recent) > self.max_events_per_ten_minute_bin:
                raise RuntimeError("simulated ten-minute journey intensity exceeds observed PL maximum")
        return {
            "event_key": (f"{self.run_id}|{state['context_id']}|"
                          f"{state['application_id']}|{state['event_sequence']}"),
            "run_id": self.run_id, "context_id": state["context_id"],
            "application_id": state["application_id"], "product_scope": PRODUCT_SCOPE,
            "creation_datetime": _timestamp(state["creation_datetime"]), "event_datetime": when,
            "event_sequence": state["event_sequence"], "event_type": kind,
            "journey_stage": state["journey_stage"], "journey_substage": state["journey_substage"],
            "decision_id": fields.get("decision_id"), "action_id": fields.get("action_id"),
            "channel": fields.get("channel"), "theme": fields.get("theme"),
            "time_bucket": fields.get("time_bucket"),
            "decision_time": fields.get("decision_time"), "resolved_send_time": fields.get("resolved_send_time"),
            "trigger_type": fields.get("trigger_type"), "policy_source": fields.get("policy_source"),
            "source_type": fields.get("source_type"), "model_source": fields.get("model_source"),
            "model_level": fields.get("model_level"), "support_count": fields.get("support_count"),
            "transition_mode": fields.get("transition_mode"), "reward": None, "cost": None,
            "success": state["success"], "done": state["done"],
            "termination_reason": state["termination_reason"],
            "latest_aip_datetime": (None if state["latest_aip_datetime"] is None else
                                    _timestamp(state["latest_aip_datetime"])),
            "ptp_classification": VALID if kind == TRANSACTION_EVENT else None,
            "is_synthetic": False,
        }

    def process_day(self, state: dict[str, Any], pending: list[dict[str, Any]], day: pd.Timestamp,
                    *, newly_activated: bool = False,
                    policy: Callable[[Mapping[str, Any], pd.Timestamp], PolicyChoice] | None = None,
                    decision_time: pd.Timestamp | None = None,
                    horizon: pd.Timestamp | None = None,
                    action_at_decision_time: bool = False,
                    ) -> tuple[list[dict], list[dict], dict[str, dict]]:
        day = day.normalize()
        if not pd.Timestamp("2026-05-01") <= day < DAY_END:
            raise ValueError("daily environment may execute May 1-31 only")
        following = day + pd.Timedelta(days=1) if horizon is None else _timestamp(horizon)
        if following.tzinfo is not None or not day < following <= DAY_END:
            raise ValueError("simulation horizon must be timezone-naive, after day start, and no later than June 1")
        events: list[dict] = []
        decisions: list[dict] = []
        actions: dict[str, dict] = {}
        self.audit_records: list[dict] = []
        if state["done"]:
            raise ValueError("terminal application cannot be processed again")
        state.setdefault("audit_sequence", 0)
        state.setdefault("ptp_generation", 0)
        state.setdefault("last_journey_datetime", state.get("activation_datetime", day.isoformat()))
        if state["events_today_date"] != day.date().isoformat():
            state["events_today_date"], state["events_today"] = day.date().isoformat(), 0
        requested = day if decision_time is None else _timestamp(decision_time)
        if requested.tzinfo is not None or requested.normalize() != day or requested >= following:
            raise ValueError("decision_time must be timezone-naive and within the simulation day")
        now = max(requested, _timestamp(state["activation_datetime"])) if newly_activated else requested
        if newly_activated:
            if state["population_origin"] in {
                "EMPIRICAL_MAY_ARRIVAL", "EMPIRICAL_MAY_ARRIVAL_V2"
            }:
                events.append(self._emit(state, now, "JOURNEY_EVENT", source_type="arrival",
                                         model_source="PL_EMPIRICAL_MAY_ARRIVAL"))
            self._schedule(state, pending, state["deadline"], "TERMINAL_EVENT")
        # Reevaluate conditional continuation and PTP risk every active day.
        # A no-event draw is temporary inactivity, never permanent failure.
        has_natural = any(item["kind"] in {"JOURNEY_EVENT", "ACTION_CONDITIONED_JOURNEY_EVENT"}
                          or item["kind"] == "CAMPAIGN_MOTIF_RELEASE_EVENT" for item in pending)
        if not has_natural:
            self._schedule_natural(state, pending, now, following)
        self._schedule_ptp_outcome(state, pending, now, following)
        # Exactly one opportunity decision per active application-day. The
        # callback sees committed visible state, never the pending queue.
        if now < _timestamp(state["deadline"]):
            state["decision_sequence"] += 1
            decision_id = f"DAILY|{self.run_id}|{state['context_id']}|{state['decision_sequence']}"
            pending_send = any(item["kind"] == "CAMPAIGN_SEND_EVENT" for item in pending)
            if policy is None:
                rng = np.random.default_rng(stable_seed(self.seed, state["application_id"],
                                                        "campaign_decision", day.date().isoformat()))
                if pending_send:
                    choice = PolicyChoice(None, policy_source="EMPIRICAL_SEND_NO_ACTION_POLICY")
                    opportunity_evidence = {
                        "model": "campaign_decision_policy", "opportunity": "ACTIVE_APPLICATION_DAY",
                        "selected_outcome": "NO_ACTION", "reason": "PENDING_SEND",
                        "support_count": None, "backoff_level": "NOT_SAMPLED_PENDING_SEND",
                    }
                else:
                    if self.daily_send_probability_override is not None or self.campaign_decision is None:
                        draw = float(rng.random())
                        send = draw < self.daily_send_probability
                        opportunity_evidence = {
                            "model": "campaign_decision_policy", "opportunity": "ACTIVE_APPLICATION_DAY",
                            "conditioning_level": "CONFIG_OVERRIDE" if self.daily_send_probability_override is not None else "GLOBAL_LEGACY",
                            "support_count": self.opportunity_manifest.get("eligible_application_days"),
                            "candidate_distribution": {"SEND": self.daily_send_probability,
                                                       "NO_ACTION": 1.0 - self.daily_send_probability},
                            "rng_draw": draw, "selected_outcome": "SEND" if send else "NO_ACTION",
                            "backoff_level": "CONFIG_OVERRIDE" if self.daily_send_probability_override is not None else "GLOBAL",
                        }
                    else:
                        sampled_decision = self.campaign_decision.sample(state, now, rng)
                        send, opportunity_evidence = sampled_decision.send, sampled_decision.evidence
                    if not send:
                        choice = PolicyChoice(None, policy_source="EMPIRICAL_SEND_NO_ACTION_POLICY")
                    else:
                        action, level = self.sim.empirical_policy.sample(
                            state, state["context_id"], state["application_id"], state["decision_sequence"])
                        choice = PolicyChoice(action, policy_source=f"EMPIRICAL_SEND_{level}")
            else:
                choice = policy(dict(state), now)
                opportunity_evidence = {
                    "model": "deterministic_active_day_opportunity",
                    "opportunity": "ACTIVE_APPLICATION_DAY", "support_count": None,
                    "selected_outcome": "EXTERNAL_POLICY_DECISION", "backoff_level": "NOT_APPLICABLE",
                }
            action = choice.action
            resolved = (None if action is None else now if action_at_decision_time and now < _timestamp(state["deadline"])
                        else resolve_bucket_time(
                            now, action.time_bucket, _timestamp(state["deadline"]),
                            np.random.default_rng(stable_seed(self.seed, state["application_id"],
                                                              state["decision_sequence"], "send_bucket"))))
            decision = {
                "run_id": self.run_id, "decision_id": choice.decision_id or decision_id,
                "decision_sequence": state["decision_sequence"], "context_id": state["context_id"],
                "application_id": state["application_id"], "product_scope": PRODUCT_SCOPE,
                "decision_time": now, "action_id": None if action is None else action.action_id,
                "channel": None if action is None else action.channel,
                "theme": None if action is None else action.theme,
                "time_bucket": None if action is None else action.time_bucket,
                "resolved_send_time": resolved, "policy_source": choice.policy_source,
                "status": "NO_ACTION" if action is None else "UNSCHEDULABLE_BEFORE_EXPIRY" if resolved is None else "SCHEDULED",
                "eligible": True, "realized_send_time": None,
                "suppression_reason": None,
                "reason": "PENDING_SEND" if action is None and pending_send else None,
                "reward": None, "cost": None, "is_synthetic": False,
            }
            decisions.append(decision)
            self._audit(state, now, "RL_DECISION", {
                "action": {"type": "NO_ACTION" if action is None else "campaign_sent",
                           "channel_id": None if action is None else action.channel,
                           "theme": None if action is None else action.theme,
                           "time_bucket": None if action is None else action.time_bucket},
                "policy_source": choice.policy_source, "decision_status": decision["status"],
                "resolved_send_time": None if resolved is None else _iso(resolved),
                "pending_send": pending_send,
                "opportunity_evidence": opportunity_evidence,
            }, decision_id=decision["decision_id"])
            if action is not None:
                actions[action.action_id] = {"action_id": action.action_id, "channel": action.channel,
                                             "theme": action.theme, "time_bucket": action.time_bucket}
                if resolved is not None:
                    self._schedule(state, pending, resolved, "CAMPAIGN_SEND_EVENT", {
                        "decision_id": decision["decision_id"], "action_id": action.action_id,
                        "channel": action.channel, "theme": action.theme, "time_bucket": action.time_bucket,
                        "decision_time": _iso(now), "resolved_send_time": _iso(resolved),
                        "policy_source": decision["policy_source"],
                    })
        while pending and not state["done"]:
            heap = [(_timestamp(item["when"]).value, item["priority"], item["order"], index)
                    for index, item in enumerate(pending)]
            _, _, _, index = min(heap)
            item = pending[index]
            when = _timestamp(item["when"])
            if when >= following:
                break
            pending.pop(index)
            kind, payload = item["kind"], item["payload"]
            if (kind in {"JOURNEY_EVENT", "ACTION_CONDITIONED_JOURNEY_EVENT", TRANSACTION_EVENT}
                    and not payload.get("ptp_outcome")
                    and payload.get("natural_generation") != state["natural_generation"]):
                continue
            if (kind == TRANSACTION_EVENT and payload.get("ptp_outcome")
                    and payload.get("ptp_generation") != state["ptp_generation"]):
                continue
            if kind == "TERMINAL_EVENT":
                suppressed = [row for row in pending if row["kind"] == "CAMPAIGN_SEND_EVENT"]
                for row in suppressed:
                    suppressed_decision = row["payload"].get("decision_id")
                    for decision in decisions:
                        if decision["decision_id"] == suppressed_decision:
                            decision["status"] = "SUPPRESSED_BY_TERMINAL"
                            decision["suppression_reason"] = EXPIRY_REASON
                    self._audit(state, when, "CAMPAIGN_SUPPRESSION", {
                        "scheduled_send_time": row["when"],
                        "selected_outcome": "SUPPRESSED",
                        "reason": EXPIRY_REASON,
                    }, decision_id=suppressed_decision)
                state.update(done=True, success=False, termination_reason=EXPIRY_REASON)
                state["terminal_datetime"] = _iso(when)
                events.append(self._emit(state, when, kind, source_type="lifecycle",
                                         model_source="OBSERVED_CREATION_PLUS_30_CALENDAR_DAYS"))
                pending.clear()
                break
            if kind == "CAMPAIGN_SEND_EVENT":
                decision_time = _timestamp(payload["decision_time"])
                # A new exposure censors unresolved responses attributed to an
                # earlier send. Those outcomes are unknown, not permanent
                # no-response observations.
                prior_exposure = [row for row in pending
                    if row["kind"] in {"CAMPAIGN_RESPONSE_EVENT", "ACTION_CONDITIONED_JOURNEY_EVENT",
                                       "CAMPAIGN_MOTIF_RELEASE_EVENT"}
                    and row.get("payload", {}).get("decision_id")
                    and row["payload"].get("decision_id") != payload.get("decision_id")]
                if prior_exposure:
                    censored_ids = sorted({row["payload"]["decision_id"] for row in prior_exposure})
                    pending[:] = [row for row in pending if row not in prior_exposure]
                    self._audit(state, when, "CAMPAIGN_RESPONSE_CENSORING", {
                        "selected_outcome": "CENSORED_BY_COMPETING_SEND",
                        "censored_decision_ids": censored_ids,
                        "censored_pending_events": len(prior_exposure),
                        "reason": "COMPETING_SEND_BOUNDARY",
                    }, decision_id=payload.get("decision_id"))
                events.append(self._emit(state, when, kind, **{**payload,
                    "decision_time": decision_time, "resolved_send_time": when,
                    "source_type": "campaign",
                    "model_source": ("EXTERNAL_RL_ACTION" if payload.get("policy_source") == "EXTERNAL_RL_ACTION"
                                     else "EMPIRICAL_DAILY_POLICY")}))
                state["campaign_sends_seen"] += 1
                state["last_campaign_send_datetime"] = _iso(when)
                for decision in decisions:
                    if decision["decision_id"] == payload.get("decision_id"):
                        decision["status"] = "REALIZED"
                        decision["realized_send_time"] = when
                next_journey_time = None
                if self.sim.campaign_motif is not None:
                    forbidden = set()
                    forbidden.add("Application Created")
                    if not can_approve_aip(state):
                        forbidden.add("AIP Approved")
                    forbidden.add("Push to Partner")
                    motif = self.sim.campaign_motif.sample(
                        state, payload["channel"], payload["time_bucket"],
                        self._rng(state, "campaign_motif"), forbidden)
                    state["natural_generation"] += 1
                    for label, delay in motif.engagement_events:
                        response_time = when + pd.Timedelta(seconds=delay)
                        if response_time < _timestamp(state["deadline"]):
                            self._schedule(state, pending, response_time, "CAMPAIGN_RESPONSE_EVENT", {
                                **payload, "response": label, "motif": True})
                    if motif.next_journey_substage is not None and motif.journey_delay_seconds is not None:
                        next_journey_time = when + pd.Timedelta(seconds=motif.journey_delay_seconds)
                        if next_journey_time < _timestamp(state["deadline"]):
                            self._schedule(state, pending, next_journey_time,
                                           "ACTION_CONDITIONED_JOURNEY_EVENT", {
                                **payload, "response": motif.next_journey_substage,
                                "natural_generation": state["natural_generation"],
                                "support_count": motif.evidence["support_count"],
                                "model_level": motif.evidence["model_level"], "motif": True,
                            })
                        else:
                            next_journey_time = None
                    if next_journey_time is None:
                        release = min(when + pd.Timedelta(seconds=motif.release_delay_seconds),
                                      _timestamp(state["deadline"]))
                        self._schedule(state, pending, release, "CAMPAIGN_MOTIF_RELEASE_EVENT", {
                            "natural_generation": state["natural_generation"], "motif": True,
                            "decision_id": payload["decision_id"]})
                    self._audit(state, when, "CAMPAIGN_RESPONSE_MOTIF", {
                        "pathway": "ACTION_CONDITIONED", "action": {
                            "channel_id": payload["channel"], "theme": payload["theme"],
                            "time_bucket": payload["time_bucket"]},
                        **motif.evidence,
                        "selected_outcome": {
                            "engagement_events": list(motif.engagement_events),
                            "next_journey_substage": motif.next_journey_substage,
                            "journey_delay_seconds": motif.journey_delay_seconds,
                            "release_delay_seconds": motif.release_delay_seconds},
                        "business_guards": {"forbidden_journey": sorted(forbidden)},
                    }, decision_id=payload["decision_id"])
                elif self.sim.action_model is not None:
                    forbidden = set()
                    forbidden.add("Application Created")
                    if not can_push_to_partner(state):
                        forbidden.add("Push to Partner")
                    if not can_approve_aip(state):
                        forbidden.add("AIP Approved")
                    if self.ptp_model is not None:
                        forbidden.add("Push to Partner")
                    sampled = self.sim.action_model.sample(payload["channel"], state["journey_substage"],
                                                           state["journey_stage"], self._rng(state, "action_response"),
                                                           forbidden,
                                                           preserve_blocked_mass=self.ptp_model is not None)
                    if sampled is not None:
                        outcome, evidence = sampled
                        response = str(outcome["next_journey_substage"])
                        next_journey_time = when + pd.Timedelta(seconds=int(evidence["sampled_delay_seconds"]))
                        if next_journey_time < _timestamp(state["deadline"]) or (next_journey_time == _timestamp(state["deadline"])
                                                                              and response == "Push to Partner"):
                            state["natural_generation"] += 1
                            conditioned_kind = TRANSACTION_EVENT if response == "Push to Partner" else "ACTION_CONDITIONED_JOURNEY_EVENT"
                            self._schedule(state, pending, next_journey_time, conditioned_kind, {
                                **payload, "response": response,
                                "natural_generation": state["natural_generation"],
                                "support_count": evidence["support_count"],
                                "model_level": evidence["model_level"],
                            })
                        else:
                            next_journey_time = None
                options = (None if self.sim.campaign_motif is not None else
                           self.sim.campaign_response.get(("STATE_CHANNEL", payload["channel"], state["journey_substage"])))
                if options is None and self.sim.campaign_motif is None:
                    options = self.sim.campaign_response.get(("STAGE_CHANNEL", payload["channel"], state["journey_stage"]))
                if options:
                    rng = self._rng(state, "campaign_engagement")
                    weights = np.asarray([float(option["probability"]) for option in options])
                    weights /= weights.sum()
                    selected = options[int(rng.choice(len(options), p=weights))]
                    if selected["response_event_type"] != "NO_OBSERVED_RESPONSE" and pd.notna(selected["median_seconds"]):
                        delay = max(1, int(round(np.interp(float(rng.random()), [0, .5, .9, .95, .99, 1],
                              [0, selected["median_seconds"], selected["p90_seconds"],
                               selected["p95_seconds"], selected["p99_seconds"], selected["p99_seconds"]]))))
                        response_time = when + pd.Timedelta(seconds=delay)
                        # Observed engagement associations precede the next
                        # journey event. A later sampled engagement is omitted.
                        next_natural = min((_timestamp(row["when"]) for row in pending
                                            if row["kind"] in {"JOURNEY_EVENT", "ACTION_CONDITIONED_JOURNEY_EVENT", TRANSACTION_EVENT}
                                            and row["payload"].get("natural_generation") == state["natural_generation"]),
                                           default=None)
                        next_boundary = min((time for time in (next_journey_time, next_natural)
                                             if time is not None), default=None)
                        if (response_time < _timestamp(state["deadline"])
                                and (next_boundary is None or response_time < next_boundary)):
                            self._schedule(state, pending, response_time, "CAMPAIGN_RESPONSE_EVENT", {
                                **payload, "response": selected["response_event_type"]})
                continue
            if kind == "CAMPAIGN_MOTIF_RELEASE_EVENT":
                if payload.get("natural_generation") == state["natural_generation"]:
                    self._schedule_natural(state, pending, when, following)
                continue
            if kind == "CAMPAIGN_RESPONSE_EVENT":
                response = str(payload["response"])
                events.append(self._emit(state, when, kind, **{**payload,
                    "decision_time": _timestamp(payload["decision_time"]),
                    "resolved_send_time": _timestamp(payload["resolved_send_time"]),
                    "model_level": response, "source_type": "campaign",
                    "model_source": "PL_OBSERVATIONAL_CAMPAIGN_RESPONSE"}))
                continue
            response = str(payload["response"])
            if response == "Application Created":
                self._audit(state, when, "ORGANIC_CONTINUATION", {
                    "pathway": "ORGANIC" if not payload.get("action_id") else "ACTION_CONDITIONED",
                    "selected_outcome": "TEMPORARY_INACTIVITY",
                    "blocked_candidate": response,
                    "reason": "ACTIVE_LIFECYCLE_ALREADY_CREATED",
                    "business_guards": {"blocked_mass_renormalized": False},
                }, decision_id=payload.get("decision_id"))
                self._schedule_natural(state, pending, when, following)
                continue
            if response == "Push to Partner" and not can_push_to_partner(state):
                raise RuntimeError("PTP without Form Filled, Professional Details and AIP")
            state["journey_substage"] = response
            state["journey_stage"] = self.sim.stage_map.get(response, state["journey_stage"])
            if response not in state["observed_journey_substages"]:
                state["observed_journey_substages"].append(response)
            if response == "Form filled":
                state["form_filled_seen"] = True
            elif response in {"Professional Details Submission", "Professional details"}:
                state["form_filled_seen"] = state["professional_details_seen"] = True
            elif response == "AIP Approved":
                state["aip_approved_seen"], state["latest_aip_datetime"] = True, _iso(when)
                state["ptp_generation"] += 1
            elif response == "Journey Close":
                state["journey_close_seen"] = True
            if response == "Push to Partner":
                state.update(done=True, success=True, termination_reason=SUCCESS_REASON)
                state["terminal_datetime"] = _iso(when)
            state["last_journey_datetime"] = _iso(when)
            conditioned = bool(payload.get("action_id"))
            events.append(self._emit(state, when, kind,
                decision_id=payload.get("decision_id"), action_id=payload.get("action_id"),
                channel=payload.get("channel"), theme=payload.get("theme"),
                time_bucket=payload.get("time_bucket"),
                decision_time=_timestamp(payload["decision_time"]) if payload.get("decision_time") else None,
                resolved_send_time=_timestamp(payload["resolved_send_time"]) if payload.get("resolved_send_time") else None,
                policy_source=payload.get("policy_source"), source_type="transaction" if state["done"] else "journey",
                model_source="PL_APPLICATION_PTP_HAZARD" if payload.get("ptp_outcome") else
                             "PL_OBSERVATIONAL_CAMPAIGN_MOTIF" if payload.get("motif") else
                             "PL_OBSERVATIONAL_ACTION_RESPONSE" if conditioned else "PL_EMPIRICAL_NATURAL",
                model_level=payload.get("model_level"), support_count=payload.get("support_count"),
                transition_mode="ACTION_CONDITIONED_DIRECT" if conditioned else "NATURAL"))
            if state["done"]:
                suppressed = [row for row in pending if row["kind"] == "CAMPAIGN_SEND_EVENT"]
                for row in suppressed:
                    suppressed_decision = row["payload"].get("decision_id")
                    for decision in decisions:
                        if decision["decision_id"] == suppressed_decision:
                            decision["status"] = "SUPPRESSED_BY_TERMINAL"
                            decision["suppression_reason"] = SUCCESS_REASON
                    self._audit(state, when, "CAMPAIGN_SUPPRESSION", {
                        "scheduled_send_time": row["when"],
                        "selected_outcome": "SUPPRESSED",
                        "reason": SUCCESS_REASON,
                    }, decision_id=suppressed_decision)
                pending.clear()
                break
            self._schedule_natural(state, pending, when, following)
            if response == "AIP Approved":
                self._schedule_ptp_outcome(state, pending, when, following)
        if any(_timestamp(item["when"]) < following for item in pending):
            raise RuntimeError("due pending event escaped its simulation day")
        return events, decisions, actions
