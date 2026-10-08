"""Authoritative external controller boundary, with no Transformer/RL dependency."""
from __future__ import annotations
from dataclasses import dataclass
import fcntl
from pathlib import Path
from typing import Any
import pandas as pd
from environment.contracts.action import EnvironmentAction
from environment.core.environment import Environment
from environment.integration import LocalClosedLoop
from environment.persistence.checkpoint import file_hash
from environment.runtime.gold_events_adapter import GOLD_COLUMNS
from environment.runtime.semantics import FINAL

@dataclass(frozen=True)
class AgentObservation:
    simulated_now: pd.Timestamp
    application_id: str
    context_id: str
    history: pd.DataFrame

@dataclass(frozen=True)
class ExternalStep:
    observation: AgentObservation
    events: pd.DataFrame
    terminal: bool
    operator_audit: dict[str,Any]

class ExternalAgentSession:
    """Persist one application's caller-controlled action/next-time decisions.

    Only ``observe()`` supplies agent input: customer-scoped Gold columns through
    committed simulated time. ``operator_audit`` is for diagnostics, never an
    agent observation. Checkpoints and future queues stay behind this boundary.
    Historical cumulative storage retains the complete immutable source; the
    agent sees only the customer's time-filtered projection, including previous
    applications. Daily crossings do not invoke a controller or invent decisions.
    """
    def __init__(self,environment:Environment,*,application_id:str,start_time:str,
                 input_history:str|Path,cumulative_history:str|Path,checkpoint:str|Path):
        if environment.runtime_version!=FINAL:raise ValueError('external contract requires corrected-v2')
        self._loop=LocalClosedLoop(environment,input_history=input_history,cumulative_history=cumulative_history,checkpoint=checkpoint)
        self.application_id=str(application_id);self._start=pd.Timestamp(start_time)
        if pd.isna(self._start) or self._start.tzinfo is not None:raise ValueError('start_time must be valid and timezone-naive')

    def _clock(self):
        if self._loop.checkpoint.exists():
            stored=self._loop.checkpoints.load(self._loop.checkpoint);self._loop._validate_identity(stored)
            if stored['application_id']!=self.application_id:raise ValueError('checkpoint application_id mismatch')
            return pd.Timestamp(stored['next_decision_time'])
        return self._start

    @staticmethod
    def _validate_lifecycles(history, now, context):
        visible = history.loc[history.context_id.astype(str).eq(context)
                              & pd.to_datetime(history.event_datetime).le(now)]
        creations = (visible.loc[visible.journey_substage.eq('Application Created')]
                     .groupby('application_id').event_datetime.min().sort_values())
        if creations.diff().dropna().lt(pd.Timedelta(days=30)).any():
            raise ValueError('customer history contains overlapping 30-day lifecycles')

    def _visible_state(self,history,now):
        state=self._loop._state(history,self.application_id,now)
        if not state.payload.get('_environment_started'):
            context=str(state.payload['context_id']);creation=pd.Timestamp(state.payload['creation_datetime'])
            visible=history.loc[history.context_id.astype(str).eq(context)&pd.to_datetime(history.event_datetime).le(now)]
            prior=visible.loc[visible.application_id.astype(str).ne(self.application_id)&visible.journey_substage.eq('Application Created')]
            state.payload['customer_kind']='RETURNING_CUSTOMER' if pd.to_datetime(prior.event_datetime).le(creation-pd.Timedelta(days=30)).any() else 'NEW_CUSTOMER'
            state.payload['prior_event_count']=int(visible.application_id.astype(str).eq(self.application_id).sum())
        return state

    def _observation(self,history,now,context):
        self._validate_lifecycles(history, now, context)
        times=pd.to_datetime(history.event_datetime)
        visible=history.loc[history.context_id.astype(str).eq(context)&times.le(now)].copy()
        return AgentObservation(now,self.application_id,context,visible.reindex(columns=list(GOLD_COLUMNS)).reset_index(drop=True))

    def observe(self)->AgentObservation:
        loop=self._loop
        loop.checkpoint.parent.mkdir(parents=True,exist_ok=True)
        with loop.checkpoint.with_suffix(loop.checkpoint.suffix+'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            history=loop._history();now=self._clock();state=self._visible_state(history,now)
            return self._observation(history,now,str(state.payload['context_id']))

    def decide(self,action:EnvironmentAction|dict, next_decision_time:str|pd.Timestamp)->ExternalStep:
        loop=self._loop;loop.checkpoint.parent.mkdir(parents=True,exist_ok=True)
        with loop.checkpoint.with_suffix(loop.checkpoint.suffix+'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            history=loop._history();action=EnvironmentAction.parse(action);end=pd.Timestamp(next_decision_time);now=self._clock()
            state=self._visible_state(history,now)
            if loop.checkpoint.exists():
                previous=loop.checkpoints.load(loop.checkpoint)
                if end==now:
                    stored_action=EnvironmentAction.parse(previous['action'])
                    if action!=stored_action:raise ValueError('retry action differs from committed decision')
                    return ExternalStep(self._observation(history,now,str(state.payload['context_id'])),pd.DataFrame(columns=list(GOLD_COLUMNS)),state.terminal,{'idempotent_retry':True})
            self._validate_lifecycles(history, now, str(state.payload['context_id']))
            result=loop.environment.advance(state,action,now,end)
            env=loop.environment
            stored=dict(run_id=env.run_id,runtime_version=env.runtime_version,seed=env.seed,
                stochastic_namespace=env.stochastic_namespace,simulation_start=None if env.simulation_start is None else env.simulation_start.isoformat(),simulation_end=None if env.simulation_end is None else env.simulation_end.isoformat(),
                input_history_sha256=file_hash(loop.input_history),artifact_hashes=env.artifact_hashes,
                application_id=self.application_id,decision_time=now.isoformat(),next_decision_time=end.isoformat(),
                action=dict(campaign_sent=action.campaign_sent,channel_id=action.channel_id,theme=action.theme,time_bucket=action.time_bucket),state=result.state.payload,pending_events=result.pending_events)
            loop._commit(stored,result.events)
            history=loop._history()
            return ExternalStep(self._observation(history,end,str(state.payload['context_id'])),pd.DataFrame(result.events).reindex(columns=list(GOLD_COLUMNS)),result.terminal,dict(decisions=result.decisions,explanations=result.explanations))
