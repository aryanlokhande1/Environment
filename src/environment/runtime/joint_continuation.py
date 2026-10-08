"""Versioned joint next-event timing and bounded same-time business packets."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd
from environment.persistence.checkpoint import file_hash
from .natural_transition import RegimeNaturalTransitionModel

@dataclass(frozen=True)
class JointOutcome:
    when: pd.Timestamp | None
    labels: tuple[str,...]
    evidence: dict[str,Any]

class JointContinuationModel:
    def __init__(self,bundle:Path):
        self.manifest=json.loads((bundle/'joint_continuation_manifest.json').read_text())
        if self.manifest.get('model_version')!='PL_JOINT_COMPETING_EVENT_SURVIVAL_V2':raise ValueError('unsupported joint continuation version')
        path=bundle/'joint_continuation.parquet'
        if file_hash(path)!=self.manifest['model_sha256']:raise ValueError('joint continuation artifact hash mismatch')
        rows=pd.read_parquet(path)
        if not np.isfinite(rows.mass).all() or rows.mass.lt(0).any() or rows.duration_seconds.lt(0).any():raise ValueError('invalid joint event mass/delay')
        self.pools={}
        for key,g in rows.groupby(['model_level','state','regime','journey_stage','ptp_eligible','source_age','customer_kind'],sort=False,dropna=False):
            g=g.sort_values(['duration_seconds','labels'],kind='stable').reset_index(drop=True)
            times=g.duration_seconds.to_numpy();mass=g.mass.to_numpy();cumulative=np.cumsum(mass)
            if cumulative[-1]>1+1e-9:raise ValueError('joint survival mass exceeds one')
            labels=[tuple(json.loads(v)) for v in g.labels]
            if any(len(set(v))!=len(v) or set(v)&{'Application Created','campaign_sent'} or ('Push to Partner' in v and len(v)!=1) for v in labels):raise ValueError('invalid finite natural packet')
            self.pools[key]=(times,cumulative,labels,int(g.denominator.iloc[0]))

    def sample(self,state:dict,now:pd.Timestamp,rng:np.random.Generator)->JointOutcome:
        source=str(state['journey_substage']);stage=str(state['journey_stage']);regime=RegimeNaturalTransitionModel.classify(state)
        candidates=[('STATE_REGIME',source,regime,''),('STATE',source,'',''),('STAGE','','',stage),('GLOBAL','','','')]
        from .base import can_push_to_partner
        eligible=can_push_to_partner(state)
        source_age='YOUNG' if (pd.Timestamp(state['last_journey_datetime'])-pd.Timestamp(state['creation_datetime'])).total_seconds()<86400 else 'LATE'
        candidates=[k+(eligible,source_age,str(state.get('customer_kind','NEW_CUSTOMER'))) for k in candidates]
        key=next((k for k in candidates if k in self.pools),None)
        if key is None:raise ValueError('joint model has no empirical backoff')
        times,cdf,labels,n=self.pools[key];origin=pd.Timestamp(state['last_journey_datetime']);elapsed=max(0.,(now-origin).total_seconds());stop=(pd.Timestamp(state['deadline'])-origin).total_seconds()
        lower=int(np.searchsorted(times,elapsed,side='left'));upper=int(np.searchsorted(times,stop,side='left'))
        before=float(cdf[lower-1]) if lower else 0.;surv=max(0.,1-before);draw=float(rng.random());target=before+draw*surv;index=int(np.searchsorted(cdf,target,side='right'))
        evidence=dict(model='joint_next_event_survival',model_level=key[0],source=source,regime=regime,ptp_eligible=eligible,source_age=source_age,support_count=n,survival_at_start=surv,rng_draw=draw,remaining_event_mass=float((cdf[upper-1] if upper else 0)-before)/surv if surv>1e-12 else 0.,reason='EMPIRICAL_INACTIVITY_THIS_EPISODE')
        if surv>1e-12 and lower<=index<upper:
            when=origin+pd.Timedelta(seconds=float(times[index]));evidence.update(reason='EMPIRICAL_JOINT_PACKET',sampled_delay_seconds=float(times[index]),labels=list(labels[index]))
            return JointOutcome(when,labels[index],evidence)
        return JointOutcome(None,(),evidence)
