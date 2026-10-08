"""One residual-life draw for an application already active at entry."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from environment.persistence.checkpoint import file_hash
from .joint_continuation import JointOutcome
from .carry_strata import AGE_BOUNDS, SILENCE_BOUNDS, DEPTH_BOUNDS


class CarryTailModel:
    def __init__(self,bundle:Path):
        manifest=json.loads((bundle/'joint_continuation_manifest.json').read_text())
        if manifest['carry_model_version']!='PL_RESIDUAL_TAIL_SURVIVAL_V2':raise ValueError('unsupported carried tail version')
        path=bundle/'carried_tail.parquet'
        if file_hash(path)!=manifest['carried_tail_sha256']:raise ValueError('carried tail hash mismatch')
        rows=pd.read_parquet(path);self.pools={}
        for key,g in rows.groupby(['weekday','model_level','state','age_band','silence_band','depth_band'],sort=False):
            g=g.sort_values(['duration_seconds','labels'],kind='stable')
            cdf=g.mass.cumsum().to_numpy()
            if g.mass.lt(0).any() or not np.isfinite(g.mass).all() or cdf[-1]>1+1e-9:raise ValueError('invalid carried residual event law')
            self.pools[key]=(g.duration_seconds.to_numpy(),cdf,[tuple(json.loads(v)) for v in g.labels],int(g.applications.iloc[0]))

    def sample(self,state,now,rng):
        age=(now-pd.Timestamp(state['creation_datetime'])).total_seconds()/86400
        silence=(now-pd.Timestamp(state['last_journey_datetime'])).total_seconds()/86400
        a=int(np.searchsorted(AGE_BOUNDS,age,side='left'));s=int(np.searchsorted(SILENCE_BOUNDS,silence,side='left'));d=int(np.searchsorted(DEPTH_BOUNDS,state.get('prior_event_count',len(state['observed_journey_substages'])),side='left'));source=state['journey_substage']
        candidates=[('SOURCE_AGE_SILENCE_DEPTH',source,a,s,d),('SOURCE_AGE_SILENCE',source,a,s,-1),('SOURCE_AGE',source,a,-1,-1),('SOURCE',source,-1,-1,-1),('AGE_SILENCE','',a,s,-1),('GLOBAL','',-1,-1,-1)]
        key=next(( (w,)+k for w in [now.weekday(),-1] for k in candidates if (w,)+k in self.pools),None)
        if key is None:raise ValueError('no supported historical carried-tail fallback')
        times,cdf,labels,n=self.pools[key];draw=float(rng.random());index=int(np.searchsorted(cdf,draw,side='right'))
        evidence=dict(model='complete_life_residual_tail',model_level=key[1],source=source,support_count=n,rng_draw=draw,reason='EMPIRICAL_INACTIVITY_THIS_EPISODE',weekday=key[0],age_band=a,silence_band=s,depth_band=d)
        if index<len(times):
            when=now+pd.Timedelta(seconds=float(times[index]))
            if when<pd.Timestamp(state['deadline']):
                evidence.update(reason='EMPIRICAL_CARRIED_REACTIVATION',labels=list(labels[index]),sampled_delay_seconds=float(times[index]))
                return JointOutcome(when,labels[index],evidence)
        return JointOutcome(None,(),evidence)
