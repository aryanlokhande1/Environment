"""Censored empirical next-event law and finite timestamp-group construction.

The product-limit calculation uses events/risk-set, the same conditional
survival convention as the frozen piecewise hazards. No simulated input is
accepted by the historical entry point.
"""
from __future__ import annotations
from collections import Counter
import json
import numpy as np
import pandas as pd
from environment.runtime.natural_transition import DOWNSTREAM, FOCUS
from environment.runtime.carry_strata import AGE_BOUNDS, SILENCE_BOUNDS, DEPTH_BOUNDS

CORE_ORDER = ['Application Created', 'Language Selection', 'Application Resume',
              'Application OTP Verification', 'Form filled', 'Professional details',
              'Professional Details Submission', 'AIP Approved', 'Product Selection',
              'Address Submission', 'Bank Approval Pending', 'Bank Approved',
              'listing', 'Push to Partner']
RANK = {s:i for i,s in enumerate(CORE_ORDER)}
CORE = frozenset(CORE_ORDER) - {'Application Created','Push to Partner'}

def packets(history: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Learn defensible unordered co-occurrence sets across both source months.

    Supported sets must contain distinct business journey labels, have March
    >=50 and April >=25 observations, and exclude creation/terminal/campaign
    labels. Source order resolves which single observation is retained in an
    unsupported group; it is not asserted to be a causal order.
    """
    ordered=history.sort_values(['application_id','event_datetime','_order'],kind='stable')
    repeated_creation=ordered.journey_substage.eq('Application Created')&ordered.loc[ordered.journey_substage.eq('Application Created')].application_id.duplicated().reindex(ordered.index,fill_value=False)
    duplicate_creation_count=int(repeated_creation.sum())
    ordered=ordered.loc[~repeated_creation]
    distinct=ordered.drop_duplicates(['application_id','event_datetime','journey_substage'])
    label_groups=distinct.groupby(['application_id','event_datetime'],sort=False).journey_substage.agg(tuple)
    last=ordered.drop_duplicates(['application_id','event_datetime'],keep='last').set_index(['application_id','event_datetime'])
    counts=Counter();groups=[]
    repeated=len(ordered)-len(distinct)
    for (app,when),labels in label_groups.items():
        counts[(tuple(sorted(labels)),when.month)]+=1
    combined=last[['context_id','application_created_at','journey_stage']].join(label_groups.rename('labels'))
    groups=((r.Index[0],r.Index[1],list(r.labels),str(r.context_id),r.application_created_at,str(r.journey_stage)) for r in combined.itertuples())
    accepted={key for key,month in counts if len(key)>1 and set(key)<=CORE
              and counts[(key,3)]>=50 and counts[(key,4)]>=25}
    rows=[];unsupported=0
    for app,when,labels,context,creation,stage in groups:
        key=tuple(sorted(labels))
        if key in accepted: labels=sorted(labels,key=lambda s:(RANK.get(s,999),s))
        else:
            if len(labels)>1:unsupported+=len(labels)-1
            labels=['Push to Partner'] if 'Push to Partner' in labels else ['Application Created'] if 'Application Created' in labels else [labels[-1]]
        rows.append(dict(application_id=app,context_id=context,event_datetime=when,
            application_created_at=creation,deadline=creation+pd.Timedelta(days=30),
            labels=json.dumps(labels,separators=(',',':')),state=labels[-1],journey_stage=stage))
    frame=pd.DataFrame(rows);seen={};regimes=[];depth=[];eligible=[]
    for r in frame.itertuples(index=False):
        prior=seen.setdefault(r.application_id,[])
        regimes.append('REVISIT' if r.state in FOCUS and set(prior)&DOWNSTREAM else 'FIRST_PASS' if r.state in FOCUS else 'UNCONDITIONED')
        prior.extend(json.loads(r.labels));depth.append(len(prior))
        eligible.append(bool(set(prior)&{'Professional details','Professional Details Submission'} and 'AIP Approved' in prior))
    frame['regime']=regimes;frame['prior_depth']=depth;frame['ptp_eligible']=eligible
    frame['source_age']=np.where((frame.event_datetime-frame.application_created_at).dt.total_seconds()<86400,'YOUNG','LATE')
    audit=dict(repeated_application_created_instrumentation_rejected=duplicate_creation_count,exact_same_substage_cotemporal_extras_rejected=repeated,
               unsupported_cotemporal_extras_omitted=unsupported,
               supported_patterns=[dict(labels=list(k),march=counts[(k,3)],april=counts[(k,4)]) for k in sorted(accepted)],
               canonical_order=CORE_ORDER,group_count=len(frame))
    return frame,audit

def episodes(p:pd.DataFrame,start:str,end:str) -> pd.DataFrame:
    """Source occurrences inside the fit window, censored at its end/expiry. PTP competes as a next event."""
    g=p.groupby('application_id',sort=False)
    x=p.assign(next_time=g.event_datetime.shift(-1),next_labels=g.labels.shift(-1),next_state=g.state.shift(-1)).copy()
    x=x.loc[x.event_datetime.ge(start)&x.event_datetime.lt(end)&x.state.ne('Push to Partner')].copy()
    stop=x.deadline.clip(upper=pd.Timestamp(end));next_before=x.next_time.notna()&x.next_time.lt(stop)
    x['observed']=next_before
    # Unobservable-prerequisite terminals consume their historical competing
    # mass but cannot be emitted. Never redistribute that mass to journey loops.
    blocked=next_before&x.next_state.eq('Push to Partner')&~x.ptp_eligible
    x.loc[blocked,'next_labels']='[]'
    x['duration_seconds']=(x.next_time.where(next_before,stop)-x.event_datetime).dt.total_seconds()
    x.loc[~x.observed,'next_labels']='[]'
    return x.loc[x.duration_seconds.gt(0)].reset_index(drop=True)

def product_limit(x:pd.DataFrame) -> pd.DataFrame:
    """Exact event-time masses with same-time events preceding censor removal."""
    if x.empty:return pd.DataFrame(columns=['duration_seconds','labels','mass','risk_set','support_count'])
    stats=x.groupby('duration_seconds',sort=True)['observed'].agg(['size','sum'])
    stats['risk_set']=len(x)-stats['size'].cumsum().shift(fill_value=0)
    stats['survival']=(1-stats['sum']/stats.risk_set).cumprod().shift(fill_value=1.)
    stats['unit_mass']=stats.survival/stats.risk_set
    observed=x.loc[x.observed,['duration_seconds','next_labels']].rename(columns={'next_labels':'labels'})
    rows=observed.groupby(['duration_seconds','labels'],sort=True).size().rename('support_count').reset_index()
    rows=rows.join(stats[['risk_set','unit_mass']],on='duration_seconds')
    rows['mass']=rows.support_count*rows.unit_mass
    return rows[['duration_seconds','labels','mass','risk_set','support_count']]

def fit_law(x:pd.DataFrame) -> pd.DataFrame:
    out=[]
    levels=[('STATE_REGIME',['state','regime'],200),('STATE',['state'],200),('STAGE',['journey_stage'],500),('GLOBAL',[],1)]
    strata=['ptp_eligible','source_age','customer_kind']
    if 'customer_kind' not in x:x=x.assign(customer_kind='NEW_CUSTOMER')
    for level,keys,minimum in levels:
        groups=x.groupby(keys+strata,sort=True,dropna=False)
        for key,g in groups:
            if len(g)<minimum:continue
            if not isinstance(key,tuple):key=(key,)
            if level=='STATE_REGIME' and key[0] not in FOCUS:continue
            z=product_limit(g)
            if z.empty:
                z=pd.DataFrame([dict(duration_seconds=0.,labels='[]',mass=0.,risk_set=len(g),support_count=0)])
            z['ptp_eligible']=bool(key[-3]);z['source_age']=str(key[-2]);z['customer_kind']=str(key[-1])
            z['model_level']=level;z['state']=str(key[0]) if level.startswith('STATE') else ''
            z['regime']=str(key[1]) if level=='STATE_REGIME' else ''
            z['journey_stage']=str(key[0]) if level=='STAGE' else ''
            z['denominator']=len(g);z['applications']=g.application_id.nunique()
            z.loc[z.labels.eq('[]'),'mass']=0.
            out.append(z)
    return pd.concat(out,ignore_index=True)

def summarize_law(train:pd.DataFrame,test:pd.DataFrame) -> dict:
    """Held-out conditional distributions; only complete one-day windows count."""
    model=fit_law(train);rows=[]
    for state,g in test.groupby('state',sort=True):
        if len(g)<200:continue
        z=model.loc[model.model_level.eq('STATE')&model.state.eq(state)].copy()
        den=z[['ptp_eligible','source_age','customer_kind','denominator']].drop_duplicates().denominator.sum()
        if den:z['mass']*=z.denominator/den
        if z.empty:continue
        # Censoring-aware held-out empirical law, not forced point targets.
        truth=product_limit(g)
        times=np.unique(np.r_[z.duration_seconds,truth.duration_seconds])
        a=z.groupby('duration_seconds').mass.sum().sort_index().cumsum().reindex(times,method='ffill').fillna(0)
        b=truth.groupby('duration_seconds').mass.sum().sort_index().cumsum().reindex(times,method='ffill').fillna(0)
        rows.append(dict(state=state,train_occurrences=int(z.denominator.iloc[0]),validation_occurrences=len(g),
            validation_applications=g.application_id.nunique(),cdf_sup_error=float(abs(a-b).max()),
            train_continuation_mass=float(z.mass.sum()),validation_continuation_mass=float(truth.mass.sum())))
    return dict(training_occurrences=len(train),validation_occurrences=len(test),state_results=rows)


def annotate_customers(p,life):
    ordered=life.dropna(subset=['context_id','application_created_at']).sort_values(['context_id','application_created_at'])
    prior=ordered.groupby('context_id').application_created_at.shift()
    returning=ordered.loc[(ordered.application_created_at-prior).dt.total_seconds().ge(30*86400),'application_id']
    p=p.copy();p['customer_kind']=np.where(p.application_id.isin(returning),'RETURNING_CUSTOMER','NEW_CUSTOMER')
    return p


def carry_landmarks(p):
    """Complete-life risk sets; no source-window expiry is treated as inactivity."""
    rows=[]
    for bound in pd.date_range('2026-03-01','2026-04-30'):
        prior=p.loc[p.event_datetime.lt(bound)].drop_duplicates('application_id',keep='last')
        prior=prior.loc[prior.deadline.gt(bound)&prior.deadline.le('2026-05-01')&prior.state.ne('Push to Partner')].copy()
        if prior.empty:continue
        future=p.loc[p.event_datetime.ge(bound)].drop_duplicates('application_id').set_index('application_id')
        prior['next_time']=pd.NaT if future.empty else prior.application_id.map(future.event_datetime)
        prior['next_labels']='[]' if future.empty else prior.application_id.map(future.labels)
        prior['observed']=prior.next_time.notna()&prior.next_time.lt(prior.deadline)
        prior['duration_seconds']=(prior.next_time.where(prior.observed,prior.deadline)-bound).dt.total_seconds()
        prior['age_band']=np.searchsorted(AGE_BOUNDS,(bound-prior.application_created_at).dt.total_seconds()/86400,side='left')
        prior['silence_band']=np.searchsorted(SILENCE_BOUNDS,(bound-prior.event_datetime).dt.total_seconds()/86400,side='left')
        prior['depth_band']=np.searchsorted(DEPTH_BOUNDS,prior.prior_depth,side='left')
        prior['weekday']=bound.weekday();prior['landmark']=bound
        prior.loc[~prior.observed,'next_labels']='[]'
        prior.loc[~prior.ptp_eligible&prior.next_labels.eq('["Push to Partner"]'),'next_labels']='[]'
        rows.append(prior)
    return pd.concat(rows,ignore_index=True)


def fit_carry(x):
    """Residual event-time survival, censored at each donor's exact expiry.

    Runtime competes the sampled event with the target application's expiry.
    Counting short-lived donors as permanent inactivity and also truncating
    target draws would count expiry twice. Fine near-expiry age strata reduce
    variation in remaining validity. No polling redraws or global scaling.
    """
    levels=[('SOURCE_AGE_SILENCE_DEPTH',['state','age_band','silence_band','depth_band']),
            ('SOURCE_AGE_SILENCE',['state','age_band','silence_band']),
            ('SOURCE_AGE',['state','age_band']),('SOURCE',['state']),('AGE_SILENCE',['age_band','silence_band']),('GLOBAL',[])]
    output=[]
    for weekday in list(range(7))+[-1]:
        subset=x if weekday==-1 else x.loc[x.weekday.eq(weekday)]
        for level,keys in levels:
            groups=subset.groupby(keys,sort=True) if keys else [((),subset)]
            for key,g in groups:
                if g.application_id.nunique()<30:continue
                if not isinstance(key,tuple):key=(key,)
                z=product_limit(g)
                z=z.loc[z.labels.ne('[]')].copy()
                if z.empty:z=pd.DataFrame([dict(duration_seconds=0.,labels='[]',support_count=0,mass=0.)])
                z['model_level']=level;z['weekday']=weekday
                for field in ['state','age_band','silence_band','depth_band']:z[field]=dict(zip(keys,key)).get(field,'' if field=='state' else -1)
                z['denominator']=len(g);z['applications']=g.application_id.nunique();output.append(z)
    return pd.concat(output,ignore_index=True)
