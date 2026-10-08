"""Historical holdout evidence and uncertainty for the final bundle.

This evaluates March-fitted laws on April observations. No simulated rows are
read. Burst-pattern support is selected using March/April stability, so April is
validation/model selection, not an untouched final test set.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from environment.empirical.joint import episodes,fit_law,product_limit
from environment.runtime.hazard import PtpHazardModel,StageContinuationModel

THRESHOLDS=np.array([0.,60.,300.,900.,3600.,86400.])

def cdf(times,mass,thresholds=THRESHOLDS):
    return np.array([float(np.asarray(mass)[np.asarray(times)<=t].sum()) for t in thresholds])

def frozen_cdf(model,rows,donors):
    times=[];masses=[];survival=1.
    for row in rows.sort_values('elapsed_lower_seconds').itertuples(index=False):
        d=donors.loc[donors.duration_seconds.ge(row.elapsed_lower_seconds)&donors.duration_seconds.lt(row.elapsed_upper_seconds),'duration_seconds'].to_numpy()
        if len(d):times.extend(d);masses.extend(np.repeat(survival*float(row.hazard)/len(d),len(d)))
        survival*=1-float(row.hazard)
    return np.array(times),np.array(masses)

def conditional_cdf_cluster_interval(group,thresholds=THRESHOLDS,repetitions=400):
    # Whole-application resampling preserves correlation between observed gaps.
    ids,indices=np.unique(group.application_id.to_numpy(),return_inverse=True);n=np.bincount(indices,minlength=len(ids));counts=np.column_stack([np.bincount(indices,weights=group.duration_seconds.le(t).to_numpy(),minlength=len(ids)) for t in thresholds]);rng=np.random.default_rng(20260502);draws=[]
    for _ in range(repetitions):
        selected=rng.integers(0,len(ids),size=len(ids));den=n[selected].sum();draws.append(counts[selected].sum(axis=0)/den)
    return np.quantile(draws,[.025,.975],axis=0).tolist()

def survival_cluster_interval(group,repetitions=400):
    ids,app=np.unique(group.application_id.to_numpy(),return_inverse=True)
    times,ti=np.unique(group.duration_seconds.to_numpy(),return_inverse=True)
    observed=group.observed.to_numpy().astype(float);rng=np.random.default_rng(20260502);draws=[]
    for _ in range(repetitions):
        multiplicity=np.bincount(rng.integers(0,len(ids),size=len(ids)),minlength=len(ids));w=multiplicity[app]
        total=np.bincount(ti,weights=w,minlength=len(times));ev=np.bincount(ti,weights=w*observed,minlength=len(times));risk=total.sum()-np.r_[0.,total.cumsum()[:-1]]
        mass=np.divide(ev,risk,out=np.zeros_like(ev),where=risk>0);cdf=1-np.cumprod(1-mass)
        draws.append([cdf[np.searchsorted(times,t,side='right')-1] if (times<=t).any() else 0. for t in THRESHOLDS])
    return np.quantile(draws,[.025,.975],axis=0).tolist()

def ptp_episodes(h,life,month):
    x=h.loc[h.application_id.isin(life.loc[life.context_id.notna()&life.application_created_at.ge(f'2026-{month:02d}-01')&life.application_created_at.lt('2026-04-01' if month==3 else '2026-05-01')].application_id)].copy();pdtime=x.loc[x.journey_substage.isin(['Professional details','Professional Details Submission'])].groupby('application_id').event_datetime.min();pt=x.loc[x.journey_substage.eq('Push to Partner')].groupby('application_id').event_datetime.min();a=x.loc[x.journey_substage.eq('AIP Approved')].drop_duplicates(['application_id','event_datetime']).copy();a=a.loc[a.application_id.map(pdtime).lt(a.event_datetime)&(a.application_id.map(pt).isna()|a.event_datetime.lt(a.application_id.map(pt)))];a['next']=a.groupby('application_id').event_datetime.shift(-1);a['stop']=pd.concat([a.next,a.application_created_at+pd.Timedelta(days=30)],axis=1).min(axis=1).clip(upper=pd.Timestamp('2026-05-01'));a['ptp']=a.application_id.map(pt);a['observed']=a.ptp.ge(a.event_datetime)&a.ptp.lt(a.stop);a['duration_seconds']=(a.ptp.where(a.observed,a.stop)-a.event_datetime).dt.total_seconds();a['next_labels']='["Push to Partner"]';a['age_bucket']=np.where((a.event_datetime-a.application_created_at).dt.total_seconds().lt(86400),'AGE_LT_1D',np.where((a.event_datetime-a.application_created_at).dt.total_seconds().lt(7*86400),'AGE_1D_7D','AGE_7D_30D'));return a.loc[a.duration_seconds.gt(0)]

def main(root:Path):
    h=pd.read_parquet(root/'historical_ordered_pl.parquet');p=pd.read_parquet(root/'historical_packets.parquet');life=pd.read_parquet('artifacts/gold_events_v2/application_lifecycle_state.parquet');train=episodes(p,'2026-03-01','2026-04-01');test=episodes(p,'2026-04-01','2026-05-01');law=fit_law(train);old=StageContinuationModel(Path('artifacts/gold_events_v2'));pairs=[]
    observed=test.loc[test.observed].copy();observed['destination']=observed.next_labels.map(lambda x:(json.loads(x) or ['BLOCKED_TERMINAL'])[0]);observed=observed.loc[observed.event_datetime.lt('2026-04-30')]
    for source,dest in [('Product Selection','Bank Approved'),('AIP Approved','Product Selection'),('Application OTP Verification','Professional Details Submission'),('Professional Details Submission','AIP Approved'),('Bank Approval Pending','Bank Approved')]:
        g=observed.loc[observed.state.eq(source)&observed.destination.eq(dest)];z=law.loc[law.model_level.eq('STATE')&law.state.eq(source)];z=z.loc[z.labels.map(lambda x:bool(json.loads(x)) and (json.loads(x) or ['BLOCKED_TERMINAL'])[0]==dest)].copy();den=z[['ptp_eligible','source_age','denominator']].drop_duplicates().denominator.sum();z['mass']*=z.denominator/den
        if len(g)<100 or z.empty:continue
        new=cdf(z.duration_seconds,z.mass/z.mass.sum());truth=np.array([g.duration_seconds.le(t).mean() for t in THRESHOLDS]);r=old._state_rows.get(source,old._global_rows);d=old._state_donors.get(source,old.donors);t,m=frozen_cdf(old,r,d);baseline=cdf(t,m/m.sum())
        pairs.append(dict(source=source,destination=dest,validation_pairs=len(g),validation_applications=g.application_id.nunique(),threshold_seconds=THRESHOLDS.tolist(),historical_cdf=truth.tolist(),historical_app_cluster_95pct_cdf=conditional_cdf_cluster_interval(g),march_joint_cdf=new.tolist(),frozen_source_only_cdf=baseline.tolist(),joint_max_threshold_error=float(abs(new-truth).max()),old_max_threshold_error=float(abs(baseline-truth).max()),scope='positive timestamp-group entry transition; internal finite zero edges validated separately'))
    ptp=PtpHazardModel(Path('artifacts/gold_events_v2'));ptp_results=[]
    for month in [3,4]:
        e=ptp_episodes(h,life,month)
        for bucket,g in e.groupby('age_bucket'):
            if len(g)<100:continue
            r=ptp._age_rows.get(bucket,ptp._global_rows);d=ptp._age_donors.get(bucket,ptp.donors);t,m=frozen_cdf(ptp,r,d);frozen=cdf(t,m);truthlaw=product_limit(g);truth=cdf(truthlaw.duration_seconds,truthlaw.mass)
            ptp_results.append(dict(month=month,age_bucket=bucket,eligible_applications=g.application_id.nunique(),eligible_episodes=len(g),observed_ptp_episodes=int(g.observed.sum()),historical_app_cluster_95pct_cdf=survival_cluster_interval(g),threshold_seconds=THRESHOLDS.tolist(),frozen_hazard_cdf=frozen.tolist(),historical_product_limit_cdf=truth.tolist(),max_threshold_error=float(abs(frozen-truth).max()),censoring='next eligible AIP, PTP, expiry or May1 truncation; no zero-at-risk episode at terminal'))
    v=life.dropna(subset=['context_id','application_created_at']).sort_values(['context_id','application_created_at']);v['prior']=v.groupby('context_id').application_created_at.shift();apr=v.loc[v.application_created_at.ge('2026-04-01')&v.application_created_at.lt('2026-05-01')].copy();apr['returning']=(apr.application_created_at-apr.prior).dt.total_seconds().ge(30*86400);apr['weekday']=apr.application_created_at.dt.weekday;identity=[]
    a=apr.loc[apr.application_created_at.lt('2026-04-11')];b=apr.loc[apr.application_created_at.ge('2026-04-11')]
    for w in range(7):
        tr=a.loc[a.weekday.eq(w)];te=b.loc[b.weekday.eq(w)]
        if min(len(tr),len(te))<30:continue
        rate=float(tr.returning.mean());p=float(te.returning.mean());n=len(te);z=1.96;center=(p+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        identity.append(dict(weekday=w,train_entrants=len(tr),train_rate=rate,validation_entrants=n,validation_rate=p,validation_95pct_wilson=[center-half,center+half]))
    result=dict(temporal_pair_timing=pairs,ptp_eligible_episode_audit=ptp_results,returning_weekday_temporal_validation=identity,ptp_assets_changed=False,ptp_runtime_changed=True,ptp_change_threshold='Require >5 percentage-point comparable conditional CDF/incidence error, outside whole-application uncertainty and robust to source-window/episode definitions. Young-age frozen CDF errors exceed this threshold in both months; late-age observed zero positives contradict the global fallback. Corrected-v2 therefore models mutually exclusive next-event causes; frozen files remain untouched.',limitations=['April is historical model-selection validation, not an untouched future test','Short observed pair gaps condition on the observed destination; late censoring is handled by survival validation, not these pair CDF intervals','PTP April comparison includes right censoring; age strata with insufficient support are omitted'])
    (root/'holdout-detail.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main(Path('data/output/finalization-validation/historical-v3'))
