"""Comparable eligible AIP episodes in the final simulated validation only.

This is evaluation, never a model-building input. Next AIP, exact expiry and
source-window end censor the episode; only first PTP can terminate it.
"""
import json
from pathlib import Path
import numpy as np,pandas as pd
from environment.empirical.joint import product_limit
A=Path('data/output/finalization-validation');e=pd.read_parquet(A/'full-metrics/simulated_pl.parquet');e=e.loc[e.creation.ge('2026-05-01')];pdtime=e.loc[e.journey_substage.isin(['Professional details','Professional Details Submission'])].groupby('application_id').event_datetime.min();pt=e.loc[e.journey_substage.eq('Push to Partner')].groupby('application_id').event_datetime.min();a=e.loc[e.journey_substage.eq('AIP Approved')].drop_duplicates(['application_id','event_datetime']).copy();a=a.loc[a.application_id.map(pdtime).lt(a.event_datetime)&(a.application_id.map(pt).isna()|a.event_datetime.lt(a.application_id.map(pt)))];a['next']=a.groupby('application_id').event_datetime.shift(-1);a['stop']=pd.concat([a.next,a.creation+pd.Timedelta(days=30)],axis=1).min(axis=1).clip(upper=pd.Timestamp('2026-06-01'));a['ptp']=a.application_id.map(pt);a['observed']=a.ptp.ge(a.event_datetime)&a.ptp.lt(a.stop);a['duration_seconds']=(a.ptp.where(a.observed,a.stop)-a.event_datetime).dt.total_seconds();a['next_labels']='["Push to Partner"]';a['age_bucket']=np.where((a.event_datetime-a.creation).dt.total_seconds().lt(86400),'AGE_LT_1D',np.where((a.event_datetime-a.creation).dt.total_seconds().lt(7*86400),'AGE_1D_7D','AGE_7D_30D'));a=a.loc[a.duration_seconds.gt(0)];thresholds=[0.,60.,300.,900.,3600.,86400.];rows=[]
for bucket,g in a.groupby('age_bucket'):
 law=product_limit(g);cdf=[float(law.loc[law.duration_seconds.le(t),'mass'].sum()) for t in thresholds];rows.append(dict(age_bucket=bucket,eligible_applications=g.application_id.nunique(),eligible_episodes=len(g),observed_ptp_episodes=int(g.observed.sum()),threshold_seconds=thresholds,simulated_product_limit_cdf=cdf))
history=json.loads((A/'historical-v3-competing/holdout-detail.json').read_text())['ptp_eligible_episode_audit'];result=dict(simulated_eligible_episodes=rows,historical_comparison=history,method='Identical latest strictly eligible observed AIP competition to next eligible AIP, PTP, expiry, or observation-window end; May entrants include explicit right censoring; no zero-exposure terminal ties')
(A/'final-ptp-episode-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows,indent=2))
