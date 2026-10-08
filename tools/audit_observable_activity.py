"""Supplement raw Phase 2 benchmarks with deterministic business-guard support.

No stochastic law is fitted here. Walk independently ordered historical packets
and stop at the first event the unchanged runtime guards could not emit. Report
both raw equal-age benchmarks and this explicitly narrower observability check.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

parser=argparse.ArgumentParser();parser.add_argument('metrics',type=Path);args=parser.parse_args()
m=json.loads(args.metrics.read_text());packets=pd.read_parquet('data/output/finalization-validation/historical-v3-subpop/historical_packets.parquet')
life=pd.read_parquet('artifacts/gold_events_v3/application_lifecycle_state.parquet').dropna(subset=['context_id','application_created_at']);prior={};rows=[];blocked={}
for row in packets.itertuples(index=False):
    if row.application_id in blocked:continue
    seen=prior.setdefault(row.application_id,set())
    for label in json.loads(row.labels):
        has_pd=bool(seen&{'Professional details','Professional Details Submission'})
        if (label=='AIP Approved' and not has_pd) or (label=='Push to Partner' and not (has_pd and 'AIP Approved' in seen)):
            blocked[row.application_id]=label;break
        rows.append((row.application_id,row.event_datetime,row.application_created_at,label));seen.add(label)
h=pd.DataFrame(rows,columns=['application_id','event_datetime','creation','journey_substage']);h['age_seconds']=(h.event_datetime-h.creation).dt.total_seconds();life['weekday']=life.application_created_at.dt.weekday
kind=packets.drop_duplicates('application_id').set_index('application_id').customer_kind
result={'method':'Deterministic historical packet prefix through first blocked AIP/PTP; same unchanged PD/AIP guards, no probability model; raw Phase 2 comparisons remain authoritative and separately reported','blocked_applications':len(blocked),'blocked_by_label':pd.Series(blocked).value_counts().to_dict(),'age':{}}
for days in [1,3,7]:
    groups={};allhist=life.loc[life.application_created_at.le(pd.Timestamp('2026-05-01')-pd.Timedelta(days=days))]
    for group,sim in [('all',m['age'][str(days)]['corrected']),('new_customer',m['returning']['medium_subpopulations']['new_customer'][str(days)]['corrected']),('returning',m['returning']['medium_subpopulations']['returning'][str(days)]['corrected'])]:
        hist=allhist if group=='all' else allhist.loc[allhist.application_id.map(kind).eq('RETURNING_CUSTOMER' if group=='returning' else 'NEW_CUSTOMER')]
        if group!='all':hist=hist.loc[hist.application_created_at.ge('2026-04-01')]
        n=h.loc[h.age_seconds.lt(days*86400)].groupby('application_id').size();hist=hist.copy();hist['events']=hist.application_id.map(n).fillna(0)
        weights={int(k):v['apps']/sim['apps'] for k,v in sim['weekday'].items()};means=hist.groupby('weekday').events.mean();mean=sum(w*means[k] for k,w in weights.items());var=sum(w*w*hist.loc[hist.weekday.eq(k),'events'].var()/len(hist.loc[hist.weekday.eq(k)]) for k,w in weights.items())
        # Derive the simulated application-cluster SE from saved Gold, not quantiles.
        se=pd.read_parquet(args.metrics.parent/'simulated_pl.parquet');ids=set(se.loc[se.creation.ge('2026-05-01')&(se.creation<=pd.Timestamp('2026-06-01')-pd.Timedelta(days=days)),'application_id'])
        arrivals=pd.read_parquet('artifacts/gold_events_v3/may_application_arrivals.parquet').set_index('application_id')
        if group!='all':ids={app for app in ids if arrivals.loc[app,'customer_kind']==('RETURNING_CUSTOMER' if group=='returning' else 'NEW_CUSTOMER')}
        counts=se.loc[se.application_id.isin(ids)&se.age_seconds.lt(days*86400)].groupby('application_id').size().reindex(list(ids),fill_value=0);sd=float(counts.std()/np.sqrt(len(counts)));difference=float(sim['events_per_app']-mean);half=1.96*np.sqrt(var+sd*sd)
        ptp_ids=set(h.loc[h.age_seconds.lt(days*86400)&h.journey_substage.eq('Push to Partner'),'application_id']);hist['ptp']=hist.application_id.isin(ptp_ids);ptp_mean=sum(w*hist.loc[hist.weekday.eq(k),'ptp'].mean() for k,w in weights.items());ptp_var=sum(w*w*hist.loc[hist.weekday.eq(k),'ptp'].var()/len(hist.loc[hist.weekday.eq(k)]) for k,w in weights.items());ptp_diff=sim['ptp_probability']-ptp_mean;ptp_half=1.96*np.sqrt(ptp_var+sim['ptp_probability']*(1-sim['ptp_probability'])/sim['apps'])
        groups[group]=dict(historical_guard_supported_ptp_rate=float(ptp_mean),simulated_ptp_rate=sim['ptp_probability'],ptp_difference_95pct_interval=[float(ptp_diff-ptp_half),float(ptp_diff+ptp_half)],historical_guard_supported_events_per_app=float(mean),simulated_events_per_app=sim['events_per_app'],difference=difference,combined_app_cluster_95pct_difference=[difference-half,difference+half],historical_apps=len(hist),simulated_apps=sim['apps'])
    result['age'][str(days)]=groups
out=args.metrics.parent/'observable-activity.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
