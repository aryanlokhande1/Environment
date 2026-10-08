"""Historical-contract intensity checks on an EXISTING certification run.

The 10% margin comes from docs/VALIDATION.md at the protected baseline,
not from simulation differences. No model or simulation output is modified.
"""
from pathlib import Path
import argparse, hashlib, json, subprocess
import numpy as np
import pandas as pd
from certification_statistics import equivalence, accepted, intensity_margin, INTENSITY_RELATIVE_MARGIN


def features(events, cohort, days):
    e=events.loc[events.application_id.isin(cohort.application_id)&events.age_seconds.lt(days*86400)].copy()
    counts=e.groupby('application_id').size()
    days_count=e.assign(age_day=np.floor(e.age_seconds/86400)).groupby(['application_id','age_day']).size().groupby(level=0).size()
    out=cohort[['application_id','application_created_at']].set_index('application_id').copy()
    out['weekday']=out.application_created_at.dt.weekday
    out['events']=counts.reindex(out.index,fill_value=0)
    out['eventful_days']=days_count.reindex(out.index,fill_value=0)
    return out


def bootstrap_difference(hist, sim, draws, seed):
    """Paired event/day features; resample whole applications within weekday."""
    rng=np.random.default_rng(seed)
    weights=sim.weekday.value_counts(normalize=True).sort_index()
    totals={}
    estimates={}
    for label,frame in [('historical',hist),('simulated',sim)]:
        point=np.zeros(2);boot=np.zeros((draws,2))
        for weekday,weight in weights.items():
            values=frame.loc[frame.weekday.eq(weekday),['events','eventful_days']].to_numpy(dtype=float)
            if len(values)<30:raise ValueError('unsupported weekday application cohort')
            point+=float(weight)*values.mean(axis=0)
            for b in range(draws):
                boot[b]+=float(weight)*values[rng.integers(0,len(values),size=len(values))].mean(axis=0)
        if point[1]<=0 or (boot[:,1]<=0).any():raise ValueError('empty eventful-day denominator')
        estimates[label]=dict(mean_sequence_length=float(point[0]),events_per_eventful_day=float(point[0]/point[1]))
        totals[label]=np.column_stack([boot[:,0],boot[:,0]/boot[:,1]])
    result={}
    for column,metric in enumerate(['mean_sequence_length','events_per_eventful_day']):
        reference=estimates['historical'][metric];actual=estimates['simulated'][metric]
        difference=actual-reference;ci=np.quantile(totals['simulated'][:,column]-totals['historical'][:,column],[.025,.975]).tolist();margin=intensity_margin(reference)
        result[metric]=dict(historical=reference,simulated=actual,difference=difference,relative_difference=difference/reference,application_bootstrap_95pct_difference=ci,equivalence_margin=margin,relative_margin=INTENSITY_RELATIVE_MARGIN,status=equivalence(difference,ci,margin))
    return result


def calculate(args):
    source=Path('docs/VALIDATION.md');text=source.read_bytes()
    original=subprocess.check_output(['git','show','cb43031e4cd0959fb59749d7a4512e903b04c4a4:docs/VALIDATION.md'])
    assert text==original and b'events/active-day and mean sequence length each within 10%' in text
    m=json.loads((args.metrics/'metrics.json').read_text());observable=json.loads((args.metrics/'observable-activity.json').read_text())
    h=pd.read_parquet(args.metrics/'historical_pl_terminal_safe.parquet');s=pd.read_parquet(args.metrics/'simulated_pl.parquet')
    h['age_seconds']=(pd.to_datetime(h.event_datetime)-pd.to_datetime(h.application_created_at)).dt.total_seconds()
    life=pd.read_parquet('artifacts/gold_events_v3/application_lifecycle_state.parquet').dropna(subset=['context_id','application_created_at']).sort_values(['context_id','application_created_at']);life['previous_creation']=life.groupby('context_id').application_created_at.shift()
    arrivals=pd.read_parquet('artifacts/gold_events_v3/may_application_arrivals.parquet');arrivals['application_created_at']=arrivals.activation_datetime
    arrivals=arrivals.loc[arrivals.application_id.isin(s.loc[s.creation.ge('2026-05-01'),'application_id'])]
    packets=pd.read_parquet('data/output/finalization-validation/historical-v3-subpop/historical_packets.parquet');kinds=packets.drop_duplicates('application_id').set_index('application_id').customer_kind
    prior={};blocked=set();rows=[]
    for row in packets.itertuples(index=False):
        if row.application_id in blocked:continue
        seen=prior.setdefault(row.application_id,set())
        for label in json.loads(row.labels):
            pd_seen=bool(seen&{'Professional details','Professional Details Submission'})
            if (label=='AIP Approved' and not pd_seen) or (label=='Push to Partner' and not(pd_seen and 'AIP Approved' in seen)):
                blocked.add(row.application_id);break
            rows.append((row.application_id,(row.event_datetime-row.application_created_at).total_seconds()));seen.add(label)
    prefix=pd.DataFrame(rows,columns=['application_id','age_seconds'])
    result={'margin_provenance':{'source':str(source),'source_baseline_commit':'cb43031e4cd0959fb59749d7a4512e903b04c4a4','source_sha256':hashlib.sha256(text).hexdigest(),'quote':'events/active-day and mean sequence length each within 10%','method':'Fixed existing 10% intensity margin, applied unchanged to all reported 1/3/7-day equal-age populations. Whole application bootstrap within weekday (400 draws, fixed weights), paired event/day features. Require entire 95% interval within +/- margin; keep existing guard-prefix cluster intervals separately.','historical_criterion_not_selected_using_simulated_errors':True},'draws':args.draws,'age':{}}
    for days in [1,3,7]:
        hc=life.loc[life.application_created_at.ge('2026-03-01')&life.application_created_at.le(pd.Timestamp('2026-05-01')-pd.Timedelta(days=days))]
        sc=arrivals.loc[arrivals.application_created_at.le(pd.Timestamp('2026-06-01')-pd.Timedelta(days=days))];groups={}
        for group in ['all','new_customer','returning']:
            hist=hc;sim=sc;guard_hist=hc
            if group!='all':
                returning=group=='returning';tag='RETURNING_CUSTOMER' if returning else 'NEW_CUSTOMER'
                hist=hc.loc[hc.application_created_at.ge('2026-04-01')&((hc.application_created_at-hc.previous_creation).dt.total_seconds().ge(30*86400) if returning else hc.previous_creation.isna())]
                guard_hist=hc.loc[hc.application_created_at.ge('2026-04-01')&hc.application_id.map(kinds).eq(tag)];sim=sc.loc[sc.customer_kind.eq(tag)]
            hs=features(h,hist,days);ss=features(s,sim,days);ps=features(prefix,guard_hist,days)
            raw=bootstrap_difference(hs,ss,args.draws,20260502+days*100+['all','new_customer','returning'].index(group))
            guard=bootstrap_difference(ps,ss,args.draws,20260502+days*1000+['all','new_customer','returning'].index(group))
            expected=m['age'][str(days)] if group=='all' else m['returning']['medium_subpopulations'][group][str(days)]
            assert abs(raw['mean_sequence_length']['historical']-expected['historical_weekday_adjusted']['events_per_app'])<1e-9
            assert abs(raw['events_per_eventful_day']['historical']-expected['historical_weekday_adjusted']['events_per_eventful_appday'])<1e-9
            assert abs(raw['mean_sequence_length']['simulated']-expected['corrected']['events_per_app'])<1e-9
            old=observable['age'][str(days)][group];margin=intensity_margin(old['historical_guard_supported_events_per_app'])
            assert abs(guard['mean_sequence_length']['historical']-old['historical_guard_supported_events_per_app'])<1e-9
            preserved={'difference':old['difference'],'application_cluster_95pct_difference':old['combined_app_cluster_95pct_difference'],'equivalence_margin':margin,'status':equivalence(old['difference'],old['combined_app_cluster_95pct_difference'],margin)}
            groups[group]={'raw':raw,'guard_prefix':guard,'preserved_guard_prefix_cluster_check':preserved,'historical_applications':len(hist),'guard_historical_applications':len(guard_hist),'simulated_applications':len(sim)}
        result['age'][str(days)]=groups
    statuses=[v['status'] for groups in result['age'].values() for group in groups.values() for comparison in ['raw','guard_prefix'] for v in group[comparison].values()]+[group['preserved_guard_prefix_cluster_check']['status'] for groups in result['age'].values() for group in groups.values()]
    result['status']='PASS WITH DOCUMENTED LIMITATION' if all(accepted(x) for x in statuses) else 'FAIL' if 'FAIL' in statuses else 'INCONCLUSIVE'
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(result['status'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--metrics',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--draws',type=int,default=400);args=parser.parse_args();calculate(args)
