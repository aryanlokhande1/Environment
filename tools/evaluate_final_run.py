"""Post-gate fidelity comparison; never used to fit empirical assets."""
from pathlib import Path
import json,gzip
import numpy as np,pandas as pd
from environment.runtime.gold_events_adapter import GOLD_COLUMNS
from environment.runtime.natural_transition import DOWNSTREAM,FOCUS
B=Path(__import__('sys').argv[3] if len(__import__('sys').argv)>3 else 'artifacts/gold_events_v3');O=Path(__import__('sys').argv[2]);O.mkdir(parents=True,exist_ok=True);R=Path(__import__('sys').argv[1]);L=Path('data/output/aws-reproduction/may-no-action-v2r2-20260502')
assert (R/'final-invariants.json').is_file(),'Correctness validation must pass before fidelity comparison'
assert json.loads((R/'final-invariants.json').read_text())['status']=='PASS'
Q=[.1,.25,.5,.75,.9,.95,.99];QN=['p10','p25','median','p75','p90','p95','p99']
def quant(v):
 x=pd.Series(v).dropna().astype(float);return dict(zip(QN,map(float,x.quantile(Q)))) if len(x) else {}
def plain(v):
 if isinstance(v,dict): return {str(k):plain(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [plain(x) for x in v]
 if isinstance(v,np.generic):return v.item()
 return v
def dump(name,value): (O/name).write_text(json.dumps(plain(value),indent=2,default=str))
def loadcp(path,date='2026-05-31'):
 with gzip.open(path/'checkpoints'/f'{date}.json.gz','rt') as f:return json.load(f)
def regcp(cp):
 x=pd.DataFrame([v for bucket in cp['lifecycle_registry'].values() for v in bucket]);x['application_created_at']=pd.to_datetime(x.creation_datetime);x['deadline']=pd.to_datetime(x.deadline);return x
life=pd.read_parquet(B/'application_lifecycle_state.parquet');life.application_id=life.application_id.astype(str)
# Reconstruct the same independently linked Phase 2 population from immutable
# source files. The evaluation has no dependency on disposable /tmp extracts.
import pyarrow.parquet as pq
from environment.persistence.checkpoint import file_hash
pieces=[];cohort_ids=set(life.application_id)
source_hashes=json.loads((B/'manifest.json').read_text())['source_files_sha256']
for filename,sha in source_hashes.items():
 source=Path('data/input/gold_history')/filename
 assert file_hash(source)==sha,'historical source hash mismatch'
 for batch in pq.ParquetFile(source).iter_batches(columns=list(GOLD_COLUMNS),batch_size=262144):
  frame=batch.to_pandas();pieces.append(frame.loc[frame.application_id.astype(str).isin(cohort_ids)].copy())
raw=pd.concat(pieces,ignore_index=True);raw.application_id=raw.application_id.astype(str)
counts=dict(cohort_applications=len(life),linked_rows=len(raw),exact_duplicate_rows=int(raw.duplicated(list(GOLD_COLUMNS)).sum()))
h=raw.merge(life[['application_id','context_id','application_created_at']],on='application_id',suffixes=('','_owner'),validate='many_to_one');h.event_datetime=pd.to_datetime(h.event_datetime)
mask=h.context_id.eq(h.context_id_owner)&h.application_created_at.notna()&h.event_datetime.ge(h.application_created_at)&h.event_datetime.lt(h.application_created_at+pd.Timedelta(days=30))
h=h.loc[mask].copy();counts['owner_lifecycle_rows']=len(h);h=h.drop_duplicates(list(GOLD_COLUMNS));counts['deduplicated_rows']=len(h)
ptp=h.loc[h.journey_substage.eq('Push to Partner')].groupby('application_id').event_datetime.min();h=h.loc[h.application_id.map(ptp).isna()|h.event_datetime.le(h.application_id.map(ptp))].copy();h=h.loc[~(h.journey_substage.eq('Push to Partner')&h.duplicated(['application_id','journey_substage']))].copy();counts['eligible_events']=len(h)
hl=life.dropna(subset=['application_created_at','context_id']).copy();hl['deadline']=hl.application_created_at+pd.Timedelta(days=30);counts['eligible_applications']=len(hl)
assert counts['eligible_events']==578324 and counts['eligible_applications']==81416,'Phase 2 historical comparison population changed'
cp={'legacy':loadcp(L),'corrected':loadcp(R)};registries={'historical':hl,**{k:regcp(v) for k,v in cp.items()}}
events={'historical':h}
for name,path in [('legacy',L),('corrected',R)]:events[name]=pd.concat([pd.read_parquet(p) for p in sorted((path/'may').glob('*/gold_events.parquet'))],ignore_index=True)
for k in events:
 e=events[k].copy();e.event_datetime=pd.to_datetime(e.event_datetime);e=e.sort_values(['application_id','event_datetime'],kind='stable').reset_index(drop=True);created=registries[k].set_index('application_id').application_created_at;e['creation']=e.application_id.map(created);e['age_seconds']=(e.event_datetime-e.creation).dt.total_seconds();events[k]=e

# Save only linked PL diagnostics, never a combined historical dataset.
h.to_parquet(O/'historical_pl_terminal_safe.parquet',index=False);events['corrected'].to_parquet(O/'simulated_pl.parquet',index=False)
def gapstats(x):
 x=x.dropna();return dict(n=len(x),seconds=quant(x),shares={str(t):float(x.le(t).mean()) for t in [0,60,300,900,3600,10800,21600,43200,86400]},over24h=float(x.gt(86400).mean()))
def loops(e,ids):
 e=e.sort_values(['application_id','event_datetime'],kind='stable');g=e.groupby('application_id',sort=False);s=e.journey_substage;t=e.event_datetime
 prev=g.journey_substage.shift();dt=g.event_datetime.diff().dt.total_seconds();positive=dt.gt(0)
 selfloop=s.eq(prev)&positive;two=s.eq(g.journey_substage.shift(2))&s.ne(prev)&positive&g.event_datetime.diff(2).dt.total_seconds().gt(dt)
 # Every edge must have positive elapsed time; ties do not define causal cycles.
 positive2=(g.event_datetime.shift(1)-g.event_datetime.shift(2)).dt.total_seconds().gt(0)
 positive3=(g.event_datetime.shift(2)-g.event_datetime.shift(3)).dt.total_seconds().gt(0)
 two &= positive2
 three=s.eq(g.journey_substage.shift(3))&s.ne(prev)&s.ne(g.journey_substage.shift(2))&prev.ne(g.journey_substage.shift(2))&positive&positive2&positive3
 firstlabel=e.groupby(['application_id','journey_substage']).event_datetime.transform('min');repeat=t.gt(firstlabel)
 firststage=e.groupby(['application_id','journey_stage']).event_datetime.transform('min');returnstage=t.gt(firststage)&e.journey_stage.ne(g.journey_stage.shift())&positive
 down=e.loc[s.isin(DOWNSTREAM)].groupby('application_id').event_datetime.min();focus=s.isin(FOCUS);dtime=e.application_id.map(down);rev=focus&dtime.lt(t);amb=focus&dtime.eq(t)
 out=dict(focus_first_pass=int((focus&~rev&~amb).sum()),focus_revisit=int(rev.sum()),focus_ambiguous_ties=int(amb.sum()),focus_revisit_share=float(rev.sum()/max(1,(focus&~amb).sum())))
 for label,mask,lag in [('self_transition',selfloop,1),('two_state_cycle',two,2),('three_state_cycle',three,3),('repeated_substage',repeat,1),('return_to_earlier_stage',returnstage,1)]:
  per=e.loc[mask].groupby('application_id').size().reindex(ids,fill_value=0);out[label]=dict(events=int(mask.sum()),applications=int(per.gt(0).sum()),mean_per_app=float(per.mean()),per_app_quantiles=quant(per),cycle_seconds=quant((t-g.event_datetime.shift(lag)).dt.total_seconds().loc[mask]))
 out['focus_revisit_per_app']=dict(mean=float(rev.sum()/len(ids)),quantiles=quant(e.loc[rev].groupby('application_id').size().reindex(ids,fill_value=0)))
 out['apps_with_focus_revisit']=set(e.loc[rev,'application_id'])
 return out

def age_metrics(name,days,subset=None):
 e=events[name];reg=registries[name];end=pd.Timestamp('2026-05-01' if name=='historical' else '2026-06-01');lo=pd.Timestamp('2026-03-01' if name=='historical' else '2026-05-01')
 l=reg.loc[reg.application_created_at.ge(lo)&reg.application_created_at.le(end-pd.Timedelta(days=days))].copy()
 if subset is not None:l=l.loc[l.application_id.isin(subset)]
 ids=l.application_id;e=e.loc[e.application_id.isin(ids)&e.age_seconds.lt(days*86400)].copy();n=e.groupby('application_id').size().reindex(ids,fill_value=0);ad=e.assign(age_day=np.floor(e.age_seconds/86400).astype(int)).groupby(['application_id','age_day']).size();fd=ad.groupby(level=0).size().reindex(ids,fill_value=0)
 l['events']=l.application_id.map(n);l['eventful_days']=l.application_id.map(fd);l['weekday']=l.application_created_at.dt.weekday;l['ptp']=l.application_id.isin(e.loc[e.journey_substage.eq('Push to Partner'),'application_id']);l['transitions']=l.events.sub(1).clip(lower=0)
 ptp_time=e.loc[e.journey_substage.eq('Push to Partner')].groupby('application_id').event_datetime.min();pd_time=e.loc[e.journey_substage.isin(['Professional Details Submission','Professional details'])].groupby('application_id').event_datetime.min();aip=e.loc[e.journey_substage.eq('AIP Approved')]
 eligible=aip.loc[aip.application_id.map(pd_time).lt(aip.event_datetime)&aip.event_datetime.le(aip.application_id.map(ptp_time)),'application_id'];l['observable_prerequisite_ptp']=l.application_id.isin(eligible)
 lp=loops(e,ids);rev=lp.pop('apps_with_focus_revisit');l['focus_revisited']=l.application_id.isin(rev)
 stats=dict(apps=len(l),events_per_app=float(n.mean()),events_per_app_quantiles=quant(n),eventful_appdays_per_app=float(fd.mean()),eventful_appday_count_quantiles=quant(fd),events_per_eventful_appday=float(len(e)/len(ad)),eventful_appday_events_quantiles=quant(ad),transitions_per_app=float(l.transitions.mean()),transition_quantiles=quant(l.transitions),ptp_probability=float(l.ptp.mean()),observable_prerequisite_ptp_probability=float(l.observable_prerequisite_ptp.mean()),expiry_probability=0.,loops=lp,
 weekday=l.groupby('weekday').agg(apps=('events','size'),events_per_app=('events','mean'),eventful_appdays_per_app=('eventful_days','mean'),ptp_probability=('ptp','mean'),observable_prerequisite_ptp_probability=('observable_prerequisite_ptp','mean'),transitions_per_app=('transitions','mean')).to_dict('index'),
 substage_events_per_app=(e.journey_substage.value_counts()/len(l)).to_dict(),event_name_events_per_app=(e.event_name.value_counts()/len(l)).to_dict(),ptp_by_focus_regime=l.groupby('focus_revisited').ptp.agg(['size','mean']).to_dict('index'))
 return stats,l,e

results=dict(population=counts,run_id=R.name,runtime='corrected-v2',seed=20260502,age={})
age_tables={}
for days in [1,3,7]:
 groups={};age_tables[days]={}
 for name in events:groups[name],age_tables[days][name],_=age_metrics(name,days)
 weights={k:v['apps']/groups['corrected']['apps'] for k,v in groups['corrected']['weekday'].items()}
 groups['historical_weekday_adjusted']={key:sum(weights[k]*groups['historical']['weekday'][k][key] for k in weights) for key in ['events_per_app','eventful_appdays_per_app','ptp_probability','observable_prerequisite_ptp_probability','transitions_per_app']}
 # Weighted empirical event-count CDF, not averages of weekday quantiles.
 hist=age_tables[days]['historical'];freq=hist.weekday.value_counts();w=hist.weekday.map({k:weights[k]/freq[k] for k in weights})
 x=hist.assign(weight=w).sort_values('events');cum=x.weight.cumsum().to_numpy();values=x.events.to_numpy()
 groups['historical_weekday_adjusted']['events_per_app_quantiles']=dict(zip(QN,[float(values[min(np.searchsorted(cum,q),len(values)-1)]) for q in Q]))
 groups['historical_weekday_adjusted']['events_per_eventful_appday']=groups['historical_weekday_adjusted']['events_per_app']/groups['historical_weekday_adjusted']['eventful_appdays_per_app']
 for field in ['eventful_days','transitions']:
  z=hist.assign(weight=w).sort_values(field);cw=z.weight.cumsum().to_numpy();vv=z[field].to_numpy()
  groups['historical_weekday_adjusted'][field+'_quantiles']=dict(zip(QN,[float(vv[min(np.searchsorted(cw,q),len(vv)-1)]) for q in Q]))
 he=events['historical'];he=he.loc[he.application_id.isin(hist.application_id)&he.age_seconds.lt(days*86400)].copy();hg=he.groupby('application_id',sort=False);hs=he.journey_substage;ht=he.event_datetime
 pdiff=hg.event_datetime.diff().dt.total_seconds().gt(0);p2=(hg.event_datetime.shift(1)-hg.event_datetime.shift(2)).dt.total_seconds().gt(0);p3=(hg.event_datetime.shift(2)-hg.event_datetime.shift(3)).dt.total_seconds().gt(0)
 masks={'self_transition':hs.eq(hg.journey_substage.shift())&pdiff,'two_state_cycle':hs.eq(hg.journey_substage.shift(2))&hs.ne(hg.journey_substage.shift())&pdiff&p2,'three_state_cycle':hs.eq(hg.journey_substage.shift(3))&hs.ne(hg.journey_substage.shift())&hs.ne(hg.journey_substage.shift(2))&hg.journey_substage.shift().ne(hg.journey_substage.shift(2))&pdiff&p2&p3,'repeated_substage':ht.gt(he.groupby(['application_id','journey_substage']).event_datetime.transform('min')),'return_to_earlier_stage':ht.gt(he.groupby(['application_id','journey_stage']).event_datetime.transform('min'))&he.journey_stage.ne(hg.journey_stage.shift())&pdiff}
 wm=hist.set_index('application_id').weekday.map({k:weights[k]/freq[k] for k in weights});adjusted_loops={}
 for label,mask in masks.items():
  per=he.loc[mask].groupby('application_id').size().reindex(hist.application_id,fill_value=0);z=pd.DataFrame({'value':per,'weight':wm}).sort_values('value');cw=z.weight.cumsum().to_numpy();vv=z.value.to_numpy()
  adjusted_loops[label]={'mean_per_app':float((per*wm).sum()),'per_app_quantiles':dict(zip(QN,[float(vv[min(np.searchsorted(cw,q),len(vv)-1)]) for q in Q]))}
 groups['historical_weekday_adjusted']['loops']=adjusted_loops

 results['age'][str(days)]=groups
 print('AGE',days,{k:(v.get('events_per_app'),v.get('ptp_probability')) for k,v in groups.items()},flush=True)

results['timing']={};results['terminal']={};results['loops_all']={}
for name,e in events.items():
 g=e.groupby('application_id',sort=False);gap=g.event_datetime.diff().dt.total_seconds();dest=e.assign(previous=g.journey_substage.shift(),gap=gap)
 transitions=[]
 for (a,b),z in dest.dropna(subset=['gap']).groupby(['previous','journey_substage'],observed=True):
  if len(z)>=200:transitions.append(dict(source=a,destination=b,**gapstats(z.gap)))
 results['timing'][name]=dict(application=gapstats(gap),by_transition=transitions,by_destination={k:gapstats(z.gap) for k,z in dest.dropna(subset=['gap']).groupby('journey_substage') if len(z)>=200},by_lifecycle_age={k:gapstats(z.gap) for k,z in dest.assign(age_bucket=pd.cut(dest.age_seconds/86400,[-1,1,3,7,14,30])).dropna(subset=['gap']).groupby('age_bucket',observed=True) if len(z)>=30})
 reg=registries[name];new=reg.loc[reg.application_created_at.ge('2026-03-01' if name=='historical' else '2026-05-01')].copy();end=pd.Timestamp('2026-05-01' if name=='historical' else '2026-06-01')
 if name=='historical':new=new.loc[new.application_created_at.lt('2026-04-01')]
 ne=e.loc[e.application_id.isin(new.application_id)];pt=ne.loc[ne.journey_substage.eq('Push to Partner')].drop_duplicates('application_id');ptimes=pt.set_index('application_id').event_datetime
 expired=~new.application_id.isin(pt.application_id)&new.deadline.le(end);censored=~new.application_id.isin(pt.application_id)&new.deadline.gt(end)
 pdtime=ne.loc[ne.journey_substage.isin(['Professional Details Submission','Professional details'])].groupby('application_id').event_datetime.min()
 aips=ne.loc[ne.journey_substage.eq('AIP Approved')].copy();aips['pd_time']=aips.application_id.map(pdtime);aips['ptp_time']=aips.application_id.map(ptimes)
 eligible=aips.loc[aips.pd_time.lt(aips.event_datetime)&aips.ptp_time.notna()&aips.event_datetime.le(aips.ptp_time)].application_id.unique()
 weak=aips.loc[aips.pd_time.le(aips.event_datetime)&aips.ptp_time.notna()&aips.event_datetime.le(aips.ptp_time)].application_id.unique()
 predecessor=dest.loc[dest.application_id.isin(pt.application_id)&dest.journey_substage.eq('Push to Partner')]
 results['terminal'][name]=dict(population='Mature March complete 30-day' if name=='historical' else 'May new, right-censored',apps=len(new),ptp_apps=len(pt),ptp_probability=len(pt)/len(new),expiry_apps=int(expired.sum()),expiry_probability=float(expired.mean()),censored_apps=int(censored.sum()),time_to_ptp_seconds=quant(pt.age_seconds),time_to_expiry_seconds=quant(np.repeat(30*86400,int(expired.sum()))),ptp_lifecycle_age_counts=pt.assign(age_bucket=pd.cut(pt.age_seconds/86400,[-1,1,3,7,14,30])).age_bucket.value_counts().sort_index().to_dict(),ptp_previous_state=predecessor.previous.value_counts().to_dict(),ptp_with_strict_observed_pd_aip=int(len(eligible)),ptp_with_weak_observed_pd_aip=int(len(weak)),ptp_without_any_observed_aip=int((~pt.application_id.isin(aips.application_id)).sum()))
 lp=loops(e,reg.application_id);lp.pop('apps_with_focus_revisit');results['loops_all'][name]=lp

# Prior historical lifecycle / identity continuity and observed entrant behavior.
ordered=hl.sort_values(['context_id','application_created_at']);ordered['prior_app']=ordered.groupby('context_id').application_id.shift();ordered['prior_creation']=ordered.groupby('context_id').application_created_at.shift();ordered['creation_gap_days']=(ordered.application_created_at-ordered.prior_creation).dt.total_seconds()/86400
april=ordered.loc[ordered.application_created_at.ge('2026-04-01')&ordered.application_created_at.lt('2026-05-01')].copy();returning=april.loc[april.creation_gap_days.ge(30)].copy();depth=h.groupby('application_id').size();last=h.groupby('application_id').event_datetime.max();firstptp=h.loc[h.journey_substage.eq('Push to Partner')].groupby('application_id').event_datetime.min()
returning['prior_ptp']=returning.prior_app.isin(firstptp.index);returning['prior_depth']=returning.prior_app.map(depth);returning['previous_last_event']=returning.prior_app.map(last);returning['terminal_time']=returning.prior_app.map(firstptp).fillna(returning.prior_creation+pd.Timedelta(days=30))
rs=dict(april_entrants=len(april),with_any_observed_prior=int(april.prior_app.notna().sum()),valid_returning=len(returning),valid_returning_fraction=len(returning)/len(april),creation_gap_days=quant(returning.creation_gap_days),gap_since_prior_last_event_days=quant((returning.application_created_at-returning.previous_last_event).dt.total_seconds()/86400),gap_since_prior_terminal_days=quant((returning.application_created_at-returning.terminal_time).dt.total_seconds()/86400),prior_observed_ptp=int(returning.prior_ptp.sum()),prior_inferred_expiry=int((~returning.prior_ptp).sum()),prior_history_depth=quant(returning.prior_depth),entrant_behavior={})
for kind,ids in [('returning',set(returning.application_id)),('no_observed_prior',set(april.loc[april.prior_app.isna(),'application_id']))]:
 rs['entrant_behavior'][kind]={str(days):age_metrics('historical',days,ids)[0] for days in [1,3,7]}
contexts=set(hl.context_id.astype(str))
rs['simulated_identity']={k:dict(new_apps=len(reg.loc[reg.application_created_at.ge('2026-05-01')]),new_apps_reusing_historical_context=int(reg.loc[reg.application_created_at.ge('2026-05-01')].context_id.isin(contexts).sum())) for k,reg in registries.items() if k!='historical'}
# The same equal-age definitions as Phase 2, separated by observed prior identity.
arrivals=pd.read_parquet(B/'may_application_arrivals.parquet')
rs['medium_subpopulations']={}
for kind,hids,sids in [
 ('returning',set(returning.application_id),set(arrivals.loc[arrivals.customer_kind.eq('RETURNING_CUSTOMER'),'application_id'])),
 ('new_customer',set(april.loc[april.prior_app.isna(),'application_id']),set(arrivals.loc[arrivals.customer_kind.eq('NEW_CUSTOMER'),'application_id']))]:
 rows={}
 for days in [1,3,7]:
  hs,ht,he=age_metrics('historical',days,hids);ss,st,se=age_metrics('corrected',days,sids)
  weights=st.weekday.value_counts(normalize=True);adjusted={}
  for field,col in [('events_per_app','events'),('eventful_appdays_per_app','eventful_days'),('transitions_per_app','transitions'),('ptp_probability','ptp')]:
   mean=sum(float(w)*ht.loc[ht.weekday.eq(k),col].mean() for k,w in weights.items())
   var=sum(float(w)**2*ht.loc[ht.weekday.eq(k),col].var()/len(ht.loc[ht.weekday.eq(k)]) for k,w in weights.items())
   adjusted[field]=float(mean);adjusted[field+'_mean_95pct_interval']=[float(mean-1.96*np.sqrt(var)),float(mean+1.96*np.sqrt(var))]
  adjusted['events_per_eventful_appday']=adjusted['events_per_app']/adjusted['eventful_appdays_per_app']
  weekday_loops={k:loops(he.loc[he.application_id.isin(ht.loc[ht.weekday.eq(k),'application_id'])],ht.loc[ht.weekday.eq(k),'application_id']) for k in weights.index}
  for label in ['self_transition','two_state_cycle','three_state_cycle','repeated_substage','return_to_earlier_stage']:
   adjusted[label+'_per_app']=sum(float(w)*weekday_loops[k][label]['mean_per_app'] for k,w in weights.items())
  rows[str(days)]={'historical_weekday_adjusted':adjusted,'corrected':ss,'historical_timing':gapstats(he.groupby('application_id').event_datetime.diff().dt.total_seconds()),'corrected_timing':gapstats(se.groupby('application_id').event_datetime.diff().dt.total_seconds())}
 rs['medium_subpopulations'][kind]=rows
results['returning']=rs

# Carried landmark comparison. Primary reproduces the original April-1 strata;
# sensitivity uses same-weekday (Friday) historical landmarks and prior depth.
snap=pd.read_parquet(B/'starting_snapshot.parquet');hist_p=h.loc[h.journey_substage.eq('Push to Partner')].groupby('application_id').event_datetime.min()
def carry_frame(name,bound):
 e=events[name];reg=registries[name];bound=pd.Timestamp(bound)
 if name=='historical':
  prior=e.loc[e.event_datetime.lt(bound)].sort_values('event_datetime').drop_duplicates('application_id',keep='last')
  x=reg.merge(prior[['application_id','event_datetime','journey_substage']],on='application_id');x=x.loc[x.application_created_at.lt(bound)&x.deadline.gt(bound)&~x.application_id.isin(hist_p.loc[hist_p.lt(bound)].index)]
  x['prior_count']=x.application_id.map(e.loc[e.event_datetime.lt(bound)].groupby('application_id').size())
 else:
  x=reg.loc[reg.application_created_at.lt(bound)].merge(snap[['application_id','snapshot_event_datetime','snapshot_substage']],on='application_id').rename(columns=dict(snapshot_event_datetime='event_datetime',snapshot_substage='journey_substage'));x['prior_count']=x.application_id.map(depth)
 x['age_days']=(bound-x.application_created_at).dt.total_seconds()/86400;x['silence_days']=(bound-x.event_datetime).dt.total_seconds()/86400;x['remaining_days']=(x.deadline-bound).dt.total_seconds()/86400
 future=e.loc[e.event_datetime.ge(bound)&e.application_id.isin(x.application_id)];n=future.groupby('application_id').size();first=future.groupby('application_id').event_datetime.min();pt=future.loc[future.journey_substage.eq('Push to Partner')].application_id
 x['active']=x.application_id.isin(n.index);x['events']=x.application_id.map(n).fillna(0);x['first_event_hours']=(x.application_id.map(first)-bound).dt.total_seconds()/3600;x['ptp']=x.application_id.isin(pt);x['expired']=~x.ptp&x.deadline.le('2026-05-01' if name=='historical' else '2026-06-01')
 x['age']=pd.cut(x.age_days,[0,7,14,21,30],include_lowest=True).astype(str);x['silence']=pd.cut(x.silence_days,[-1,1,7,14,21,31]).astype(str);x['depth']=pd.cut(x.prior_count,[0,3,6,10,np.inf],include_lowest=True).astype(str);x['landmark']=bound
 return x
carry={};frames={}
for name,bound in [('historical','2026-04-01'),('legacy','2026-05-01'),('corrected','2026-05-01')]:
 x=carry_frame(name,bound);frames[name]=x;carry[name]=dict(initialized=len(x),age_days=quant(x.age_days),silence_hours=quant(x.silence_days*24),remaining_days=quant(x.remaining_days),starting_substage=x.journey_substage.value_counts().to_dict(),prior_event_depth=quant(x.prior_count),active_apps=int(x.active.sum()),active_fraction=float(x.active.mean()),events=int(x.events.sum()),ptp_apps=int(x.ptp.sum()),expiry_apps=int(x.expired.sum()),inactive_expiry_apps=int((x.expired&~x.active).sum()),time_to_first_event_hours=quant(x.first_event_hours),events_per_reactivated_app=float(x.loc[x.active,'events'].mean()),events_per_reactivated_app_quantiles=quant(x.loc[x.active,'events']),events_per_all_app_quantiles=quant(x.events))
keys=['journey_substage','age','silence'];g=frames['historical'].groupby(keys).active.agg(['size','mean']);matched={}
for name in ['legacy','corrected']:
 x=frames[name].join(g,on=keys);z=x.loc[x['size'].ge(30)];matched[name]=dict(supported_apps=len(z),coverage=len(z)/len(x),observed_active=int(z.active.sum()),expected_active=float(z['mean'].sum()),relative_deficit=1-float(z.active.sum())/float(z['mean'].sum()))
 counts_by=z.groupby(keys).size().rename('may_n');rates_with=g.join(counts_by).dropna(subset=['may_n'])
 estimation_variance=float((rates_with.may_n**2*rates_with['mean']*(1-rates_with['mean'])/rates_with['size']).sum())
 predictive_variance=estimation_variance+float((rates_with.may_n*rates_with['mean']*(1-rates_with['mean'])).sum())
 expected=matched[name]['expected_active']
 matched[name]['approx_95pct_expected_interval']=[max(0,expected-1.96*np.sqrt(estimation_variance)),expected+1.96*np.sqrt(estimation_variance)]
 matched[name]['approx_95pct_predictive_active_interval']=[max(0,expected-1.96*np.sqrt(predictive_variance)),expected+1.96*np.sqrt(predictive_variance)]
fridays=pd.concat([carry_frame('historical',day) for day in pd.date_range('2026-03-06','2026-04-17',freq='7D')],ignore_index=True)
# Only complete lifecycle follow-up; avoid treating late-April censoring as expiry.
fridays=fridays.loc[fridays.deadline.le('2026-05-01')];keys2=keys+['depth'];rates=fridays.groupby(keys2).agg(opportunities=('active','size'),unique_apps=('application_id','nunique'),rate=('active','mean'));sensitivity={}
for name in ['legacy','corrected']:
 x=frames[name].join(rates,on=keys2);z=x.loc[x.unique_apps.ge(30)];sensitivity[name]=dict(supported_apps=len(z),coverage=len(z)/len(x),observed_active=int(z.active.sum()),expected_active=float(z.rate.sum()),relative_deficit=1-float(z.active.sum())/float(z.rate.sum()))
carry['matched_april1']=matched;carry['matched_fridays_depth']=sensitivity;carry['friday_reference']=dict(landmarks=7,opportunities=len(fridays),unique_apps=fridays.application_id.nunique(),active_opportunities=int(fridays.active.sum()))
results['carried']=carry
# Runtime suppression/support is not inferred from a small realized row count.
from environment.runtime.hazard import StageContinuationModel
from environment.runtime.natural_transition import RegimeNaturalTransitionModel
model=StageContinuationModel(B);nat=RegimeNaturalTransitionModel(B);init=loadcp(R,'2026-04-30');support=dict(no_natural_successor=0,no_remaining_timing_donor=0,beyond_last_donor=0)
for st in init['states'].values():
 elapsed=(pd.Timestamp('2026-05-01')-pd.Timestamp(st['last_journey_datetime'])).total_seconds();stop=(pd.Timestamp(st['deadline'])-pd.Timestamp(st['last_journey_datetime'])).total_seconds();sub=st['journey_substage'];stage=st['journey_stage']
 if nat.lookup(st) is None:support['no_natural_successor']+=1
 donors=model._state_donors.get(sub) if sub in model._state_rows else model._stage_donors.get(stage) if stage in model._stage_rows else model.donors
 durations=donors.duration_seconds.to_numpy() if donors is not None else np.array([])
 support['no_remaining_timing_donor']+=int(not ((durations>=elapsed)&(durations<stop)).any());support['beyond_last_donor']+=int(len(durations)>0 and elapsed>durations.max())
results['carried']['unused_frozen_v2_model_support']=support
# Extract exact diagnostic reasons without interpreting audit draws as realized events.
from collections import Counter
for name,path in [('legacy',L),('corrected',R)]:
 c=Counter();cc=Counter();carryids=set(loadcp(path,'2026-04-30')['states'])
 for p in sorted((path/'audit').glob('*/explanations.jsonl.gz')):
  with gzip.open(p,'rt') as f:
   for line in f:
    row=json.loads(line)
    if row['category']=='ORGANIC_CONTINUATION':
     ex=row['explanation'];reason=ex.get('reason',ex.get('selected_outcome','UNKNOWN'));c[reason]+=1
     if row['application_id'] in carryids:cc[reason]+=1
 results.setdefault('organic_audit',{})[name]=dict(all=dict(c),carry=dict(cc))
dump('metrics.json',results)
print('ANALYSIS COMPLETE',flush=True)
