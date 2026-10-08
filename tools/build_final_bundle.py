"""Build gold_events_v3 exclusively from hash-verified March-April Gold.

No output/simulation directory is read. Frozen bundles are copied, never edited.
The new joint continuation law uses exact-time product-limit empirical survival
and finite defensible co-occurrence groups; PTP hazard assets remain byte-exact.
"""
from __future__ import annotations
import argparse,json,shutil,sys
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from environment.empirical.joint import packets,episodes,fit_law,summarize_law,annotate_customers,carry_landmarks,fit_carry
from environment.persistence.checkpoint import file_hash
from environment.empirical.identity import eligible_gap_neighborhood
from environment.runtime.gold_events_adapter import GOLD_COLUMNS
from environment.runtime.base import stable_seed

SOURCES={'events_gold_1_bucket_000_20260331.parquet':'be321119e240ab62781751258b958160dbf51d10b36a51527b7074900cdbd2fa','events_gold_2_bucket_000_20260430.parquet':'957cc2adaa480ac98055848b21ebfe0d4c2659e641d4e703556f4e384367d0b8'}

def write_json(p,value):p.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8')

def history(root:Path,base:Path) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    life=pd.read_parquet(base/'application_lifecycle_state.parquet');ids=set(life.application_id);pieces=[];offset=0
    for name,sha in SOURCES.items():
        path=root/name
        if file_hash(path)!=sha:raise ValueError('immutable historical hash mismatch: '+name)
        source=pq.ParquetFile(path)
        for b in source.iter_batches(columns=list(GOLD_COLUMNS),batch_size=262144):
            x=b.to_pandas();x['_order']=np.arange(offset,offset+len(x));offset+=len(x);pieces.append(x.loc[x.application_id.isin(ids)].copy())
    h=pd.concat(pieces,ignore_index=True);n=len(h);dups=int(h.duplicated(list(GOLD_COLUMNS)).sum());h=h.drop_duplicates(list(GOLD_COLUMNS));h=h.merge(life[['application_id','context_id','application_created_at']],on='application_id',suffixes=('','_owner'),validate='many_to_one')
    h=h.loc[h.context_id.eq(h.context_id_owner)&h.application_created_at.notna()&h.event_datetime.ge(h.application_created_at)&h.event_datetime.lt(h.application_created_at+pd.Timedelta(days=30))].copy()
    h=h.sort_values(['application_id','event_datetime','_order'],kind='stable');pt=h.loc[h.journey_substage.eq('Push to Partner')].drop_duplicates('application_id').set_index('application_id');pttime=h.application_id.map(pt.event_datetime);ptorder=h.application_id.map(pt._order)
    h=h.loc[pttime.isna()|h.event_datetime.lt(pttime)|(h.event_datetime.eq(pttime)&h._order.le(ptorder))].copy()
    if h.journey_substage.isin(['campaign_sent','campaign_open_read','campaign_clicked']).any():raise ValueError('campaign attribution audit required before fitting joint organic law')
    return h,life,dict(raw_source_rows=offset,linked_rows=n,exact_duplicates_removed=dups,eligible_ordered_rows=len(h),eligible_applications=int(life.dropna(subset=['context_id','application_created_at']).shape[0]))

def ptp_audit(h,life):
    mature=life.loc[life.context_id.notna()&life.application_created_at.ge('2026-03-01')&life.application_created_at.lt('2026-04-01')];x=h.loc[h.application_id.isin(mature.application_id)].copy();pdtime=x.loc[x.journey_substage.isin(['Professional Details Submission','Professional details'])].groupby('application_id').event_datetime.min();pt=x.loc[x.journey_substage.eq('Push to Partner')].groupby('application_id').event_datetime.min();aip=x.loc[x.journey_substage.eq('AIP Approved')].drop_duplicates(['application_id','event_datetime']).copy();aip=aip.loc[aip.application_id.map(pdtime).lt(aip.event_datetime)];aip=aip.loc[aip.application_id.map(pt).isna()|aip.event_datetime.lt(aip.application_id.map(pt))];aip['next_aip']=aip.groupby('application_id').event_datetime.shift(-1);aip['ptp']=aip.application_id.map(pt);lifeend=aip.application_id.map(mature.set_index('application_id').application_created_at)+pd.Timedelta(days=30);aip['end']=pd.concat([aip.next_aip,lifeend.rename('life_end')],axis=1).min(axis=1);aip['event']=aip.ptp.ge(aip.event_datetime)&aip.ptp.lt(aip.end)
    return dict(model_changed=False,reason='Frozen PTP files preserved; corrected-v2 competing next-event law supersedes independent clocks after conditional March/April audit demonstrated young CDF errors >7pp and unsupported late-age global PTP backoff. Raw PTP marginal remains confounded by unobservable prerequisites.',runtime_changed=True,mature_apps=len(mature),raw_ptp_apps=len(pt),eligible_aip_apps=aip.application_id.nunique(),eligible_aip_episodes=len(aip),eligible_ptp_apps=aip.loc[aip.event].application_id.nunique(),eligible_ptp_episodes=int(aip.event.sum()),raw_ptp_without_observable_eligible_aip=int(len(pt)-aip.loc[aip.event].application_id.nunique()),censoring='latest eligible AIP to next eligible AIP, first PTP, or exact creation+30d; March complete-life cohort; source truncation and April immature applications excluded',modification_threshold='Material conditional held-out incidence/CDF error outside application-cluster uncertainty, with matching eligibility and risk-episode competition; raw all-entrant marginal difference is insufficient',frozen_manifest_eligible_episodes=55199,frozen_manifest_ptp_episodes=23731)

def arrivals(base:Path,out:Path,h:pd.DataFrame,life:pd.DataFrame):
    schedule=pd.read_parquet(base/'may_application_arrivals.parquet').copy();valid=life.dropna(subset=['application_created_at']).sort_values(['context_id','application_created_at']);valid['prior_creation']=valid.groupby('context_id').application_created_at.shift();valid['prior_app']=valid.groupby('context_id').application_id.shift();apr=valid.loc[valid.application_created_at.ge('2026-04-01')&valid.application_created_at.lt('2026-05-01')].copy();apr['returning']=(apr.application_created_at-apr.prior_creation).dt.total_seconds().ge(30*86400);apr['weekday']=apr.application_created_at.dt.weekday
    # Match the historical returner prior-history distribution using donor prior
    # lifecycles. Historical context identities stay real; applications stay new.
    last=valid.sort_values('application_created_at').drop_duplicates('context_id',keep='last').set_index('context_id');depth=h.groupby('application_id').size();pt=h.loc[h.journey_substage.eq('Push to Partner')].groupby('application_id').event_datetime.min()
    # Vector arrays avoid copying the entire historical registry per arrival.
    context_ids=last.index.astype(str).to_numpy();created_ns=last.application_created_at.astype('int64').to_numpy();prior_ids=last.application_id.astype(str).to_numpy();prior_ptp=np.isin(prior_ids,pt.index);prior_depth=last.application_id.map(depth).fillna(0).to_numpy().astype(int)
    # Exclude immutable source anomalies with overlapping historical lifecycles.
    invalid_contexts=set(valid.loc[(valid.application_created_at-valid.prior_creation).dt.total_seconds().lt(30*86400),'context_id'])
    used=np.isin(context_ids,list(invalid_contexts));records=[];rejected=0;weekday_groups={}
    for w in range(7):
        group=apr.loc[apr.weekday.eq(w)];group=group if len(group)>=200 else apr
        returning=group.loc[group.returning]
        weekday_groups[w]=(float(group.returning.mean()),(returning.application_created_at-returning.prior_creation).dt.total_seconds().to_numpy()/86400,returning.prior_app.isin(pt.index).to_numpy(),returning.prior_app.map(depth).fillna(0).to_numpy().astype(int))
    for row in schedule.sort_values(['activation_datetime','application_id'],kind='stable').to_dict('records'):
        when=pd.Timestamp(row['activation_datetime']);rng=np.random.default_rng(stable_seed(20260502,row['application_id'],'returning-identity-v3'));rate,gaps,outcomes,depths=weekday_groups[when.weekday()]
        row['customer_kind']='NEW_CUSTOMER';row['prior_application_id']=None;row['identity_gap_backoff']='NOT_RETURNING';row['target_prior_gap_days']=None
        if rng.random()<rate:
            d=int(rng.integers(len(gaps)));target=float(gaps[d]);candgap=(when.value-created_ns)/86400e9;row['target_prior_gap_days']=target;row['identity_gap_backoff']='EXACT_DAY'
            eligible=(candgap>=30)&~used
            outcome_eligible=eligible&(prior_ptp==outcomes[d])
            row['target_prior_ptp']=bool(outcomes[d]);row['prior_outcome_supported']=bool(outcome_eligible.any())
            matching,row['identity_gap_backoff']=eligible_gap_neighborhood(candgap,outcome_eligible if outcome_eligible.any() else eligible,target)
            selected=matching&(prior_ptp==outcomes[d])&(prior_depth//4==depths[d]//4)
            if selected.sum()<10:selected=matching&(prior_ptp==outcomes[d])
            if not selected.any():selected=matching
            indices=np.flatnonzero(selected)
            if len(indices):
                i=int(indices[int(rng.integers(len(indices)))]);used[i]=True
                row.update(snapshot_context_id=str(context_ids[i]),customer_kind='RETURNING_CUSTOMER',prior_application_id=str(prior_ids[i]))
            else:rejected+=1
        records.append(row)
    x=pd.DataFrame(records);x.to_parquet(out/'may_application_arrivals.parquet',index=False);manifest=json.loads((base/'may_arrival_manifest.json').read_text());manifest.update(model_version='PL_EMPIRICAL_MAY_ARRIVAL_V3',schedule_sha256=file_hash(out/'may_application_arrivals.parquet'),identity_model='empirical April weekday returner fraction; eligible terminal historical context, empirical day-gap/outcome/depth matching with nearest supported eligible gap range; no within-May context reuse',source_files_sha256=SOURCES,historical_april_entrants=len(apr),historical_valid_returners=int(apr.returning.sum()),generated_returners=int(x.customer_kind.eq('RETURNING_CUSTOMER').sum()),ineligible_draws_backed_off_to_new=rejected,history_continuity='Immutable March-April events stay associated with the reused context; fresh synthetic application IDs for May')
    write_json(out/'may_arrival_manifest.json',manifest);return manifest

def build(args):
    base=Path(args.base);out=Path(args.output);audit=Path(args.audit);audit.mkdir(parents=True,exist_ok=True)
    if out.exists():raise ValueError('new output bundle must not exist; do not mutate frozen artifacts')
    h,life,counts=history(Path(args.history),base);print('historical eligible',counts,flush=True);h.to_parquet(audit/'historical_ordered_pl.parquet',index=False)
    p,bursts=packets(h);p=annotate_customers(p,life);p.to_parquet(audit/'historical_packets.parquet',index=False);print('packets',bursts,flush=True)
    train=episodes(p,'2026-03-01','2026-04-01');test=episodes(p,'2026-04-01','2026-05-01');validation=summarize_law(train,test);production=episodes(p,'2026-03-01','2026-05-01');law=fit_law(production)
    # Eligibility/age/customer strata retain their own empirical risk sets.
    # Returning lifecycles are only observable in April; validate those on an
    # April early/late split rather than pretending March supplies returners.
    validation['returning_validation_method']='April1-10 source episodes vs April11-30; complete observation horizon separately censored'
    shutil.copytree(base,out);carry_risk=carry_landmarks(p);carry_risk.to_parquet(audit/'historical_carry_risk.parquet',index=False);fit_carry(carry_risk).to_parquet(out/'carried_tail.parquet',index=False);law.to_parquet(out/'joint_continuation.parquet',index=False)
    ptpa=ptp_audit(h,life);write_json(audit/'ptp-audit.json',ptpa);write_json(audit/'historical-validation.json',validation);write_json(audit/'burst-audit.json',bursts)
    write_json(out/'joint_continuation_manifest.json',dict(model_version='PL_JOINT_COMPETING_EVENT_SURVIVAL_V2',model_sha256=file_hash(out/'joint_continuation.parquet'),carried_tail_sha256=file_hash(out/'carried_tail.parquet'),carry_model_version='PL_RESIDUAL_TAIL_SURVIVAL_V2',carry_backoff='weekday source+age+silence+depth -> source+age+silence -> source+age -> source -> age+silence -> global; >=30 unique historical applications; then pooled weekday',source_files_sha256=SOURCES,source_period='Immutable March-April 2026 only; no May, simulated, synthetic or PostgreSQL rows',fit_window=['2026-03-01','2026-05-01'],validation_method='March source episodes censored April1 fit; April source episodes censored May1 holdout; product-limit CDF by source; application clustering required for inference',validation=validation,population=counts,build_configuration=dict(state_min=200,stage_min=500,regime_min=200,customer_kind='observable prior lifecycle gap >=30d; unobserved prior treated as new',carry_min_unique_apps=30,carry_age_bounds=[7,14,21,22,23,24,25,26,27,28,29],carry_expiry='right censor donor at exact expiry; sample residual event once and compete with target expiry',carry_landmarks='daily March1-April30, complete lifecycle deadline <=May1',burst_march_min=50,burst_april_min=25),backoff=['supported source+regime within eligibility/age','supported source within eligibility/age','supported source stage within eligibility/age','global historical within eligibility/age'],silence_semantics='condition exact empirical survival on elapsed silence; one draw per source episode through expiry; no polling redraw',same_time_order=bursts['canonical_order'],burst_patterns=bursts['supported_patterns'],ptp_hazard_changed=False,ptp_runtime_changed=True,terminal_competition='next journey packet and eligible PTP are mutually exclusive next outcomes; hidden-prerequisite PTP mass becomes inactivity without guard relaxation; frozen PTP files retained for replay but unused in corrected-v2',risk_strata=['observed PD+AIP eligibility','source lifecycle age <1d or >=1d','observed prior lifecycle >=30d earlier'],ptp_audit_evidence='historical-v3/holdout-detail.json: source-window and AIP-renewal censoring; >5pp conditional frozen CDF error in March and April, unsupported late-age global backoff'))
    arrival=arrivals(base,out,h,life);snap=pd.read_parquet(out/'starting_snapshot.parquet');snap['prior_event_count']=snap.application_id.map(h.groupby('application_id').size()).fillna(0).astype(int);snap['customer_kind']=snap.application_id.map(p.drop_duplicates('application_id').set_index('application_id').customer_kind);snap.to_parquet(out/'starting_snapshot.parquet',index=False)
    mm=json.loads((out/'model_manifest.json').read_text());mm['starting_snapshot_sha256']=file_hash(out/'starting_snapshot.parquet');write_json(out/'model_manifest.json',mm)
    manifest=json.loads((out/'manifest.json').read_text());manifest.update(bundle_version='gold_events_v3',runtime_semantics='corrected-v2',source_files_sha256=SOURCES,build_configuration=dict(builder='tools/build_final_bundle.py',historical_only=True),validation_window='March training / April validation with source-window censoring')
    for r in manifest['artifacts']:
        r['sha256']=file_hash(out/r['filename'])
        if r['filename'] in ['may_application_arrivals.parquet','may_arrival_manifest.json']:r['model_artifact_version']='PL_EMPIRICAL_MAY_ARRIVAL_V3'
        if r['filename']=='may_application_arrivals.parquet':
            r['conditioning_keys']=['weekday','positive empirical daily residual','April weekday returner rate','eligible prior lifecycle gap','prior PTP outcome','prior history depth']
            r['backoff_hierarchy']=['WEEKDAY','GLOBAL','PRIOR_OUTCOME_EXACT_DAY','PRIOR_OUTCOME_WITHIN_3_DAYS','PRIOR_OUTCOME_NEAREST_SUPPORTED_RANGE','ELIGIBLE_OUTCOME_UNAVAILABLE']
    template=dict(manifest['artifacts'][0]);template.update(logical_name='joint_continuation',filename='joint_continuation.parquet',sha256=file_hash(out/'joint_continuation.parquet'),model_artifact_version='PL_JOINT_COMPETING_EVENT_SURVIVAL_V2',support_unit='historical source occurrence at risk',conditioning_keys=['source','regime','ptp_eligible','source_age','customer_kind','destination packet','exact elapsed time'],backoff_hierarchy=['STATE_REGIME','STATE','STAGE','GLOBAL'])
    manifest['artifacts'].append(template);template=dict(template);template.update(logical_name='joint_continuation_manifest',filename='joint_continuation_manifest.json',sha256=file_hash(out/'joint_continuation_manifest.json'));manifest['artifacts'].append(template)
    template=dict(template);template.update(logical_name='carried_tail',filename='carried_tail.parquet',sha256=file_hash(out/'carried_tail.parquet'),model_artifact_version='PL_RESIDUAL_TAIL_SURVIVAL_V2',support_unit='complete historical lifecycle at a daily landmark',conditioning_keys=['weekday','source','age','silence','prior depth'],backoff_hierarchy=['WEEKDAY_SOURCE_AGE_SILENCE_DEPTH','WEEKDAY_SOURCE_AGE_SILENCE','WEEKDAY_SOURCE_AGE','WEEKDAY_SOURCE','WEEKDAY_AGE_SILENCE','WEEKDAY_GLOBAL','POOLED_WEEKDAY']);manifest['artifacts'].append(template)
    write_json(out/'manifest.json',manifest);write_json(out/'hashes.json',{p.name:file_hash(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='hashes.json'})
    write_json(audit/'build-summary.json',dict(population=counts,bursts=bursts,validation=validation,ptp_audit=ptpa,returning=dict(historical=arrival['historical_valid_returners']/arrival['historical_april_entrants'],generated=arrival['generated_returners']/len(pd.read_parquet(out/'may_application_arrivals.parquet')))))
    print('BUILD COMPLETE',out,flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',default='artifacts/gold_events_v2');parser.add_argument('--output',default='artifacts/gold_events_v3');parser.add_argument('--history',default='data/input/gold_history');parser.add_argument('--audit',default='data/output/finalization-validation/historical');build(parser.parse_args())
