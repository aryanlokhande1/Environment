"""Meaningful empirical/runtime invariants for the new bundle."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from environment.empirical.joint import product_limit
from environment.models.artifact_loader import ArtifactLoader
from environment.runtime.joint_continuation import JointContinuationModel
from environment.persistence.checkpoint import file_hash
B=Path(__file__).parents[1]/'artifacts/gold_events_v3'

def test_product_limit_accounts_for_censoring_and_no_event_mass():
    x=pd.DataFrame(dict(duration_seconds=[5.,10.,30.],observed=[True,True,False],next_labels=['["A"]','["B"]','[]']))
    z=product_limit(x);assert z.mass.tolist()==pytest.approx([1/3,1/3]);assert z.mass.sum()==pytest.approx(2/3)
    x=pd.DataFrame(dict(duration_seconds=[5.,10.,20.],observed=[True,False,True],next_labels=['["A"]','[]','["B"]']))
    z=product_limit(x);assert z.mass.tolist()==pytest.approx([1/3,2/3])

def test_new_bundle_preserves_frozen_ptp_and_source_contract():
    assert len(ArtifactLoader(B).validate())==31
    for name in ['ptp_hazard.csv','ptp_hazard_event_delays.parquet','ptp_hazard_manifest.json']:
        assert file_hash(B/name)==file_hash(B.parent/'gold_events_v2'/name)
    m=json.loads((B/'joint_continuation_manifest.json').read_text());assert m['ptp_hazard_changed'] is False
    assert m['ptp_runtime_changed'] is True
    assert len(m['source_files_sha256'])==2
    for p in B.iterdir():
        if p.suffix in ['.json','.csv']:assert b'\r' not in p.read_bytes()

def test_finite_supported_groups_keep_zero_edges_and_no_terminal_or_duplicate():
    m=JointContinuationModel(B);multi=[]
    for times,cdf,labels,n in m.pools.values():
        for packet in labels:
            if len(packet)>1:multi.append(packet)
            assert len(packet)==len(set(packet))
            assert not set(packet)&{'Application Created','campaign_sent'}
            if 'Push to Partner' in packet:assert packet==('Push to Partner',)
            if 'AIP Approved' in packet and 'Professional Details Submission' in packet:
                assert packet.index('Professional Details Submission')<packet.index('AIP Approved')
    assert ('Product Selection','Bank Approved') in multi
    assert max(map(len,multi))==2  # observed accepted groups, not unbounded walks

def test_conditional_silence_tail_has_support_and_polling_never_resamples():
    m=JointContinuationModel(B);state=dict(journey_substage='listing',journey_stage='Application',last_journey_datetime='2026-04-02',creation_datetime='2026-04-02',deadline='2026-05-02',observed_journey_substages=['listing'],form_filled_seen=False,professional_details_seen=False,aip_approved_seen=False,done=False)
    now=pd.Timestamp('2026-05-01');out=m.sample(state,now,np.random.default_rng(7))
    assert out.evidence['survival_at_start']>0
    # A historical late successor exists even for old finite donor tails.
    times,_,labels,_=m.pools[('STATE','listing','','',False,'YOUNG','NEW_CUSTOMER')]
    assert (times>86400).any()
    for seed in range(300):
        sampled=m.sample(state,now,np.random.default_rng(seed))
        if sampled.when is not None:
            assert now<=sampled.when<pd.Timestamp(state['deadline'])

def test_returning_arrivals_have_real_prior_context_fresh_application_and_gap():
    x=pd.read_parquet(B/'may_application_arrivals.parquet');r=x.loc[x.customer_kind.eq('RETURNING_CUSTOMER')];l=pd.read_parquet(B/'application_lifecycle_state.parquet').dropna(subset=['application_created_at','context_id']).set_index('application_id')
    assert 0.15<len(r)/len(x)<0.30
    assert x.application_id.is_unique and r.snapshot_context_id.is_unique
    assert r.prior_application_id.isin(l.index).all()
    assert (r.snapshot_context_id.to_numpy()==r.prior_application_id.map(l.context_id).to_numpy()).all()
    assert ((r.activation_datetime-r.prior_application_id.map(l.application_created_at)).dt.total_seconds()>=30*86400).all()
    assert not x.application_id.isin(l.index).any()
    # A May1 carry context may return after its old lifecycle expires; the
    # arrival-time gap above, not permanent exclusion, is authoritative.


def test_competing_terminal_is_observed_not_independent_censoring():
    from environment.empirical.joint import episodes,fit_law
    # Equal source risk: one natural, one terminal, one complete inactive life.
    rows=[]
    for app,dest in [('a','listing'),('b','Push to Partner'),('c',None)]:
        rows.append(dict(application_id=app,event_datetime=pd.Timestamp('2026-03-01'),deadline=pd.Timestamp('2026-03-31'),state='AIP Approved',labels='["AIP Approved"]',journey_stage='Application',regime='UNCONDITIONED',ptp_eligible=True,source_age='YOUNG'))
        if dest:rows.append(dict(rows[-1],event_datetime=pd.Timestamp('2026-03-01 00:01'),state=dest,labels=json.dumps([dest])))
    e=episodes(pd.DataFrame(rows),'2026-03-01','2026-04-01');e=e.loc[e.state.eq('AIP Approved')]
    z=product_limit(e)
    assert z.mass.sum()==pytest.approx(2/3)
    assert z.set_index('labels').mass['["Push to Partner"]']==pytest.approx(1/3)
    # The old natural-only KM would censor terminal and inflate late natural mass.
    assert e.loc[e.next_state.eq('Push to Partner'),'observed'].all()
    e['ptp_eligible']=False
    # Reconstructing an unobservable terminal must retain inactivity, no guard relaxation.
    p=pd.DataFrame(rows);p['ptp_eligible']=False
    blocked=episodes(p,'2026-03-01','2026-04-01')
    assert blocked.loc[blocked.next_state.eq('Push to Partner'),'next_labels'].eq('[]').all()


def test_late_age_fallback_cannot_inherit_young_ptp():
    m=JointContinuationModel(B)
    for key,(times,cdf,labels,n) in m.pools.items():
        if key[-2]=='LATE':assert all('Push to Partner' not in p for p in labels)
        if not key[-3]:assert all('Push to Partner' not in p for p in labels)


def test_carried_tail_artifact_keeps_complete_life_denominators():
    from environment.runtime.carried_tail import CarryTailModel
    model=CarryTailModel(B)
    for times,cdf,labels,n in model.pools.values():
        assert n>=30 and cdf[-1]<=1+1e-9
        assert (times>=0).all()
        assert all(len(set(p))==len(p) and 'Application Created' not in p for p in labels)


def test_residual_tail_does_not_count_donor_expiry_as_permanent_inactivity():
    from environment.empirical.joint import fit_carry
    rows=[]
    for i in range(45):
        observed=15<=i<30
        rows.append(dict(application_id=str(i),weekday=4,state='listing',age_band=10,
                         silence_band=4,depth_band=0,observed=observed,
                         duration_seconds=1. if i<15 else 5. if observed else 10.,
                         next_labels='["Application Resume"]' if observed else '[]'))
    law=fit_carry(pd.DataFrame(rows))
    pool=law.loc[law.weekday.eq(4)&law.model_level.eq('SOURCE_AGE_SILENCE_DEPTH')]
    # Fifteen donors have already expired before the event-time risk set.
    assert pool.mass.sum()==pytest.approx(.5)


def test_carried_artifact_schema_is_checked_before_runtime(tmp_path):
    import shutil
    shutil.copytree(B,tmp_path/'bundle');bundle=tmp_path/'bundle'
    path=bundle/'carried_tail.parquet'
    pd.read_parquet(path).drop(columns=['weekday']).to_parquet(path,index=False)
    manifest=json.loads((bundle/'manifest.json').read_text())
    for record in manifest['artifacts']:
        if record['logical_name']=='carried_tail':record['sha256']=file_hash(path)
    (bundle/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='artifact schema mismatch.*weekday'):
        ArtifactLoader(bundle).validate()


def test_returning_gap_backoff_preserves_near_donors_and_eligibility():
    from environment.empirical.identity import eligible_gap_neighborhood
    gaps=np.array([29.,31.,32.,33.,80.,90.]);eligible=gaps>=30
    chosen,reason=eligible_gap_neighborhood(gaps,eligible,30.,minimum=3)
    assert reason=='WITHIN_3_DAYS'
    assert gaps[chosen].tolist()==[31.,32.,33.]
    chosen,reason=eligible_gap_neighborhood(gaps,eligible,40.,minimum=3)
    assert reason=='NEAREST_SUPPORTED_RANGE'
    assert gaps[chosen].tolist()==[31.,32.,33.]
    chosen,reason=eligible_gap_neighborhood(gaps,np.zeros(6,dtype=bool),40.)
    assert not chosen.any() and reason=='NO_ELIGIBLE_CONTEXT'
