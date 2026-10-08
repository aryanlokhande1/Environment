"""Corrected-v2 integration contracts, exercised through real empirical assets."""
from copy import deepcopy
from pathlib import Path
import json
import pandas as pd
import pytest
from environment import Environment,EnvironmentState,EnvironmentAction,ExternalAgentSession
from environment.runtime.semantics import FINAL
from environment.runtime.gold_events_adapter import GOLD_COLUMNS
from environment.runtime.joint_continuation import JointOutcome

B=Path(__file__).parents[1]/'artifacts/gold_events_v3'
START=pd.Timestamp('2026-06-01T09:00:00')

def env(run='test',ns='contract-reference'):
    return Environment(B,run_id=run,stochastic_namespace=ns,runtime_version=FINAL)

def state(e,app='new-contract-app'):
    raw=e._runtime.arrivals.iloc[0].to_dict();raw.update(application_id=app,application_created_at=START,snapshot_context_id='contract-context',snapshot_event_datetime=START)
    return EnvironmentState(e._runtime.new_state(raw,activation=START))

def logical(rows):return [{k:v for k,v in r.items() if k!='run_id'} for r in rows]

@pytest.mark.parametrize('hours',[1,3,12,24])
def test_generic_calendar_wait_segmentation(hours):
    e=env();initial=state(e);end=START+pd.Timedelta(days=3)
    whole=deepcopy(initial);one=e.advance(whole,EnvironmentAction.wait(),START,end)
    split=deepcopy(initial);cursor=START;events=[]
    while cursor<end and not split.terminal:
        stop=min(end,cursor+pd.Timedelta(hours=hours));r=e.advance(split,EnvironmentAction.wait(),cursor,stop);events+=r.events
        split=EnvironmentState(json.loads(json.dumps(split.payload,default=str)),json.loads(json.dumps(split.pending_events,default=str)));cursor=stop
    assert one.events==events
    assert whole.pending_events==split.pending_events
    for k in ['random_counter','event_sequence','natural_generation','ptp_generation','done','success','journey_substage']:
        assert whole.payload.get(k)==split.payload.get(k)

def test_namespace_separates_storage_from_stochastic_identity():
    a=env('operational-a');b=env('operational-b');c=env('operational-a','different-realization')
    ra=a.advance(state(a),EnvironmentAction.wait(),START,START+pd.Timedelta(days=1));rb=b.advance(state(b),EnvironmentAction.wait(),START,START+pd.Timedelta(days=1));rc=c.advance(state(c),EnvironmentAction.wait(),START,START+pd.Timedelta(days=1))
    assert logical(ra.events)==logical(rb.events)
    assert ra.state.payload==rb.state.payload and ra.pending_events==rb.pending_events
    assert logical(ra.events)!=logical(rc.events) or ra.state.payload['last_rng_reference']!=rc.state.payload['last_rng_reference']

def test_namespace_state_mismatch_rejected_before_mutation():
    a=env();s=state(a);before=deepcopy(s)
    with pytest.raises(ValueError,match='namespace'):env(ns='wrong').advance(s,EnvironmentAction.wait(),START,START+pd.Timedelta(hours=1))
    assert s==before

def session(tmp_path,*,e=None):
    e=e or env();source=tmp_path/'source.parquet'
    if not source.exists():
        # Same customer, prior lifecycle, future same-customer and other-customer
        # rows; none of the future or other-customer observations may leak.
        rows=[dict(application_id='prior',context_id='contract-context',event_datetime=START-pd.Timedelta(days=40),journey_stage='Application',journey_substage='Application Created'),dict(application_id='new-contract-app',context_id='contract-context',event_datetime=START,journey_stage='Application',journey_substage='Application Created'),dict(application_id='future',context_id='contract-context',event_datetime=START+pd.Timedelta(days=40),journey_stage='Application',journey_substage='Push to Partner'),dict(application_id='other',context_id='another',event_datetime=START-pd.Timedelta(hours=1),journey_stage='Application',journey_substage='Application Created')]
        pd.DataFrame(rows).to_parquet(source,index=False)
    return ExternalAgentSession(e,application_id='new-contract-app',start_time=str(START),input_history=source,cumulative_history=tmp_path/'history.parquet',checkpoint=tmp_path/'checkpoint.json')

def test_observation_is_gold_only_customer_scoped_and_time_filtered(tmp_path):
    s=session(tmp_path);o=s.observe()
    assert list(o.history.columns)==list(GOLD_COLUMNS)
    assert set(o.history.application_id)=={'prior','new-contract-app'}
    assert o.history.event_datetime.max()<=o.simulated_now
    assert not hasattr(o,'pending_events') and not hasattr(o,'state')

@pytest.mark.parametrize('boundary',['after_history','after_checkpoint','after_commit'])
def test_external_boundary_crash_recovery(tmp_path,boundary):
    s=session(tmp_path)
    def crash(point):
        if point==boundary:raise RuntimeError('injected crash')
    s._loop._commit_hook=crash
    end=START+pd.Timedelta(hours=3)
    with pytest.raises(RuntimeError,match='injected'):s.decide(EnvironmentAction.wait(),end)
    recovered=session(tmp_path);step=recovered.decide(EnvironmentAction.wait(),end)
    reference_path=tmp_path/'reference';reference_path.mkdir();reference=session(reference_path);expected=reference.decide(EnvironmentAction.wait(),end)
    pd.testing.assert_frame_equal(step.observation.history,expected.observation.history)
    assert step.observation.simulated_now==end
    stored=json.loads((tmp_path/'checkpoint.json').read_text());assert stored['stochastic_namespace']=='contract-reference';assert stored['state']['_stochastic_namespace']=='contract-reference'
    retry=recovered.decide(EnvironmentAction.wait(),end);assert retry.events.empty


def test_multiple_decisions_and_midnight_are_caller_controlled(tmp_path):
    s=session(tmp_path)
    # This fixture is organically quiet so all three decisions remain possible.
    h=pd.read_parquet(s._loop.input_history);quiet=h.loc[h.application_id.eq('new-contract-app')].copy();quiet['journey_substage']='UNSUPPORTED_CONTRACT_STATE';pd.concat([h,quiet],ignore_index=True).to_parquet(s._loop.input_history,index=False)
    s._loop.environment._runtime.sim.campaign_motif=None
    s._loop.environment._runtime.sim.action_model=None
    s._loop.environment._runtime.sim.campaign_response={}
    s._loop.environment._runtime.joint.sample=lambda state,now,rng: JointOutcome(None,(),{'reason':'fixture'})
    first=s.decide(EnvironmentAction.campaign('SMS','MORNING'),START+pd.Timedelta(hours=3))
    assert len(first.operator_audit['decisions'])==1
    assert first.events.journey_substage.eq('campaign_sent').sum()==1
    second=s.decide(EnvironmentAction.wait(),START+pd.Timedelta(hours=9));assert len(second.operator_audit['decisions'])==1
    third=s.decide(EnvironmentAction.wait(),START+pd.Timedelta(days=1,hours=3));assert len(third.operator_audit['decisions'])==1
    assert third.observation.simulated_now==START+pd.Timedelta(days=1,hours=3)
    assert json.loads(s._loop.checkpoint.read_text())['state']['decision_sequence']==3


def test_future_queue_is_not_observation_and_boundary_events_are_visible(tmp_path):
    s=session(tmp_path);e=s._loop.environment
    e._runtime.joint.sample=lambda state,now,rng: JointOutcome(None,(),{"reason":"fixture"})
    original=e._runtime.process_day
    def process(*args,**kwargs):
        payload,pending,day=args[:3]
        if payload['decision_sequence']==0:
            e._runtime._schedule(payload,pending,START+pd.Timedelta(hours=3),'CAMPAIGN_RESPONSE_EVENT',dict(response='campaign_open_read',decision_id='prior-send',decision_time=START.isoformat(),resolved_send_time=START.isoformat()))
        return original(*args,**kwargs)
    e._runtime.process_day=process
    before=s.observe();assert not before.history.journey_substage.eq('campaign_open_read').any()
    step=s.decide(EnvironmentAction.wait(),START+pd.Timedelta(hours=3))
    assert step.observation.history.journey_substage.eq('campaign_open_read').sum()==1
    assert step.observation.history.event_datetime.max()<=step.observation.simulated_now


def test_resume_rejects_namespace_change(tmp_path):
    s=session(tmp_path);s.decide(EnvironmentAction.wait(),START+pd.Timedelta(hours=1))
    with pytest.raises(ValueError,match='namespace'):session(tmp_path,e=env(ns='wrong')).observe()


def test_explicit_next_time_required_for_v2():
    e=env()
    with pytest.raises(ValueError,match='explicit next decision'):e.step(state(e),EnvironmentAction.wait(),START)


def test_namespace_send_outcomes_and_customer_history_match_across_run_ids(tmp_path):
    (tmp_path/'left').mkdir();(tmp_path/'right').mkdir()
    left=session(tmp_path/'left',e=env('storage-left'))
    right=session(tmp_path/'right',e=env('storage-right'))
    for action,end in [(EnvironmentAction.send('SMS','MORNING'),START+pd.Timedelta(hours=3)),
                       (EnvironmentAction.wait(),START+pd.Timedelta(hours=9))]:
        a=left.decide(action,end);b=right.decide(action,end)
        pd.testing.assert_frame_equal(a.events,b.events)
        pd.testing.assert_frame_equal(a.observation.history,b.observation.history)
        assert a.terminal==b.terminal
        if a.terminal:break


def test_carried_wait_segmentation_retains_one_residual_risk_draw():
    e=env();raw=e._runtime.arrivals.iloc[0].to_dict()
    raw.update(application_id='carried-contract',application_created_at=START-pd.Timedelta(days=28),
               snapshot_context_id='carried-context',snapshot_event_datetime=START-pd.Timedelta(days=28),
               snapshot_substage='listing',snapshot_stage='Application',prior_event_count=3,
               observed_journey_substages=['Application Created','listing'])
    initial=EnvironmentState(e._runtime.new_state(raw,activation=START))
    whole=deepcopy(initial);one=e.advance(whole,EnvironmentAction.wait(),START,START+pd.Timedelta(days=1))
    split=deepcopy(initial);events=[]
    for hour in [0,6,12,18]:
        if split.terminal:break
        result=e.advance(split,EnvironmentAction.wait(),START+pd.Timedelta(hours=hour),START+pd.Timedelta(hours=hour+6));events+=result.events
        split=EnvironmentState(json.loads(json.dumps(split.payload,default=str)),json.loads(json.dumps(split.pending_events,default=str)))
    assert one.events==events and whole.pending_events==split.pending_events
    assert whole.payload['_carry_tail_evaluated'] and split.payload['_carry_tail_evaluated']
    assert whole.payload['random_counter']==split.payload['random_counter']
