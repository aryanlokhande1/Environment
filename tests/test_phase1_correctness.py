"""Corrected world-time and manifest-last persistence regressions."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from environment import Environment, EnvironmentAction, EnvironmentState, LocalClosedLoop
from environment.persistence.checkpoint import CheckpointStore, file_hash
from environment.runtime.hazard import _PiecewiseHazard
from environment.runtime.semantics import CORRECTED, LEGACY
from environment.simulation import MaySimulationRunner

BUNDLE = Path(__file__).parents[1] / 'artifacts/gold_events_v2'
START = pd.Timestamp('2026-05-01')
WAIT = EnvironmentAction.no_action()


class GridDraw:
    def __init__(self, draw):
        self.draw = draw

    def random(self):
        return self.draw


def test_hazard_conditional_interval_semantics_and_no_event_mass():
    model = _PiecewiseHazard(pd.DataFrame(), pd.DataFrame(), {})
    rows = pd.DataFrame([
        dict(elapsed_lower_seconds=0, elapsed_upper_seconds=100, risk_set=100, hazard=.4),
        dict(elapsed_lower_seconds=100, elapsed_upper_seconds=200, risk_set=60, hazard=.5),
    ])
    donors = pd.DataFrame({'duration_seconds': [10, 20, 150, 190]})
    model.corrected = True
    def outcomes(start, stop):
        values = []
        for draw in (np.arange(1000) + .5) / 1000:
            result = model._sample(rows, donors, origin=START,
                now=START + pd.Timedelta(seconds=start),
                horizon=START + pd.Timedelta(seconds=stop),
                deadline=START + pd.Timedelta(seconds=200), rng=GridDraw(draw),
                model_level='TEST', support_key={})
            values.append(None if result.when is None else int((result.when - START).total_seconds()))
        return values
    # S(200)=(1-.4)(1-.5)=.3. Timing donors partition bin mass,
    # rather than creating caller-window-dependent probability holes.
    assert outcomes(0, 200).count(None) == 300
    assert outcomes(0, 200).count(10) == 200
    # Conditional on T>=15, S=.8: remaining incidence=.5/.8=.625.
    assert outcomes(15, 200).count(None) == 375
    assert outcomes(15, 200).count(20) == 250
    assert outcomes(21, 99).count(None) == 1000  # legitimately no donor mass here


def _state(env, raw=None):
    raw = raw or env._runtime.initial_population()[0]
    return EnvironmentState(env._runtime.new_state(raw, activation=START))


def _run(env, original, boundaries, resume=False):
    state, events = deepcopy(original), []
    cursor = START
    for end in boundaries:
        if state.terminal:
            break
        result = env.advance(state, WAIT, cursor, end)
        events.extend(result.events)
        cursor = end
        if resume and end == boundaries[len(boundaries) // 2]:
            # Same JSON representation as CheckpointStore, including absence mass.
            data = json.loads(json.dumps({'payload': state.payload, 'pending': state.pending_events}, default=str))
            state = EnvironmentState(data['payload'], data['pending'])
            env = Environment(BUNDLE, run_id=env.run_id, runtime_version=CORRECTED)
    return state, events


@pytest.mark.parametrize('hours', [1, 6, 24])
def test_hazard_wait_segmentation_and_checkpoint_replay(hours):
    env = Environment(BUNDLE, run_id='partition-test')
    raws = env._runtime.initial_population()[:8]
    for index in range(12):
        raw = env._runtime.arrivals.iloc[index].to_dict()
        raw['application_created_at'] = START
        raw['snapshot_event_datetime'] = START
        original = _state(env, raw)
        original.payload.update(creation_datetime=START.isoformat(), deadline=(START+pd.Timedelta(days=30)).isoformat(),
            last_journey_datetime=START.isoformat(), application_id=f'partition-app-{index}')
        if index % 2:
            original.payload.update(journey_substage='AIP Approved', journey_stage='Application',
                form_filled_seen=True, professional_details_seen=True, aip_approved_seen=True,
                latest_aip_datetime=START.isoformat())
        raws.append(original)

    for raw in raws:
        original = raw if isinstance(raw, EnvironmentState) else _state(env, raw)
        # New arrivals are activated at the chosen start for this API probe.
        if pd.Timestamp(original.payload['creation_datetime']) > START:
            continue
        end = min(START + pd.Timedelta(days=4), pd.Timestamp(original.payload['deadline']))
        if end <= START:
            continue
        boundaries = list(pd.date_range(START + pd.Timedelta(hours=hours), end, freq=f'{hours}h'))
        if not boundaries or boundaries[-1] != end:
            boundaries.append(end)
        whole, whole_events = _run(env, original, [end])
        split, split_events = _run(env, original, boundaries)
        resumed, resumed_events = _run(env, original, boundaries, resume=True)
        assert whole_events == split_events == resumed_events
        for key in ['done', 'success', 'terminal_datetime', 'random_counter', 'journey_substage',
                    'natural_generation', 'ptp_generation', 'event_sequence']:
            assert whole.payload.get(key) == split.payload.get(key) == resumed.payload.get(key)
        assert whole.pending_events == split.pending_events == resumed.pending_events


def _expiry_state(env):
    state = _state(env)
    deadline = START + pd.Timedelta(hours=3)
    state.payload.update(creation_datetime=(deadline - pd.Timedelta(days=30)).isoformat(),
        deadline=deadline.isoformat(), journey_substage='UNSUPPORTED_TEST_STATE',
        journey_stage='UNSUPPORTED_TEST_STAGE', form_filled_seen=False,
        professional_details_seen=False, aip_approved_seen=False, latest_aip_datetime=None)
    return state, deadline


@pytest.mark.parametrize('offset', [-1, 0, 1])
def test_expiry_before_exact_crossing_and_resume(offset):
    env = Environment(BUNDLE, run_id='expiry')
    state, deadline = _expiry_state(env)
    state.pending_events.append(dict(when=(deadline+pd.Timedelta(hours=1)).isoformat(),
        priority=3, order=999, kind='CAMPAIGN_RESPONSE_EVENT',
        payload={'response':'campaign_open_read'}))
    end = deadline + pd.Timedelta(nanoseconds=offset)
    result = env.advance(state, WAIT, START, end)
    assert result.events == []
    assert state.terminal == (offset >= 0)
    if offset < 0:
        stored = json.loads(json.dumps({'payload':state.payload,'pending':state.pending_events},default=str))
        state = EnvironmentState(stored['payload'], stored['pending'])
        result = env.advance(state, WAIT, end, deadline)
    assert state.terminal
    assert state.payload['terminal_datetime'] == deadline.isoformat()
    assert state.pending_events == []
    assert sum(x['category'] == 'REALIZED_EVENT' for x in result.explanations) <= 1
    with pytest.raises(ValueError, match='terminal application'):
        env.advance(state, WAIT, deadline, deadline + pd.Timedelta(hours=1))


def test_expiry_equal_time_ptp_preserves_transaction_priority():
    env = Environment(BUNDLE, run_id='ptp-tie')
    state, deadline = _expiry_state(env)
    state.payload.update(form_filled_seen=True, professional_details_seen=True,
                         aip_approved_seen=True, latest_aip_datetime=START.isoformat())
    env._runtime._schedule(state.payload, state.pending_events, deadline, 'TRANSACTION_EVENT',
                           {'response':'Push to Partner','ptp_outcome':True,'ptp_generation':0})
    result = env.advance(state, WAIT, START, deadline)
    assert state.terminal and state.payload['success']
    assert len(result.events) == 1 and result.events[0]['journey_substage'] == 'Push to Partner'
    assert not state.pending_events


class Transformer:
    def encode(self, history):
        return np.zeros(192)


class SendPolicy:
    def choose_action(self, embedding):
        return EnvironmentAction.campaign('SMS', 'MORNING')


def _loop(tmp_path, version=CORRECTED, **kwargs):
    source = tmp_path / 'source.parquet'
    if not source.exists():
        pd.DataFrame([dict(application_id='app', context_id='ctx', event_datetime=START,
            journey_stage='Application', journey_substage='Application Created')]).to_parquet(source,index=False)
    env = Environment(BUNDLE, run_id='persist', runtime_version=version, **kwargs)
    return LocalClosedLoop(env, input_history=source, cumulative_history=tmp_path/'history.parquet',
                           checkpoint=tmp_path/'checkpoint.json')


@pytest.mark.parametrize('boundary', ['after_history', 'after_checkpoint', 'after_commit'])
def test_crash_recovery_exactly_once(tmp_path, boundary):
    loop = _loop(tmp_path)
    def crash(point):
        if point == boundary:
            raise RuntimeError('injected crash')
    loop._commit_hook = crash
    with pytest.raises(RuntimeError, match='injected'):
        loop.run_step('app', START, Transformer(), SendPolicy())
    resumed = _loop(tmp_path)
    resumed.run_step('app', START, Transformer(), SendPolicy())
    history = pd.read_parquet(resumed.cumulative_history)
    simulated = history.loc[history.cumulative_row_key.notna()]
    assert len(simulated) > 0
    assert not simulated.cumulative_row_key.duplicated().any()
    reference_dir = tmp_path / 'reference'
    reference_dir.mkdir()
    reference = _loop(reference_dir)
    reference.run_step('app', START, Transformer(), SendPolicy())
    pd.testing.assert_frame_equal(history, pd.read_parquet(reference.cumulative_history))
    actual_state = resumed.checkpoints.load(resumed.checkpoint)
    expected_state = reference.checkpoints.load(reference.checkpoint)
    assert actual_state['state'] == expected_state['state']
    assert actual_state['pending_events'] == expected_state['pending_events']
    digest = file_hash(resumed.cumulative_history)
    class NeverCall:
        def encode(self, *args): raise AssertionError('already committed')
        def choose_action(self, *args): raise AssertionError('already committed')
    retried = resumed.run_step('app', START, NeverCall(), NeverCall())
    assert retried.result.events == []
    assert file_hash(resumed.cumulative_history) == digest
    assert resumed.checkpoints.load(resumed.checkpoint)['state']['decision_sequence'] == 1
    if not retried.result.terminal:
        resumed.run_step('app', START + pd.Timedelta(days=1), Transformer(), SendPolicy())
        assert resumed.checkpoints.load(resumed.checkpoint)['state']['decision_sequence'] == 2


@pytest.mark.parametrize('target', ['projection','snapshot','checkpoint','version','seed','models'])
def test_restore_rejects_corruption_and_identity_mismatch(tmp_path, target):
    loop = _loop(tmp_path)
    loop.run_step('app', START, Transformer(), SendPolicy())
    stored = loop.checkpoints.load(loop.checkpoint)
    if target == 'projection':
        loop.cumulative_history.write_bytes(b'bad parquet')
    elif target == 'snapshot':
        (tmp_path/stored['local_commit']['history']).write_bytes(b'bad snapshot')
    elif target == 'checkpoint':
        stored['state']['random_counter'] += 1
        loop.checkpoints.save(stored, loop.checkpoint)
    elif target == 'version':
        loop.environment.runtime_version = LEGACY
    elif target == 'seed':
        loop.environment.seed += 1
    else:
        loop.environment.artifact_hashes = {}
    with pytest.raises(ValueError, match='mismatch|disagrees'):
        loop.run_step('app', START + pd.Timedelta(days=1), Transformer(), SendPolicy())


def test_runtime_version_resume_and_legacy_boundary():
    env = Environment(BUNDLE, run_id='legacy', runtime_version=LEGACY)
    state, deadline = _expiry_state(env)
    env.advance(state, WAIT, START, deadline)
    assert not state.terminal  # frozen defect is deliberately retained
    corrected = Environment(BUNDLE, run_id='legacy')
    state.payload['_runtime_version'] = LEGACY
    with pytest.raises(ValueError, match='runtime_version'):
        corrected.advance(state, WAIT, deadline, deadline + pd.Timedelta(hours=1))


def test_pending_response_and_rng_survive_crash_after_commit(tmp_path):
    loop = _loop(tmp_path)
    # An unsupported observed state keeps this fixture organically quiet.
    history = pd.read_parquet(loop.input_history)
    quiet = history.iloc[-1].to_dict()
    quiet.update(journey_substage='UNSUPPORTED_TEST_STATE', journey_stage='UNSUPPORTED_TEST_STAGE')
    pd.concat([history, pd.DataFrame([quiet])], ignore_index=True).to_parquet(loop.input_history, index=False)
    # Add a previously scheduled engagement before the commit boundary.
    step = loop.environment.step
    def pending_step(state, action, when):
        result = step(state, WAIT, when)
        loop.environment._runtime._schedule(state.payload, state.pending_events,
            START + pd.Timedelta(days=1, hours=1), 'CAMPAIGN_RESPONSE_EVENT', {
                'decision_id':'pending', 'action_id':'SMS|THEME_PLACEHOLDER|NIGHT',
                'channel':'SMS','theme':'THEME_PLACEHOLDER','time_bucket':'NIGHT',
                'decision_time':START.isoformat(), 'resolved_send_time':START.isoformat(),
                'response':'campaign_open_read'})
        result.pending_events[:] = state.pending_events
        return result
    loop.environment.step = pending_step
    def crash(point):
        if point == 'after_commit': raise RuntimeError('injected')
    loop._commit_hook = crash
    with pytest.raises(RuntimeError):
        loop.run_step('app', START, Transformer(), SendPolicy())
    stored = loop.checkpoints.load(loop.checkpoint)
    resumed = _loop(tmp_path)
    resumed.run_step('app', START, Transformer(), SendPolicy())
    state = resumed._state(resumed._history(), 'app', START)
    assert state.pending_events == stored['pending_events']
    assert state.payload['random_counter'] == stored['state']['random_counter']
    result = resumed.environment.advance(state, WAIT, START+pd.Timedelta(days=1),
        START+pd.Timedelta(days=1,hours=2))
    assert sum(row['journey_substage']=='campaign_open_read' for row in result.events)==1


def test_real_checkpoint_roundtrip_at_expiry(tmp_path):
    env = Environment(BUNDLE, run_id='expiry-checkpoint')
    state, deadline = _expiry_state(env)
    before = deadline - pd.Timedelta(nanoseconds=1)
    env.advance(state, WAIT, START, before)
    store = CheckpointStore()
    path = tmp_path/'state.json'
    store.save({'runtime_version':CORRECTED,'state':state.payload,'pending':state.pending_events},path)
    data = store.load(path)
    restored = EnvironmentState(data['state'],data['pending'])
    env = Environment(BUNDLE, run_id='expiry-checkpoint',runtime_version=data['runtime_version'])
    env.advance(restored,WAIT,before,deadline)
    store.save({'state':restored.payload,'pending':restored.pending_events},path)
    assert store.load(path)['state']['done']
    assert store.load(path)['pending']==[]



def test_committed_history_cannot_resume_without_marker(tmp_path):
    loop = _loop(tmp_path)
    loop.run_step('app', START, Transformer(), SendPolicy())
    loop.checkpoint.unlink()
    with pytest.raises(ValueError, match='no committed checkpoint'):
        _loop(tmp_path).run_step('app', START, Transformer(), SendPolicy())
