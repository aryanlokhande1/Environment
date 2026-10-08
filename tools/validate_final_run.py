"""Check final-version lifecycle, namespace and durable day-commit invariants."""
from pathlib import Path
import argparse,gzip,json
import pandas as pd
from environment.simulation import MaySimulationRunner
from environment.persistence.checkpoint import file_hash

def validate(run_dir:Path):
    runner=MaySimulationRunner.open(run_dir.name,output_root=run_dir.parent);report=runner.validate_run();namespace=runner.stochastic_namespace
    previous=None;registry={};expired=set();ptp=set();event_rows=0
    for marker in sorted((run_dir/'checkpoints').glob('*.commit.json')):
        commit=json.loads(marker.read_text());path=run_dir/commit['checkpoint'];assert file_hash(path)==commit['checkpoint_sha256']
        with gzip.open(path,'rt') as f:cp=json.load(f)
        assert cp['runtime_version']=='corrected-v2' and cp['stochastic_namespace']==namespace
        reg={x['application_id']:x for records in cp['lifecycle_registry'].values() for x in records}
        if previous:
            old,prior=previous;assert commit['previous_checkpoint_sha256']==prior['checkpoint_sha256']
            day=pd.Timestamp(cp['completed_day']);end=day+pd.Timedelta(days=1)
            gold=pd.read_parquet(run_dir/commit['gold_partition']);assert file_hash(run_dir/commit['gold_partition'])==commit['gold_partition_sha256'];event_rows+=len(gold)
            for row in gold.itertuples(index=False):
                life=reg[str(row.application_id)];when=pd.Timestamp(row.event_datetime);deadline=pd.Timestamp(life['deadline'])
                assert str(row.context_id)==life['context_id']
                assert pd.Timestamp(life['creation_datetime'])<=when<=deadline
                assert when<deadline or row.journey_substage=='Push to Partner'
                assert str(row.application_id) not in ptp,'post-PTP row, including same-time'
                if row.journey_substage=='Push to Partner':ptp.add(str(row.application_id))
            removed=set(old['states'])-set(cp['states']);removed|={app for app in reg if app not in registry and app not in cp['states']}
            for app in removed-ptp:
                assert day<=pd.Timestamp(reg[app]['deadline'])<=end
                assert app not in expired;expired.add(app)
            assert cp['terminal_counts'].get('APPLICATION_EXPIRED_30_DAYS',0)==len(expired)
            assert cp['terminal_counts'].get('PUSH_TO_PARTNER_SUCCESS',0)==len(ptp)
            for app,state in cp['states'].items():
                assert state['_runtime_version']=='corrected-v2' and state['_stochastic_namespace']==namespace
                assert not state['done'] and pd.Timestamp(state['_simulation_time'])==end
                assert pd.Timestamp(state['deadline'])>end and state['context_id']==reg[app]['context_id']
                for pending in cp['pending'][app]:assert end<=pd.Timestamp(pending['when'])<=pd.Timestamp(state['deadline'])
        previous=(cp,commit);registry=reg
    assert event_rows==report['may_rows']
    report.update(stochastic_namespace=namespace,checkpoint_hash_chain=True,exact_expiry_removals=len(expired),ownership_violations=0,post_expiry_rows=0,post_ptp_including_same_time=0,runtime_namespace_state_verified=True)
    (run_dir/'final-invariants.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_dir',type=Path);args=parser.parse_args();print(json.dumps(validate(args.run_dir),indent=2))
