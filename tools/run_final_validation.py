"""Run a medium candidate, or the single full run after an explicit fidelity gate."""
import argparse
import json
import shutil
from pathlib import Path
import pandas as pd
from environment.simulation import MaySimulationRunner
from environment.persistence.checkpoint import file_hash

parser=argparse.ArgumentParser()
parser.add_argument('--run-id',required=True)
parser.add_argument('--full',action='store_true')
parser.add_argument('--gate',type=Path)
args=parser.parse_args()
root=Path('data/output/finalization-validation/full' if args.full else 'data/output/finalization-validation/medium')
if args.full:
    if args.gate is None:raise ValueError('full run requires accepted medium fidelity gate')
    gate=json.loads(args.gate.read_text())
    if gate['status']!='PASS' or gate['bundle_sha256']!=file_hash(Path('artifacts/gold_events_v3/manifest.json')):raise ValueError('medium gate is not accepted for this exact bundle')
    if root.exists() and any(root.iterdir()):raise ValueError('the single full run already exists')
if (root/args.run_id).exists():raise ValueError('run already exists; do not overwrite validation evidence')
runner=MaySimulationRunner(run_id=args.run_id,stochastic_namespace='corrected-v2-reference-20260502',artifact_dir='artifacts/gold_events_v3',runtime_version='corrected-v2',seed=20260502,arrival_limit=None if args.full else 4096,historical_sources=[str(p) for p in sorted(Path('data/input/gold_history').glob('events_gold_*_2026*.parquet'))],output_root=root)
for day in pd.date_range('2026-05-01','2026-05-31'):
    if shutil.disk_usage('/').free<2*1024**3:raise RuntimeError('insufficient free storage')
    if day.day==16:runner=MaySimulationRunner.open(args.run_id,output_root=root)
    result=runner.run_day(day)
    print(day.date(),result['gold_rows'],result['ptp'],flush=True)
print(json.dumps(runner.validate_run()),flush=True)
