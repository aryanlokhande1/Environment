"""Report finite timestamp patterns separately from duplicate instrumentation."""
import argparse,json
from pathlib import Path
import pandas as pd

parser=argparse.ArgumentParser();parser.add_argument('metrics',type=Path);args=parser.parse_args()
A=Path('data/output/finalization-validation');audit=json.loads((A/'historical-v3-competing/burst-audit.json').read_text())
h=pd.read_parquet(A/'historical-v3-subpop/historical_packets.parquet');s=pd.read_parquet(args.metrics.parent/'simulated_pl.parquet')
patterns={tuple(sorted(p['labels'])):p for p in audit['supported_patterns']}
groups=s.groupby(['application_id','event_datetime']).journey_substage.agg(tuple);ties=groups.loc[groups.map(len).gt(1)]
counts=ties.map(lambda v:tuple(sorted(v))).value_counts();unsupported={str(k):int(v) for k,v in counts.items() if k not in patterns}
result={'method':'Distinct within-application exact timestamp groups; historical supported packets exclude duplicate instrumentation. Counts are aggregate evidence, not an equal-age marginal calibration target. March/April pattern support is reported separately.',
        'simulated_groups':len(groups),'simulated_tie_groups':len(ties),'simulated_extra_rows':int(ties.map(len).sub(1).sum()),'unsupported_simulated_groups':unsupported,
        'patterns':[dict(**p,simulated=int(counts.get(k,0)),historical_packets=int(h.labels.map(lambda v:tuple(sorted(json.loads(v)))==k).sum())) for k,p in patterns.items()]}
for name,mask in [('new_applications',s.creation.ge('2026-05-01')),('carried_applications',s.creation.lt('2026-05-01'))]:
    x=s.loc[mask];g=x.groupby(['application_id','event_datetime']).size();result[name]={'rows':len(x),'timestamp_groups':len(g),'tie_groups':int(g.gt(1).sum()),'extra_rows':int(g.sub(1).sum())}
(args.metrics.parent/'burst-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if unsupported:raise SystemExit('unsupported simulated timestamp pattern')
