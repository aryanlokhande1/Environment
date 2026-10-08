from pathlib import Path
import json
import numpy as np,pandas as pd
from environment.empirical.joint import episodes,summarize_law,fit_carry
A=Path('data/output/finalization-validation/historical-v3-subpop');p=pd.read_parquet(A/'historical_packets.parquet');r=p.loc[p.customer_kind.eq('RETURNING_CUSTOMER')];returning=summarize_law(episodes(r,'2026-04-01','2026-04-11'),episodes(r,'2026-04-11','2026-05-01'))
x=pd.read_parquet(A/'historical_carry_risk.parquet');train=x.loc[x.landmark.lt('2026-04-01')];test=x.loc[x.landmark.ge('2026-04-01')].copy();law=fit_carry(train);pools={}
for key,g in law.groupby(['weekday','model_level','state','age_band','silence_band','depth_band']):pools[key]=(g.duration_seconds.to_numpy(),g.mass.to_numpy())
pred=[]
for row in test.itertuples(index=False):
 a=row.age_band;s=row.silence_band;d=row.depth_band;source=row.state
 candidates=[('SOURCE_AGE_SILENCE_DEPTH',source,a,s,d),('SOURCE_AGE_SILENCE',source,a,s,-1),('SOURCE_AGE',source,a,-1,-1),('SOURCE',source,-1,-1,-1),('AGE_SILENCE','',a,s,-1),('GLOBAL','',-1,-1,-1)]
 key=next(((w,)+k for w in [row.weekday,-1] for k in candidates if (w,)+k in pools),None)
 t,m=pools[key];pred.append(m[t<(row.deadline-row.landmark).total_seconds()].sum())
test['prediction']=pred;test['visible']=test.observed&test.next_labels.ne('[]');test['error']=test.visible.astype(float)-test.prediction
# Resample entire historical applications, including correlated daily landmarks.
g=test.groupby('application_id').agg(error=('error','sum'),n=('error','size'));rng=np.random.default_rng(20260502);draws=[]
for _ in range(400):
 z=g.iloc[rng.integers(0,len(g),size=len(g))];draws.append(z.error.sum()/z.n.sum())
result=dict(returning_temporal_validation=returning,carry_temporal_validation=dict(train_landmarks=len(train),train_unique_applications=train.application_id.nunique(),validation_landmarks=len(test),validation_unique_applications=len(g),observed_incidence=float(test.visible.mean()),predicted_incidence=float(test.prediction.mean()),observed_minus_predicted_app_cluster_95pct_interval=np.quantile(draws,[.025,.975]).tolist(),source_window='Complete creation+30d <= May1 lifecycles only; March landmark fit / April landmark validation',limitation='Correlated application landmarks are clustered; April also used for model selection'))
(A/'subpopulation-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
