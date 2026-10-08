# Corrected-v2 materiality certification

The simulator and `artifacts/gold_events_v3` were unchanged during this
certification correction. No new May simulation or empirical refit was run.
The recorded decision and measured effects are in
[corrected-v2-certification.json](corrected-v2-certification.json).
This certification concerns the existing organic training environment and
external controller contract; it does not certify a historical campaign policy.

## Pre-existing activity margin

The protected baseline commit
`cb43031e4cd0959fb59749d7a4512e903b04c4a4` already contains this Intensity gate
in [VALIDATION.md](VALIDATION.md):

> events/active-day and mean sequence length each within 10%

This is the first-priority evidence for the equivalence margin. It was not
estimated from simulated errors, enlarged to obtain a pass, or replaced by a
March/April-derived tolerance. The source text and baseline commit are checked
by `tools/calculate_activity_equivalence.py` and recorded with SHA256.

For each population (all new applications, new customers, returning customers)
and each complete 1/3/7-day horizon, the margin is exactly
`0.10 * historical_reference_mean`. The seven-day intensity contract is applied
unchanged to the additional diagnostic horizons. Sequence length includes the
same Gold events used by Phase 2. Eventful days retain the existing Phase 2
age-day denominator, not a newly selected calendar denominator.

Both raw weekday-adjusted historical comparisons and the supplementary
executable guard-prefix comparisons must pass separately. For raw sequence
length and events/eventful day, resample whole applications within weekday
strata, retaining paired event/day counts and fixed cohort weekday weights.
Use 400 bootstrap draws, consistent with existing application-level historical
validation. Require the entire 95% difference interval within `[-margin,+margin]`.
Apply the same criterion to the guard-prefix bootstrap comparisons and preserve
and check the previously reported application-cluster intervals independently.
The supplementary diagnostic never replaces the raw historical benchmark.

## PTP materiality

The eligible-episode audit declared material conditional error above five
percentage points, with matching risk competition and uncertainty.
[VALIDATION.md](VALIDATION.md) also pre-declares a five-point funnel reach margin.
Use the existing `0.05` absolute margin for conditional population rates and
require the complete application-cluster difference interval inside it.
Statistically nonzero differences below that margin remain visible limitations.
The denominator, risk competition, raw PTP deficit and all original intervals
remain reported. No PTP asset, guard, or runtime change accompanies this rule.

## Loop inflation

A positive point difference cannot establish inflation. Use the existing
application-cluster difference intervals; intervals including zero do not
establish a positive increase. Overall earlier-stage returns differ by
`0.00788/app` with interval `[-0.02931,0.04507]`; returning-customer differences
are `0.02710/app` with interval `[-0.07711,0.13131]`.

No numeric loop materiality margin was invented. If a future interval establishes
a positive increase without a pre-specified materiality margin, the checker
returns INCONCLUSIVE and requires a materiality audit. A material failure requires
supported evidence beyond an independently justified margin.

## Classification and retained limitations

- PASS: practically equivalent within historical support, or no supported loop inflation.
- PASS WITH DOCUMENTED LIMITATION: statistically nonzero, but the entire interval is inside the justified equivalence margin.
- FAIL: uncertainty supports a discrepancy outside the justified materiality margin.
- INCONCLUSIVE: the interval crosses a materiality boundary; equivalence is not certified.

The existing gate contains 17 checks, including historical burst support.
No check was removed to change the reported denominator. The four previous
failures were certification-rule defects: two point-only loop comparisons and
two zero-difference tests that conflated statistical and practical significance.
The original strict gate, raw effects and all intervals are retained locally.

Known limitations remain: Wednesday carry benchmark sensitivity and deficit;
underrepresented >24-hour timing tails; returning prior-gap p95 drift
(51.56 versus historical 46.00 days); observational campaign-response assets;
raw PTP deficit partly caused by prerequisite observability; April used for
validation/model selection. Conditional activity/PTP differences are disclosed.

## Reproduce certification without simulation

Use the existing run at
`data/output/finalization-validation/certification-full/may-corrected-v2-final-20260502`
and its saved metrics. Never run May again for this certification correction.

```bash
.venv/bin/python tools/calculate_activity_equivalence.py \
  --metrics data/output/finalization-validation/certification/full-metrics \
  --output data/output/finalization-validation/freeze-certification/activity-equivalence.json
.venv/bin/python -m pytest -o addopts= -q tests/test_certification_statistics.py
.venv/bin/python -m pytest -o addopts= -q
```

Record the passing test logs and run `tools/certify_frozen_candidate.py` with
`--activity`, `--pytest-log`, `--focused-log`, and `--output`. The full run and
historical inputs remain local immutable evidence; generated runs are excluded
from Git. Freeze evidence is outside v3 so every empirical bundle byte remains
identical to the medium/full-certified candidate.
