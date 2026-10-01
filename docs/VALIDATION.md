# Fidelity validation contract

## Current status

The latest preserved seven-day campaign-fidelity candidate failed its campaign
acceptance gate. This limits statistical claims about reproducing the historical
campaign scheduler, but it does not block external RL-selected timing or the
explicit WAIT/SEND demonstration driver.

Execution validation is separate from campaign fidelity. A completed local May
run must have 31 hash-linked daily commits, immutable historical source stats,
unique simulated row keys, chronological/terminal-safe May events, no June rows,
canonical combined schema, and exact historical + May = combined row counts.

## Pre-declared gates

These gates apply independently to carried-forward and newly arriving
applications. A candidate passes only when every invariant passes and sampling
uncertainty does not explain a material miss.

| Family | Seven-day acceptance gate |
|---|---|
| Integrity | zero post-terminal, duplicate, out-of-order, cross-application prerequisite, or guard violations |
| Replay | byte-equivalent Gold rows and equivalent checkpoint state after restart |
| Natural distributions | substage TV <= 0.05 and transition TV <= 0.10 |
| Funnel reach | absolute error <= 5 percentage points for PD, AIP, PTP and AIP-to-PTP |
| Resume/OTP | absolute error <= 5 percentage points, split by FIRST_PASS/REVISIT |
| Intensity | events/active-day and mean sequence length each within 10% |
| Campaign volume | opportunities, send-days, and sends each within 10%; sends/send-day within 10% |
| Campaign response | open/read, click, and campaign-to-journey each within 15% and historical 95% interval |
| Timing | reported p50/p90 latency relative error <= 20%, with terminal-safe censoring |

Proportions use Wilson 95% intervals. Count intervals use a cohort bootstrap by
application, not by row. TV comparisons use the union of observed categories.
Metrics must report numerator, denominator, excluded post-terminal rows,
right-censored rows, and carried/new cohort separately.

## Bounded execution

1. Run package tests and bundle validation.
2. Validate all reference hashes and schemas.
3. Fit a new, separately versioned candidate bundle; never modify
   `gold_events_v1`.
4. Run May 1-7 with seed `20260502`, capped at 5,000 carried-forward and 5,000
   arrivals, writing baseline and candidate to different run directories.
5. Evaluate every gate above. Stop if any gate fails.
6. Run corrected full-May only after the seven-day report is an unconditional
   PASS.

Historical lifecycle comparisons remove all rows after the first PTP or the
30-day deadline. Competing-send response windows are censored. PTP denominators
include only same-application eligible AIP risk time.
