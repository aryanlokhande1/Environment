# Known Limitations

## Campaign opportunity cadence

Historical Personal Loan data has about 1.52 raw sends per send-day. The current validated environment primarily represents one decision opportunity per active application-day. Campaign cadence therefore is not claimed to be perfectly calibrated.

`get_decision_opportunities(state, start_time, end_time)` is the stable extension point.

`TODO_INTRA_DAY_CAMPAIGN_OPPORTUNITY_MODEL`: fit and validate an immutable-history intra-day opportunity process before returning multiple timestamps. Do not emulate the missing process by target forcing or repeated arbitrary draws.

## Time/product scope

The frozen bundle is Personal Loan evidence from March-April 2026 and the packaged closed-loop scheduler supports May 2026 decision dates. Broader products or calendar periods require a separately validated versioned bundle.
