# Known Limitations

## Campaign opportunity cadence

Historical matched Personal Loan data has about 1.52 raw sends per send-day. The current environment represents one decision opportunity per active application-day. Campaign cadence therefore is not accepted as calibrated.

`get_decision_opportunities(state, start_time, end_time)` is the stable extension point.

Gold Events contain realized sends but not latent decisions, NO_ACTION outcomes,
eligibility rejections, or scheduled sends suppressed before delivery. Those
rows cannot identify an environment opportunity process independently of the
historical policy. Supply the decision log specified in
`config/reference_inputs.json`, then fit a new versioned bundle. Do not emulate
the missing process by target forcing or repeated arbitrary draws.

The seven-day campaign-fidelity gate remains failed, so the corrected full-May
campaign-fidelity comparison remains unrun. This is distinct from the working
operational full-May WAIT/SEND simulation and combined-history export. See
`docs/VALIDATION.md`.

## Time/product scope

The frozen bundle is Personal Loan evidence from March-April 2026 and the packaged closed-loop scheduler supports May 2026 decision dates. Broader products or calendar periods require a separately validated versioned bundle.
