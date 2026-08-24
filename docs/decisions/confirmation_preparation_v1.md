# Confirmation preparation decision (v1)

Date frozen: 2026-08-14

## Decision

Prepare the six outcome-sealed experiments as a **noncanonical prospective
confirmation panel**. Generate simulator recommendations and outcome-free
diagnostics before any human outcome reveal. Do not fit or deploy a calibrated
trust threshold at this sample size.

The panel is:

- `tcg8p`
- `pb2rr`
- `z358z`
- `ShannonS2`
- `Blair1131`
- `KlarS44`

The exact model matrix contains 1,152 base calls and 312 primary-model prompt
perturbation calls. An outcome-free convergence rule can add at most 236 calls,
for 1,700 total attempts. Automatic retries and semantic repair are disabled.
The frozen incremental compute ceiling is $125; this record does not authorize
execution, downloads, spending, participant-row access, or outcome reveal.

## Simulator suite

Text, sequence, and continuous tasks use Qwen3-8B as the primary simulator,
with Qwen3-14B and Qwen2.5-14B as generic comparators. Socrates is included only
where the released SocSci210 checkpoint mapping confirms the task or equivalent
fielding was unseen. It is excluded for `pb2rr` and `z358z` because those studies
were in its training split. `pb2rr` uses Qwen3-VL-8B as primary, Qwen2.5-VL-7B
as a vision comparator, and Qwen3-8B accessible-text output as a modality
ablation.

The development-fitted hashed-ridge treatment-effect baseline is frozen and
applied outcome-blind to the five normalized tasks. It is not applied to
`tcg8p`, whose uncapped dollar outcome is not commensurate with normalized
ordinal utility.

## Trust analysis

The trust result is a fixed ranking, not a classifier or calibrated probability.
The ranking averages direction-aligned midranks of five prespecified diagnostics:
top-two margin, resampled winner stability, prompt/interface sensitivity,
cross-model winner agreement, and cross-model arm-rank dispersion. Evaluation
after reveal uses all six experiments as six units, with fixed coverage counts
of 3/6, 5/6, and 6/6. There is no learned acceptance threshold and no deployed
accept/abstain policy.

## Human fallback

Budgets remain 0, 10, 25, 50, 100, and 250 total human observations. Synthetic
only and balanced humans only are the primary comparisons. The previously
frozen fixed-pseudocount, empirical-Bayes, balanced, and hedged fusion methods
remain labeled negative-result ablations; no further tuning is permitted after
their failure to improve development regret. `tcg8p` is reported separately in
raw USD per month and is excluded from pooled normalized regret.

## Claim boundary

Six independent experimental units can provide valuable prospective evidence
and a decisive demonstration of leakage-safe research practice. They cannot
support a universal trust-calibration claim or the repository's larger canonical
benchmark claim. Null or negative simulator, diagnostic, and fallback results
must be preserved.

