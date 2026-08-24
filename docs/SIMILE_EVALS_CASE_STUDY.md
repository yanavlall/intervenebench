# InterveneBench: Behavioral Simulation Evals Case Study

This case study is organized around the central question in Simile's current
[Evaluations — Member of Technical Staff](https://jobs.ashbyhq.com/simile/33d75074-c23b-4a1f-bfdb-129bcc5be662)
role: how do we know when a simulation of human behavior is good enough to trust?

## Evaluation judgment

The first design choice was to make the eval decision-relevant. Matching average
survey responses can be useful, but it does not tell a customer whether a model
would choose the right product, message, or policy. InterveneBench therefore
scores a frozen intervention recommendation against a randomized human
experiment.

The primary metric is decision regret:

```text
human utility of the best arm - human utility of the simulator-selected arm
```

This exposes a failure mode hidden by exact-choice accuracy. In the prospective
panel, exact choice was only 3/6 and chance-compatible, yet all mistakes were
near-ties and mean normalized regret was `0.0035` versus `0.0410` for uniform
choice. The eval reports both; neither is allowed to stand in for the other.

## Noisy human ground truth

The project treats a randomized experiment as evidence about a population
quantity, not an infallible label. Each task freezes its utility scale, estimand,
weighting and missingness rules, tie rule, and practical tolerance before
scoring. Questionnaire bounds—not observed held-out minima and maxima—define
normalization.

Headline uncertainty resamples experiments rather than pretending that model
calls, arms, seeds, or participant rows create independent benchmark examples.
Finite action-space enumeration provides an additional exact random-policy
comparison for the five normalized tasks.

## Leakage-safe model evaluation

The pipeline enforces the following state transition:

```text
source-verified task
-> outcome-blind prompt bundle
-> model outputs and strict parsing
-> frozen recommendation and diagnostics
-> separate human-outcome reveal authorization
-> scoring and release decision
```

Recommendation artifacts are immutable and hash-bound. Target outcomes cannot
enter prompts, diagnostics, model selection, thresholds, or feature engineering.
Malformed model outputs are retained as unavailable cells—there is no semantic
repair that quietly changes the evaluation.

The completed confirmation run planned 1,464 calls. Exactly 1,404 outputs were
strictly valid; one prespecified Socrates task cell was recorded unavailable
without a rerun. Human outcomes and participant rows were never uploaded to
Modal.

## Model comparison and regression thinking

The suite compares model families, sizes, prompt and answer-order perturbations,
and text/vision interfaces. It also includes no-effect and classical baselines.

A retrospective Mistral comparator illustrates why model quality must be tied to
the customer decision. Mistral improved treatment-effect MAE (`0.0503` versus
`0.0637`) but did not improve exact choice (`2/5` for both) and slightly worsened
regret (`0.0043` versus `0.0035`). “Better effect estimates” did not mean “better
action.”

The repository includes an experiment-paired model-version regression gate so a
new version cannot pass by pooling correlated calls or by improving one aggregate
metric while materially worsening worst-case regret or schema validity.

## Trust and calibration

Outcome-free trust features included margin, entropy, repeated-generation
stability, answer-order sensitivity, and model disagreement. The prespecified
confirmation ranking failed: AURC was `0.711` versus `0.500` under random
abstention, and exact-choice AUROC was `0.222`.

The response was to hold the feature—not search for a favorable threshold after
seeing the labels. Practical-reliability classification also had no negative
cases, so calibration was marked non-estimable. The resulting product decision
is explicit: no confidence-based abstention policy is deployed.

## Human/model hybrid evaluation

The fallback system asks whether a small human pilot repairs an uncertain
synthetic decision. Pilot observations are disjoint from evaluation observations,
budgets are nested and sampled without replacement, and policies share the same
folds and seeds.

The result was a meaningful negative finding. Human-only, fixed-pseudocount,
empirical-Bayes, and outcome-free hedged policies all had higher point regret
than synthetic-only at every tested nonzero budget. Human-only pilots were
statistically worse at 10, 25, 50, and 100 observations.

The negative empirical-Bayes result repeated directionally from nine development
experiments to five untouched normalized confirmation experiments. Mechanism
analysis showed six harmful, three corrective, and six unchanged task-by-budget
cells; average harm was `13.6×` average correction. The project labels this an
exploratory pattern, not a causal decomposition.

## Product-quality decision

The evidence is converted into four separate release scopes:

| Scope | Decision |
|---|---|
| Limited research-stage candidate screening | Allowed |
| Autonomous intervention selection | Hold |
| Confidence-based abstention | Hold |
| Small-sample human fallback | Hold |

This prevents a narrow research success from silently authorizing a broader
customer-facing capability.

## What this demonstrates

- Evaluation taste: the metric is tied to action utility, with explicit gaming
  and claim boundaries.
- Statistical judgment: experiment-level inference, noisy winners, practical
  equivalence, sampling uncertainty, and non-estimable calibration are visible.
- LLM fluency: strict structured outputs, model-family comparisons, exposure
  caveats, prompt sensitivity, and architecture-specific failure handling.
- Research engineering: hash-bound artifacts, deterministic replay, fail-closed
  authorization, Modal execution, tests, and release gates.
- Behavioral-science rigor: source-verified RCTs, survey scales, weighting,
  causal estimands, and preserved population definitions.

## What it does not demonstrate

Six prospective experiments are not enough for a calibrated universal trust
model. Most current tasks are survey or behavioral-intention settings rather than
transactions or long-horizon field behavior. Public study materials may also
have appeared in foundation-model pretraining, so the defensible claim is
outcome-blind evaluation—not guaranteed pretraining absence.

Those limitations define the next real research step: a larger independently
precommitted panel with more behavioral outcomes. They are not solved by adding
more calls to the same six experiments.

## Reproduce the release

```bash
PYTHONPATH=src python3 -m intervenebench.public_cli verify --root .
```

The portable command verifies the aggregate findings, public case study, scoped
release decisions, public documents, figure, and frozen zero-authority program.
It requires neither local run artifacts nor raw data, makes no model calls, and
accesses no participant rows. Maintainers can add `--deep-replay` to rebuild the
synthesis from the restricted hash-pinned aggregate provenance.
