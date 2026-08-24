# Independent Replication Stage v1

**Status:** active outcome-blind qualification stage; no paid inference or
human-outcome reveal authorized by this document.

## Decision

InterveneBench will attempt a second, independent prospective panel to test the
preliminary low-regret signal from the six-experiment confirmation. The goal is
not to increase call count on experiments whose outcomes are already known. It
is to add genuinely independent randomized experiments whose recommendations
and diagnostics are frozen before any target human outcome is opened.

The target is **16 new experiments**. A panel of **12** is the minimum
analyzable prospective replication. Fewer than 12 qualified experiments do not
authorize a reveal or a positive replication claim. A 12--15 experiment panel
may provide a bounded replication result but cannot satisfy the strong tier
defined below.

Every retained experiment must have a distinct fielding cluster and a distinct
conservative paradigm group. The panel must not reuse any of the nine revealed
development experiments or six revealed confirmation experiments.

SocSci210 remains primary: require at least 8 SocSci210 experiments in a
12-task panel and at least 10 in a 16-task panel. Unless at least four tasks use
behavioral or consequential outcomes, any positive conclusion is narrowed to
attitudinal/message-policy decisions rather than cross-outcome intervention
selection.

## Why another panel is warranted

In the first confirmation, exact choice was compatible with uniform random
selection: 3/6 exact winners, with `P(random count >= 3) = 0.373`. The
decision-regret result was more promising. On five normalized tasks, primary
mean regret was `0.00352` versus `0.04096` under uniform random choice. Only
`12/216` complete random-action combinations had equally low mean regret.

That is a legitimate signal but not a reliable general result. Five comparable
experiments are too few to distinguish a stable low-regret capability from a
favorable task sample. The new panel is the prespecified test of that question.

## Qualification gates

Candidates pass in this order:

1. stable source identity, fielding-level deduplication, and no overlap with a
   revealed task;
2. actions that a stable decision maker can deploy under a fixed factual world;
3. a source-grounded, bounded, monotone utility that can be normalized to
   `[0,1]` without target outcomes;
4. exact source-faithful treatment, stimulus, personalization, order, and
   sequence reconstruction;
5. an outcome-blind human-data mapping with assignment, missingness, weights,
   and nuisance standardization frozen without projecting the target outcome;
6. a tested simulator adapter covering every retained arm and required nuisance
   path.

A hard failure stops work on that candidate. The panel will not be padded with
measurement experiments, target attributes, fabricated factual worlds,
unstable utilities, missing source assets, exposed results, or multiple modules
from one respondent fielding.

## Frozen primary question

For each experiment `i`, let `R_i(S)` be normalized regret from the frozen
primary simulator policy and `E[R_i(U)]` be exact expected regret from choosing
uniformly over that experiment's admissible actions.

The primary estimand is the experiment-level paired mean:

```text
Delta_U = mean_i(R_i(S) - E[R_i(U)])
```

Lower is better. Exact winner agreement is secondary because near-tied arms can
make it unstable without creating meaningful decision loss.

Primary uncertainty treats the experiment as the unit. It includes a paired
experiment-cluster bootstrap, a fixed-panel uniform-action randomization test,
a magnitude-aware exact paired sign-flip test, and leave-one-experiment and
leave-one-paradigm-out sensitivity. Model draws, arms, nuisance paths, and
prompt variants never increase the independent sample size.

## Minimum replication gate

A 12--15 experiment panel supports a bounded positive replication only if all
of the following hold:

- normalized mean regret improves over uniform action by at least `0.01`;
- the one-sided 95% experiment-cluster upper bound for `Delta_U` is below zero;
- the fixed-panel uniform-action randomization probability is at most `0.05`;
- every leave-one-experiment and leave-one-paradigm mean `Delta_U` remains
  negative;
- the exact paired sign-flip probability is at most `0.10` as a supportive
  robustness check;
- versus both always-control and the pre-reveal frozen classical policy, the
  primary policy has point mean regret no larger and the one-sided 95% upper
  bound on excess regret is below `+0.01`.

These criteria require better-than-random decision value without a material
disadvantage to implementable simple policies. They do not establish universal
simulator trust.

## Strong replication gate

The stronger conclusion requires at least 16 qualified experiments, every
minimum criterion above, an exact paired sign-flip probability at most `0.05`,
and a one-sided 95% upper bound below zero for regret differences versus both
always-control and the frozen classical policy.

If the primary uniform-action gate passes but operational baselines only meet
noninferiority, report a bounded low-regret replication—not superiority over
simple decision policies.

For a negative completion, distinguish evidence from low power. A one-sided
95% lower bound above `-0.01` supports no practically meaningful improvement;
a lower bound above zero supports harm. Otherwise, a failed positive gate is
reported as non-replication or inconclusive rather than evidence of equivalence.

## Secondary outcomes

- exact intervention-choice accuracy under the heterogeneous arm-count null;
- worst-case normalized regret;
- treatment-effect magnitude and sign fidelity;
- selected-arm optimality under participant bootstrap;
- common practical-regret sensitivities at `0`, `0.0025`, `0.005`, `0.01`,
  `0.025`, and `0.05`;
- source stratum, modality, action count, and paradigm results, reported before
  any pooled interpretation.

No task-specific tolerance may be changed after reveal.

## Simulator policy

The replication headline retains the existing primary policy: Qwen3 8B for
text tasks and Qwen3-VL 8B for exact-image tasks. A frozen model-plurality
ensemble is secondary. Model expansion is used to test architecture diversity
and outcome-free disagreement, not to choose a winner after human outcomes.

The current six pinned checkpoints may be reused. The outcome-blind execution
plan may add one genuinely independent public text family and one genuinely
independent public vision family only after exact revisions, per-file manifests,
licenses, runtimes, and adapter canaries are frozen. Source/reverse answer order
is mandatory. Semantics-preserving prompt variants are limited and frozen.

## Trust analysis

The 15 revealed experiments are development data for one new trust procedure.
Feature definitions, directions, model form, threshold policy, and all
hyperparameters must be selected and frozen using only those experiments.
The new replication outcomes may evaluate that frozen procedure but may not
change it.

With 12--16 test experiments, risk--coverage and continuous-regret ranking are
primary. Calibration or universal trust claims remain out of scope. If label
balance is inadequate, classification metrics are reported as not estimable.

## Human fallback

Fallback candidates may be revised only on the 15 revealed experiments and
must be frozen before the new panel is revealed. Pilot and evaluation people
remain disjoint, budgets remain nested, and benchmark uncertainty resamples
experiments. A negative result is retained if limited human evidence again
fails to reduce regret.

## Compute boundary

This decision authorizes `$0` of spending and zero model calls. A separate,
hash-bound authorization is required after the contracts, assets, model
revisions, prompts, exact call plan, convergence gates, runtime attestations,
and cost ceiling are frozen.

The recommended high-evidence design is expected to use roughly 3,200--4,200
calls for a 12-task panel and roughly 4,200--5,500 for a 16-task panel. The
exact builder-derived count replaces these planning ranges after the corpus is
frozen. A maximum-evidence ceiling may not exceed `$500` and is allowed only
for prespecified cross-family or large-family tests.
Unused budget is not a failure. Redundant same-task generations that do not
change a prespecified diagnostic are not authorized.

## Stopping and interpretation

- If fewer than 12 experiments pass every gate, stop before inference/reveal
  and report corpus insufficiency.
- If 12--15 pass, run the bounded minimum replication and label inconclusive
  outcomes honestly.
- If 16 pass, run the strong replication protocol.
- If the minimum primary gate fails, the current six-experiment low-regret
  signal is not replicated. Do not search thresholds, swap primary models, or
  redefine utilities on the revealed panel.
- If the uniform gate passes but control/classical noninferiority fails, do not
  claim practical simulator value.
- If the strong gate passes, the project may claim replicated low-regret
  intervention selection across the tested paradigms, while still avoiding a
  universal trust claim.
