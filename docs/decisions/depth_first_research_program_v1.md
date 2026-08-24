# Depth-First Research Program v1

**Decision date:** 2026-08-13  
**Status:** active scope; no new outcome reveal or paid execution authorized  
**Machine-readable scope:** `data/manifests/research/depth_first_v1.json`
**Live contract batch:** `data/manifests/audits/depth_first_contract_batches.csv`

## Decision

InterveneBench will pursue a bounded 15-experiment research study before considering another broad external-corpus search. The study retains the three substantive contributions:

1. source-audited DecisionTask corpus construction;
2. multi-family simulator evaluation for intervention choice and regret; and
3. prospective abstention with independent human fallback.

The trust component is explicitly exploratory. Fifteen experiments cannot support the repository's full general trust-model claim, which remains gated at 100 independent experiments, 25 paradigm groups, and 20 untouched test experiments.

## Three Evidence Tiers

### Discovery: six already revealed experiments

The five portfolio experiments and the completed Phase 1 smoke experiment are the discovery set. Their outcomes may be reused to formulate methods and screen simulator configurations, but they cannot provide prospective validation for choices made after their reveal.

### Prospective development: three new experiments

Complete source-faithful contracts for `nj5dx`, `es4xw`, and `e2pyb`. The original third candidate, `jtgyq`, was stopped because SocSci210 omits its randomized no-message control and no local source participant file can restore it. `a42yg` remains scientifically eligible but was replaced under the bounded speed rule because source-faithful reconstruction needs a substantially larger interactive sequence adapter, whereas `e2pyb` reused the frozen ordinal-PNG path. Before revealing any prospective-development human outcomes, freeze the simulator panel, diagnostic definitions, fallback candidates, seeds, and report fields being evaluated. These experiments determine whether discovery-derived methods show enough promise to carry forward.

### Sealed confirmation: six existing runnable experiments

Preserve `tcg8p`, `pb2rr`, `z358z`, `ShannonS2`, `Blair1131`, and `KlarS44` for one prospective confirmation. Every task belongs to a distinct conservative paradigm group. The complete model, diagnostic, abstention, fallback, seed, and reporting policy must be frozen before these outcomes are revealed.

This is a noncanonical discovery/development/confirmation design. It is not relabeled as the project's 65/15/20 canonical split.

## Why This Route

The repository already contains 38 source-audited candidates. Six tasks are completed discovery evidence, six more have runnable sealed contracts, and multiple eligible SocSci210 candidates need bounded contract completion. Searching another broad external corpus would therefore add document-retrieval cost before using the higher-yield work already completed.

The initial contract queue favors fixed-media experiments with local source materials and reusable adapters. It defers video, longitudinal, physical-experience, missing-asset, and unresolved-recode tasks.

The live batch is evaluated by `verify-contract-progress`. A candidate counts as runnable only after Gate 4; surviving a scientific screen does not increase the corpus size. On 2026-08-13, `a42yg` survived the scientific gates but was deferred because source-faithful reconstruction requires a new interactive, sequence-aware multimodal adapter. `nj5dx` then reached Gate 4 after exact infographics were recovered and a reusable ordinal-PNG adapter was added. `jtgyq` failed the full-arm scoring gate and was replaced by `es4xw`; `es4xw` reached Gate 4 through the same image adapter plus a separate recommendation-bound, four-column source-SAV reader. `e2pyb` reused the image adapter and became the third runnable task, replacing the deferred `a42yg` under the speed rule. No target outcome was opened in these decisions.

## Progress Accounting

Only these outputs count as progress:

- a new runnable, outcome-sealed DecisionTask with one primary outcome and complete source trace;
- a simulator family that passes its parser/cost gate and tests a distinct decision-level hypothesis;
- an outcome-free diagnostic artifact whose hashes and direction were frozen before reveal;
- a fallback policy evaluated with pilot and scoring participants kept disjoint; or
- an analysis artifact replayable without another model call.

Document retrieval, model calls, prompt variants, arms, stochastic draws, and participant resamples do not increase the experiment count.

## Stop and Pivot Rules

### Corpus

- Pause a candidate lane after one three-candidate batch yields zero runnable contracts.
- Stop the current queue if two consecutive candidates fail for the same structural reason.
- Cap focused contract-completion work at 20 hours for the three required additions.
- Stop corpus work when 15 independent tasks are ready; 18 requires a new scope decision.
- If 15 cannot be reached inside the cap, freeze the smaller corpus and narrow claims instead of restoring ambiguous tasks.

### Simulators

- Run new configurations first on two discovery experiments.
- Require at least 98% semantic parse success for a full development run.
- Drop a configuration that exceeds its frozen cost cap or duplicates a cheaper configuration without a distinct decision hypothesis.
- Stop adding generic models after two consecutive candidates neither improve mean development regret by 0.002 nor produce a meaningfully different intervention recommendation.
- Do not fine-tune unless the prompted suite is complete and a specific decision-level deficiency motivates the run.

### Trust diagnostics

- Primary diagnostics are winner margin, ranking stability, and cross-model disagreement.
- Evaluate a fixed rank composite and simple scalar rules before fitting a classifier.
- If development data lack at least three reliable and three unreliable experiments, classification and calibration claims are non-estimable; analyze continuous regret ranking only.
- If no frozen diagnostic improves development risk-coverage and regret over random abstention, stop feature proliferation. Confirm and report the negative result, then rely on conservative human validation.

### Human fallback

- Preserve budgets 0, 10, 25, 50, 100, and 250.
- Human pilot records must be disjoint from scoring records in every replicate.
- Compare synthetic-only, human-only, uniform human validation, and one synthetic-informed allocation/fusion policy.
- If synthetic-informed fusion fails to improve paired regret at every 25/50/100-person budget, stop tuning it and preserve the negative result.
- If uncertainty allocation does not beat uniform allocation on both mean and upper-tail regret at two or more budgets, remove "intelligent allocation" from the primary claim.

## Compute Boundary

This decision authorizes no paid inference, Modal job, model download, external API call, fine-tuning, participant-level transmission, or sealed-outcome reveal. It freezes the research direction and permits local response-free implementation, tests, dry-run manifests, and already-authorized development analysis. A later execution authorization must bind exact models, calls, checkpoints, expected cost, and a hard ceiling.

## Completion Criteria

Before sealed confirmation:

- 15 independent tasks and their paradigm roles verify;
- all six confirmation outcomes remain sealed;
- source/action/outcome/utility/estimator contracts are complete;
- simulator checkpoints, prompts, parsers, rosters, perturbations, and sampling rules are frozen;
- diagnostics and trust direction/threshold procedure are frozen, including a possible "no validated abstention policy" result;
- fallback folds, allocations, fusion, budgets, feasibility, and seeds are frozen;
- experiment-clustered uncertainty and report skeleton are implemented;
- the complete test suite and dry-run replay pass; and
- paid execution and outcome reveal receive separate explicit authorization.
