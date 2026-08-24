# Project Specification

# InterveneBench: Knowing When Human Simulations Can Be Trusted for Decisions

## One-Sentence Thesis

InterveneBench evaluates whether behavioral simulators can prospectively make low-regret intervention decisions on benchmark-held-out randomized experiments, whether outcome-free diagnostics can identify unreliable synthetic decisions, and how much independent human evidence is needed when the simulator should abstain.

## Why This Project Exists

Behavioral simulators are commonly evaluated by how closely their responses resemble human responses. That is necessary but not sufficient for decision use. A simulator can reproduce an overall response distribution while getting the contrast between interventions wrong, selecting the wrong arm, or failing confidently.

InterveneBench asks a stricter question:

> If synthetic humans are used before human outcomes are known, would they cause a decision-maker to choose the same or a practically equivalent intervention as a randomized human experiment, how much human utility would a mistake sacrifice, and can that risk be estimated in advance?

The central error is not implausible text or an inaccurate individual prediction. It is a wrong intervention choice with measurable decision regret.

## Distinct Contribution

Existing work has already shown that LLMs can predict response distributions and treatment effects in social-science experiments, and recent work evaluates intervention selection and human-plus-LLM pilot studies. Persona or user drift has also been formalized as a source of confounding in synthetic experiments.

InterveneBench therefore does not claim novelty from treatment-effect prediction alone. Its intended contribution is an integrated, prospective selective-decision benchmark with four properties:

1. **Frozen decisions before reveal.** Simulator predictions, diagnostics, recommendations, and trust decisions are committed before target human outcomes are available to scoring.
2. **Decision loss rather than correlation alone.** Headline results are normalized regret, practical reliability, and best-arm selection under human uncertainty.
3. **Prospective abstention.** A trust policy is evaluated by risk-coverage and regret at coverage, not only retrospective error correlations.
4. **Independent human rescue.** Human observations used to update a decision are disjoint from those used to score it.

## Locked Scope

Primary dataset:

- SocSci210.

Primary split and uncertainty unit:

- Entire randomized experiments, grouped by experimental paradigm.

Primary scored unit:

- A predeclared `DecisionTask`: one experiment, one decision-relevant outcome or prespecified utility, and one admissible arm set.

Primary outcome:

- Decision reliability and normalized regret on benchmark-held-out experimental paradigms.

Primary contribution:

- A reproducible benchmark, outcome-free failure diagnostics, a prospective trust/abstention policy, and an independent human-fallback analysis.

Not the project:

- A generic "LLM imitates people" benchmark.
- A claim that benchmark holdout proves absence from foundation-model pretraining.
- A dashboard-first demo.
- A single-model leaderboard.
- A large fine-tune without decision-level evaluation.
- A reimplementation of Socrates' distributional-fidelity results.

## Research Questions

### RQ1: Descriptive Fidelity

Can a simulator reproduce human response distributions or individual responses?

This is a supporting layer, not the headline claim.

### RQ2: Intervention Fidelity

Can a simulator correctly predict how normalized human utility changes when an intervention changes?

For arm `j` relative to control:

```text
tau_H[j] = E[U(Y) | A=j] - E[U(Y) | A=control]
tau_S[j] = E[U(Y_hat) | A=j] - E[U(Y_hat) | A=control]
```

### RQ3: Decision Fidelity

If the simulator is used to choose an arm, does it select the human-best or a practically equivalent intervention, and what regret follows when it does not?

### RQ4: Prospective Failure Prediction

Before target human outcomes are revealed, can outcome-free diagnostics predict exact choice correctness, practical reliability, or expected regret?

### RQ5: Independent Human Fallback

When the simulator abstains or is unreliable, how much disjointly evaluated human evidence is needed to correct or validate the decision?

## Dataset Reality and Audit Gate

SocSci210 is a flattened response-prediction dataset, not a ready-made causal benchmark. The released table contains identifiers, demographics, combined stimuli, integer responses, condition and task indices, prompts, generated reasoning, and study IDs. It does not explicitly encode several fields InterveneBench needs, including design type, control arm, outcome family, scale bounds, orientation, deployable action set, or survey weights.

Before implementation:

- Pin the dataset revision and verify file hashes.
- Record the data-use basis for each access regime. Private local analysis may rely on the public-repository terms; participant-level API transmission, cloud fine-tuning, and raw-data redistribution require separate decisions.
- Trace every eligible experiment to original TESS/OSF materials.
- Verify condition, participant, outcome, and randomization mappings.
- Reconstruct structured response options and questionnaire bounds.
- Identify between-subject, within-subject, factorial, repeated-measures, and other designs.
- Determine whether survey weights are recoverable.
- Create a response-blinded eligibility registry.
- Create a response-blinded paradigm taxonomy.

The working audit is maintained in `docs/audits/data_audit.md`; the comparative dataset decision and quantitative qualification gates are in `docs/audits/dataset_qualification.md`.

### Dataset Architecture

SocSci210 supplies benchmark scale, not automatic ground-truth semantics. InterveneBench uses four evidence layers:

1. **Primary benchmark:** source-verified SocSci210 decision tasks.
2. **Gold audit stratum:** SocSci210 studies matched to the independently curated 70-experiment Nature/TESS archive, used to measure reconstruction agreement and strengthen Phase 1 selection.
3. **Preregistered augmentation:** source-verified randomized experiments de-duplicated against SocSci210 and added before the canonical split to supply enough independent experiments for trust-model development and evaluation.
4. **Post-freeze replication:** later temporal or otherwise untouched experiments evaluated only after the policy is frozen, if available.

The outcome-blind structural audit found 121 SocSci210 studies with a plausible between-arm task and stable arm-specific stimulus, but this is an upper bound. A frozen source audit of 40 studies from the strict structural pool initially identified 5 provisional simple-core tasks and 23 valid extension tasks while excluding 12. Source-primary-outcome mapping moved three provisional cores to source-data, utility-sensitivity, and composite extensions, leaving 2 core-simple, 26 extension, and 12 ineligible studies without changing the 28-study scientifically usable total. Combined with the earlier curated-overlap audit, the strict SocSci210 pool projects to roughly 50--59 scientifically usable experiments and only about 10--12 untouched test experiments. SocSci210 therefore remains the primary stratum but cannot by itself carry a strong general trust-model claim.

The first external archive census is frozen in `data/manifests/audits/external_candidate_universe_v1.csv` and fully adjudicated. Its 31 rows now contain 7 scientifically usable modules (5 sealed and 2 development-only), 3 source-blocked rows, and 21 ineligible rows including 5 exact SocSci210 duplicates. The metadata orphan `willer845` was recovered and design-excluded; `system_threat` was confirmed as the same fielding as SocSci210 `345ms`. `KlarS44` is usable but shares respondents and fielding with SocSci210 `xtvu5`, so the census contributes only 6 distinct usable external fieldings. Participant records were never opened. Incidental published result-text exposure is logged and bars affected studies from untouched evaluation; a qualitative abstract exposed while resolving `system_threat` also makes `345ms` development-only.

The cost-controlled Benchmark v1 scope remains frozen in `data/manifests/benchmark/benchmark_v1_candidates.csv` and `data/manifests/benchmark/benchmark_v1_freeze.json`. It contains 38 audited candidate modules: 31 SocSci210 and 7 external. After the separate five-task portfolio reveal, 27 remain outcome sealed, 22 remain potentially eligible for a later canonical test after contract completion, and 11 are development-only because of the smoke reveal, result exposure, or the portfolio reveal. The five portfolio tasks are permanently development-only; this was recorded before their declared human outcomes were read. Six runnable contracts remain sealed: `tcg8p`, `z358z`, `pb2rr`, `Blair1131`, `ShannonS2`, and `KlarS44`. Every full-benchmark split remains unassigned. The canonical-split preflight now fails closed at six provisional independent runnable fieldings, six paradigms, and one projected test experiment. The full benchmark is therefore a possible expansion, not a claim made by the completed portfolio.

The active depth-first research scope is frozen separately in `data/manifests/research/depth_first_v1.json`. It uses six already revealed experiments as discovery evidence, three newly runnable outcome-sealed prospective-development tasks (`nj5dx`, `es4xw`, and `e2pyb`), and the six runnable tasks above for one prospective confirmation. The 15-experiment corpus target is met, so additional corpus search stops under the frozen yield rule. This design is deliberately noncanonical: it uses grouped development analysis and a sealed confirmation panel rather than presenting a statistically fragile 65/15/20 split as a general benchmark. Trust prediction remains exploratory, and the 100-experiment full-claim gate does not change.

That depth-first confirmation has since been completed. Its six experiments
remain the project's prospective case study, while all 15 revealed experiments
are development evidence for future method design. A separately frozen attempt
to assemble a 12--16 experiment independent replication panel was stopped
before inference or reveal after both public-source search lanes failed their
outcome-blind yield gates. The historical protocol and machine gate remain in
`docs/decisions/independent_replication_stage_v1.md` and
`data/manifests/research/independent_replication_protocol_v1.json`; they are not
current execution authority or a completion dependency.

Corpus expansion now follows explicit no-progress rules. A metadata-blind
random TESS-root lane closed at its versioned order-20 checkpoint with zero
clean passes, three conditional rows, and seventeen terminal or blocked rows.
Rather than inspect another ten random roots, the project froze a
high-precision title-only action-oriented lane that preserves the original hash
order, permits at most twelve audits, and stops after orders 6 or 9 if clean
scientific yield is inadequate. No human outcome or simulator result informed
either the stop or the replacement selection.

The targeted lane subsequently closed at order 9 with one strict survivor,
`dvwu7`, below its minimum of two. That task remains outcome sealed and moves
to mechanical mapping and adapter completion; targeted orders 10--12 remain
unopened. Open-ended external corpus search is therefore stopped.

The active scope is the role-focused evaluation product frozen in
`docs/decisions/role_focused_evaluation_program_v1.md` and
`data/manifests/research/role_focused_evaluation_program_v1.json`. Completion
now means a replayable behavioral-simulator evaluation lifecycle, scoped
release gate, aggregate-only public case study, experiment-paired model-version
regression suite, results explorer, and technical report. This scope change
does not strengthen the empirical claim: the confirmation remains a small,
noncanonical prospective panel; confidence-based abstention and limited-human
fallback remain held after their negative results.

An aggregate-only post-reveal development analysis adds a uniform-random arm baseline and leave-one-experiment-out effect attenuation. Uniform random choice has expected exact accuracy `0.333` and mean regret `0.0293` after respecting each task's arm count. Cross-fitted attenuation reduces local-model treatment-effect MAE from `0.0467` to `0.0394` without changing rankings, but the no-effect policy remains better calibrated at `0.0361`. This is a development diagnostic, not a sealed-test result or a frozen calibration policy for the six remaining tasks.

Open-ended external source retrieval is closed. The historical bounded-batch
procedure in `docs/protocol/external_audit_batch_protocol.md` remains available
for reproducibility and for a future explicitly reopened study, but it grants
no present corpus-audit authority.

The full claim requires at least 100 independent eligible experiments across at least 25 paradigm groups in the combined, de-duplicated benchmark, at least 20 frozen test experiments, and sufficient correct/incorrect recommendation variation. SocSci210 and external-stratum results must be reported separately before pooling. If the combined audit misses those gates, narrow the claim and uncertainty analysis rather than treating outcomes, arms, models, prompts, or seeds as independent experiments. See `docs/reports/phase2_viability_report.md`.

### Forbidden SocSci210 Fields

Treat the released `reasoning` field as outcome-derived oracle data because it was generated using the observed human response. It is forbidden in target prompts, simulator inputs, embeddings, diagnostics, and trust features.

Generate prompts from an explicit allowlist. Do not assume that a released `prompt` field is safe without validation.

## Decision-Task Registry

Every primary decision task must declare before test outcomes are opened:

- Experiment and source identifiers.
- Paradigm group.
- Target population and data-access regime.
- Randomization unit and design.
- Outcome and outcome family.
- Questionnaire bounds, labels, and orientation.
- Utility transformation.
- Admissible arms.
- Whether control is an admissible action.
- Control/reference arm.
- Estimand.
- Missing-data and weighting rules.
- Deterministic tie rule.
- Practical regret tolerance.

The primary benchmark contains at most one primary decision task per experiment. Secondary outcomes may be analyzed, but they remain clustered within experiment and cannot inflate the effective benchmark sample size.

## Experiment Eligibility

Eligibility is determined without treatment-effect estimates, significance, model performance, or winners.

Initial eligible tasks should have:

- Verified random assignment.
- A recoverable and decision-relevant outcome.
- A defensible utility orientation.
- A meaningful set of deployable arms.
- Sufficient observations for arm-level estimation.
- Text-based or otherwise faithfully representable interventions.
- No unmodeled design feature that invalidates a mean contrast.

Phase 1 is restricted further to clean between-subject, one-factor, binary or ordinal tasks.

## Splitting

Split paradigm groups while keeping every experiment intact:

- Approximately 65% train experiments.
- Approximately 15% validation experiments.
- Approximately 20% held-out test experiments.

Group integrity and paradigm separation take precedence over exact percentages.

Where paradigm constraints permit, balance the independently curated overlap stratum across splits. This is design-time stratification, not outcome-based seed selection. If validation contains any eligible overlap task, Phase 1 selects from that stratum using the frozen deterministic task rule.

Use train for:

- Project behavioral-model fitting.
- Classical baseline fitting.
- Out-of-fold development recommendations.
- Trust-model fitting labels generated through cross-fitting.

Use validation for:

- Prompt selection.
- Hyperparameter and model selection.
- Calibration choices.
- Trust threshold selection.
- Phase 1 smoke testing and debugging.

Use test only after the full protocol, report skeleton, and analysis plan are frozen.

### Checkpoint-Specific Exposure

A benchmark split does not retroactively hold data out from a released checkpoint. For Socrates or any other fine-tuned model, record its training-study mapping. Evaluate it as held out only on experiments confirmed absent from that training set; otherwise retrain it, restrict to a safe intersection, or mark it non-primary and potentially exposed.

### Foundation-Model Contamination

Benchmark-held-out experiments may still appear in a base model's pretraining. Remove titles, authors, citations, and result-bearing paper text from prompts. Record publication dates and model knowledge cutoffs where possible, run prespecified contamination proxies, and report lower-risk temporal or unpublished subsets. Use the term `benchmark-held-out` unless stronger absence can be demonstrated.

## Prospective Freeze/Reveal Protocol

For each target decision task:

1. Create a blinded simulator bundle using only allowlisted design, intervention, outcome, response-option, metadata, and permitted pre-treatment population fields.
2. Run all frozen simulator variants.
3. Compute synthetic arm means, treatment effects, rankings, diagnostics, and recommendation.
4. Apply the frozen trust/abstention policy.
5. Save an immutable, hashed recommendation artifact with full provenance.
6. Verify the artifact.
7. Only then allow the scoring process to load human outcomes.
8. Estimate human arm utilities and effects.
9. Score intervention choice, regret, and trust-policy behavior.

The recommendation artifact is part of the scientific evidence, not merely a log file.

## Population-Access Regimes

Every result is labeled with one of three regimes:

- `DESIGN_ONLY`: public/design population information; no target participant records.
- `TARGET_COVARIATES`: target pre-treatment covariates or demographic distribution; no target outcomes.
- `HUMAN_PILOT_BUDGET_k`: exactly `k` target human outcome observations are additionally available to the fallback policy.

Within a task, simulate the same persona roster under every arm. Do not reuse original treatment assignment to create different synthetic populations across arms.

## Utility and Human Estimand

For declared questionnaire bounds `[L, H]`:

```text
higher-is-better: U(Y) = (Y - L) / (H - L)
lower-is-better:  U(Y) = (H - Y) / (H - L)
```

Never derive bounds or orientation from held-out response values.

The initial estimand is the unadjusted intention-to-treat difference in mean normalized utility. Control is included in the decision set by default; excluding it requires a documented operational constraint.

Later support for survey weights, covariate adjustment, repeated measures, factorial contrasts, and noncompliance must use explicit design-specific estimators. Do not silently apply the Phase 1 estimator to unsupported designs.

For uncapped continuous outcomes, preserve the source unit and source-aligned estimand rather than deriving a scale from target responses. Freeze missing codes, primary and robust location estimands, utility direction, raw-unit tolerances, and uncertainty before reveal. Such tasks may be scored for choice, raw-unit effects, and raw-unit regret, but they cannot enter pooled normalized-regret claims until a training/development-only normalization rule is frozen. The first `tcg8p` contract uses primary mean USD/month, median robustness, and source-declared missing codes; its human outcomes remain sealed until canonical split assignment and recommendation freeze.

## Simulator Suite

### A. No-Effect Decision Baseline

Predict the same normalized expected utility under every arm and select control using the frozen tie rule. This is a transparent default policy and cannot use target outcomes.

### B. Classical Cross-Experiment Baseline

Use structured metadata and frozen text representations of treatments/outcomes. Condition and task indices are study-local labels and are not meaningful cross-experiment features by themselves.

Candidate models include regularized linear/logistic/ordinal regression, random forest, or gradient boosting. Complexity must be justified by experiment-level validation.

### C. Generic Prompted LLM

Prompt with the permitted population profile, experiment context, assigned intervention, outcome question, and valid response options. Request a probability distribution where technically reliable, or use repeated constrained samples.

Store model/checkpoint version, prompt hash, parser version, raw outputs, seeds, temperature, top-p, timestamp, and access regime.

### D. Behavioral Specialist

Evaluate Socrates or another behavioral specialist only under a verified checkpoint-compatible held-out set.

### E. Project Fine-Tune

Fine-tune an open 7B/14B-class model with LoRA only after the benchmark and cross-fitting design work. Fine-tuning is optional until it serves a clear decision-level hypothesis. Reproducing Socrates' distributional improvement is not itself a sufficient contribution.

If a complex simulator cannot beat a simple baseline on intervention or decision metrics, preserve and emphasize that result.

## Evaluation Layer 1: Descriptive Fidelity

Use outcome-appropriate metrics:

- Accuracy and MAE.
- Log loss and Brier score when probabilities are available.
- Wasserstein, total variation, or Jensen-Shannon distance.
- Correlation.
- Subgroup error with adequate support.

Test whether descriptive fidelity predicts treatment-effect error and regret. Do not infer that it does.

## Evaluation Layer 2: Intervention Fidelity

For each arm relative to control, report:

- Human and synthetic normalized effects.
- Absolute treatment-effect error.
- Sign correctness.
- Effect-size and ranking correlation across tasks where defined.
- Subgroup effect error where support permits.
- Within-task and across-experiment uncertainty.

Relative effect error is secondary and must use a prespecified stabilized denominator because it is undefined or misleading near zero.

## Evaluation Layer 3: Decision Reliability and Regret

Let:

- `j_hat` be the synthetic-selected arm.
- `j_star` be the human point-estimate best arm.

Normalized regret is:

```text
R = U_human(j_star) - U_human(j_hat)
```

Report:

- Exact point-estimate best-arm accuracy.
- Practical reliability `1[R <= delta]`.
- Probability the selected arm is optimal under a human bootstrap.
- Top-2 accuracy where meaningful.
- Ranking correlation.
- Mean, median, upper-tail, and worst-case normalized regret.
- Subgroup regret where powered and ethically appropriate.

Near-tied arms should not be presented as substantively different without uncertainty. Report a prespecified sensitivity grid for `delta`, including exact choice.

## Failure Diagnostics

Every diagnostic must be computable before target outcomes are revealed.

### Predictive Uncertainty

- Response entropy.
- Synthetic winner margin.
- Confidence or posterior interval width derived without target outcomes.

### Sampling Instability

- Repeated-generation variance.
- Arm-ranking consistency.
- Winner stability.

### Model Disagreement

- Arm-level effect disagreement.
- Winner disagreement.
- Divergence between simple baselines and behavioral models.

### Prompt Sensitivity

- Formatting and wording perturbations.
- Condition-order perturbations.
- Answer-order perturbations with correct inverse mapping.

### OOD and Support

- Distance from training paradigm/text/metadata representations.
- Outcome-family and intervention-type novelty.
- Demographic support and missingness.

### Persona/User Drift

- Stable-answer changes.
- Negative-control outcome shifts.
- KL divergence or embedding distance in treatment-invariant attributes.

Drift is a diagnostic hypothesis, not a guaranteed contribution. The important question is whether it predicts held-out error or regret beyond simpler diagnostics.

## Trust Model

The trust model predicts whether a frozen synthetic recommendation is reliable. Required targets are:

```text
P(j_hat == j_star | outcome-free diagnostics)
P(R <= delta | outcome-free diagnostics)
```

Expected normalized regret is an optional continuous target.

Allowed features:

- Simulator identity.
- Prespecified experiment metadata and outcome family.
- Entropy and synthetic winner margin.
- Repeated-sample instability.
- Model disagreement.
- Prompt sensitivity.
- Persona drift.
- OOD distance.
- Demographic support and missingness.

Forbidden features:

- Target experiment responses.
- Human effects, winner, regret, significance, or confidence intervals.
- Any transformation fitted using target outcomes.

### Cross-Fitting Requirement

For fine-tuned simulators, recommendations used as trust-model training examples must be out of fold. The behavioral model producing a development experiment's recommendation cannot have trained on that experiment. Use grouped cross-fitting and keep all tasks from an experiment in one fold.

### Trust Evaluation

Report:

- Calibration plots and Brier score.
- AUROC/AUPRC when supported by label balance.
- Risk-coverage curve.
- Area under the risk-coverage curve.
- Regret at fixed coverage levels.
- Coverage achievable at fixed maximum risk.
- Comparison with random abstention and simple single-diagnostic policies.

A failed trust model is a valid result if the protocol is prospective and adequately powered.

## Independent Human Fallback

Budgets:

- 0
- 10
- 25
- 50
- 100
- 250

Compare:

- Humans only.
- Synthetic only.
- Synthetic plus uniformly allocated human evidence.
- Synthetic plus outcome-free adaptive allocation.

For every replicate, pilot participants consumed by the policy are disjoint from the remaining participants used for evaluation. Repeat nested acquisition with deterministic seeds.

Any rule that fuses synthetic and human estimates must be fitted or calibrated on development experiments only. Candidate adaptive rules may allocate toward arms with small predicted margins, high instability, high disagreement, or high expected value of information. Do not call a rule intelligent merely because it uses information unavailable before target outcomes.

Primary fallback outputs:

- Normalized regret versus human outcomes acquired.
- Probability of exact and practical reliability versus cost.
- Marginal regret reduction per additional human observation.
- Results by trust-policy acceptance/abstention status.

## Statistical Rigor

- Treat experiments as the independent unit for benchmark claims.
- Use experiment-clustered bootstrap intervals.
- Preserve paired comparisons between simulators on the same experiments.
- Use multiple seeds for stochastic simulation and acquisition.
- Keep all outcomes and simulators from one experiment clustered.
- Report effective experiment counts and label balance.
- Report effect sizes and uncertainty, not only p-values.
- Correct or clearly scope multiple subgroup comparisons.
- Preserve null and negative results.

Do not treat millions of response rows as millions of independent benchmark observations.

## Pre-Registration and Test Reveal

Before opening test outcomes, freeze:

- Dataset revision and data-use decision.
- Eligibility registry and exclusions.
- Paradigm taxonomy and split manifest.
- Primary decision task per experiment.
- Utilities, action sets, controls, tie rules, and tolerances.
- Simulator suite and checkpoint revisions.
- Prompt templates and parsers.
- Diagnostics.
- Trust-model fitting and threshold procedure.
- Human-fallback allocation, fusion, and scoring rules.
- Seeds, bootstrap procedure, metrics, and report skeleton.

After reveal, any correction must be logged with the original result retained. The corrected test is no longer a fresh untouched test.

## Project Milestones

### Immediate Portfolio Milestone: Five-Experiment Decision Pilot

The immediate deliverable is a finished five-experiment study, not the full trust-model program. The frozen tasks are `5vm8g`, `xc4yq`, `de5hx`, `turagaS11`, and `wallaceS12`. The no-effect policy and local prompted simulator ran outcome-blind, every recommendation was frozen, and a separate decision then authorized only these five outcomes as development evidence. The completed result reports human and synthetic effects, intervention choice, regret, uncertainty, distribution fidelity, utility sensitivity, and disjoint human fallback.

This milestone did not create a canonical split or validate a trust model and required no corpus expansion, fine-tuning, Modal, dashboard, or paid inference. The local model chose the exact human-best arm in 3/5 experiments and reduced mean normalized regret from 0.0308 for the no-effect/control-tie policy to 0.0038. Its treatment-effect MAE was nevertheless worse (0.0467 versus 0.0361), and the frozen simple human-fallback policies did not outperform synthetic-only on average. The five tasks remain valid development evidence but are no longer untouched test evidence. The other six runnable tasks stay sealed while expansion is evaluated.

### Phase 1: Validation Smoke Test

One eligible validation decision task completes the two-stage freeze/reveal path with a no-effect baseline, one non-oracle simulator, treatment effects, recommendation, regret, provenance, and focused tests.

### Optional Phase 2: Eligible Benchmark Registry and Breadth

If the portfolio result justifies expansion, improve simulator and fallback policies on the five development tasks, but preserve still-sealed tasks for prospective evaluation. Complete the highest-yield contracts and add external experiments only if pursuing a general trust-model claim. The canonical-split preflight must pass before any full benchmark reveal; the current six-fielding sealed runnable set is explicitly insufficient. Do not tune fallback or trust thresholds on future target outcomes.

### Phase 3: Cross-Fitted Diagnostics and Trust

Generate out-of-fold development recommendations, compute outcome-free diagnostics, fit simple trust models, and freeze the abstention policy.

### Phase 4: Test Evaluation and Human Fallback

Reveal the untouched test once, evaluate decision and trust metrics, run independent human-acquisition simulations, and produce the final research package.

### Optional Phase 5: Post-Freeze Replication

Evaluate the frozen policy on later temporal or otherwise untouched experiments not used in the augmentation split. Treat this as the strongest evidence that trust diagnostics generalize beyond the combined benchmark.

## Deliverables

- Audited and versioned decision-task registry.
- Leakage-resistant benchmark implementation.
- Frozen recommendation artifacts.
- Baseline and simulator comparisons.
- Experiment-level effect and decision results.
- Risk-coverage and calibration evaluation.
- Independent human-fallback cost-regret curves.
- Reproducible repository and model/data cards.
- Six-to-eight-page research paper or equivalent technical report.
- Demo only after the scientific pipeline is complete.

## Main Figures

1. Prospective freeze/reveal benchmark protocol.
2. Descriptive fidelity versus effect error and regret.
3. Human versus synthetic treatment effects.
4. Regret distributions by simulator and access regime.
5. Diagnostic value beyond synthetic winner margin.
6. Trust-policy risk-coverage and calibration.
7. Independent human evidence versus regret.
8. Optional external-validation risk-coverage.

## Go/No-Go Criteria After Data Audit

Proceed with the full trust-model claim only if the combined, de-duplicated audit yields at least 100 independent eligible experiments across at least 25 paradigm groups, at least 20 frozen test experiments, and enough failure labels for meaningful held-out evaluation. Dataset source must remain visible in splitting, diagnostics, uncertainty, and reporting. If the eligible test set is smaller or nearly all simulators make the same decision, label the trust analysis exploratory and use nested repeated grouped cross-validation on development data while preserving the untouched test as descriptive evidence.

The Phase 2 viability audit triggered the augmentation branch: SocSci210 remains primary and the first manually verified external archive is included in the frozen Benchmark v1 candidate scope. The first archive cannot close the scale gap alone, so the universal trust-model claim is not currently authorized. A second corpus is a contingent expansion, not an immediate milestone. It is reconsidered only if the cost-capped v1 pilot demonstrates that additional independent experiments would materially strengthen the claim. External experiments must satisfy the same source trace, action-set, utility, leakage, and one-primary-task rules.

## Claim Discipline

Do not claim:

- That models understand humans.
- That synthetic participants replace human participants.
- That benchmark holdout proves pretraining absence.
- Individual counterfactual validity.
- Universal confidence calibration.
- Trust outside the evaluated tasks, populations, models, and access regimes.

Claim only:

- How evaluated simulators performed on declared decision tasks.
- Whether frozen outcome-free diagnostics predicted held-out reliability.
- How regret changed under a frozen abstention policy.
- How much independent human evidence changed regret.

## Related Work That Defines the Bar

- Kolluri, Wu, Park, and Bernstein. *Finetuning LLMs for Human Behavior Prediction in Social Science Experiments*. EMNLP 2025. <https://aclanthology.org/2025.emnlp-main.1530/>
- Ashokkumar, Hewitt, Ghezae, and Willer. *Large language models can predict the results of social science experiments*. Nature, 2026. <https://doi.org/10.1038/s41586-026-10742-x>
- Lin et al. *The Illusion of Intervention: Your LLM-Simulated Experiment is an Observational Study*. 2026. <https://arxiv.org/abs/2605.20767>

InterveneBench should cite and compare directly with these works. Its value comes from prospective selective decision-making, not from presenting their questions as unstudied.
