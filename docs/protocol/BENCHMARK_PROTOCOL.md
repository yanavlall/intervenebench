# InterveneBench Benchmark Protocol

## Status and Authority

This document is the normative evaluation protocol for InterveneBench. `PROJECT_SPEC.md` defines the project scope, `PHASE_1.md` defines the first implementation milestone, and this file defines what a valid benchmark run must do.

If an implementation convenience conflicts with this protocol, the protocol wins. Any change to a locked decision must be recorded before held-out test outcomes are opened.

## Scientific Claim

InterveneBench evaluates a prospective decision process:

> Given an experiment's design, intervention arms, target-population information allowed by the declared access regime, and outcome definition—but not its human outcomes—can a simulator choose an intervention that has low regret in the corresponding randomized human experiment, and can a policy know when to abstain or acquire human evidence?

The benchmark does not claim that a simulator understands people, substitutes for human participants, or identifies individual counterfactuals.

## Units of Analysis

### Experiment

An `Experiment` is the indivisible unit for splitting, generalization, cross-fitting, and headline uncertainty. All rows, participants, conditions, and outcomes belonging to an experiment stay in the same split.

### Paradigm Group

A `ParadigmGroup` groups experiments that share a materially similar experimental task, intervention mechanism, or outcome paradigm. Paradigm groups, not individual rows, are assigned across train, validation, and test. This operationalizes the requirement that test paradigms be unseen.

The grouping rule must be created from design metadata and stimulus/outcome text while blinded to human response values, effect sizes, significance, and winners. The taxonomy or clustering procedure is frozen before outcomes are analyzed.

### Decision Task

A `DecisionTask` is the scored object:

```text
experiment
+ target population and access regime
+ one decision-relevant outcome or prespecified utility
+ admissible intervention arms
+ control/reference arm
+ outcome orientation and scale
+ estimand
+ tie and practical-equivalence rules
```

An experiment may contain several outcomes, but the primary benchmark uses at most one predeclared primary decision task per experiment. Secondary multi-outcome analyses must keep outcomes clustered within experiment and must not treat them as independent evidence.

When an otherwise eligible experiment has several candidate outcomes, select the primary outcome without consulting response values, treatment-effect estimates, or simulator outputs. Apply this hierarchy in order:

1. Keep only supported post-treatment binary, ordinal, bounded numeric, count, or continuous outcomes that correspond to the intervention's stated target and have a defensible utility direction. Continuous and count outcomes require a source-defined scale or unit, a frozen missing-code policy, a robust estimand, and uncertainty tests before any response is revealed.
2. Exclude manipulation checks, treatment-only questions, pre-treatment measures, unsupported repeated-choice or composite tasks, and outcomes whose wording or scale cannot be verified in the final source questionnaire.
3. In the independently curated overlap, prefer outcomes independently catalogued in `RA_outcome_features.csv`.
4. Select the earliest remaining outcome in the final fielded questionnaire; break a true tie by lexicographic source question ID.

Record the selected source question, SocSci210 task mapping, scale, orientation, and the hashes of the materials used to make the choice. The rule is applied before the canonical split and cannot be revised after any outcome reveal.

### Admissible Arm-Set Selection

Use the full randomized arm set when it defines one coherent decision. A source-randomized subset may define the single primary `DecisionTask` only when the full set mixes a deployable action factor with fixed-context, nuisance, or non-deployable factors and all of the following are satisfied without consulting outcomes:

1. Within the subset, the target population, decision-maker, beneficiary, background context, outcome, utility direction, and estimand are fixed.
2. The retained cells differ only in the deployable action factor and preserve a relevant control or no-action option when one was randomized.
3. At least two action levels remain, assignment to every retained cell was randomized, and the exact fielded stimuli are recoverable.
4. Every omitted cell and the reason for omission are recorded. Omitted contexts do not become separate independent experiments.
5. Enumerate all subsets satisfying the rule. Prefer a source-designated primary contrast; otherwise choose the subset with the most deployable action levels, then the lexicographically smallest sorted source-cell identifiers.

The subset rule and selected cell IDs are frozen before the canonical split. Never choose a subset using response distributions, treatment effects, significance, simulator behavior, or which contrast appears substantively favorable.

## Eligibility Registry

Before splitting or inspecting human effect estimates, create an eligibility registry for every candidate experiment. Record:

- Stable experiment and source identifiers.
- Original source and dataset revision.
- Randomization unit and study design.
- Between-subject, within-subject, factorial, repeated-measures, or other structure.
- Arm identifiers and treatment descriptions.
- Whether each arm is a feasible decision action.
- Control/reference arm.
- Candidate primary outcome and outcome family.
- Questionnaire scale bounds and labels.
- Outcome orientation.
- Missing-data and survey-weight availability.
- Text, image, video, interactive, or other modality.
- Paradigm group.
- Eligibility decision and reason.

Eligibility may depend on design and data availability, but not on observed treatment-effect size, direction, significance, simulator performance, or regret.

## Canonical Split

The target split is approximately:

- 65% train experiments.
- 15% validation experiments.
- 20% test experiments.

Exact percentages may move slightly to preserve paradigm groups. The split manifest must contain:

- Dataset revision and content hashes.
- Eligibility-registry hash.
- Grouping rule and version.
- Split seed.
- Experiment-to-paradigm and experiment-to-split mappings.
- Creation timestamp.
- A flag indicating whether test outcomes remain sealed.

The validation split is used for prompt selection, model selection, trust thresholds, calibration choices, and pipeline debugging. Test is evaluated only after the benchmark configuration is frozen.

### Noncanonical Portfolio Milestone

A small response-free engineering split may be used to prove multi-experiment orchestration before a canonical benchmark is possible. Its train, validation, and test labels are software-boundary labels only: they do not authorize outcome access, real-outcome model fitting, or scientific split claims.

The current five-task portfolio milestone may reveal outcomes only after all frozen recommendations verify and a separate decision permanently designates those tasks as development evidence. Such a reveal does not reduce the number of experiments, but it removes their future untouched-test status. No result from five experiments may be presented as validation of a general trust model.

That development reveal is now complete. Its authorization freezes only `5vm8g`, `xc4yq`, `de5hx`, `turagaS11`, and `wallaceS12`; it does not authorize any outcome access for the six remaining runnable sealed contracts. The scored portfolio is reusable for development and case analysis but excluded from any later canonical test count.

### Third-Party Checkpoint Compatibility

For every pretrained or fine-tuned behavioral checkpoint, record the experiments used in its training when known. A checkpoint may be called experiment-held-out only on experiments confirmed absent from its fine-tuning data.

If a released Socrates checkpoint used a different split:

- Evaluate it only on a leakage-safe intersection;
- Retrain it under the InterveneBench split; or
- Report it as a non-primary, potentially exposed comparison.

Never silently imply that the benchmark split also held out an experiment from a third-party checkpoint.

## Data Access Regimes

Every run declares one of these regimes:

1. `DESIGN_ONLY`: experiment design, treatment text, outcome question, response options, and a public population distribution; no target participant records.
2. `TARGET_COVARIATES`: the above plus target pre-treatment covariates or demographic distribution; no target outcomes.
3. `HUMAN_PILOT_BUDGET_k`: the above plus exactly `k` independently sampled target human outcome observations used by the fallback policy.

Do not compare regimes as if they consume the same human information. Human fallback budgets refer to outcome observations; any target-covariate access must be reported separately.

## Blinded and Revealed Data Views

### Blinded View

Simulator, diagnostic, and trust-policy code may receive only explicitly allowlisted fields, such as:

- Experiment and decision-task identifiers.
- Treatment descriptions.
- Outcome question and valid response options.
- Questionnaire scale bounds and orientation.
- Allowed pre-treatment participant attributes.
- Prespecified experiment metadata.

It must not receive:

- Human response values.
- Human condition means or effect estimates.
- Human winner, regret, significance, or confidence intervals.
- SocSci210 `reasoning` fields.
- Any feature derived from target outcomes.
- Result-bearing paper text, abstracts, tables, or conclusions.

Generate prompts from the allowlist. Do not assume a released prompt column is safe without validation.

### Revealed View

Human outcomes become accessible only to the scoring stage after recommendation artifacts are frozen and verified. Revealed outputs may be used for scoring and later aggregate scientific analysis, but never retroactively to change a frozen test simulator, diagnostic, threshold, action set, utility, or feature.

## Synthetic Population Construction

Within a decision task, the same synthetic persona roster must be evaluated under every admissible arm. This targets intervention differences in a fixed population and prevents compositional drift from masquerading as a treatment effect.

For each roster, record:

- Source population and access regime.
- Sampling seed and weights.
- Included attributes and missingness handling.
- Whether profiles are empirical, reweighted, or generated.
- Any exclusions.

Actual target treatment assignment must not determine which personas are simulated under an arm.

## Outcome Utility and Estimands

For a response with declared questionnaire bounds `[L, H]`, normalize utility without consulting held-out response values:

```text
higher-is-better: U(Y) = (Y - L) / (H - L)
lower-is-better:  U(Y) = (H - Y) / (H - L)
```

For arm `j`:

```text
mu_H[j]  = E[U(Y) | randomized arm j]
mu_S[j]  = E[U(Y_hat) | simulated arm j]
tau_H[j] = mu_H[j] - mu_H[control]
tau_S[j] = mu_S[j] - mu_S[control]
```

The Phase 1 estimand is the unadjusted intention-to-treat difference in mean normalized utility for a clean between-subject experiment. Survey weighting, covariate adjustment, repeated measures, noncompliance, and factorial contrasts require explicit later estimands and tests.

Control is included in the decision set by default because choosing no intervention can be the correct decision. Excluding it requires a documented real-world constraint.

### Uncapped Continuous Outcomes

An uncapped continuous outcome is not assigned an artificial normalized utility merely to enter a pooled metric. Before reveal, its task contract must freeze the source-defined unit, valid domain, missing codes, primary location estimand, robust sensitivity estimand, utility direction, tie rule, raw-unit regret tolerance grid, and bootstrap procedure.

For `tcg8p` Q11, the source proposal's analysis targets mean monthly willingness to pay, so mean USD/month is primary and median USD/month is a required robustness analysis. Source codes `77777`, `99998`, and `99999` are missing. Lower willingness to pay is better for the declared notification-policy objective. Effects and regret are initially reported in USD/month, with exact choice primary and tolerances of 0, 5, 10, and 20 USD/month reported as frozen sensitivities.

Because Q11 has no source upper bound, `tcg8p` is excluded from pooled normalized-regret summaries until a scale is defined using training/development evidence only and frozen before canonical test reveal. Target observed minima, maxima, quantiles, variances, effects, or winners are forbidden normalization inputs. This task remains outcome sealed and canonical-split unassigned until that split is frozen.

## Recommendation and Freeze Artifact

The simulator selects:

```text
j_hat = argmax_j mu_S[j]
```

Before human outcomes can be revealed, write an immutable recommendation artifact containing:

- Dataset, registry, split, decision-task, and code-version hashes.
- Simulator identity and checkpoint revision.
- Prompt-template and parser hashes.
- Persona-roster identity.
- Raw-output artifact references.
- Synthetic arm means and treatment effects.
- Ranked arms and selected arm.
- Outcome-free diagnostics.
- Trust probability, threshold, and accept/abstain decision when applicable.
- Random seeds and sampling parameters.
- Timestamp and artifact hash.

Scoring must reject a missing, malformed, unhashed, or subsequently modified recommendation artifact.

## Human Ground Truth and Uncertainty

Human-best intervention is estimated from randomized observations, but the sample argmax is noisy. Report:

- Point-estimate best arm.
- Human arm means and treatment effects.
- Arm sample sizes and missingness.
- Within-experiment uncertainty appropriate to the design.
- Bootstrap probability that each arm is optimal.
- Probability that the synthetic-selected arm is optimal.

For the Phase 1 between-subject design, resample participants within arms. For headline benchmark comparisons, resample experiments as clusters and keep all tasks from one experiment together.

## Decision Metrics

Primary metrics are:

- Correct point-estimate best-arm selection.
- Normalized decision regret:

```text
R = max_j mu_H[j] - mu_H[j_hat]
```

- Probability the selected arm is optimal under the human bootstrap.
- Practical reliability `1[R <= delta]`, where `delta` is declared before test evaluation.
- Regret at fixed trust-policy coverage levels.
- Area under the risk-coverage curve relative to random abstention.

Secondary metrics include top-2 accuracy, arm-ranking correlation, absolute treatment-effect error, sign correctness, and subgroup regret.

Report a sensitivity grid for practical regret tolerance, including exact choice (`delta = 0`) and predeclared normalized tolerances. Do not choose a tolerance after viewing test performance.

## Trust Model

The trust model is a meta-evaluator. It does not predict human responses directly.

Required targets:

```text
P(j_hat equals point-estimate human best | outcome-free diagnostics)
P(R <= delta | outcome-free diagnostics)
```

Expected normalized regret may be an additional target.

Allowed inputs include simulator identity, experiment metadata, outcome family, entropy, synthetic winner margin, repeated-sample instability, model disagreement, prompt sensitivity, persona drift, OOD distance, demographic support, and missingness. All must be computable without target human outcomes.

For a fine-tuned simulator, development recommendations used to train the trust model must be out of fold. Use grouped cross-fitting so the simulator that produces a recommendation did not train on that experiment. Fit and tune the trust model with experiment groups, simple regularized models, and calibration methods appropriate to the effective number of experiments.

The full prospective trust claim requires at least 100 independent eligible experiments across at least 25 paradigm groups in the combined, de-duplicated benchmark, at least 20 frozen test experiments, and enough positive and negative reliability labels to evaluate calibration. SocSci210 remains the primary named stratum; source-stratum results are reported separately before pooling, and dataset identity is an allowed prespecified diagnostic. If these gates fail, label trust modeling exploratory and use nested repeated grouped cross-validation on development experiments while preserving the smaller untouched test for descriptive confirmation.

Evaluate:

- Calibration and Brier score.
- AUROC/AUPRC where label balance supports them.
- Risk-coverage curve.
- Regret at fixed coverage.
- Coverage at fixed maximum risk.
- Comparison with random abstention and simple one-feature policies.

## Human Fallback

For budgets `0, 10, 25, 50, 100, 250`, compare:

- Human only.
- Synthetic only.
- Synthetic plus uniformly allocated human observations.
- Synthetic plus an outcome-free adaptive allocation rule.

Within each acquisition replicate:

1. Sample pilot participants according to the declared budget and allocation rule.
2. Update or validate the intervention choice using only those pilot outcomes and the frozen synthetic evidence.
3. Score the resulting decision on disjoint remaining participants.
4. Repeat with deterministic seeds and aggregate at the experiment level.

The fusion rule between synthetic and human evidence must be learned or calibrated only on development experiments. Report regret and probability of correct/practically reliable choice against human-data cost.

## Model and Data Contamination

Experiment-level benchmark holdout does not prove absence from a foundation model's pretraining. For each evaluated model:

- Record knowledge cutoff and checkpoint date when available.
- Remove title, author, citation, and result-bearing text from prompts.
- Record experiment publication or public-posting date when available.
- Test simple contamination proxies, such as title/author recognition, without using them to tune on test outcomes.
- Report results for lower-risk temporal or unpublished subsets when possible.

Use the phrase `benchmark-held-out` unless stronger absence from model training can be demonstrated.

## Freeze Checklist Before Test Reveal

The test outcome vault remains sealed until all of the following are committed:

- Dataset revision and data-use decision.
- Eligibility registry and exclusions.
- Paradigm grouping and split manifest.
- Primary decision task per experiment.
- Utilities, orientations, action sets, controls, and tie rules.
- Simulator suite and checkpoint revisions.
- Prompt templates and parsers.
- Diagnostic definitions.
- Trust-model fitting and threshold procedure.
- Human-fallback sampling, allocation, fusion, and scoring rules.
- Seeds and bootstrap procedures.
- Primary metrics, coverage levels, and statistical comparisons.
- Test report skeleton and claims language.

After reveal, corrections are permitted only for genuine implementation defects. Report every correction, preserve the original result, and do not reuse the corrected test as a fresh untouched test.

## Claim Discipline

Valid claims concern the evaluated decision tasks, models, access regimes, and experiment distribution. InterveneBench may show that:

- A simulator often or rarely selects low-regret interventions.
- Descriptive fidelity does or does not predict intervention reliability.
- Outcome-free diagnostics do or do not support useful abstention.
- Limited independent human evidence reduces regret by a measured amount.
- Simple baselines outperform or underperform complex simulators.

It must not claim general human substitutability, individual counterfactual validity, universal confidence calibration, or reliability outside the evaluated distribution.
