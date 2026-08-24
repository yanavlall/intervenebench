# Phase 1: Prospective Validation Smoke Test

## Goal

Get one eligible SocSci210 validation decision task through a complete leakage-safe path:

```text
pinned data
-> blinded eligibility registry
-> paradigm-group split
-> blinded simulator inputs
-> baseline/simulator predictions
-> synthetic treatment effects
-> frozen intervention recommendation
-> reveal validation human outcomes
-> human treatment effects
-> decision correctness and regret
-> reproducible report
```

Phase 1 proves protocol and metric correctness. It does not establish scientific performance or justify opening held-out test outcomes.

## Non-Negotiables

- Use SocSci210 as the primary dataset.
- Resolve the dataset-use gate before downloading records or sending them to an external API.
- Pin the dataset revision and save content hashes.
- Split whole paradigm groups while keeping experiments intact.
- Select the smoke task from validation only.
- Choose eligibility, outcome, arms, control, orientation, and estimand without effect estimates or winners.
- Use a clean between-subject randomized task for the first run.
- Build prompts from an allowlist.
- Exclude `response`, `reasoning`, human aggregates, effect estimates, and result-bearing text from the blinded stage.
- Use the same synthetic persona roster under every arm.
- Freeze and hash the synthetic recommendation before human outcomes are loaded.
- Normalize utility using questionnaire bounds, not observed values.
- Fail explicitly on unsupported or ambiguous design/outcome types.
- Keep test outcomes sealed throughout Phase 1.
- Do not build a dashboard, fine-tune a model, or launch large-scale inference.

## Stage 0: Freeze the Data-Access Policy

Complete the access-policy section of `docs/audits/data_audit.md`:

- Record the public-repository terms supporting private local analysis.
- Keep participant-level records, demographics, responses, and generated reasoning local.
- Permit simulator calls to contain only verified design-level experiment material and synthetic or public population specifications.
- Defer cloud fine-tuning and participant-level transmission until separately approved.
- Do not redistribute the raw dataset; publish code, hashes, non-sensitive metadata, and aggregate results only.
- Record attribution requirements and any source-study restrictions discovered during reconstruction.

An explicit dataset license remains desirable, especially before commercial use or redistribution, but its absence does not block the private local Phase 1 analysis under the recorded working decision. It does block sending participant records to external APIs. Model prompting in Phase 1 uses the `DESIGN_ONLY` regime.

## Stage 1: Pin and Audit SocSci210

Pin:

```text
socratesft/SocSci210
revision 048481111a4425ed83dc0eacf15f8431f252b21a
```

Record:

- Shard names, byte sizes, and hashes.
- Row and study counts.
- Released column schema.
- Mapping-file hashes.
- Identifier uniqueness and scope.
- Missingness by structural field, without treatment-effect summaries.

Trace candidate experiments to original TESS/OSF study materials and verify the reconstruction without reading reported results.

The full Benchmark v1 canonical split is guarded by `data/manifests/splits/benchmark_v1_canonical_split_preflight.json`. After the separately authorized five-task development reveal, six runnable sealed fieldings remain. They do not pass the exploratory or full trust-model gates, so every canonical assignment remains unassigned. The five-task `supported_ordinal_pilot.json` remains an engineering split only; it never became a canonical train/validation/test split. The $25 ceiling still does not authorize paid inference, Modal, or fine-tuning.

The immediate post-Phase-1 milestone is complete. The reduced portfolio scope in `data/manifests/benchmark/portfolio_pilot_scope.json` governed 60 zero-cost local calls and five outcome-blind frozen recommendations. The separate `portfolio_pilot_development_reveal.json` then authorized only those five declared outcomes and permanently made them development-only. The local model selected the exact human-best intervention in 3/5 tasks, with mean normalized regret 0.0038; the no-effect/control-tie policy selected 0/5 with regret 0.0308. Treatment-effect MAE nevertheless favored the no-effect policy, and the prespecified simple fallback policies did not improve on synthetic-only on average. These are portfolio findings, not a general trust-model test.

The next milestone is the noncanonical depth-first research program in `data/manifests/research/depth_first_v1.json`. It treats the portfolio five plus the completed smoke experiment as discovery evidence, adds three newly runnable outcome-sealed experiments (`nj5dx`, `es4xw`, and `e2pyb`) as prospective-development evidence, and keeps six runnable contracts sealed for confirmation. The 15-task corpus target is therefore met and corpus search stops. Simulator execution, diagnostic freezing, fallback freezing, and the prospective-development reveal remain separate later authorizations; the expansion does not make the canonical split preflight pass and does not authorize outcome reveal or paid compute.

Later scope decision: the six-task confirmation is complete. A subsequent
attempt to qualify a 12--16 experiment independent prospective replication
panel stopped before inference or reveal after both source-search lanes failed
their outcome-blind yield rules. That attempt remains historical planning
evidence under `docs/decisions/independent_replication_stage_v1.md`; it is no
longer required for project completion and grants no model-call, participant,
or outcome authority.

The targeted lane closed at order 9 with one strict scientific survivor,
`dvwu7`, below its frozen minimum of two. Orders 10--12 remain unopened. The
project preserves `dvwu7` for mechanical mapping and adapter work but stops
open-ended external corpus search rather than relaxing source standards.

The current milestone is the role-focused evaluation product in
`docs/decisions/role_focused_evaluation_program_v1.md`: preserve the completed
prospective evidence, expose a single verified evaluation lifecycle and scoped
release gate, support experiment-paired model regression, and ship an
aggregate-only case study, results explorer, and technical report. The pivot
changes completion work, not the claim boundary or interpretation of revealed
results.

The post-reveal aggregate analysis also evaluates a uniform-random arm policy and leave-one-experiment-out effect attenuation without reopening participant records. Random choice has expected exact accuracy 0.333 and mean regret 0.0293. Cross-fitted attenuation preserves all five choices and lowers local-model treatment-effect MAE from 0.0467 to 0.0394, still above the no-effect policy's 0.0361. It is explicitly development-only and does not authorize applying a calibrated policy to sealed tasks.

Use `docs/audits/dataset_qualification.md` as the qualification record. Start source validation with the 17 strict candidates in the independently curated overlap; this strengthens the smoke test without selecting on outcomes.

## Stage 2: Build the Blinded Experiment Registry

Create a structured registry containing only design and availability information:

- `experiment_id`
- `source_id`
- `paradigm_group`
- `design_type`
- `randomization_unit`
- `condition_ids`
- `condition_descriptions`
- `deployable_arm_ids`
- `control_arm_id`
- `primary_outcome_id`
- `outcome_family`
- `response_options`
- `scale_lower`
- `scale_upper`
- `higher_is_better`
- `weighting_rule`
- `missingness_rule`
- `eligible`
- `exclusion_reason`

No registry field may contain human responses, arm means, effect sizes, significance, or winners.

### Phase 1 Eligibility Rule

The first decision task must have:

- Verified between-subject randomization.
- One experimental factor.
- Two to four deployable arms.
- Text-only treatment material.
- One binary or ordinal primary outcome.
- Explicit response options and questionnaire bounds.
- Defensible outcome orientation.
- Clear control/reference.
- At least 100 valid observations per arm.
- No carryover, order, conjoint, list, multi-wave, or unmodeled factorial complication.

If validation contains eligible tasks from the independently curated overlap, restrict the Phase 1 selection pool to that audit stratum. Otherwise use all eligible validation tasks. In either case, select the lexicographically smallest stable task ID after eligibility filtering. Do not hand-pick the most interesting task.

## Stage 3: Freeze the Paradigm-Group Split

Target approximately 65/15/20 train/validation/test experiments while assigning entire paradigm groups.

The split manifest must include:

- Dataset revision.
- Registry hash.
- Grouping-rule version.
- Seed.
- Experiment-to-paradigm mapping.
- Experiment-to-split mapping.
- Counts and achieved percentages.
- Manifest hash.

Assertions:

- Every eligible experiment appears exactly once.
- Every experiment's rows remain together.
- Every paradigm group appears in exactly one split.
- Repeating with the same input and seed produces the same manifest.
- Development commands cannot change test membership.

## Stage 4: Declare the Decision Task

Write a frozen decision-task record before simulation:

- Experiment ID and source.
- Validation split membership.
- Target-population access regime.
- Admissible arms and whether control is eligible.
- Control/reference arm.
- Outcome and response family.
- Questionnaire bounds and orientation.
- Utility transform.
- Intention-to-treat estimand.
- Missing-data rule.
- Weighting rule.
- Tie rule.
- Practical regret tolerance and sensitivity grid.

For Phase 1, use unadjusted mean normalized utility. If weights are unavailable, explicitly label the estimand as the released analytic sample rather than the national population.

## Stage 5: Construct the Blinded Simulator Bundle

Create a separate artifact containing only:

- Decision-task identifiers.
- Treatment descriptions.
- Outcome question.
- Valid response options and scale labels.
- Permitted pre-treatment population profiles.
- Outcome-free experiment metadata.

Exclude:

- `response`
- `reasoning`
- Released prompts unless reconstructed and verified safe.
- Actual target condition assignment when using a common persona roster.
- Human arm counts derived after outcome exclusions, unless declared as design information.
- Human means, effects, uncertainty, significance, winner, or regret.
- Paper title, authors, abstract, results, tables, and conclusions.

Hash the blinded bundle and assert that the simulator process has no path to the human-outcome store.

## Stage 6: Run Two Simulator Paths

### Required Baseline: No-Effect Control Policy

- Predict normalized expected utility `0.5` under every arm.
- Produce zero synthetic treatment effects.
- Select control using the frozen tie rule.

This is the population/no-treatment-effect baseline. It must not use target arm outcomes.

### Required Non-Oracle Simulator

Run one real simulator through a common adapter:

```text
simulate(decision_task, persona_roster, arm) -> response probabilities or samples
```

For the first run:

- Use at most 100 fixed personas.
- Evaluate every persona under every arm.
- Use three constrained stochastic draws per persona-arm pair, or one probability distribution if the model provides stable valid probabilities.
- Set a hard cost/token cap.
- Save every raw output and parser result.

If credentials or dataset-use permission are unavailable, a deterministic mock may exercise tests, but the scientific smoke test remains incomplete until a real non-oracle simulator runs.

Required provenance:

- Simulator and checkpoint revision.
- Prompt-template hash.
- Parser version.
- Persona-roster hash.
- Seed, temperature, top-p, and sample count.
- Timestamp.
- Raw-output references.
- Parse-failure count.

Do not silently impute malformed responses.

## Stage 7: Compute Synthetic Effects and Freeze the Recommendation

For each predicted response distribution, compute expected normalized utility. Aggregate over the identical persona roster:

```text
mu_S[j]  = mean predicted normalized utility under arm j
tau_S[j] = mu_S[j] - mu_S[control]
```

Rank arms and select:

```text
j_hat = argmax_j mu_S[j]
```

Write and hash an immutable recommendation artifact containing:

- All input/provenance hashes.
- Synthetic arm means and effects.
- Arm ranking and selected arm.
- Frozen tie rule.
- Any outcome-free uncertainty diagnostics computed in Phase 1.
- Artifact timestamp and final hash.

This artifact must exist and verify successfully before scoring can load validation outcomes.

## Stage 8: Reveal Validation Outcomes and Estimate Human Effects

Only after Stage 7:

- Load observed responses for the selected validation task.
- Apply the frozen eligibility, missingness, scale, orientation, and weighting rules.
- Estimate human arm utilities and effects:

```text
mu_H[j]  = mean observed normalized utility under randomized arm j
tau_H[j] = mu_H[j] - mu_H[control]
```

- Identify the point-estimate human-best arm.
- Bootstrap participants within randomized arms using a fixed seed.
- Estimate the probability that each arm is optimal and that the synthetic-selected arm is optimal.

Do not change the action set, outcome, utility, simulator, parser, or tie rule after reveal.

## Stage 9: Score the Decision

Compute:

- Human and synthetic arm means.
- Human and synthetic effects.
- Absolute treatment-effect error.
- Sign correctness.
- Exact point-estimate best-arm correctness.
- Probability selected arm is optimal under bootstrap.
- Normalized regret:

```text
regret = max_j mu_H[j] - mu_H[j_hat]
```

- Practical reliability for the frozen tolerance grid.

Regret must be non-negative apart from negligible floating-point tolerance.

## Stage 10: Produce the Smoke Report

Generate `docs/reports/phase_1_smoke_test.md` with:

- Dataset revision and license status.
- Registry, split, decision-task, blinded-bundle, and recommendation hashes.
- Experiment, outcome, design, action set, and access regime.
- Human and synthetic arm summaries.
- Treatment effects and errors.
- Synthetic recommendation and point-estimate human best.
- Exact correctness, bootstrap optimality probability, and regret.
- Simulator cost and parse failures.
- Known caveats.
- A prominent statement that one validation task is an engineering smoke test, not a benchmark conclusion.

Replaying the report from frozen raw outputs must not require another model call.

## Minimal Phase 1 Structure

```text
pyproject.toml
configs/
  phase1.yaml
src/intervenebench/
  schemas.py
  socsci210.py
  splits.py
  protocol.py
  simulators.py
  evaluation.py
  cli.py
data/manifests/
  audits/
    experiment_registry.csv
  contracts/
    decision_task.json
    blinded_bundle.json
  splits/
    split.json
tests/
  fixtures/
  test_schemas.py
  test_splits.py
  test_leakage.py
  test_effects.py
  test_decisions.py
  test_protocol.py
  test_smoke_pipeline.py
docs/
  audits/
    data_audit.md
  reports/
    phase_1_smoke_test.md
```

This is the evolved post-smoke layout. Generated render QA and the unused pre-smoke directory scaffold are archived under `.work/` rather than presented as active implementation. Trust, human-fallback, Modal, dashboard, and fine-tuning implementation remain deferred until the benchmark registry and split are ready.

## Test-First Order

### 1. Schemas

- Invalid bounds or orientation fail.
- Control must be an admissible arm.
- Composite identifiers are unique.
- Unsupported designs fail explicitly.

### 2. Splits

- Experiments and paradigms are disjoint across splits.
- Coverage is complete.
- Determinism holds.
- Test membership is immutable during development.

### 3. Leakage and Protocol States

- Blinded artifacts exclude forbidden columns and derived values.
- `reasoning` is rejected.
- Training artifacts contain no validation/test experiments.
- Scoring cannot run without a verified frozen recommendation.
- Mutation after freeze is detected.
- Test reveal requires an explicit evaluation release.

### 4. Effects

- Known two-arm and multi-arm fixtures produce exact results.
- Orientation reversal works.
- Missingness follows the declared rule.
- Bootstrap results reproduce by seed.

### 5. Synthetic Aggregation

- Probabilities are valid.
- Expected outcomes are correct.
- Persona rosters match across arms.
- Invalid responses fail visibly.

### 6. Choice and Regret

- Correct choice yields zero regret.
- Wrong choice yields the exact expected regret.
- Regret is non-negative.
- Control inclusion/exclusion and ties follow the task record.
- Practical tolerance is applied correctly.

### 7. End-to-End Fixture

A toy experiment completes blind -> recommend -> freeze -> reveal -> score -> replay with known artifacts and metrics.

## Exact Acceptance Criteria

Status: **completed for the validation-only engineering smoke path on `jf46x:task-0`**. The frozen report is `docs/reports/phase_1_smoke_test.md`. This completion does not convert the three-study smoke stratum into the canonical benchmark split or authorize test-outcome reveal.

Phase 1 passes only when:

- [x] Dataset revision and hashes are recorded.
- [x] The selected task has a source trace; the CDC/DHS reconstruction discrepancy was resolved before reveal.
- [x] Data-use permission is documented.
- [x] The eligibility registry contains no response-derived fields.
- [x] The paradigm-group smoke split is deterministic and complete.
- [x] The selected task is in validation and satisfies every Phase 1 eligibility rule.
- [x] Test outcomes have never been loaded or summarized.
- [x] The simulator bundle contains only allowlisted fields.
- [x] `response` and `reasoning` are inaccessible to simulation.
- [x] The same aggregate population roster is evaluated under every arm.
- [x] A no-effect baseline and one real non-oracle local simulator ran.
- [x] Every prediction is valid or counted as an explicit failure; this run had zero parse failures.
- [x] Synthetic arm means, effects, ranking, and recommendation are saved.
- [x] The recommendation was hashed and timestamped before outcome reveal.
- [x] Human means and effects match independent exact toy-fixture calculations and direct binary proportions.
- [x] Choice correctness and regret match an independent toy-fixture calculation.
- [x] Bootstrap optimality results reproduce with a fixed seed.
- [x] All tests pass.
- [x] The report replays exactly from frozen outputs without a model call.
- [x] The report labels the run as validation-only and non-conclusive.

## Out of Scope

- Revealing or scoring the held-out test split.
- Full SocSci210 audit completion beyond what is needed to freeze the registry procedure and select one validation task.
- Trust-model training.
- Human-fallback simulation.
- Socrates integration.
- Behavioral LoRA fine-tuning.
- Modal-scale inference.
- Dashboard.
- Paper claims based on one task.

## Phase 1 Stop Conditions

Stop and report rather than improvise if:

- Dataset-use permission is unresolved for the proposed operation.
- No candidate task has recoverable randomization and outcome metadata.
- Outcome direction or action eligibility is ambiguous.
- The selected study is within-subject, factorial, or otherwise unsupported.
- The blinded bundle cannot be separated from response-derived fields.
- The split cannot preserve paradigm groups.
- A real simulator cannot be run within the declared cost/privacy constraints.
