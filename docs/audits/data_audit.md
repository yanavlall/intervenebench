# SocSci210 Data Audit

## Status

**Gate status: open for private local Phase 1 analysis under the working decision below. Participant-level external transmission, cloud fine-tuning, and raw-data redistribution remain blocked pending separate review.**

This document records verified dataset facts, open questions, and the eligibility audit required before InterveneBench implementation. It is not a results document and must not contain treatment-effect estimates or experiment winners.

Current access summary: a separately authorized development reveal has opened only the declared primary outcomes for `5vm8g`, `xc4yq`, `de5hx`, `turagaS11`, and `wallaceS12` after their synthetic recommendations were frozen. Those five are permanently development-only. The aggregate results live in `docs/reports/portfolio_pilot_development_results.md`, not in this audit. Twenty-seven Benchmark v1 candidates remain sealed, including six runnable contracts; every canonical split remains unassigned.

The later six-task confirmation has also been revealed only after its frozen
recommendations and diagnostics were materialized. It remains the project's
small noncanonical prospective case study. A subsequent 12--16 experiment
independent-replication attempt stopped before inference or reveal: the
random-root lane closed at order 20 and the title-only action lane closed at
order 9 under their outcome-blind no-progress rules. Neither lane grants
present source-access authority, and open-ended corpus search is closed.

## Audit Log

### 2026-08-20: role-focused evaluation-product pivot

- Closed the independent-replication completion queue without inference or
  human-outcome reveal after both external qualification lanes failed their
  frozen clean-yield gates.
- Made the completed six-task prospective confirmation the bounded empirical
  case study for a behavioral-simulator evaluation and release-gating system.
- Kept the universal trust-model, autonomous-deployment, confidence-abstention,
  and successful-small-pilot claims unauthorized.
- Froze the active scope in
  `data/manifests/research/role_focused_evaluation_program_v1.json`; it
  authorizes zero spend, zero model calls, zero participant-row access, and no
  additional outcome reveal.

### 2026-08-14: independent replication and external-lane pivot

- Froze the aggregate-only independent replication gate, with 12 experiments as
  the minimum analyzable panel and 16 as the strong target. SocSci210 remains
  primary and every retained task must have a distinct fielding and paradigm.
- Audited the first 20 rows of a metadata-blind, hash-ordered public TESS-root
  lane. It yielded 0 clean passes, 3 conditional rows, 10 design-ineligible
  rows, 1 duplicate, 2 source-blocked rows, and 4 prospective failures after
  incidental result-text exposure.
- Applied the versioned checkpoint and left original orders 21--30 unopened.
  The closure uses source-audit dispositions only, not human outcomes or model
  performance.
- Before opening another source, froze a high-precision title-only external
  lane for explicit messages, frames, rhetoric, disclosure, affirmation,
  feedback, reminders, and related actions. It preserves the original hash
  order, excludes structural measurement titles, caps audit at 12 rows, and
  has no-progress checkpoints after orders 6 and 9.
- No participant member was extracted or opened, no model call was made, and
  no human-outcome reveal was authorized by this work.
- The targeted lane then closed at order 9 with one strict scientific
  survivor, `dvwu7`, below the frozen minimum of two. Orders 10--12 remain
  unopened. `dvwu7` stays outcome sealed and requires only mapping and adapter
  implementation before it can be considered runnable.

### 2026-08-11: Pinned snapshot and outcome-blind structural qualification

- Downloaded all 17 Parquet shards at revision `048481111a4425ed83dc0eacf15f8431f252b21a` for private local analysis.
- Verified every shard SHA-256 against its upstream LFS object hash.
- Verified 2,901,390 rows, 210 study IDs, 400,491 study-participant pairs, 1,194 study-condition pairs, and 1,197 study-task pairs.
- Confirmed that `(study_id, sample_id)` is unique and that the other released numeric identifiers are study-local.
- Without reading or summarizing response values, found 175 structurally between-subject studies, 161 studies with at least two conditions and a task shared across all conditions, 121 that also have a stable single stimulus per arm for at least one shared task, and 74 that additionally have two to four arms and at least 100 participants per arm.
- Matched 40 SocSci210 studies to an independently curated 70-experiment archive using public OSF titles; 17 are strict Phase 1 structural candidates.
- Confirmed from the introducing paper that automated reconstruction success established executable, non-empty parsing rather than manual semantic agreement.
- Locked SocSci210 as the conditional primary source, the curated overlap as a gold audit stratum, and de-duplicated non-overlap studies as possible external validation.

The complete qualification decision, thresholds, and crosswalk counts are recorded in `docs/audits/dataset_qualification.md`.

### 2026-08-11: Frozen 40-study viability audit

- Deterministically selected 40 of the 57 previously unaudited studies in the strict 74-study structural pool using the frozen hash rule in `docs/protocol/phase2_viability_protocol.md`.
- Recovered and inspected original questionnaires, proposals, codebooks, or treatment materials for all 40 without loading the SocSci210 `response` column or consulting results.
- Initially classified 5 as provisional simple core, 23 as scientifically valid extensions, and 12 as ineligible. Source-primary-outcome mapping then moved three provisional cores into explicit extension tracks, leaving 2 core-simple, 26 extension, and 12 ineligible studies while preserving the 28-study scientifically usable total.
- Projected approximately 50--59 scientifically usable experiments in the full strict SocSci210 pool after combining the sample with the earlier source audit. A 20% test would contain only about 10--12 experiments before paradigm grouping.
- Triggered the preregistered augmentation branch: SocSci210 remains the primary named stratum, but the full trust-model claim requires a de-duplicated external randomized-experiment stratum audited under the same rules.

The frozen registry and decision are recorded in `data/manifests/audits/phase2_candidate_registry.csv` and `docs/reports/phase2_viability_report.md`. All 40 outcomes remain sealed.

### 2026-08-11: Metadata-only repository verification

- Verified the public Hugging Face repository and pinned revision without downloading participant response records.
- The visible dataset card contains dataset structure and loading instructions but no `license` declaration or downstream use terms.
- The pinned revision is the repository's visible README update commit `048481111a4425ed83dc0eacf15f8431f252b21a`.
- The repository reports approximately 1.45 GB of downloadable files and one released `train` table.
- The metadata directory contains `participant_mapping.json` (approximately 1.71 kB), `condition_mapping.json` (approximately 10.1 MB), and `task_mapping.json` (approximately 7.31 MB).
- The public preview of `participant_mapping.json` contains 170 `seen` and 40 `unseen` study IDs. This is the Socrates study-wise mapping, not the InterveneBench paradigm-group split.
- No participant records were downloaded, no response values were inspected, and no records were transmitted to an external model or API during this check.

Evidence:

- Dataset repository: <https://huggingface.co/datasets/socratesft/SocSci210/tree/main>
- Pinned commit: <https://huggingface.co/datasets/socratesft/SocSci210/commit/048481111a4425ed83dc0eacf15f8431f252b21a>
- Dataset card: <https://huggingface.co/datasets/socratesft/SocSci210/blob/main/README.md>
- Metadata directory: <https://huggingface.co/datasets/socratesft/SocSci210/tree/main/metadata>

Gate decision: the public-repository terms and the authors' explicit research release are sufficient for private local Phase 1 analysis. They are not treated as blanket approval to transmit participant records to unrelated APIs, redistribute the raw files, or place the data in third-party fine-tuning infrastructure.

### 2026-08-12: Source-verified continuous contract for `tcg8p`

- Verified source Q11 and SocSci210 task 0 as an open nonnegative integer monthly willingness-to-pay outcome across three advance-notice policies.
- Verified source missing codes `77777`, `99998`, and `99999` from the codebook without reading SocSci210 response values.
- Locked source-aligned mean monthly WTP as the primary location estimand and median monthly WTP as a required robustness analysis.
- Implemented raw USD/month treatment effects and regret, deterministic bootstrap uncertainty, freeze/reveal authorization, and fixture replay.
- Did not assign `tcg8p` to train, validation, or test. Its outcomes remain sealed. Because Q11 has no source upper bound, pooled normalized regret remains blocked pending a development-only scale frozen before canonical test reveal.

### 2026-08-12: Recode blockers closed conservatively and two ordinal contracts frozen

- Completed a bounded provenance search for `gx6hp` and `hgmu6` across the pinned source bundles, released SocSci210 metadata, public dataset documentation, associated publication metadata, and public code/repository indexes.
- Recovered no construction program or variable-level crosswalk that proves whether stored responses were reversed when the released prompt labels were changed. No response value or distribution was inspected.
- Kept both studies in the historical scientific-viability census but barred them from Benchmark v1 scoring unless author- or dataset-maintainer-supplied provenance arrives before the canonical split. The machine-readable decision is `data/manifests/audits/response_recode_adjudications.csv`.
- Froze source-faithful, outcome-sealed ordinal decision-task candidates and blinded bundles for `5vm8g:task-2` and `xc4yq:task-7`. Both remain split-unassigned and reveal-unauthorized; contract construction used instrument text, metadata, and structural row counts only.
- Opened no new human outcomes, made no paid simulator calls, and spent no cloud compute.

### 2026-08-12: External source audit started

- Audited the first clear candidate in the frozen external-universe order, `Bougher893`, using its public OSF questionnaire, codebook, and design materials.
- Classified it as ineligible because its personalized 3-by-3 factorial cells jointly alter candidate party composition and participant-relative issue agreement, while randomized candidate labels leave no stable beneficiary or decision utility. Treating the nine cells as deployable arms would violate the action-set rule.
- Did not extract or open the participant data files contained in the materials archive. Published aggregate result text was incidentally visible on the landing page and in the proposal/manuscript while locating source materials; the exposure is logged, `Bougher893` is ineligible, and it is barred from any untouched evaluation stratum. The remaining 28 clear candidates remain uninspected for outcomes.
- Logged an accidental encounter with result-bearing text on the public study page and proposal/manuscript; it did not affect the frozen audit order or the design-based exclusion.
- Switched subsequent source qualification to bounded three-candidate batches under `docs/protocol/external_audit_batch_protocol.md`. The first batch audited frozen orders 2–4 concurrently and committed them centrally in order.
- Excluded `brandtS1`, `patriot_act`, and `Melin1066` on action-set or stable-utility grounds. No participant response file was opened. Incidental aggregate-result text exposure during discovery was logged for all three, barring them from untouched evaluation. Twenty-five clear first-census candidates remain unaudited.
- Completed parallel batch 2 for `ThorsonS42`, `Harbridge-Yong1032`, and `converseS16` using direct-source retrieval. `ThorsonS42` is an exact source-level duplicate of SocSci210 `xtvu5`; `converseS16` is source-blocked because one fielded experiment cannot be identified; and `Harbridge-Yong1032` is the first scientifically eligible external extension.
- Froze Harbridge-Yong source cells 1–3 as a three-arm ignored-alternative disclosure task under the new outcome-blind admissible-arm subset rule. Its source Q2 congressional-confidence outcome is development-only because prior-pilot result text was incidentally exposed. No participant file was opened. Twenty-two clear first-census candidates remain unaudited.
- Completed parallel batch 3 for `Krupnikov719`, `ShannonS2`, and `AnsonBRIEF60`. `Krupnikov719` is ineligible because no source outcome supplies a stable utility and substantive outcomes are selected by a treatment-affected continuation gate. `ShannonS2` is retained as a sealed six-arm factorial/repeated-message extension, and `AnsonBRIEF60` as a development-only personalized-policy extension.
- Temporary mixed archives were used only to isolate design files; participant members were never extracted or opened and the archives were removed. Incidental result/process-statistic exposure was logged for Krupnikov and prior-pilot exposure for Anson. Nineteen clear first-census candidates remain unaudited.
- Completed parallel batch 4 for `Craig735`, `Iles1294`, and `Blair1131`, with all target outcomes sealed. Craig is a distinct fielding from SocSci210 `345ms` but is ineligible for lack of a defensible policy utility; Iles is ineligible because its cells change the factual evidence state; Blair is retained as a deterministic three-arm crisis-action subset extension with a narrow public-approval utility.
- Freeze order 11 (`system_threat`) remains a separate duplicate-adjudication hold. Sixteen clear first-census candidates remain unaudited.
- Completed sealed parallel batch 5 for `KrupnikovS34`, `turagaS11`, and `SchaadS62`. Krupnikov is an ineligible measurement-wording experiment; Turaga is retained as a core-simple three-arm environmental information-policy extension; and Schaad is an exact duplicate of SocSci210 `9nphm` and independently lacks deployable actions.
- No participant or result-bearing file was opened. Thirteen clear first-census candidates remained unaudited at that checkpoint.

### Completed first external census

- Replaced exhaustive source reconstruction with the back-tested two-stage funnel in `docs/protocol/external_audit_batch_protocol.md`: hard instrument-level exclusions stop after minimal evidence, while every survivor receives a complete decision-contract audit.
- Completed all 29 clear candidates. The final 31-row universe contains 7 scientifically usable modules, 3 source-blocked rows, and 21 ineligible rows including 5 exact SocSci210 duplicates.
- Five usable modules remain sealed; two are development-only because prior result text was incidentally encountered. `KlarS44` shares respondents and fielding with SocSci210 `xtvu5`, leaving 6 additional distinct usable fieldings rather than 7 independent experiments.
- Resolved `system_threat` as the same TESS2 107 fielding as SocSci210 `345ms`. A qualitative abstract was unexpectedly exposed during DOI identity checking; no participant values or arm statistics were opened, but `345ms` is conservatively development-only and excluded from canonical test evaluation.
- Recovered metadata orphan `willer845` as TESS/OSF node `d3agv` and excluded it because its arms substitute different factual evidence/world descriptions rather than deployable actions under one fixed world.
- No participant response record was opened anywhere in the census. The completed yield confirms that a second independent external corpus is required.

The current external decisions and source hashes are recorded in `data/manifests/audits/external_source_audit_registry.csv`, `data/manifests/audits/external_source_bundles.csv`, and `docs/audits/external_source_audit_log.md`.

### 2026-08-12: Benchmark v1 candidate-scope freeze

- Froze the exact already-audited Benchmark v1 scope without opening any new outcomes: 38 candidate modules, comprising 31 SocSci210 and 7 external modules.
- Thirty-two candidates remain sealed. Two are barred by unresolved response-recode provenance and three are barred by instrument-level structural adjudications, leaving 27 potentially eligible for canonical test after task-contract completion. Five are development-only after result exposure, and the completed Phase 1 validation task is development-only after its authorized validation reveal.
- Deliberately left every canonical train/validation/test assignment unassigned. Primary task contracts, shared-fielding dependencies, and cross-stratum paradigm groups must be resolved first.
- Recorded deterministic candidate selection hashes, source-registry hashes, outcome-access states, contract blockers, and the known shared TESS 8041.040/043 fielding for `KlarS44`.
- Opened no participant outcome, made no paid simulator call, and spent no Modal compute during the freeze.
- Paused a second-corpus search. It becomes a contingency after a cost-capped Benchmark v1 pilot rather than the next milestone.

### 2026-08-12: cross-audit reconciliation and readiness freeze

- Reconciled `willer845` to SocSci210 source `d3agv` and `relihan1399` to SocSci210 source `yp736`. The later final-instrument audits are stronger than the earlier provisional design screen and prove that both experiments vary fabricated or incompatible factual world states rather than deployable actions. No outcome was used in either adjudication.
- Preserved both rows in the 38-module historical candidate census but barred them from Benchmark v1 scoring through `data/manifests/audits/cross_audit_adjudications.csv`. A subsequent complete-instrument audit also barred `ftwqy`: its randomized passages assert incompatible 25% versus 55% occupational-composition facts, not deployable actions under one world. Later result-exposure dispositions for `mzm26` and `egmxd` reduce the canonical-test-potential count to 27.
- Added sealed outcome-blind simulator contracts for `turagaS11` and the deterministic cells 1--3 subset of `Blair1131`. Neither is scoreable until source variable, missing-code, weight, allocation, and realized-support mappings are verified.
- Froze `data/manifests/benchmark/benchmark_v1_readiness.csv`: 3 runnable sealed contracts, 2 simulator-ready external contracts with scoring mappings pending, 33 conservative paradigm clusters, and no canonical split assignments.
- Opened no participant outcome, made no paid simulator call, and spent no Modal compute.

### 2026-08-12: outcome-blind external scoring mappings

- Completed source-data mappings for `turagaS11`, `Blair1131`, and `wallaceS12` using a zero-row SAV dictionary or codebook followed only by assignment, weight, and prespecified nuisance columns. No target outcome value or outcome-derived summary was opened.
- Froze exact assignment/outcome/weight fields, missing codes, retained-cell support, archive hashes, and temporary-file deletion attestations in `data/manifests/audits/external_schema_mappings.csv`.
- Completed weighted Hájek decision-task contracts and blinded simulator bundles for all three. Added the six-arm `wallaceS12` military commitment/compliance action subset with source Q1 as the reversed reputation utility.
- Added generic weighted-arm means and within-arm weighted bootstrap support with synthetic fixture tests.
- Regenerated the readiness freeze: 6 runnable sealed contracts, 33 conservative paradigm clusters, and no canonical split assignments. `Blair1131` remains an extension because two retained arms have fewer than 100 assigned rows.
- Opened no human outcome, made no paid simulator call, and spent no Modal compute.

The machine-readable artifacts are `data/manifests/benchmark/benchmark_v1_candidates.csv` and `data/manifests/benchmark/benchmark_v1_freeze.json`; the decision record is `docs/decisions/benchmark_v1_scope_freeze.md`.

### 2026-08-12: supported-ordinal pilot and sequence-aware readiness

- Completed the sealed `de5hx` source contract, including exact article stimuli, randomized response-order variants, a Jack Tucker primary utility, and the mandatory inverse Gary Rogers sensitivity utility.
- Completed structural human mappings for `ShannonS2` and `KlarS44` without reading outcomes. Both are deferred from runnable status because their focal modules were embedded in randomized multi-module surveys; an isolated prompt would not faithfully reconstruct treatment exposure.
- Regenerated the readiness freeze with 7 runnable sealed contracts and 2 sequence-adapter-deferred mappings across 33 conservative paradigm clusters.
- Froze a five-task bounded-ordinal engineering pilot with three train, one validation, and one sealed test label. This pilot is noncanonical and does not authorize any human-outcome reveal.
- Added a no-effect policy artifact and a deterministic hashed-ridge classical baseline with an experiment-to-split guard; the real baseline remains unfitted until an authorized development reveal.
- Opened no human outcome, made no paid simulator call, and spent no Modal compute.

### 2026-08-13: governed five-task development reveal

- Verified the hash-bound local run and all five frozen recommendations before reading any target outcome.
- Recorded a separate machine-readable authorization that permanently assigns `5vm8g`, `xc4yq`, `de5hx`, `turagaS11`, and `wallaceS12` to development-only use and fixes the outcome columns, bootstrap, fallback allocation, and fusion rule.
- Read only the three declared SocSci210 study/task slices and, for the two external tasks, assignment, the predeclared Q6a/Q1 fields, and weight. Wrote no participant row to the score artifact.
- Preserved the other six runnable contracts as sealed and regenerated the Benchmark v1 scope, readiness, and canonical-split preflight. The preflight now sees six runnable sealed fieldings, six paradigms, and one projected test experiment, so it remains blocked.
- Corrected one fallback-evaluation implementation issue before the final report: the first artifact changed its evaluation remainder with budget. The corrected version fixes a disjoint evaluation third per replicate across every nonzero budget. The superseded artifact remains for provenance.
- Used no paid inference and no Modal compute.

### 2026-08-12: source-programmed survey-sequence adapters

- Added fail-closed sequence bundle validation and deterministic nuisance episodes that are paired across every intervention arm for the same synthetic persona.
- Promoted `KlarS44` with both Klar/Thorson block orders, both Thorson wording and outcome-block orders, and all programmed item-list permutations.
- Promoted `ShannonS2` with all six whole-module orders, six within-Shannon vignette orders, 48 programmed Powell paths plus independent Q3 answer order, and all four Farrow conditions.
- Kept both sequence tasks outside the first five-task cost-capped pilot because their longer prompts make them a separate compute tier; this is a cost decision rather than a scientific-readiness blocker.
- Regenerated the outcome-blind readiness freeze with 9 runnable sealed contracts across 33 conservative paradigm clusters at that checkpoint; the later `z358z` adapter raises the current total to 10.

### 2026-08-12: `z358z` fixed-context human contract

- Read and visually verified the complete 24-page final Kalla/Nayak/Saperstein questionnaire. The Nayak module is a participant-randomized 2-by-2 design crossing research context with consent-policy proposal.
- Froze one deterministic action subset without consulting outcomes: SocSci conditions 0 and 1, which map exactly to the CTD-versus-TRT drug scenario with general notification versus verbal consent. Conditions 2 and 3 are the omitted dosing-time context and cannot become another independent experiment.
- Selected source Q3a / SocSci task 2 under the frozen hierarchy: it is the earliest supported common downstream item and measures perceived value of conducting the randomized drug study on a 1--7 scale. The decision utility is limited to public legitimacy or acceptability and does not establish clinical, legal, autonomy, welfare, or ethical superiority.
- Projected SocSci210 only for structural columns and read the official SAV only at zero rows plus `XTESS175`, `DOV_OPTION`, `weight`, and the row identifier. The four-cell join had zero mapping mismatches; no Q1--Q7 value, response distribution, effect, significance result, or winner was opened.
- Added a source-programmed sequence bundle covering all four Kalla/Nayak/Saperstein orders. Its structured Kalla generator produces three randomized candidate pairs with office context, randomized trait order, and all six source attributes for both candidates; its Saperstein branch preserves both question orders. The same deterministic nuisance episode is paired across both consent-policy arms. This promotes `z358z` to the tenth runnable sealed contract while keeping it outside the first low-cost pilot. The released Socrates mapping marks `z358z` as seen, so that checkpoint cannot be described as experiment-held-out on this task.
- Opened no human outcome, made no paid simulator call, and spent no Modal compute.

### 2026-08-12: `pb2rr` image and source-behavior contract

- Read and visually verified the complete 12-page final questionnaire and both one-page source PDF stimuli. The design independently randomizes an article prime and one of 16 recipient names before an incentivized $0–$10 dictator-game transfer.
- Applied the frozen primary-outcome hierarchy to source Q4, the first behavioral target of the experiment. SocSci210 omits Q4 and the recipient-name factor, retaining only later identity and threat measures; those later tasks may not substitute for the incentivized behavioral primary.
- Froze the deployable decision as the Hispanic-population-growth article versus the attention-matched iPhone-growth control article. The estimator computes a study-weighted Hajek mean within each article-by-name cell and averages the 16 name cells equally within article arm. This preserves the planned 50/50 White-name versus Black-name distribution and uniform implementation within each name-race level.
- Added a fail-closed image/PDF bundle that pins both fielded assets by SHA-256, pairs every recipient name across both simulated arms, requires a complete arm-by-name prediction grid, and parses the full bounded $0–$10 response distribution.
- Used the already-local mixed archive only to isolate a temporary participant SAV. Read zero rows for labels, then projected only `XTESS187`, `DOV_INSERT_NAME`, and `weight` to verify 781 source rows, positive weights, and support in all 32 randomized cells. Q4 was never projected or opened; no outcome missingness count, distribution, arm statistic, effect, significance result, or winner was computed. The temporary SAV was deleted.
- Regenerated the outcome-blind readiness freeze with 11 runnable sealed contracts. `pb2rr` remains outside the first cheap five-task pilot because it requires image-capable simulation and source-data scoring. No paid simulator call or Modal compute was used.

### 2026-08-12: categorical menu contract and workbook fail-closed guard

- Completed `egmxd` as an exact-image, three-arm menu-label contract. Extracted and hash-pinned the three embedded fielded menus, froze the 14-option categorical outcome, and added participant-level one-hot reconstruction plus weighted arm utility.
- Projected only P_COND and WEIGHT from the temporary source DTA and structural fields from SocSci210. No participant outcome row was opened; the temporary DTA was deleted.
- A workbook advertised as a codebook contained unweighted and weighted frequency sheets. An over-broad preview accidentally exposed aggregate target menu-choice frequencies. Inspection stopped. None of those values informed task selection, utility, prompts, diagnostics, features, models, thresholds, or tests; `egmxd` is permanently development-only.
- Added a fail-closed XLSX metadata check that reads sheet names from `xl/workbook.xml` and rejects frequency/result-bearing workbooks before any cell inspection.
- Mapped `mzm26` source Q12 as an incentivized $0--$10 donation, but deferred its simulator contract because all four exact fielded videos are absent from the official source deposit. Prior-pilot result text in its proposal also makes it development-only.
- Regenerated the 38-module freeze: 32 sealed, 27 canonical-test-potential, 6 development-only, 5 terminally barred, and still 11 runnable sealed contracts. No paid simulator call or Modal compute was used.

### 2026-08-13: 4w9pz source mapping and fail-closed scale gates

- Rendered and checked the complete 17-page Hecht teen questionnaire. Verified the final two-arm teacher-policy fielding, source-only behavioral primary `T70_14`, and randomized Campbell-versus-Hecht block order.
- Used a result-free codebook and zero-row DTA dictionary, then projected only `P_COND70` and `WEIGHT`: 392 undermining / 411 affording rows and positive finite weights. No `T70_14` value, missingness count, distribution, arm outcome, effect, or winner was opened. The temporary DTA and render directory were deleted.
- Froze `4w9pz_decision_task_candidate.json` and a recommendation-bound source-binary CSV reader. The task remains simulator-blocked because the exact co-fielded Campbell teen module and photographs are absent; the separately deposited adult Campbell experiment is not a faithful substitute.
- Added a canonical-split preflight. It blocks assignment at 11 provisional independent runnable fieldings, 11 paradigms, and two projected test experiments; both the 60/20 exploratory and 100/25/20 full-claim gates fail.
- Froze a $25 engineering-pilot ceiling without authorizing spending: $18 prompted-inference tier plus $7 interrupted-call/parser reserve. Paid inference, Modal, fine-tuning, real-outcome classical fitting, and outcome reveal remain unauthorized.
- All 144 tests pass. No paid simulator call or Modal compute was used.

### 2026-08-13: reduced portfolio scope and response-free five-task run

- Froze the immediate deliverable as a five-experiment portfolio study while preserving the publication-scale benchmark as optional future work.
- Authorized local outcome-blind inference only. Human reveal, paid inference, Modal, fine-tuning, real-outcome classical fitting, and a trust-model claim remain blocked.
- Added a common bounded-ordinal runner and explicit relative-weight parser. The first strict probability-sum response and a count-format diagnostic failed closed; neither produced a recommendation. Before the successful run, the parser contract was explicitly changed to retain non-negative raw relative weights and deterministically normalize them.
- Completed 60 structured local calls with Ollama `llama3.2:3b`, froze five recommendations and their raw outputs, and hash-bound the run to the scope, engineering split, tasks, bundles, local model digest, and code.
- Outcome-free winner stability was 1/3 for four tasks and 2/3 for one task. These are diagnostics only; no correctness claim is possible before reveal.
- Human outcomes opened: no. Paid cost: $0. Modal used: no. Human reveal remains unauthorized.

## Public Snapshot

- Dataset: `socratesft/SocSci210`
- Pinned revision: `048481111a4425ed83dc0eacf15f8431f252b21a`
- Public location: <https://huggingface.co/datasets/socratesft/SocSci210>
- Paper: Kolluri et al., *Finetuning LLMs for Human Behavior Prediction in Social Science Experiments*, EMNLP 2025, <https://aclanthology.org/2025.emnlp-main.1530/>
- Reported size: approximately 2.9 million response rows, 400,491 participants, 210 studies, 1,194 conditions, and 1,197 outcomes.
- Public storage format: 17 Parquet shards plus three mapping files.
- Public table split: a single `train` table; benchmark mappings are stored separately.

## Released Columns

The public dataset server reports:

```text
sample_id: int64
participant: int64
demographic: struct
stimuli: string
response: int64
condition_num: int64
task_num: int64
prompt: string
reasoning: string
study_id: string
```

The released schema does not explicitly encode:

- Outcome family.
- Valid response scale and labels as structured fields.
- Outcome orientation.
- Control arm.
- Whether an arm is a deployable intervention.
- Between-subject versus within-subject design.
- Factorial or repeated-measures structure.
- Randomization unit.
- Survey weights.
- Source publication date or contamination metadata.

These fields must be reconstructed and manually verified against original study materials before causal or decision scoring.

## Known Methodological Risks

### Mixed Designs

The SocSci210 paper states that source studies include both between-subject and within-subject designs. A simple difference in condition means is not a universal estimator. Phase 1 therefore admits only a verified, clean between-subject randomized experiment.

### Outcome-Derived Reasoning

The `reasoning` field contains oracle rationales generated using the observed human response. It is forbidden in all target-experiment simulator inputs, prompts, embeddings, diagnostics, and trust features.

### Prompt Safety

The released `prompt` field appears intended for response prediction, but InterveneBench will not treat it as automatically safe. Prompts must be regenerated from an explicit allowlist of verified design, treatment, outcome, response-option, and permitted demographic fields.

### Identifier Scope

The public schema does not document whether `participant`, `sample_id`, `condition_num`, or `task_num` are globally unique. Until verified, use composite identifiers that include `study_id`.

### Response Encoding and Recode Provenance

`response` is stored as an integer across heterogeneous studies. Do not infer binary, ordinal, or numeric family from dtype. Verify valid values and questionnaire bounds from source materials. Never use observed held-out minima or maxima to define the scale.

Store three distinct objects for every scored outcome: the source variable coding and labels; the released SocSci prompt coding and labels; and the exact transformation, with provenance, that maps stored responses to the canonical scale. If source and prompt labels disagree and the transformation cannot be recovered independently, the task remains unscored. Never infer the transformation from response distributions, arm means, expected effect direction, or a reported result.

### Sampling Weights

No survey-weight field appears in the released schema. Unless weights can be recovered from original data, InterveneBench must describe human estimates as unweighted estimates for the released analytic sample rather than weighted nationally representative population estimates.

### Reconstruction Error

SocSci210 was built with an automated reconstruction pipeline. Every benchmark-eligible experiment needs a source trace and a manual check that participant-condition-outcome mappings agree with the original data and codebook.

The SocSci210 paper's automated success check required the parser to execute and produce non-empty output. It did not report manual semantic verification of every reconstructed condition, task, or response scale.

A shared `task_num` is not sufficient evidence of one scalar outcome: some study-participant-condition-task keys contain repeated items or varying stimulus text. The registry must either reconstruct the intended composite utility or exclude the task.

## Data-Access Decision

The public Hugging Face dataset card does not currently state a dataset-specific license. However, the repository is intentionally public, its card provides loading instructions, and Hugging Face's public-repository terms grant users a broad license to use public repository content through the service. The SocSci210 project also explicitly describes the data and models as released research artifacts.

Working decision for this private research project (not legal advice):

- **Private local download and analysis:** permitted for Phase 1.
- **External simulator prompting:** permitted only in the `DESIGN_ONLY` regime using verified treatment text, outcome questions, response options, and synthetic or public population specifications. Do not transmit participant rows or target participant demographics.
- **Participant-level external API transmission:** prohibited unless separately approved.
- **Local derived artifacts:** hashes, split manifests, code, non-sensitive reconstructed metadata, and aggregate metrics may be retained.
- **Raw-data redistribution:** prohibited; direct future users to the pinned upstream repository.
- **Cloud fine-tuning or uploading raw records to managed compute:** deferred until a separate data-use decision.
- **Commercial use:** outside the current project decision and requires renewed review.

Author clarification is still worthwhile before public release, cloud fine-tuning, or commercial use, but it is no longer a blocker for local Phase 1 work.

Evidence for the working decision:

- Hugging Face public-repository terms: <https://huggingface.co/terms-of-service>
- Hugging Face license guidance: <https://huggingface.co/docs/hub/repositories-licenses>
- SocSci210 release page: <https://stanfordhci.github.io/socrates/>

## Split Compatibility Gate

The released `participant_mapping.json` lists 170 seen and 40 unseen study IDs for the Socrates work. This mapping may be useful for a secondary comparison, but it does not by itself guarantee InterveneBench paradigm separation.

Before using a released Socrates checkpoint:

- Record its exact fine-tuning mapping.
- Confirm that every primary evaluation experiment was absent from its fine-tuning set.
- Otherwise restrict evaluation to a safe intersection, retrain, or label the comparison as potentially exposed.

## Experiment Audit Template

Complete one row per source experiment while blinded to human response values and reported results.

| Field | Required entry |
|---|---|
| `experiment_id` | Stable SocSci210 `study_id` plus source identifier |
| Source | TESS/OSF URL and local source hash |
| Design | Between-subject, within-subject, factorial, repeated-measures, other |
| Randomization unit | Participant, cluster, item, other |
| Conditions | Verified arm IDs and descriptions |
| Deployable arms | Arms a decision-maker could actually choose |
| Control/reference | Verified arm and rationale |
| Primary outcome | Preregistered/designated primary outcome or blinded fixed rule |
| Outcome family | Binary, ordinal, bounded numeric, count, categorical, text |
| Scale | Allowed values, lower bound, upper bound, labels |
| Orientation | Higher better, lower better, or no defensible utility |
| Missingness | Definition and planned handling, without effect summaries |
| Weights | Weight field and estimand, or explicit unweighted analysis |
| Modality | Text-only, image, video, interactive, mixed |
| Paradigm group | Blinded taxonomy label |
| Phase 1 eligible | Yes/no with design-based reason |
| Notes | Reconstruction discrepancies or unresolved issues only |

## Phase 1 Selection Rule

The smoke-test decision task must be selected from validation after the split is frozen and must satisfy:

- Verified between-subject randomization.
- One experimental factor.
- Two to four deployable arms.
- Text-only treatment.
- One binary or ordinal outcome with explicit response options and direction.
- Clear control/reference and action set.
- At least 100 valid observations per arm.
- No selection based on effect size, significance, simulator performance, or winner.

If several validation tasks qualify, choose using a frozen outcome-free rule such as lexicographic experiment ID after eligibility filtering.

## Unresolved Questions

- [x] Record a working basis and restrictions for private local Phase 1 analysis.
- [ ] Obtain explicit clarification before participant-level external transmission, cloud fine-tuning, raw-data redistribution, or commercial use.
- [x] Download the pinned snapshot and verify shard hashes.
- [x] Confirm that all 210 mapped study IDs appear in the complete data.
- [x] Determine identifier scope and uniqueness.
- [ ] Reconstruct condition and outcome metadata for all studies. The 17-study strict curated-overlap stratum is complete; 40 additional studies have completed design screening, but most retained studies still require primary-outcome mapping.
- [ ] Verify which `study_id` values contain one versus multiple experiments.
- [ ] Recover design type and randomization unit for the broader registry. Complete for the 17-study strict curated-overlap stratum and the frozen 40-study viability batch.
- [ ] Recover questionnaire bounds, labels, and orientation for the broader registry. Complete for the three eligible smoke-stratum tasks.
- [ ] Locate survey weights or lock an unweighted estimand.
- [x] Create and freeze the outcome-blind Phase 1 smoke-stratum paradigm taxonomy.
- [x] Freeze the 17-study Phase 1 eligibility registry and three-task smoke split. This is not the canonical full-benchmark split.
- [x] Select and score the first validation smoke-test task using the frozen rule (`jf46x:task-0`).
