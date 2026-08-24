# Phase 2 Dataset-Viability Audit Protocol

**Freeze date:** 2026-08-11  
**Dataset revision:** `048481111a4425ed83dc0eacf15f8431f252b21a`  
**Selection label:** `phase2-viability-v1:2102026`  
**Outcome status:** sealed

## Purpose

Before scaling simulators or fitting a trust model, this audit estimates how much of SocSci210 can support defensible intervention decisions. It tests the dataset and reconstruction burden, not model performance. No human response, treatment effect, reported result, simulator output, or intervention winner may be used to select or classify a study.

The frozen batch contains 40 of the 57 previously unaudited studies that passed the strict outcome-blind structural screen: between-participant condition assignment; two to four released conditions; at least one task shared across conditions with one stable released stimulus per condition; and at least 100 study-local participants in every condition. The 17 independently curated-overlap studies audited in Phase 1 were excluded before selection.

Selection is deterministic. For every remaining candidate, compute:

```text
SHA256("phase2-viability-v1:2102026:" + experiment_id)
```

Then sort ascending by digest and take the first 40. The result is stored in `data/manifests/audits/phase2_viability_batch.csv`. Changing audit classifications does not change batch membership.

## Permitted Evidence

- Released study ID, participant ID, condition number, task number, and treatment stimulus.
- Pinned OSF node metadata, including title and source URL.
- Original proposals, questionnaires, codebooks, stimuli, preregistrations, and randomization documentation.
- Design and measurement sections of papers when no better fielded instrument exists.

Human response columns, arm means, effect estimates, significance tests, reported intervention winners, and model predictions are forbidden. Source documents that mingle design and results must be inspected only for the minimum design facts needed; any accidentally encountered result must be logged and must not influence eligibility or task choice.

## Required Reconstruction Fields

Each experiment must record:

1. original source identity and exact fielded instrument;
2. randomization unit and whether assignment is between-participant;
3. complete arm semantics, including factorial or nested assignment;
4. treatment modality and whether SocSci faithfully preserves it;
5. all plausible pre-treatment-declared outcomes and their order/randomization;
6. response scale, missing codes, and utility direction;
7. whether arms form a coherent, realistically selectable action set;
8. whether the estimator is a simple arm mean, factorial contrast, order-aware model, or another design-specific estimand; and
9. unresolved discrepancies between source materials and SocSci210.

Primary outcome selection must follow a frozen rule that does not use response values: prefer a source-designated primary outcome; otherwise the first post-treatment behavioral outcome in fielded order; otherwise the first post-treatment intention outcome; otherwise the first post-treatment attitudinal outcome. Randomized outcome order, multi-item constructs, or multiple co-primary outcomes require an order-aware or composite task specification; they may not be collapsed by convenience.

## Audit Tracks

The audit distinguishes scientific eligibility from implementation complexity:

- `core_simple`: clean between-participant, one-factor, text-preserved intervention with a supported binary or ordinal outcome and coherent action set.
- `extension_factorial`: valid randomized factorial design requiring predeclared marginal or cell-level decisions.
- `extension_continuous`: valid design with a numeric outcome requiring an explicit continuous scale, missing-code policy, and robust estimator.
- `extension_order`: valid design requiring order-aware outcome handling.
- `extension_multimodal`: valid intervention whose fielded content depends on images, charts, audio, or video.
- `extension_interactive`: valid sequential, nested, or participant-input-dependent treatment.
- `ineligible`: no coherent intervention decision, treatment cannot be reconstructed, assignment is not usable, outcome semantics are not recoverable, or source trace is inadequate.

A study may carry more than one extension flag. Extension studies are not scored by the simple Phase 1 estimator until the corresponding estimator, simulator interface, and tests exist.

## Decision Rules

- Never reinterpret survey-question wording as a deployable intervention merely to increase sample size.
- Never treat cells of an undeclared factorial as interchangeable one-factor actions.
- Never paraphrase away a fielded non-text treatment and call it source faithful.
- Never use different topics or populations as competing actions unless the source design establishes a common decision maker, objective, and target population.
- Never infer outcome utility direction from observed arm performance.
- An unresolved source trace is a failed audit, not permission to trust the flattened SocSci representation.

## Viability Gates

After all 40 classifications are frozen, estimate track prevalence with a finite-population uncertainty interval and project counts to the full 74-study strict pool. Proceed as follows:

- If the projected lower bound supports at least 100 eligible tasks only after including the 121-study broader structural pool, audit a second frozen batch from that broader pool before simulator scaling.
- If at least 60 source-faithful decision tasks across at least 20 paradigm groups appear attainable, proceed with a narrower benchmark and describe the trust model as exploratory unless the final untouched test contains at least 20 experiments.
- If fewer than 40 defensible tasks or fewer than 15 paradigm groups appear attainable from SocSci210, keep SocSci210 as a diagnostic source but add a preregistered external randomized-experiment archive before making a general trust-model claim.
- Regardless of count, do not multiply experiments into pseudo-independent units by outcomes, arms, models, prompts, or seeds.

The full-project target of 100 tasks remains aspirational, not a pass/fail condition that can override scientific eligibility. The claim must contract if the data do not support it.

## Audit Outputs

- Frozen selection manifest with source links and structural facts.
- Candidate-level audit registry with exclusion reasons and extension flags.
- Source-file hashes and a design-only audit log.
- A viability report giving observed and projected counts, paradigm diversity, source-reconstruction failure rates, and the resulting go/narrow/augment decision.
- No human outcomes or simulator results.

## Completed Decision and Source-Mapping Update

The frozen 40-study audit is complete. Its initial design screen found 5 provisional `core_simple`, 23 scientifically valid extension, and 12 ineligible studies. Source-primary-outcome mapping then moved three provisional cores to extensions without changing scientific eligibility: `4w9pz` needs source-data recovery, `de5hx` needs utility-sensitivity analysis, and `345ms` needs a composite estimator. The current registry therefore contains 2 `core_simple`, 26 scientifically valid extension, and 12 ineligible studies.

The outcome seal remained intact. The two remaining core tasks are not yet score-ready because their released prompt scale direction conflicts with source codebooks and the canonical recode has not been proven. Most eligible extension studies still require source-verified primary-outcome mapping. The audit therefore qualifies the dataset architecture and records implementation readiness separately from scientific eligibility.

The first de-duplicated external census is frozen in `data/manifests/audits/external_candidate_universe_v1.csv`; its protocol and implications are in `docs/decisions/external_universe_freeze.md`.
