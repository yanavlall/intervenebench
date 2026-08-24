# Benchmark v1 Candidate-Scope Freeze

**Freeze ID:** `benchmark-v1-scope-20260812`  
**Freeze date:** 2026-08-12  
**Status:** candidate scope frozen; canonical task registry and split unassigned

## Decision

Benchmark v1 will proceed using only experiments already source-audited in this repository. A second external-corpus search is paused. It becomes a later contingency only if results from Benchmark v1 show that additional experiments would materially improve the scientific claim.

This freeze does not claim that every retained module is ready for simulation or scoring. It fixes the exact outcome-blind candidate scope while preserving unresolved contract and dependency work.

## Frozen Scope

The manifest `data/manifests/benchmark/benchmark_v1_candidates.csv` contains 38 scientifically retained candidate modules:

- 31 from the primary SocSci210 stratum;
- 7 from the first frozen external archive;
- 32 with outcomes still sealed;
- 22 currently potentially eligible for canonical test after task-contract completion;
- 2 sealed but barred from scoring because stored-response recode provenance could not be recovered;
- 3 sealed but barred after instrument-first audits proved that their cells change fabricated or incompatible factual world states rather than deployable interventions;
- 5 development-only because result text or aggregate target frequencies were exposed; and
- 1 development-only because its validation outcomes were revealed for the Phase 1 smoke test.

The candidate count is not the final effective experiment count. Shared fieldings, multiple experiment modules, unresolved mappings, and unsupported estimators may reduce it. Arms, outcomes, simulator variants, personas, and seeds do not increase it.

## What Is Frozen

- Candidate identities and dataset strata.
- Source-registry identity and SHA-256 hashes.
- Existing response-blind paradigm labels.
- Current task track, contract blocker, terminal recode adjudication, and cross-audit structural adjudication.
- Outcome-access and canonical-test eligibility status.
- Known fielding dependency for `KlarS44`, which shares respondents with the TESS 8041.040/043 fielding containing SocSci210 `xtvu5`.
- Deterministic freeze order and per-candidate selection hashes.

The machine-readable freeze record is `data/manifests/benchmark/benchmark_v1_freeze.json`.

## What Is Deliberately Not Frozen

- Canonical train, validation, or test membership.
- A final claim that every candidate represents an independent fielding.
- Final task contracts for candidates with pending outcome, recode, weight, order, modality, or estimator work.
- Simulator suite or paid-inference configuration.
- Trust-model thresholds.
- Human-fallback policy.

Every row therefore has `canonical_split=unassigned`. Assigning splits now would risk separating co-fielded or related experiments and would create pressure to preserve tasks that later fail their contract.

The deterministic outcome-blind readiness and dependency map is `data/manifests/benchmark/benchmark_v1_readiness.csv`, with its self-checking record in `benchmark_v1_readiness.json`. It consolidates the 38 candidate labels into 33 conservative paradigm clusters while leaving all splits unassigned.

## Required Gates Before the Canonical Split

1. Keep `gx6hp` and `hgmu6` out of scoring unless the documented author-supplied provenance condition is met before splitting. Keep `d3agv`, `yp736`, and `ftwqy` out permanently under their instrument-level structural adjudications. The `ftwqy` final instrument itself proves that the arms assert incompatible 25% versus 55% occupational-composition facts. These gates are resolved conservatively, not guessed.
2. Complete a primary outcome and utility contract for each remaining task intended for v1 scoring. After the separately authorized portfolio reveal, six runnable contracts remain sealed: `tcg8p`, `pb2rr`, `z358z`, `Blair1131`, `ShannonS2`, and `KlarS44`. The five portfolio tasks remain valid development contracts but cannot enter an untouched canonical test. The `jf46x` contract remains the earlier validation smoke task.
3. Record all known shared-fielding and shared-participant clusters.
4. Consolidate paradigm groups across SocSci210 and the external stratum.
5. Select the supported estimator and modality tier for the first scaled pilot.
6. Exclude or defer tasks whose contract remains unresolved; do not infer missing mappings from outcomes.
7. Freeze the de-duplicated task registry and only then assign paradigm groups to train, validation, and test.

The executable preflight in `data/manifests/splits/benchmark_v1_canonical_split_preflight.json` currently blocks this step: six runnable sealed tasks imply at most six provisional independent fieldings, six paradigm groups, and one test experiment. It also requires every runnable fielding cluster to be finalized before authorization. Do not replace this gate with a nominal short split.

## Compute Boundary

This scope update opened no participant outcome row, made no paid simulator call, and spent no Modal compute. It did accidentally expose aggregate `egmxd` target frequencies while previewing a workbook advertised as a codebook. Inspection stopped, the exposure is recorded, and `egmxd` is barred from canonical test evaluation. The next paid step remains blocked until a small supported task tier, canonical development split, simulator configuration, and hard cost cap are frozen.

The first scaled pilot completed with no-effect and local response-free baselines, followed by a separate five-task development reveal. It used no paid inference or Modal. The original $25 record remains a ceiling only; any stronger simulator or additional reveal still requires explicit authorization.

The bounded-ordinal engineering pilot was frozen separately from the full canonical split. Its train/validation/test labels were orchestration fixtures only. The five declared outcomes were later opened under `portfolio_pilot_development_reveal.json`, so all five are now development-only; the labels never became canonical assignments.
