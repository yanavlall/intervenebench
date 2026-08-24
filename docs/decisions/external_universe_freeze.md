# External Candidate Universe v1

**Freeze date:** 2026-08-12  
**Freeze label:** `external-universe-v1:20260812`  
**Source:** CC0 70-study Nature/TESS feature archive  
**Outcome status:** sealed

## Purpose

This freeze defines the first external augmentation census before any external human outcome, treatment effect, reported winner, simulator output, or regret is used. It is a candidate universe for source audit, not a declaration that every row is eligible.

The source tables are pinned locally under `data/raw/nature_treatment_effects/4427d699-e1a1-4c93-9cf8-7374e6607e18_v1.0/`. `RA_study_features.csv` contains 70 study rows. `RA_outcome_features.csv` contains 71 unique study IDs, including the orphan `willer845`, which has no study-feature row.

## De-duplication Rule

The crosswalk compares normalized titles from the 70-row archive with the 199 pinned SocSci210 OSF metadata records available locally. It freezes:

- 39 normalized exact-title matches; and
- one prespecified high-confidence replication-suffix match, `KlarBRIEF70` to SocSci210 `wn3y9`.

Those 40 underlying studies are excluded from the external candidate universe and recorded in `data/manifests/audits/external_archive_crosswalk.csv`. De-duplication must ultimately operate at the fielded-experiment level, so title matching is only the first pass.

Every remaining study-feature row receives a deterministic hash:

```text
SHA256("external-universe-v1:20260812:" + archive_study_id)
```

The universe is sorted by that hash. No performance or outcome field participates in ordering.

## Frozen Census

`data/manifests/audits/external_candidate_universe_v1.csv` contains 31 rows:

| Status | Count | Meaning |
|---|---:|---|
| Pending source audit | 29 | Clear first-pass non-overlaps with sufficient study metadata |
| Pending duplicate adjudication | 1 | `system_threat` may be the same fielded experiment as SocSci210 `345ms` |
| Insufficient metadata | 1 | `willer845` appears only in the outcome-feature table |

The universe therefore provided 29 immediately auditable candidates, not 31 automatically eligible experiments. The completed audit subsequently confirmed `system_threat` as the same fielding as SocSci210 `345ms` and recovered `willer845` as TESS/OSF node `d3agv`, where it failed the fixed-world action-set gate.

## Consequence for the Full Project

This archive is valuable as an independently curated audit and augmentation source, but its completed yield is small: 7 scientifically usable modules, 3 source-blocked rows, and 21 ineligible rows including 5 exact SocSci210 duplicates. One usable module shares respondents and fielding with an existing SocSci210 experiment, leaving only 6 distinct additional usable fieldings. A second independent external candidate corpus is therefore required before a 100-experiment trust-model claim can be made.

The next external step is not to loosen eligibility. It is to freeze another randomized-experiment corpus, de-duplicate it against both SocSci210 and this archive at the fielded-experiment level, and apply the same source, action-set, utility, and outcome-leakage rules.
