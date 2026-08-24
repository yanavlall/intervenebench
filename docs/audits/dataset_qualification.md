# Dataset Qualification Decision

**Decision date:** 2026-08-11  
**Decision:** SocSci210 is conditionally qualified as InterveneBench's primary source dataset. It is not qualified as a ready-made causal benchmark or, by itself, as a sufficiently powered trust-model benchmark. The project will pair its scale with source-level manual verification, an independently curated overlap stratum, and a preregistered de-duplicated external augmentation stratum.

This decision was made from repository metadata, source-study metadata, design text, and structural counts. No treatment-effect estimate, arm mean, significance test, intervention winner, or simulator result was used.

## Bottom Line

SocSci210 remains the best available primary source for the intended project because it provides unusual breadth across independent randomized experiments and links to source materials. It does not, however, retain enough source-faithful decision tasks by itself to support a strong prospective trust-model claim. The cleaner 70-experiment Nature/TESS archive is also too small alone, and megastudies contain too few independent experiments even when they contain many intervention arms. The defensible solution is a de-duplicated combined benchmark with SocSci210 retained as the named primary stratum.

SocSci210's scale does not remove its central weakness: it was reconstructed automatically, and its public table omits causal metadata that InterveneBench needs. The introducing paper defines a successful reconstruction operationally as parsing code that executes and produces non-empty output. That does not establish that arm semantics, outcome identity, response direction, deployable action sets, or causal contrasts are correct. Every scored decision task therefore requires a source trace and a blinded design audit.

The resulting dataset architecture is:

1. **Primary benchmark source:** eligible, source-verified SocSci210 experiments.
2. **Gold audit stratum:** SocSci210 experiments independently matched to the manually curated 70-experiment Nature/TESS archive.
3. **External validation:** de-duplicated curated experiments not used in SocSci210, plus later temporal or megastudy stress tests, evaluated only after the policy is frozen.

## Sources Compared

| Candidate | Strength | Limitation | Role |
|---|---|---|---|
| SocSci210 | 210 experiments and participant-level randomized outcomes across varied social-science domains | Automatic reconstruction; missing structured causal metadata; public dataset card lacks a dataset-specific license declaration | Primary benchmark source, conditional on source verification |
| Curated Nature/TESS archive | 70 manually coded experiments; clear study and outcome metadata; CC0 release | Too few independent experiments for a strong train/validation/test trust-model evaluation on its own; the benchmark and results are already public | Gold audit stratum and de-duplicated external validation |
| UAS | Rich longitudinal and population data | Repeated panel structure, fragmented interventions, and data-use constraints make it a poor core fit | Optional later generalization dataset |
| Intervention megastudies | Many arms and direct decision relevance | Usually too few independent studies for experiment-level trust-model generalization | Later high-arm stress tests |

## Verified SocSci210 Snapshot

- Repository: `socratesft/SocSci210`
- Revision: `048481111a4425ed83dc0eacf15f8431f252b21a`
- Storage: 17 valid Parquet shards and three mapping files
- Rows: 2,901,390
- Study IDs: 210
- Study-participant pairs: 400,491
- Study-condition pairs: 1,194
- Study-task pairs: 1,197
- Released columns: `sample_id`, `participant`, `demographic`, `stimuli`, `response`, `condition_num`, `task_num`, `prompt`, `reasoning`, and `study_id`

All 17 local shard SHA-256 hashes match the upstream LFS object hashes. The mapping-file hashes are:

| File | SHA-256 |
|---|---|
| `participant_mapping.json` | `b436801983a6c31dd408bab65d5b2b79fb5515db129df2ff07acb781549ecbee` |
| `condition_mapping.json` | `ea0a2b7ffdac8cb128e7d3dc599148e711ebb72a0297f40473925ddf6870347d` |
| `task_mapping.json` | `b38799df03c844089e772b5768cef21affdc6c22af21f91be936ac2a09c3c40e` |

Identifier audit:

- `sample_id`, `participant`, `condition_num`, and `task_num` are not globally unique.
- `(study_id, sample_id)` is unique for all 2,901,390 rows and is the canonical released-row key.
- `(study_id, participant)` identifies 400,491 study-local participants.
- `(study_id, participant, condition_num, task_num)` is not a unique row key; some tasks contain repeated items or varying stimulus text.
- All registry and metric code must therefore use explicit composite identifiers and must not assume that one task index is one simple scalar outcome.

## Outcome-Blind Structural Screen

These counts are an upper bound on eligibility, not a final causal audit:

| Structural property | Studies |
|---|---:|
| All released studies | 210 |
| Structurally between-subject assignment | 175 |
| Within-subject or mixed assignment pattern | 35 |
| At least two released conditions | 205 |
| At least two conditions and a task present in every condition | 182 |
| Between-subject, at least two conditions, and a task present in every condition | 161 |
| Previous row plus one stable stimulus per condition for at least one common task | 121 |
| Previous row restricted to 2–4 arms and at least 100 participants per arm | 74 |

The fall from 161 to 121 matters. Some `task_num` values aggregate repeated items or changing stimulus text. Such tasks may require a composite utility or a design-specific estimator and must not be silently treated as a single outcome.

The 74-study count is only a Phase 1 structural upper bound. It does not yet establish one-factor randomization, text-only modality, binary or ordinal response semantics, outcome direction, control identity, deployability, or agreement with the original source files.

## Curated-Archive Crosswalk

The CC0 Nature archive contains:

- 70 study-feature rows: 50 marked TESS and 20 marked non-TESS/Coppock;
- 134 outcome-feature rows across 71 distinct outcome study IDs; and
- one outcome-only study ID, `willer845`, with no corresponding study-feature row. This inconsistency must be handled explicitly rather than silently joined away.

Public OSF metadata resolved for 199 of the 210 SocSci210 IDs before rate limiting. Normalized exact-title matching plus one high-confidence title-suffix match identified 40 overlaps with the 70-study curated archive. Of these:

- 36 pass the broad SocSci210 structural screen;
- 17 pass the stricter Phase 1 structural upper-bound screen; and
- none is considered benchmark-eligible until its design mapping is checked against both the original source and the curated coding.

The 17 Phase 1 cross-check candidates are identified by SocSci210 ID:

```text
5vm8g  9263n  bsd7j  fxcn4  j6xgs  jf46x  ncs7k  nhgxf  nk9jd
v6nhw  vemrp  vz5r4  xc4yq  xfmrn  xy8jw  y9nb7  yg958
```

This list was generated without effect estimates or response summaries. It is a candidate audit stratum, not a hand-picked performance set.

## Qualification Gates

| Gate | Threshold | Current status |
|---|---|---|
| Structural decision-task scale | At least 100 plausible tasks before manual exclusions | **Passed only as an upper-bound screen:** 121 studies meet the stable common-task structural screen |
| Independent gold audit | A meaningful source-validated overlap | **Provisional pass:** 40 title matches, including 17 strict Phase 1 structural candidates |
| Paradigm diversity | At least 25 defensible paradigm groups | **Likely with augmentation:** the audit sample contains 28 eligible design-level groups; final grouping and de-duplication remain pending |
| Untouched test size | At least 20 eligible test experiments for the full trust claim | **SocSci-only fail:** strict-pool projection implies about 10--12 before grouping; augmentation required |
| Causal reconstructability | Arms, randomization, outcomes, scales, and utility direction recoverable | **Mixed:** 28/40 remain scientifically usable and 12/40 are excluded; primary mapping moved three provisional cores to extensions and exposed two unresolved response-recode conflicts |
| Prospective leakage control | Target outcomes absent from selection, prompts, diagnostics, and tuning | **Pass by protocol; implementation tests still required** |
| Private local use | Sufficient basis for local Phase 1 work | **Open with restrictions:** see `docs/audits/data_audit.md` |

## Locked Consequences for the Project

- SocSci210 remains primary, but only source-verified decision tasks enter headline scoring.
- The 70-study curated archive does not replace SocSci210 as the primary benchmark.
- The curated-overlap indicator is a design-time audit stratum, never a response-derived trust feature.
- Paradigm groups and eligibility are frozen before outcomes are summarized.
- The grouped split should balance the curated-overlap audit stratum where paradigm constraints permit, without selecting a seed for favorable effects or model performance.
- The Phase 1 validation task must come from the curated-overlap stratum if any eligible overlap task lands in validation. Selection remains deterministic after the split.
- Test tasks receive independent source verification. Prefer two human coders for all test tasks; at minimum, use a primary coder plus independent adjudication of every ambiguity and a second-code audit of all test tasks and a random development subset.
- Titles, authors, citations, and reported results are excluded from simulator prompts. Publication status and model-cutoff exposure are recorded for contamination sensitivity analyses.
- The full trust-model claim requires at least 100 independent eligible experiments in the combined, de-duplicated benchmark, at least 25 paradigm groups, at least 20 frozen test experiments, and usable failure-label balance.
- SocSci210 and external-stratum results must be reported separately before pooled estimates. Dataset identity remains visible in diagnostics and uncertainty analysis.
- The project must not manufacture sample size by treating outcomes, arms, simulators, prompts, or seeds from one experiment as independent experiments.

## Why This Is Good Enough—and What Would Make It Exceptional

The project is good enough to proceed because the scientific question is decision-relevant, the prospective leakage design is strong, and the source audits expose rather than hide reconstruction error. The 40-study viability audit shows that SocSci210 alone is not yet good enough for a strong trust-model paper claim: it projects to roughly 50--59 usable strict-pool experiments and about 10--12 test experiments. That is a dataset-power limitation, not a failure of the research question.

The exceptional version is not “run more LLMs.” It is:

1. publish a versioned, outcome-blind decision-task registry with exclusion reasons;
2. quantify SocSci210 reconstruction agreement against source files and the curated overlap;
3. preserve an untouched paradigm-held-out test and a real freeze/reveal audit trail;
4. make normalized regret, practical reliability, and risk-coverage the headline metrics;
5. evaluate independent human rescue without reusing the scoring sample; and
6. report null or negative results when trust diagnostics or complex simulators do not beat simple baselines.

## Immediate Next Step

The 17 strict overlap candidates have now been source-validated without consulting human effects; three pass the Phase 1 gate. The resulting response-blind smoke split assigns `5vm8g` to train, `jf46x` to validation, and `xc4yq` to test. The validation smoke task completed the prospective freeze/reveal path, while `xc4yq` remains sealed. See `docs/reports/phase_1_smoke_test.md`.

The five provisional core mappings are now frozen: two remain structurally core-simple, but a bounded provenance search found no construction artifact or variable-level recode crosswalk that resolves their opposite source-versus-release scale labels. They are therefore barred from Benchmark v1 scoring unless author-supplied provenance arrives before the canonical split. Three others require source-data, utility-sensitivity, or composite extensions. The first de-duplicated external universe is also frozen and fully adjudicated. Of 31 rows, 7 are scientifically usable modules, 3 are source-blocked, and 21 are ineligible including 5 exact SocSci210 duplicates. Five usable modules remain sealed and two are development-only; `KlarS44` shares respondents with SocSci210 `xtvu5`, so the external census contributes only 6 distinct usable fieldings. Participant records were never opened; incidental published result-text exposure is logged and bars affected studies from untouched evaluation. The qualitative abstract exposed during `system_threat` deduplication also makes SocSci210 `345ms` development-only.

The cost-controlled Benchmark v1 scope records 38 audited modules: 31 SocSci210 and 7 external. After the separately governed portfolio reveal, 27 remain sealed, 22 remain potentially canonical-test eligible after contracts, and 11 are development-only. Every full-benchmark split remains unassigned. Six runnable contracts remain sealed: `tcg8p`, `z358z`, `pb2rr`, `Blair1131`, `ShannonS2`, and `KlarS44`. The five-task engineering labels never became canonical assignments; their later reveal is explicitly development-only. A second corpus remains optional unless the project pursues a general trust-model claim.

The candidate-level audit trail is maintained in `docs/audits/source_audit_log.md` and `data/manifests/audits/phase1_candidate_registry.csv`. Source verification can reduce the preliminary structural count: for example, `v6nhw` contains two randomized factors and randomized question order even though its released structure initially resembles a simple four-condition experiment.

## Evidence

- Kolluri et al., *Finetuning LLMs for Human Behavior Prediction in Social Science Experiments*, EMNLP 2025: <https://aclanthology.org/2025.emnlp-main.1530/>
- SocSci210 pinned repository: <https://huggingface.co/datasets/socratesft/SocSci210/tree/048481111a4425ed83dc0eacf15f8431f252b21a>
- Ashokkumar et al., *Large language models can predict the results of social science experiments*, Nature 2026: <https://www.nature.com/articles/s41586-026-10742-x>
- Curated archive: <https://codeocean.com/capsule/9843791/tree/v1>
- TESS: <https://www.tessexperiments.org/>
