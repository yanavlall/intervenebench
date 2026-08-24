# Phase 2 Source-Audit Log

This log applies the frozen `docs/protocol/phase2_viability_protocol.md` to `data/manifests/audits/phase2_viability_batch.csv`. It contains design and measurement facts only. Human responses, arm summaries, treatment effects, significance tests, intervention winners, simulator outputs, and regret are forbidden.

## Batch-level completion record

Original questionnaires, proposals, codebooks, or treatment materials were recovered and inspected for all 40 frozen studies. The directory contents are locally preserved under `data/raw/sources/<study_id>/`; each study has at least two source files. The design-screen classifications and concise source discrepancies are frozen in `data/manifests/audits/phase2_candidate_registry.csv`, and source-bundle counts and SHA-256 digests are in `data/manifests/audits/phase2_source_bundles.csv`.

For each source-bundle digest, sort repository-relative file paths within `data/raw/sources/<study_id>/`, compute SHA-256 for each file, serialize the standard `shasum -a 256` lines in that order, and SHA-256 hash that serialized manifest. The aggregate digest detects a changed file, filename, or bundle membership; it is provenance, not a substitute for upstream hashes.

The design screen produced 5 provisional simple-core studies, 23 valid extension studies, and 12 exclusions. A later source-primary-outcome audit tightened three of the five provisional core labels: `4w9pz` requires source-data recovery, `de5hx` requires utility-sensitivity analysis, and `345ms` requires a composite estimator. The resulting current registry contains 2 core-simple, 26 valid extension, and 12 ineligible studies. The most consequential discrepancies were not response-related: `c956y` has 16 source assignments while the released reconstruction exposes two conditions; `jtgyq` omits the source control arm; and multiple released studies flatten question-order, factorial, multimodal, or multi-item assignments into apparently simple condition labels. Full count projections and the scope decision are recorded in `docs/reports/phase2_viability_report.md`.

All batch rows were outcome sealed during the original audit. Three are now conservatively development-only after later source work: `345ms` after qualitative article text exposure; `mzm26` after prior-pilot result text appeared in its proposal; and `egmxd` after aggregate target frequencies appeared in a workbook advertised as a codebook. No participant outcome row was opened. The five provisional core studies, `tcg8p`, `z358z`, `pb2rr`, `mzm26`, and `egmxd` now have source-level primary-outcome decisions; most other extension rows remain design-screen classifications pending complete primary-outcome mapping. This distinction is encoded in the registry.

### `mzm26`: development-only video-asset blocker

- Final questionnaire SHA-256: `4970abb7b784c745a9c0748dfd132d658e94f68a80c78d9cdfe484f9c1cba261`.
- The final instrument verifies participant randomization to Black versus White homeless targets crossed with no-information versus empathy videos. Q12 is the first post-video behavioral endpoint and transfers whole dollars from a real $10 participation fee to Miriam's Kitchen.
- The deployable action is no-information versus empathy, standardized over the randomized target-race nuisance factor. Race is not treated as a selectable intervention.
- The official OSF deposit contains the questionnaire and participant data but not `BLACK_NOINFO_FINAL.mov`, `BLACK_INFO_FINAL.mov`, `WHITE_NOINFO_FINAL.mov`, or `WHITE_INFO_FINAL.mov`. SocSci210 text descriptions cannot replace those videos. No simulator bundle is authorized until the exact assets are recovered.
- A proposal paragraph unexpectedly exposed prior-pilot result text. Inspection stopped. No target TESS outcome was opened or used, but the task is conservatively development-only.

### `egmxd`: complete development-only categorical multimodal contract

- Final questionnaire SHA-256: `dc08a3a01bea7435a852e9b23a12a3927ac6773e431fc82e069fe1a18d121da2`.
- The exact QR-control, low-impact-label, and high-impact-warning menu images were extracted from the source questionnaire and hash-pinned under `data/derived/stimuli/egmxd/`.
- Source Q1 requires one choice among 14 menu items. The primary participant utility is 1 for source options 2 and 7--14, which the source treatment identifies as lower climate impact, and 0 for options 1 and 3--6. The SocSci task must group all 14 flattened option rows by participant before any arm statistic is computed.
- The source DTA was projected only for P_COND and WEIGHT; SocSci210 was projected only for structural fields. No participant outcome was selected.
- A workbook advertised as a codebook contained `Unweighted Frequencies` and `Weighted Frequencies` worksheets. An over-broad preview exposed aggregate target menu-choice frequencies. Inspection stopped, those values were not retained in any contract or computation, and the task is permanently barred from canonical test evaluation.
- The repository now inspects XLSX sheet names directly from container metadata and rejects frequency/result sheets before any worksheet cell is read.

## Primary-outcome audit of the five provisional core studies

The second audit was completed on 2026-08-12 using final fielded instruments, source codebooks or zero-row variable metadata, and released SocSci210 treatment/task metadata. The SocSci210 `response` column was never projected or loaded. Exact mappings are frozen in `data/manifests/audits/phase2_primary_outcome_mappings.csv`.

The later `z358z` adjudication freezes source Q3a / SocSci task 2 within the source drug-study action subset only. Structural projection established the exact map `condition 0 = XTESS175 1 / DOV_OPTION 1`, `condition 1 = XTESS175 1 / DOV_OPTION 2`, `condition 2 = XTESS175 2 / DOV_OPTION 1`, and `condition 3 = XTESS175 2 / DOV_OPTION 2`, with zero mismatches. Only conditions 0 and 1 are retained. A sealed source-programmed adapter now marginalizes over all four Kalla/Nayak/Saperstein orders with identical nuisance paths paired across the two retained arms. No human outcome was opened.

The later `pb2rr` adjudication freezes source Q4, an incentivized whole-dollar transfer from $0 through $10, rather than substituting one of the later attitude tasks retained by SocSci210. The policy contrast is the Hispanic-population-growth article versus the iPhone-growth control article. The human estimator computes study-weighted Hajek means inside all 32 article-by-recipient-name cells and averages the 16 names equally within each article. Both one-page source PDFs are hash-pinned in a sealed multimodal bundle. Zero-row source metadata and only assignment, name, and weight columns verified positive-weight support before outcome missingness; Q4 was never projected or opened.

### `4w9pz`: source-data extension

- Final questionnaire SHA-256: `b925ffa716d2f32463cf91e1e9f606f93086b66cd2491eadeb8604a7382c446c`.
- The final fielded experiment is a two-arm teacher-speech contrast (`P_COND70=1` undermining, `P_COND70=2` affording), despite the earlier project description's larger factorial proposal.
- The frozen fallback hierarchy selects T70_14, the first actual post-treatment behavioral choice: easy review versus hard challenge for equal extra credit.
- SocSci210 omits T70_14. Its task 2 contains the two earlier help-seeking intentions, and its remaining scalar tasks are perceptions or attitudes.

Classification correction: retain scientific eligibility but move from `core_simple` to `extension_source_data`. The benchmark must recover the governed source records or exclude the task; it may not substitute a more convenient SocSci outcome.

Outcome-blind completion on 2026-08-13 recovered the source scoring schema without opening `T70_14`: `P_COND70=1` is undermining, `P_COND70=2` is affording, `T70_14=1` is easy review, `T70_14=2` is hard challenge, missing codes are 77/98/99, and `WEIGHT` is positive and finite. Assignment support is 392/411 before outcome missingness. A recommendation-bound source CSV reader now fails closed unless a validation split, matching task, and simulator recommendation are frozen.

The final teen questionnaire also freezes a new dependency: Campbell and Hecht blocks were randomized in order. The exact co-fielded Campbell teen module and its photographs are not present in the Hecht deposit. A separate adult Campbell deposit cannot be substituted because population, questionnaire, and fielding differ. Therefore the human mapping is complete, but simulator promotion is blocked as `sequence_source_assets_pending`; the task does not increase the runnable count.

### `de5hx`: utility-sensitivity extension

- Final questionnaire SHA-256: `3d59c81363d588e2744dedefd20e8dbb56df0525502dd96a3894786acabe53c8`.
- Source Q8 / SocSci task 0 is the source-first candidate-preference outcome, coded from definite Jack Tucker support to definite Gary Rogers support.
- The three arms are negative status-quo framing, positive status-quo framing, and a neutral article. The complete source articles must replace SocSci210's ellipses.
- A campaign can choose a message, but “better” reverses depending on whether the decision maker represents Jack or Gary. No neutral welfare ordering is implied by the response scale.

Classification correction: retain scientific eligibility but move from `core_simple` to `extension_utility_sensitivity`. A primary beneficiary and the opposite-beneficiary sensitivity must be frozen before scoring.

### `345ms`: composite extension

- Final questionnaire SHA-256: `dde3bf62d888198f627308f0964bb3a2b83d7ae2425833bb8db6cb1a4ec24344`.
- The proposal designates a three-item system-threat construct. The final instrument fields the items as Q3, Q4, and Q5.
- SocSci210 task 2 bundles all three questions and mixes two seven-point scales with one five-point scale.

Classification correction: retain scientific eligibility but move from `core_simple` to `extension_composite`. The estimator must identify items, normalize source-defined bounds, orient each item, freeze the aggregation rule, and preserve experiment-level uncertainty. Selecting only Q3 would violate the source-primary hierarchy.

### `gx6hp`: core-simple, recode provenance pending

- Final questionnaire SHA-256: `3a2e66e5cd41ad29799d999ed0e7149c33a8e97aa2524149eee299f010de3f22`.
- Source Q6 / SocSci task 0 is candidate favorability after violent versus peaceful campaign rhetoric.
- The source instrument and zero-row source variable metadata code 1 as strongly favorable and 5 as strongly unfavorable. The released SocSci prompt states the reverse direction.

Classification: retain `core_simple`, but block scoring until the canonical response transformation is recovered from construction code or independently verified metadata. Prompt text is not sufficient evidence that stored responses were recoded.

### `hgmu6`: core-simple, recode provenance pending

- Final questionnaire SHA-256: `110e4786d28d55ef6b6d490f4305e92dcd8237c5c532c6b30e823eabe2e63278`.
- The two arms compare a general COVID-19 death-toll message with the same context plus racial-disparity information.
- No single source-primary item is declared. The frozen fallback therefore selects final-fielded Q14 / SocSci task 0, the first post-treatment attitudinal item.
- The source instrument and codebook code 1 as strongly agree and 5 as strongly disagree. The released SocSci prompt again states the reverse direction.

Classification: retain `core_simple`, with the same recode-provenance block as `gx6hp`.

### Construction-code provenance check

On 2026-08-12, the official Socrates repository identified by the project website (`akaashkolluri/socrates`, tree `8c9aef227ab1a5b315e797c811a122a12d8bb88a`) contained only `README.md` and `assets/teaser.png`. Its README links to the released Hugging Face dataset and model artifacts but does not publish dataset-construction or response-recode code. Therefore the official repository does not resolve either scale-direction discrepancy. Both studies remain outcome sealed and unscored; resolution requires a construction artifact from the dataset authors or independent variable-level provenance that explicitly maps the stored response values.

### Accidental result encounters

During minimum necessary text extraction, the `hgmu6` proposal PDF exposed a short pilot-results paragraph and the `gx6hp` proposal exposed pilot-result text and a pilot table. These encounters occurred after the 40-study design-screen sample and before the five-study primary-outcome mappings were frozen. They were not used for eligibility, track assignment, outcome selection, utility, reconstruction, prioritization, or any model decision. The affected decisions above follow only the final fielded instruments and the frozen source-primary/fallback rule.

## `tcg8p`: source verified; continuous-outcome extension

Source title: *Do notifications affect consumers willingness to incur power outages? Evidence from Public Safety Power Shutoffs in California*.

Files inspected:

- Proposal PDF SHA-256: `3e8a05847f41bef0f0af997cd598874744c7d39594ea72ba3b796a82ecaa88af`
- Materials archive SHA-256: `e2fb3db433dfd45088bee2f9a17f309d817040ba19eb1affe55f6922830d6d87`
- Final programmed and final simple questionnaires from that archive; the rendered simple questionnaire was visually checked at the randomized item.

Verified design:

- The proposal defines participant-level random assignment to three advance-notice policies: no notice, 24 hours, or one week before two specified wildfire-related power outages.
- The fielded questionnaire implements the treatment at Q11 after common pre-treatment questions. It asks monthly willingness to pay to avoid the two outages.
- The outcome is an open numeric dollar amount, not binary or ordinal. Its utility direction is lower willingness-to-pay for avoiding the same outages, because lower valuation implies that the notification policy reduces the burden of those outages. This direction follows the stated decision construct, not observed responses.
- The three arms are a coherent policy action set for a utility or regulator choosing how much notice to provide.
- No treatment-varying image, chart, audio, video, interactive input, nested arm, factorial crossing, or randomized focal outcome order appears in the fielded Q11 treatment.

SocSci210 reconstruction check:

- SocSci210 condition 0 maps to no advance notice, condition 1 to 24 hours, and condition 2 to one week.
- SocSci task 0 preserves the three notification scenarios and requests an integer dollar response. It slightly standardizes the fielded wording but preserves the randomized content and common outage scenario.
- The source codebook independently encodes `77777` as don't know, `99998` as skipped on web, and `99999` as refused for Q11. The questionnaire and codebook declare a nonnegative numeric entry but no substantive upper bound.
- The source proposal's planned analysis targets differences in mean willingness to pay. The continuous contract therefore uses mean monthly WTP as the primary location estimand, median WTP as a mandatory robustness analysis, participant-within-arm bootstrap uncertainty, and raw USD/month effects and regret. Exact choice is primary; raw-dollar tolerances of 0, 5, 10, and 20 are frozen sensitivities, not a post-outcome threshold search.
- Because Q11 is uncapped, the task must not be entered into pooled normalized-regret summaries until a development-only scale is declared before canonical test reveal. No observed target minimum, maximum, quantile, or standard deviation may define that scale.

Implementation status: the continuous parser, source-missingness policy, mean and median estimators, treatment-effect error, recommendation, raw-dollar regret, participant-within-arm bootstrap, immutable artifact verification, and no-call replay are implemented and fixture-tested. `tcg8p` itself remains canonical-split unassigned, outcome sealed, and reveal unauthorized. No human response value or reported result was inspected.
