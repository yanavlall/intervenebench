# Phase 1 Source-Audit Log

This log records outcome-blind reconstruction checks for the strict SocSci210 candidates in the independently curated overlap. It contains design facts only. Human response values, treatment-effect estimates, significance tests, winners, and regret are forbidden here.

The machine-readable status is in `data/manifests/audits/phase1_candidate_registry.csv`. A candidate is eligible for the Phase 1 smoke test only after its original materials verify every locked eligibility field.

## `5vm8g`: eligible for Phase 1

Source title: *Beliefs about Racial Discrimination*.

Files inspected:

- Proposal PDF SHA-256: `e9ea3f47447f5fca60d097ab8640af46ed04777c26dcec03c46715731f4bd1f0`
- Materials archive SHA-256: `7ce5df7c7066c60664b088ad879a4804a878f417f3ce307030645d2330afb179`
- Final questionnaire SHA-256: `45e90954d9e1564abc4eb44c2faaf9a2a1da598f841bad252728550091224ed6`
- Field report SHA-256: `15b83c5232b3f499073ceb57ebf447a9a9e52c62de2a0b863a62cb131bb7befe`
- Codebook SHA-256: `9dfa1244ce6d2986028c56726c232d8878745969fdc1a8b86a33832eaed36d5a`

Verified design:

- Two-arm, between-subject assignment at the participant level (`GROUP=1` treatment, `GROUP=2` control in the source codebook).
- Both groups first report incentivized beliefs about callback discrimination.
- The treatment group then sees a fixed factual message: black-sounding names required 15 applications per callback versus 10 for white-sounding names, described as a 50% difference. The final fielded questionnaire does not personalize this message using the participant's elicited belief.
- The control group does not see that factual message.
- The final questionnaire and field report identify no randomized question-order variable.
- The post-treatment outcomes are text-only binary or ordinal items, followed by a repeated donation-choice task.

SocSci210 reconstruction check:

- `condition_num=0` is control and `condition_num=1` is treatment. Only condition 1 contains source Q8, the treatment-only interpretation check, which independently verifies this mapping without response values.
- SocSci210 task 2 maps to source Q2: perceived seriousness of labor-market discrimination on a five-point scale from `1 = A very serious problem` to `5 = Not a problem at all`.
- The independently curated outcome registry includes this perceived-discrimination outcome.
- Applying the frozen outcome-selection hierarchy chooses source Q2 / SocSci210 task 2: it is the earliest supported post-treatment outcome, has verified bounds, and measures the stated proximal target. Utility is `6 - raw_response`, so greater utility means greater recognition of discrimination as serious.
- The common-task `stimuli` released by SocSci210 are identical across the two conditions and omit the treatment message. Using those strings directly would force a no-treatment prompt and invalidate the synthetic contrast. The Phase 1 bundle must instead reconstruct and hash the exact treatment and control context from the final questionnaire.

Leakage safeguard: the proposal includes results from a separate pilot. Those result-bearing pages are prohibited prompt, diagnostic, feature-engineering, and model-selection context. No SocSci210 human response value was inspected during this audit.

Phase 1 decision: eligible. The admissible actions are showing the verified factual information or showing no additional information, with source Q2 / SocSci210 task 2 as the predeclared primary outcome.

## `9263n`: excluded from Phase 1; retain for later factorial evaluation

Source title: *To Do, to Have, or to Share? Valuing Experiences and Material Possessions by Involving Others*.

Files inspected:

- Proposal PDF SHA-256: `efa403f68bb54fafaf654be2064139219f75453343a525bb0f6811816f78311f`
- Materials archive SHA-256: `199e45abea34ed02dc118e8ce0847f3b4f91b404ff2ead8adc2e1f55e77a5edb`
- Final questionnaire SHA-256: `4379006209d80be53e588c3dab3a67446f4c516b4c9c2f6bcbe6c9611d1dc8d6`

Verified design:

- Four-cell, between-subject assignment at the participant level.
- One factor asks participants to recall an experiential purchase versus a material possession.
- A second factor asks participants to recall a shared purchase versus a solitary purchase.
- The source questionnaire encodes the resulting cells as social experience, solitary experience, social material possession, and solitary material possession.
- Primary outcomes include happiness and relatedness ratings on verified ordinal scales.

SocSci210 reconstruction check:

| `condition_num` | Purchase type | Social context |
|---:|---|---|
| 0 | experience | shared |
| 1 | experience | solitary |
| 2 | material possession | shared |
| 3 | material possession | solitary |

The four-level condition field is a factorial-cell code. Comparing the four cells as if they were four levels of one intervention would discard the source estimands and overstate Phase 1's clean-design coverage.

Phase 1 decision: excluded with reason `factorial_design`. Retain for a later factorial estimator that explicitly models the two main effects and their interaction. No human response value was inspected during this audit.

## `bsd7j`: excluded from Phase 1; retain for multimodal evaluation

Source title: *The Taxpayer Gap*.

Files inspected:

- Proposal PDF SHA-256: `8f5130a3fd313d447f1867957fb3e6913443ea6395b743db1fd69914fba53078`
- Final questionnaire SHA-256: `e6d85649d82e65d01e09a0af2ed503f492a562aa6290c060e364c2e91a589e5c`
- Field report SHA-256: `78a35e3021a8489c3f00af0c6a94e13f26850b683609336264f9901d27af2fcc`
- Codebook SHA-256: `ec938e676653699d17fc10fda1b43ab83d4ecf0bd4205b4fcda38d2aa7afa9c2`

Verified design:

- Three-arm, between-subject assignment at the participant level, optionally blocked by party identification.
- The federal-income-tax arm presents a chart and text emphasizing that many low-income households have negative federal individual income-tax rates.
- The all-taxes arm presents a different chart and text emphasizing that almost all households pay some combination of federal, state, and local taxes.
- The active control presents an unrelated Internet-usage chart and comprehension item.
- Post-treatment outcomes include beliefs about who pays taxes and policy-spending preferences.

SocSci210 maps `condition_num=0` to the Internet control, `condition_num=1` to federal-income-tax information, and `condition_num=2` to all-taxes information. Its `stimuli` strings summarize the charts in prose, but the final fielded interventions require the charts themselves.

Phase 1 decision: excluded with reason `non_text_treatment`. Replacing the condition-specific charts with prose would change the intervention rather than faithfully reconstruct it. Retain the study for a later multimodal benchmark. No human response value was inspected during this audit.

## `fxcn4`: excluded from Phase 1; retain for factorial and multimodal evaluation

Source title: *Do Victims' Race and Gender Identity Interact to Predict the Perceived Credibility of Sexual Harassment Claims?*

Files inspected:

- Proposal PDF SHA-256: `f31be9ba91870baaa62dcc164972ba2041909e666c67852ec99ef6f85e32ff8f`
- Materials archive SHA-256: `1553de0ef3e242cfc27190d377447e05830548a963d7d2172161ac995fee31ae`
- Final simplified questionnaire SHA-256: `71e35aa90b1103c9b5feed10514559e26fe614ffd155fb9db74010ee99a205b5`
- Final programming questionnaire SHA-256: `531642411094ff79c5f9907eb3a1370d0fa9aed6f57fc13be003a2d796cde90c`
- Codebook SHA-256: `b4fedc3fbe3b9bffd20b2d9d8e6ecf3126eb63456ffa7ae0614b89fb9290d31f`

Verified design:

- Four-cell, between-subject assignment at the participant level.
- Claimant race is randomized as Black versus White.
- Claimant gender identity is independently randomized as cisgender versus transgender.
- The four incident reports are presented as condition-specific images.
- The order of the claim-believability, claim-credibility, and claim-truthfulness items is randomized.

The source therefore has three separate Phase 1 incompatibilities: a `2 x 2` factorial treatment, required image presentation, and randomized focal-outcome order. Any one of these is sufficient for exclusion from the clean one-factor text-only smoke path.

Phase 1 decision: excluded with reason `factorial_order_and_modality`. Retain for later estimators that model the factorial assignment, order, and image inputs explicitly. No human response value was inspected during this audit.

## `jf46x`: eligible for Phase 1

Source title: *Smallpox Vaccine Recommendations: Is Trust a Shot in the Arm?*

Files inspected:

- Proposal PDF SHA-256: `9f01ba3dfca5204551e20fc7a6b33ae9d13467d424fa8b5e735d596e42369e8e`
- Materials archive SHA-256: `67bca81538cec97f824c8aa8151dc5a68f1d80b947471d99da14b0e564fb6032`
- Final questionnaire SHA-256: `40a96a1acbcb8a6197e07b979b7c6729b067ea90207674fad1daa5d71691c502`

Verified design:

- Two-arm, post-test-only, between-subject assignment at the participant level.
- Both arms present the same fictional smallpox outbreak, vaccine facts, and vaccination recommendation.
- The shared-values arm emphasizes careful, compassionate, open, and honest decision-making.
- The past-performance arm emphasizes expertise and previously successful outbreak-control approaches.
- Both messages are text-only communication strategies and are admissible intervention actions.
- The source identifies vaccine cooperation as the primary dependent variable; source Q2 asks whether the participant would get the recommended vaccine (`Yes` or `No`). No randomized outcome order is specified.

SocSci210 reconstruction check:

- `condition_num=0` maps to the shared-values or trust message.
- `condition_num=1` maps to the past-performance or confidence message.
- SocSci210 task 0 maps to source Q2, the binary vaccine-intention outcome. Task 0 has 260 participants in condition 0 and 259 in condition 1.
- Applying the frozen outcome-selection hierarchy chooses source Q2 / SocSci210 task 0. Utility is `1` for `Yes` and `0` for `No`.
- The released SocSci210 `stimuli` refer to the CDC, while the final fielded questionnaire refers to the Department of Homeland Security. This is a material reconstruction discrepancy. The benchmark must use a hashed transcription of the final DHS questionnaire and must not pass the released CDC wording to simulators.

Phase 1 decision: eligible. The decision is which verified public-health message framing to deploy, with vaccination intention as the predeclared primary outcome. No human response value was inspected during this audit.

## `xc4yq`: eligible for Phase 1

Source title: *Individual Differences in Responses to Terrorist Threat Messages*.

Files inspected:

- Proposal PDF SHA-256: `6ddf11b27494a9cecdb674ded8a38c4c59d0f154986f176b8c751cc8c7c019d5`
- Materials archive SHA-256: `39ff9475a90573039d37f49a48f0a581584af3d87f439cdbb70b39e2e1612f51`
- Final questionnaire SHA-256: `b7467979e24199e1d9e19373c6a8d1aae981e524ab62ea767a2b07f72ae8f5d2`

Verified design:

- Three-arm, between-subject assignment at the participant level.
- The high-fear arm describes specific modes and consequences of terrorist attack before recommending preparedness actions.
- The plain low-fear arm gives the preparedness recommendations without the high-fear lead-in.
- The low-fear positive-protection arm adds an efficacy-oriented example of a family that prepared successfully.
- All three interventions are text-only messages. No randomized outcome order is specified.
- The study distinguishes appropriate protective actions from overreactions such as avoiding air travel or public gatherings.

SocSci210 maps `condition_num=0` to plain low fear, `condition_num=1` to high fear, and `condition_num=2` to positive protection. The released strings abbreviate the messages with ellipses, so source-faithful prompts must be reconstructed from the final questionnaire.

The primary outcome-selection hierarchy excludes fear and manipulation-check items, the explicitly overreactive actions, and the vague suspicious-person reporting item whose utility is not defensibly monotone. The first remaining clean target action is making a home emergency kit: source behavior item 3 / SocSci210 task 7. Its scale runs from `1 = Very unlikely` through `7 = Very likely`, with `8 = I already do`; higher values indicate greater preparedness. Each arm has more than 180 task-7 observations.

Phase 1 decision: eligible. The decision is which of three verified preparedness messages to deploy, with home emergency-kit readiness as the predeclared primary outcome. No human response value was inspected during this audit.

## `j6xgs`: excluded from Phase 1

Source title: *Introducing a Novel Framework for Understanding the Relationships Between Busyness, Idleness, and Happiness*.

Files inspected:

- Proposal PDF SHA-256: `704d34064784d0ae6343fbebc7d0e13001d3bbbd520d7ca06801ec02b8d1e229`
- Materials archive SHA-256: `8b7b8e75c89f0fd5219b3f987f8096a1fe948a95945478f891d99fffa91a57f7`

Verified design:

- Two-cell participant-randomized study.
- Participants reflect on either a highly busy or a minimally busy time.
- Outcomes include subjective state items and happiness or fulfillment ratings.
- The proposal's central hypotheses are moderated by age and future-time perspective.

Phase 1 decision: excluded with reason `action_set_ambiguous`. The reflection prompt is an elicitation or framing manipulation, not yet a defensible pair of real-world intervention actions for the benchmark's decision-regret claim. This exclusion is unrelated to observed effects.

## `v6nhw`: excluded from Phase 1; retain for later factorial evaluation

Source title: *Burden Sharing and Collective Action: A Study of Opinion on Opioid Treatment Funding*.

Files inspected:

- Proposal PDF SHA-256: `0218a77fafa5047e84fdfe4d95fc7931d82cba1cde9cef35f64a71e0f050beca`
- Materials archive SHA-256: `5127cada40da44bb4b737ce5c96b7dbaf9adeb0da090f5525d35cb00abdaf305`
- Field report SHA-256: `032f2d855b3482bbfa43f9f38a7361c5a8a01b15a745b7d739b24fb02f92ffe4`
- Final questionnaire SHA-256: `906aad6993f6575b6092e6be0c775d547a025538a08235d2dec8ec5b6e462654`
- Codebook SHA-256: `66ea52180c2be0cc8756041f4bb509b51236e0582c7a5eb0eda7e088ed35d7c0`

Verified design:

- `P_BASED` randomizes Q1 between a needs-based funding rule and a resource-based funding rule.
- `P_DISTANCE` independently randomizes Q2 between a clinic one-quarter mile away and a clinic two miles away.
- `RND_01` independently randomizes whether Q1 or Q2 appears first.
- ZIP-area overdose status and respondent income are pre-treatment inserts rather than intervention assignments.
- Both outcomes use a five-point support-to-oppose scale where lower raw values mean greater support.

SocSci210 reconstruction check:

| `condition_num` | Q1 funding factor | Q2 distance factor |
|---:|---|---|
| 0 | needs-based | one-quarter mile |
| 1 | needs-based | two miles |
| 2 | resource-based | one-quarter mile |
| 3 | resource-based | two miles |

The released four-level condition field is therefore a factorial-cell code, not a four-arm action set. Treating the cells as four competing policies would be a methodological error.

Phase 1 decision: excluded with reason `factorial_and_order_complication`. The study can support prespecified marginal contrasts in a later estimator that explicitly handles factorial assignment and randomized order, but it violates the locked Phase 1 requirement for a clean one-factor task without order complications. This exclusion is unrelated to observed effects.

## `ncs7k`: excluded from Phase 1

Files inspected: proposal PDF (`beec90d7815d09ddc5bdfc4262972ef1cac52f744ec057353d155bd4ce399384`), materials archive (`741902f3c4db43aa6d17b018f289d595a32844c51436551356d204d4d699042a`), and final simple questionnaire (`69617c1f7d60161b527f3ec1db001f0c297fea66cbdfbe9cd8ccdd9bb0fe170f`).

The source independently randomizes group identity (Antifa versus Proud Boys) and cancel-culture framing, producing a `2 x 2` factorial design. It also randomizes the order of the focal outcome items. Phase 1 decision: excluded with reason `factorial_and_order_complication`. No human response value was inspected.

## `nhgxf`: excluded from Phase 1

Files inspected: proposal PDF (`7331608f545673607ab8ef51c106f42fdd137bdfa6d32a1c3157e206119cf78e`), materials archive (`fbb6b01ef0d74da74b08d4a705f096fc870fd3ff0088e6c53def44f95c54b6ff`), and final questionnaire (`5cbcdf92f29dd6e948d4ae29ba0bb64dc7a00fc992c78c86ad6e7fc11ecec3c8`).

The three arms use anger, fear, and relaxation autobiographical-writing inductions. The paired candidate-evaluation outcome block is presented in randomized order. Phase 1 decision: excluded with reason `outcome_order_complication`; retain for a later order-aware analysis. No human response value was inspected.

## `nk9jd`: excluded from Phase 1

Files inspected: proposal PDF (`cf565fc777d361bea9727a4d080d6b95f82529eb2fc956e51a6895015b90124a`), materials archive (`6991311308006593b880efa2c2b865ebb24c6ac657b0fd00eb44d972c1969020`), and final questionnaire (`6909e4fd096fc468ba53120ecf45b5a449d121aff1c6e20fd9dd7c4746a07ff5`).

The fielded source contains four arms: national-identity affirmation, partisan self-distancing, self-affirmation, and a vacation-writing control. The control contains nested random assignment, and the Democratic/Republican thermometer order is randomized. SocSci210 exposes only two condition labels for this study, so the original action set cannot be reconstructed faithfully from the released condition field. Phase 1 decision: excluded with reason `incomplete_arm_reconstruction_and_order`. No human response value was inspected.

## `vemrp`: excluded from Phase 1

Files inspected: proposal PDF (`44d9b878adeffdc43d40b3412eb3f0061c269b01f6811bd6e1bada54c035336c`), materials archive (`9b9939c0d89b70872436126ed6ba1037714e49d1e7db8dfa8b51b7845c3d254e`), and final questionnaire (`6c61727e341939601b0c9ad84aa30819043c4ca5c439865fb7a69ceeae0a63ab`).

The source crosses target income (high versus low) with environmental behavior (green versus not green) in a `2 x 2` design and randomizes the order of the two focal judgments. Phase 1 decision: excluded with reason `factorial_and_order_complication`. No human response value was inspected.

## `vz5r4`: excluded from Phase 1

Files inspected: proposal PDF (`13f2d5291172e0e716252d6467d9ee07f216ee428621e1dde9327f6bbdb39795`), materials archive (`847c14a5b4e09f6c4d84465445fee1191c0feb48ef2b5fc83147c47d47da070b`), and final questionnaire (`71d4f54dfe6f3b801008f71f0140fa2c3b9c142b2fe6aea30be233535f065d0a`).

The four conditions ask participants to complete different multi-step elicitation tasks about traits, issues, social groups, or a control topic. Party presentation and the focal outcome order are randomized. Phase 1 decision: excluded with reason `complex_multistage_and_order`. No human response value was inspected.

## `xfmrn`: excluded from Phase 1

Files inspected: proposal PDF (`6fef82102a1e163156bc361ae14cd3996ef72ee47570df23046cf3f296974fc1`), materials archive (`6f1d56240dcde7be7ddb346f31332fb83a441360d5e9f2e89aa86af897509384`), and final questionnaire (`d2676414e8b711bc69c8eb54cee1fb65b3085374e46493cd73f91f062fd5b93d`).

The four arms are control, published-response reminder, private-response reminder, and imagined-friends-reading reminder. The Democratic/Republican rating pair and the Democratic/Republican trust pair each have independently randomized order. Phase 1 decision: excluded with reason `outcome_order_complication`. No human response value was inspected.

## `xy8jw`: excluded from Phase 1

Files inspected: proposal PDF (`6ec76a0ae69d5a667d4327bace3f31643e4f3c0c1babc02d14f10773f474b9bd`) and final questionnaire (`14ef19d66b36cc79fd7353a94ad487dbcbe424923f47ef4b467d02a9a058d045`).

The anger and fear treatments require photographs of women's faces; the control has no corresponding image. Candidate-evaluation items are also randomized. Phase 1 decision: excluded with reason `non_text_treatment_and_order`; a prose substitute would change the fielded intervention. No human response value was inspected.

## `y9nb7`: excluded from Phase 1

Files inspected: proposal PDF (`7782fd00954d0ed14232d9f034b37b99b950fab2ed419d386b2bf5e8102c56ac`), materials archive (`23625b097e01df3bf55d27e2a2fc4dcc3e90aadc5b15483a38b58dee58866633`), and final simple questionnaire (`6295643c8d7d7828856524bbf204cdc7a410d5f43fae9d9f7681738df8c8d9c1`).

The four conditions are no story, a nonpartisan women-candidate story, a Democratic women-candidate story, and a Republican women-candidate story. Treatment arms include three candidate photographs, which the SocSci210 text reconstruction only summarizes, and the focal candidate grid is randomized. Phase 1 decision: excluded with reason `non_text_treatment_and_order`. No human response value was inspected.

## `yg958`: excluded from Phase 1; retain for factorial evaluation

Files inspected: proposal PDF (`a2349298183c868b26f221d2baf7e10f3bb898b1e36fd783057426fb19102035`), materials archive (`4d7f1888f17be2828f1cb23861f0e5aed277d99832d4350489389c7207850f3d`), and final simple questionnaire (`c8d3ab77b43f59358bd4e12f212e097f1d30294eb9a4e54a88bfd40516dde827`).

The four arms combine descriptive they/them-pronoun information and an injunctive dignity norm in a `2 x 2`-like structure. Personal-pronoun selection is bundled into the descriptive-information arms but absent from the other arms, so it is also an interactive post-message component confounded with that factor. Although the independently curated registry verifies the study and its pronoun outcomes, this is not a clean one-factor text-only task. Phase 1 decision: excluded with reason `factorial_and_interactive_treatment`. No human response value was inspected.

## Qualification result

All 17 strict curated-overlap candidates have now received a source-level, outcome-blind audit. Three survive the locked Phase 1 gate: `5vm8g`, `jf46x`, and `xc4yq`. Fourteen are excluded for design, modality, order, reconstruction, or action-set reasons—not because of their observed effects. This three-task audit stratum is sufficient to exercise a train/validation/test protocol smoke test, but it is not large enough for scientific trust-model claims. The broader SocSci210 registry must be audited before the full benchmark proceeds.
