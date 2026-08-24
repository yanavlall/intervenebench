# Modal discovery preflight freeze v1

Status: frozen locally, non-executing, zero authority, zero spend.

This freeze turns the earlier broad simulator budget into a safe first launch unit. It does not install or authenticate Modal, download weights, invoke a model, inspect a human outcome, or authorize any prospective experiment.

## First future execution

The first remotely executed stage is limited to 40 parser-preflight calls: 10 calls for each of four models over the five development-only ordinal experiments. For every model, the plan uses the first and last source-order arm from each experiment. This gives two prompts per experiment and covers 4-, 5-, and 8-option response schemas plus a fielded message-variant task.

The four exact public, ungated Apache-2.0 checkpoints are:

- `Qwen/Qwen3-8B` at commit `b968826d9c46dd6066d109eabc6255188de91218`;
- `Qwen/Qwen3-14B` at commit `40c069824f4251a91eefaf281ebe4c544efd3e18`;
- `Qwen/Qwen2.5-14B-Instruct` at commit `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`; and
- `socratesft/socrates-qwen2.5-14b-sft` at commit `6666d399b373dd37a2691a921550732f2fdddb20`.

Model repository metadata, configuration, tokenizer, chat-template, and weight-manifest hashes are recorded in the freeze. Qwen3 thinking is disabled. Socrates is explicitly marked training-exposed on `xc4yq` and `de5hx`, so those rows may test plumbing but cannot support a primary clean specialist comparison.

## Runtime and ceiling

The frozen runtime is Python 3.11 with exact Modal, Torch, Transformers, Accelerate, and Safetensors versions on one Nvidia L40S per model container. No GPU fallback is permitted. Modal-built images expose a provider image ID rather than a documented OCI digest, so a future execution authorization must bind that Modal image ID together with the image-recipe and dependency-lock hashes. CUDA and GPU details must also be attested before the first model call.

At the official Modal L40S rate checked on 2026-08-13 (`$0.000542` per second), the preflight allows at most 7,200 aggregate GPU-container seconds, or `$3.9024` in GPU time. The hard all-in ceiling is `$5.00`. The run must stop before dispatch if another call could breach a call, time, attempt, or cost limit.

## Pass and stop rules

- Each model must produce 10 of 10 strictly parseable outputs to advance.
- A malformed response is retained as a failure and is not repaired or semantically reprompted.
- At most one byte-identical transport retry is allowed per model.
- Any runtime-attestation mismatch, unlisted experiment, outcome-bearing field, duplicate call ID, artifact overwrite, or budget overrun stops the stage.
- Passing this preflight does not automatically launch the 240-call aggregate screen. That requires a separate authorization.

## Authentication boundary

Modal authentication happens only after this local package is verified. Credentials remain in Modal's local profile or another untracked credential store; no token or credential file belongs in the repository. Execution uses three fail-closed authority stages: (1) hydrate the exact locked inference image without model downloads or GPU calls, (2) bind the resulting Modal image ID and cache the four exact public checkpoint revisions while GPU inference remains denied, then (3) bind the image ID and the four cache-attestation hashes before the exact 40-call parser preflight. None of those authorities permits prospective-task inference, outcome access, fine-tuning, or the larger simulator run.

The offline check is:

```bash
intervenebench verify-modal-preflight-freeze
```

The source of truth is `configs/simulators/modal_discovery_preflight_v1.json`; the explicit 40-call plan is `data/manifests/simulators/modal_preflight_call_plan_v1.json`.

## Execution disposition (2026-08-13)

The original execution reached all four model groups but closed before accepting any output because a PyTorch version object could not be deserialized by the deliberately lightweight local environment. That transport failure is frozen in the v1 failure manifest and its returned content was unavailable and unused.

A clean v2 retained the exact tasks, calls, prompts, seeds, model revisions, and generation settings, changed only the transport to plain JSON, and reduced its remaining compute ceiling so the combined attempts stayed within the original `$5.00` cap. V2 reached strict local parsing and failed the predeclared 10/10 gate because at least one response did not contain exactly the required `probabilities` object. No malformed response was repaired, extracted, or reprompted; no call output was accepted into scientific artifacts; and the larger 240-call screen was not authorized or launched. The active source of truth for the closed v2 attempt is `configs/simulators/modal_discovery_preflight_v2.json`.

## Constrained canary disposition (2026-08-13)

After the two preflight failures, a separate four-call canary tested whether grammar-constrained decoding could establish a reliable response interface before any larger spend. The canary used one identical blinded `5vm8g` prompt for each frozen model, reused the four hash-verified cached checkpoints, required positive integer relative weights for answer values 1–5, normalized those weights deterministically, prohibited retries and repair, capped incremental spend at `$1.25`, and denied sealed-task inference, outcome access, fine-tuning, and automatic continuation.

The first image-materialization attempt failed before image creation or inference because Modal requires a minimum two-second scale-down window. That failure was recorded, the single infrastructure value was corrected from one to two seconds, the freeze was re-hashed, and all 240 tests passed before dispatch. The corrected image then materialized with zero model calls.

All four canary attempts were dispatched. The canary failed and stopped when remote strict JSON validation raised `JSONDecodeError: Expecting ',' delimiter` for a constrained output. Two response identities had returned to the wrapper before the exception surfaced, but no response was accepted or serialized as a scientific output. Because the malformed raw string was rejected inside the remote boundary, the artifact does not distinguish backend constraint failure from generation truncation; no causal diagnosis should be claimed without a new, separately frozen test. The failure is recorded at `artifacts/modal_constrained_canary/constrained_canary_20260813_v1/failure_manifest.json`. The 240-call screen remains unauthorized and unlaunched.

## Parser-free forced-choice disposition (2026-08-13)

The next separately frozen canary removed text generation entirely. Each model received the same blinded `5vm8g` prompt and one source arm. The response choices were mapped to codes A–E. Before inference, every tokenizer had to prove that each code appended to the fully rendered chat prompt as exactly one distinct token and decoded back to that exact code. The simulator distribution was then defined as the temperature-1 softmax of the model's final-position logits restricted to those five token IDs. No output text, structured decoder, parser, semantic retry, repair, or sampled generation was involved.

The freeze permitted exactly four single-forward-pass attempts, reused only the existing hash-verified checkpoint cache, capped incremental spend at `$0.90`, denied outcome access and sealed-task inference, and prohibited automatic continuation. The canary passed 4/4. All four tokenizers mapped A–E to the same five distinct token IDs (32–36), all returned distributions were finite and normalized, and the hash-bound run completed in 183.96 wall-clock seconds. The final manifest is `artifacts/modal_forced_choice/forced_choice_canary_20260813_v1/final_manifest.json`.

This establishes a reliable machine interface, not simulator validity. The four distributions differ sharply in entropy and modal choice, which is a useful future disagreement diagnostic but cannot be interpreted as accuracy without human outcomes. The larger simulator screen remains unauthorized and unlaunched.

## Forty-call parser-free screen disposition (2026-08-13)

The passed forced-choice interface was generalized from A–E to deterministic A–H prefixes and applied to the first and last source-order arms of the five blinded development experiments. Four model workers ran concurrently; each loaded its checkpoint once and completed ten sequential forward passes. The screen was independently frozen at exactly 40 attempts, prohibited generation, retry, repair, human-outcome access, prospective-task inference, and automatic continuation, and carried a `$1.75` incremental authorization ceiling.

The run passed 40/40 in 82.74 wall-clock seconds. All output envelopes, tokenizer contracts, probability normalization, runtime identities, cache attestations, prompt hashes, and the final manifest verified. An outcome-blind discovery artifact then computed only predeclared-style synthetic diagnostics. Across the ten arm prompts, the four models had one unanimous modal response and two prompts where all four modal responses differed. Their screened-pair choice agreed unanimously for three of five experiments. These are disagreement and confidence observations, not accuracy results. Multi-arm experiments remain incomplete because the screen intentionally included only two arms, so no full intervention recommendation or regret may be reported from this artifact.

The frozen report is `docs/reports/forced_choice_discovery_screen.md`; the run manifest is `artifacts/forced_choice_screen/discovery_screen_20260813_v1/final_manifest.json`; and the outcome-blind diagnostic artifact is `artifacts/forced_choice_screen/discovery_screen_20260813_v1/outcome_blind_diagnostics.json`.

## Answer-order canary disposition (2026-08-13)

Before expanding to all arms, a separate outcome-blind canary tested whether the parser-free distribution depended materially on response-option order. It paired the completed 40 source-order calls with exactly 40 full-reverse calls, inverse-mapped every reverse distribution to the original response values, reused the verified checkpoint cache, prohibited generation, retry, repair, human-outcome access, and automatic continuation, and carried a `$1.75` incremental authorization ceiling.

The reverse run completed 40/40 in 234.93 wall-clock seconds. The comparison failed every prespecified robustness threshold: median total variation was `0.263` against a maximum of `0.10`; nearest-rank p90 total variation was `0.954` against `0.25`; modal-response stability was `0.400` against a minimum of `0.75`; and screened-pair choice stability was `0.750` against `0.80`.

The frozen failure pivot is active: do not scale the single-order method. The next full-action-set estimator must balance source and reversed answer orders after inverse mapping. This is a method-robustness result, not a human-accuracy result, and no additional job was launched automatically. The report is `docs/reports/answer_order_canary.md`; the completion manifest is `artifacts/answer_order_canary/answer_order_canary_20260813_v1/final_manifest.json`; and the paired diagnostic artifact is `artifacts/answer_order_canary/answer_order_canary_20260813_v1/paired_robustness_diagnostics.json`.

## Balanced full-action freeze disposition (2026-08-13)

The prescribed pivot was implemented without further model calls. Each accepted source and reverse distribution is inverse-mapped to source values, renormalized, and averaged with fixed equal weights. Replaying the 40 existing pairs produced 40 balanced arm distributions and 20 screened-pair choices, all bound to the 80 component-output hashes. This operation is algebraically invariant to exchanging source and reverse order; it does not claim invariance to arbitrary permutations or accuracy against humans.

The complete five-experiment action matrix contains 17 arms, four models, and two orders, for 136 logical ordered calls and 68 balanced arm predictions. Eighty verified calls are reusable. The zero-authority completion freeze schedules only 56 missing calls, exactly 14 per model, with a `$1.75` future incremental ceiling and no retry, repair, outcome access, or automatic continuation. It would yield 20 full-action recommendations after completion. No new inference was executed under this freeze.

The method and call accounting are reported in `docs/reports/balanced_full_action_freeze.md`; the plan is `data/manifests/simulators/balanced_full_action_plan_v1.json`; and the freeze is `configs/simulators/balanced_full_action_v1.json`.

## Balanced full-action execution disposition (2026-08-13)

After all local tests passed, a separately hash-bound image authorization materialized Modal image `im-eLaRTHMO9bCNpaVEoteCIs` with zero inference. A second authorization then permitted only the 56 missing calls under the frozen `$1.75` ceiling. Four cached model groups completed 14 calls each, yielding 56/56 valid outputs in 67.53 wall-clock seconds with no retry, repair, model download, or human-outcome access.

The deterministic combiner verified and joined those 56 outputs with the 80 pre-existing component hashes. It produced 68 balanced arm distributions and 20 full-action recommendations across the five experiments and four models. The models were unanimous on `5vm8g` and agreed three-to-one on `xc4yq`, `de5hx`, `turagaS11`, and `wallaceS12`. These are outcome-blind disagreement observations, not human accuracy results. The run stopped after completion and authorized no scoring, reveal, threshold selection, or later stage.

The completion manifest is `artifacts/balanced_full_action/balanced_full_action_20260813_v1/final_manifest.json`; the recommendation artifact is `artifacts/balanced_full_action/balanced_full_action_20260813_v1/full_action_recommendations.json`.
