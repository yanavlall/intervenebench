# Cross-Family Regression Freeze v1

## Decision

InterveneBench will run one tightly scoped architecture-robustness study using
`mistralai/Mistral-Small-3.1-24B-Instruct-2503` as a genuinely independent
text-and-vision model family. The study replays exactly the 624 logical calls
used by the prospective confirmation panel's primary Qwen policy:

| Experiment | Calls |
|---|---:|
| `tcg8p` | 120 |
| `pb2rr` | 128 |
| `z358z` | 64 |
| `ShannonS2` | 192 |
| `Blair1131` | 24 |
| `KlarS44` | 96 |
| **Total** | **624** |

There are 312 base calls and 312 frozen alternate-format calls. No adaptive
reserve call is included.

## Why this is the next high-value experiment

The existing confirmation result could be specific to closely related Qwen
checkpoints. One independent architecture that handles both text and images
tests that failure mode without changing tasks, interventions, nuisance cells,
answer order, prompt perturbations, assets, or the primary response interface.
That makes the comparison paired and interpretable.

The candidate is the public, ungated, Apache-2.0 Mistral Small 3.1 24B model at
commit `68faf511d618ef198fef186659617cfd2eb8e33a`. The official repository
describes the checkpoint as a 24B text-and-vision model and recommends the
original Mistral format with vLLM. The freeze therefore selects only
`consolidated.safetensors` plus the required tokenizer/configuration files. It
explicitly excludes the duplicate ten-file Transformers shard set.

## Claim boundary

This is **retrospective cross-family robustness**. The six human experiments
were already revealed after the original prospective recommendations were
frozen. Therefore:

- it does not add an independent experiment;
- it does not increase the prospective confirmation sample size;
- it cannot validate a universal trust model; and
- it can test whether the observed low-regret decision result is robust to an
  independent simulator architecture.

The Mistral recommendation artifacts must still be frozen before the separate
regression scorer can read the existing aggregate human score. This preserves a
clean model-execution/scoring boundary even though the study is retrospective.

## Frozen interface

- Forced-choice tasks retain the exact source/reverse code mapping and use
  next-token logits only if every answer code is one exact distinct Mistral
  token. This gate must pass at 100% before target dispatch.
- `tcg8p` retains strict nonnegative-integer generation with temperature 0.7,
  top-p 0.9, and at most 32 new tokens. No clamping or semantic repair is
  allowed.
- The system instruction is static and hash-bound. The repository's dynamic
  current-date default system prompt is disabled.
- Invalid required model-task grids are marked unavailable. The original Qwen
  primary is never silently replaced.
- Automatic retries and reserve calls are zero. A missing transport call needs
  a separate byte-identical retry authorization.

## Regression gate

The comparison reuses `ModelVersionRegressionThresholds`:

- paired mean normalized-regret noninferiority margin: 0.01;
- worst-regret increase margin: 0.01;
- maximum exact-choice-rate drop: 0.10;
- maximum practical-reliability-rate drop: 0.10; and
- maximum schema-validity-rate drop: 0.01.

With only six experiments, results remain paired descriptive robustness
evidence. A pass does not authorize autonomous deployment or broaden the
current limited-research release decision.

## Compute boundary

The intended runtime is one Modal A100-80GB because the official Mistral card
reports about 55 GB of GPU memory for BF16/FP16 inference. The freeze caps
aggregate GPU time at 100,000 seconds. At Modal's recorded A100-80GB price of
$0.000694/second, that is $69.40; the hard incremental all-in cap is $90.
Spending is a ceiling, not a target.

## Current authority

Every authority bit is false. This freeze permits no model download, Modal
resource creation, inference, retry, reserve call, participant-row access,
human-outcome access, or regression scoring.

The zero-authority cache/materialization/canary package is now implemented and
documented in `cross_family_modal_preflight_v1.md`. The next permitted state
change is a separate image-materialization authorization, followed by separate
cache and three-call synthetic-canary authorizations. Only after those pass and
the one-token masked-logprob interface is explicitly adjudicated may a distinct
624-call inference authorization be considered.

## Frozen artifacts

- `data/manifests/research/cross_family_regression_protocol_v1.json`
- `data/manifests/simulators/mistral_small_3_1_24b_source_manifest_v1.json`
- `data/manifests/simulators/cross_family_call_plan_v1.json`
- `src/intervenebench/cross_family_regression.py`
- `tests/test_cross_family_regression.py`

Verify without model or outcome access:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_cross_family_call_plan.py
```
