# Prospective multimodal development freeze v3

**Decision date:** 2026-08-13  
**Status:** locally frozen and tested; remote materialization blocked pending explicit asset-egress approval; target outcomes sealed

## Scope

The next prospective-development stage contains three independent source-audited image experiments:

| Experiment | Arms | Exact source asset type |
|---|---:|---|
| `nj5dx` | 2 | class-inequality infographics |
| `es4xw` | 4 | candidate management-team images |
| `e2pyb` | 3 | racial-disparity infographics |

There are nine total arms. Every model scores every arm in source and reverse response-option order.

## Frozen simulator panel

- **Primary:** `Qwen/Qwen3-VL-8B-Instruct` at commit `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` with the exact PNG attached.
- **Vision comparator:** `Qwen/Qwen2.5-VL-7B-Instruct` at commit `cc594898137f460bfe9f0759e9844b3ce807cfb5` with the exact PNG attached.
- **Modality ablation:** `Qwen/Qwen3-8B` at commit `b968826d9c46dd6066d109eabc6255188de91218`, receiving only the frozen accessible description.

The modality ablation is the model selected on the five revealed text-only discovery tasks. It is not allowed to masquerade as a vision model. For `es4xw`, meaningless picture numbers are removed from the text ablation so they cannot create fake arm information.

Socrates is excluded from this prospective primary panel because it is text-only and its released participant mapping marks all three experiments as training-seen.

## Exact call arithmetic

- 9 arms × 2 option orders × 2 VLMs = 36 exact-image calls.
- 9 arms × 2 option orders × 1 text ablation = 18 text calls.
- **Total: 54 deterministic next-token-softmax calls.**

There is no free-text generation, repair, semantic retry, or automatic continuation. The estimator inverse-maps both answer orders to the source values and averages the two distributions equally before choosing the arm with maximum expected normalized utility.

## Outcome-free diagnostics frozen before reveal

Primary diagnostics are the Qwen3-VL winner margin, source/reverse choice stability, mean arm-level source/reverse total variation, two-VLM complete-action agreement, and vision-versus-accessible-text choice agreement. Response entropy and per-arm VLM utility dispersion are secondary. Directions are frozen in `configs/simulators/prospective_multimodal_v3.json`.

Three experiments cannot support classifier calibration. The prospective-development analysis is limited to continuous regret ranking, exact/practical reliability, and fixed risk-coverage descriptions. It cannot be called a canonical trust-model test.

## Runtime and budget

The runtime is pinned to Python 3.11, Torch 2.9.1, Transformers 4.57.6, Pillow 11.3.0, and one NVIDIA L40S per model group. The two public VLM revisions require approximately 34 GB of checkpoint storage. The text checkpoint is reused from the existing verified cache. The frozen execution permits at most 54 attempts, three model loads, 3,600 aggregate GPU-seconds, zero automatic retries, and a hard `$5.00` incremental ceiling.

## Egress boundary

The Modal image must contain the inference code, dependency lock, model-file manifest, and nine exact fielded PNG assets (about 4 MB total). A remote materialization attempt was rejected locally before upload because this asset egress requires explicit user approval. No image, stimulus, checkpoint, or human outcome was transmitted, and no paid inference ran.

Human outcomes for all three experiments remain sealed. Even after inference succeeds, a separate reveal authorization is required; the inference authorization cannot open outcomes.

## Machine-readable artifacts

- Call plan: `data/manifests/simulators/prospective_multimodal_plan_v1.json`
- Model-file manifest: `data/manifests/simulators/multimodal_model_file_manifests_v1.json`
- Zero-authority freeze: `configs/simulators/prospective_multimodal_v3.json`
- Dependency lock: `infra/modal/multimodal-requirements.lock`
- Modal worker: `infra/modal/prospective_multimodal_app.py`
- Authority wrapper: `scripts/run_prospective_multimodal.py`
