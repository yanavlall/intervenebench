# Prospective multimodal development freeze v4

**Decision date:** 2026-08-13  
**Status:** completed 54/54; recommendations and diagnostics aggregated; target outcomes remain sealed

The v3 execution stopped before returning any model group because Transformers'
`AutoVideoProcessor` requires Torchvision even for the frozen still-image Qwen-VL
path. The create-only failure artifact is preserved at
`artifacts/prospective_multimodal/prospective_multimodal_20260813_v3/failure_manifest.json`.

Version 4 changes only the runtime package and its resulting integrity bindings:

- add `torchvision==0.24.1`, the release paired with `torch==2.9.1`;
- include the authorizer script in the frozen implementation hashes;
- require and attest the exact Torchvision version remotely;
- materialize a new Modal image and create new staged authorizations.

The scientific contract is unchanged: the same three sealed experiments, nine
public PNGs, three pinned models, source/reverse option order, exact 54-call plan,
outcome-free diagnostics, no retry/repair, and a $5 incremental ceiling. No v3
output may be reused. Human outcomes and participant records are not uploaded,
opened, or authorized.

Active machine-readable freeze:
`configs/simulators/prospective_multimodal_v4.json`.

## Execution result

- Modal image: `im-OaOlZh0TMVQ4aT8Uup7P4b`.
- Final run: `prospective_multimodal_20260813_v4`.
- Result: 54 strict forced-choice outputs, 18 per model, with zero automatic
  retries and no authorized next stage.
- Runtime attestation: Python 3.11, Torch 2.9.1, Torchvision 0.24.1,
  Transformers 4.57.6, Pillow 11.3.0, CUDA 12.8, NVIDIA L40S.
- Local aggregation: 27 balanced arm predictions, nine model decisions, and
  three experiment-level outcome-free diagnostic rows.
- Repository verification after aggregation: 298 tests passed.

Execution artifacts are under
`artifacts/prospective_multimodal/prospective_multimodal_20260813_v4/`.
The final manifest and every call envelope state that target human outcomes were
not accessed. Correctness, treatment-effect error, and decision regret remain
unknown until a separate development-reveal authorization is frozen and used.
