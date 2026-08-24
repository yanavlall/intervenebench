# Cross-Family Modal Preflight v1

## Outcome

The target-free Modal package for the frozen Mistral robustness replay is
implemented and locally verified. It still grants zero authority: no image was
materialized, no checkpoint was downloaded, no Modal function was invoked, no
inference call was made, and no target or human data was accessed.

The package separates three future authorizations:

1. materialize one hash-pinned runtime image with zero download and zero
   inference;
2. cache only the public Mistral checkpoint at commit
   `68faf511d618ef198fef186659617cfd2eb8e33a`; and
3. run exactly three synthetic canaries on one A100-80GB.

None of those stages can authorize any of the 624 target calls. A successful
canary also does not automatically advance the study.

## Runtime decision

The original Mistral checkpoint format is paired with vLLM 0.8.5,
`mistral-common` 1.5.4, Python 3.11, Torch 2.6.0, and Transformers 4.53.3. The
complete 154-package Linux dependency graph is hash-pinned in
`infra/modal/cross-family-requirements.lock`. The versions follow the official
model card's original-format vLLM route rather than the explicitly untested
Transformers route.

The runtime is provisional until the synthetic canary proves that it loads the
exact original checkpoint and produces valid text, vision, and continuous
outputs. A runtime mismatch fails closed.

## Interface adjudication

vLLM 0.8.5 exposes allowed-token masking and per-output-token log probabilities
through a one-token engine probe. The canary therefore checks whether all
answer codes are exact, distinct tokenizer tokens and whether the engine returns
a complete normalized distribution over only those codes.

This is not silently declared equivalent to the original direct-logit Qwen
implementation. The canary records the engine probe explicitly. Even if it
passes, a separate local adjudication must freeze whether the masked probe is an
acceptable implementation of the already-frozen forced-choice estimand before
any target execution package can exist.

## Synthetic-only canaries

- one text forced-choice request with three answer codes;
- one exact-PNG vision forced-choice request using a one-pixel embedded synthetic
  PNG; and
- one strict nonnegative-integer request.

The image, prompts, seeds, parsers, and hashes are embedded in the preflight
freeze. No confirmation prompt, study stimulus, participant record, or human
aggregate is embedded in the Modal app.

## Resource ceilings

- cache stage: exact allowlisted files only, at most the manifest's declared byte
  total, hard incremental ceiling $12;
- canary stage: exactly three attempts, no retries, hard incremental ceiling $15;
- target stage: absent and unauthorized.

The model volume is referenced with `create_if_missing=False`. Cache inference
uses no GPU. Canary inference uses one A100-80GB, a read-only model mount,
blocked network access, one container, and zero automatic retries.

## Artifacts

- `configs/simulators/cross_family_modal_preflight_v1.json`
- `infra/modal/cross-family-requirements.in`
- `infra/modal/cross-family-requirements.lock`
- `infra/modal/cross_family_app.py`
- `src/intervenebench/cross_family_modal.py`
- `scripts/build_cross_family_modal_freeze.py`
- `scripts/run_cross_family_preflight.py`
- `tests/test_cross_family_modal.py`

Local verification, with no Modal import or remote action:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_cross_family_modal_freeze.py
PYTHONPATH=src .venv/bin/python -m pytest tests/test_cross_family_modal.py -q
```

The next state change requires a new, explicit user authorization. Merely
running either verification command above cannot spend money or contact Modal.
