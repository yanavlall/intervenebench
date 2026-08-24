"""Second versioned overlay requesting a wider allowed-token logprob window."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import modal


V1_PATH = Path(__file__).with_name("cross_family_target_seedfix_app.py")
_spec = importlib.util.spec_from_file_location(
    "infra.modal.cross_family_target_seedfix_app", V1_PATH
)
if _spec is None or _spec.loader is None:
    raise RuntimeError("could not load the pinned v1 seed-fix app")
v1 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = v1
_spec.loader.exec_module(v1)
base = v1.base

if modal.is_local():
    ROOT = Path(__file__).resolve().parents[2]
    SEEDFIX_V2_PATH = ROOT / "configs/simulators/cross_family_seedfix_v2.json"
    SOURCE_PATH = Path(__file__)
else:
    SEEDFIX_V2_PATH = Path("/opt/intervenebench/cross_family_seedfix_v2.json")
    SOURCE_PATH = Path("/root/infra/modal/cross_family_target_seedfix_v2_app.py")

REMOTE_SOURCE_PATH = Path("/root/infra/modal/cross_family_target_seedfix_v2_app.py")
V2_ENVELOPE = json.loads(SEEDFIX_V2_PATH.read_text(encoding="utf-8"))
if set(V2_ENVELOPE) != {"payload", "sha256"}:
    raise RuntimeError("embedded v2 seed-fix envelope is malformed")
V2 = V2_ENVELOPE["payload"]
if base._payload_hash(V2) != V2_ENVELOPE["sha256"]:
    raise RuntimeError("embedded v2 seed-fix envelope hash is invalid")
if V2.get("parent_seedfix_v1_payload_sha256") != v1.SEEDFIX_ENVELOPE["sha256"]:
    raise RuntimeError("v2 seed-fix is bound to another v1 overlay")

app = modal.App("intervenebench-cross-family-seedfix-v2", include_source=False)
seedfix_image = (
    v1.seedfix_image.add_local_file(
        SEEDFIX_V2_PATH,
        "/opt/intervenebench/cross_family_seedfix_v2.json",
        copy=True,
    ).add_local_file(SOURCE_PATH, str(REMOTE_SOURCE_PATH), copy=True)
)


def _forced_choice_v2(llm: Any, request: dict[str, Any]) -> dict[str, Any]:
    from vllm import SamplingParams

    codes = request["answer_codes"]
    token_ids = base._exact_answer_token_ids(llm.get_tokenizer(), codes)
    effective_seed = 0 if request.get("generation_seed") is None else int(
        request["generation_seed"]
    )
    params = SamplingParams(
        temperature=1.0,
        max_tokens=1,
        min_tokens=1,
        seed=effective_seed,
        allowed_token_ids=token_ids,
        logprobs=max(20, len(token_ids)),
        detokenize=False,
    )
    outputs = llm.chat(base._messages(request), sampling_params=params, use_tqdm=False)
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError("v2 forced-choice call returned an unexpected output count")
    generated = outputs[0].outputs[0]
    if len(generated.token_ids) != 1 or len(generated.logprobs or []) != 1:
        raise RuntimeError("v2 forced-choice call did not return one token")
    entries = generated.logprobs[0]
    log_values: list[float] = []
    for token_id in token_ids:
        entry = entries.get(token_id)
        if entry is None or not math.isfinite(float(entry.logprob)):
            raise RuntimeError("v2 forced-choice call omitted an allowed-token logprob")
        log_values.append(float(entry.logprob))
    maximum = max(log_values)
    weights = [math.exp(value - maximum) for value in log_values]
    denominator = sum(weights)
    return {
        "probabilities_by_code": {
            code: weight / denominator
            for code, weight in zip(codes, weights, strict=True)
        },
        "candidate_token_ids": token_ids,
        "free_generation_used": False,
        "engine_probe_tokens": 1,
        "null_seed_normalized": request.get("generation_seed") is None,
        "effective_generation_seed": effective_seed,
        "requested_logprob_count": max(20, len(token_ids)),
    }


def _runtime(
    *, expected_image_id: str, cache_attestation: dict[str, Any]
) -> dict[str, Any]:
    return {
        **v1._runtime(
            expected_image_id=expected_image_id,
            cache_attestation=cache_attestation,
        ),
        "seedfix_v2_payload_sha256": V2_ENVELOPE["sha256"],
        "seedfix_v2_policy": "null_seed_zero_and_twenty_logprob_window",
    }


@app.function(
    image=seedfix_image,
    gpu="A100-80GB:1",
    volumes={base.MODEL_ROOT: base.model_volume.with_mount_options(read_only=True)},
    timeout=3600,
    startup_timeout=1800,
    scaledown_window=60,
    retries=0,
    max_containers=1,
    block_network=True,
    restrict_modal_access=True,
    name="run_cross_family_seedfix_canary",
    include_source=False,
)
def run_cross_family_seedfix_canary(
    expected_parent_freeze_payload_sha256: str,
    expected_modal_image_id: str,
    expected_cache_attestation_sha256: str,
    expected_seedfix_payload_sha256: str,
) -> dict[str, Any]:
    if (
        base._payload_hash(base.CONFIG) != expected_parent_freeze_payload_sha256
        or V2_ENVELOPE["sha256"] != expected_seedfix_payload_sha256
    ):
        raise RuntimeError("v2 seed-fix canary authorization mismatch")
    cache = base._load_cache_attestation(expected_cache_attestation_sha256)
    runtime = _runtime(
        expected_image_id=expected_modal_image_id,
        cache_attestation=cache,
    )
    request = {
        "prompt": (
            "This is a synthetic parser check, not a study. Choose exactly one "
            "response code. A = option one; B = option two; C = option three. "
            "Return one code only."
        ),
        "modality": "text",
        "asset_path": None,
        "asset_sha256": None,
        "answer_codes": ["A", "B", "C"],
        "generation_seed": None,
    }
    result = _forced_choice_v2(base._load_llm(), request)
    return {
        "schema_version": "intervenebench.cross_family_seedfix_canary_result.v1",
        "status": "passed_target_free_null_seed_forced_choice_stop",
        "result": result,
        "runtime_attestation": runtime,
        "attempt_count": 1,
        "target_prompts_or_assets_accessed": False,
        "target_calls_made": 0,
        "model_downloaded": False,
        "human_outcomes_accessed": False,
        "participant_rows_read": 0,
        "automatic_next_stage": False,
    }


@app.function(
    image=seedfix_image,
    gpu="A100-80GB:1",
    volumes={base.MODEL_ROOT: base.model_volume.with_mount_options(read_only=True)},
    timeout=10800,
    startup_timeout=1800,
    scaledown_window=60,
    retries=0,
    max_containers=1,
    block_network=True,
    restrict_modal_access=True,
    name="run_cross_family_target_group",
    include_source=False,
)
def run_cross_family_target_group(
    requests: list[dict[str, Any]],
    expected_freeze_payload_sha256: str,
    expected_modal_image_id: str,
    expected_cache_attestation_sha256: str,
) -> dict[str, Any]:
    if base._payload_hash(base.CONFIG) != expected_freeze_payload_sha256:
        raise RuntimeError("v2 target group is bound to another parent freeze")
    if not requests:
        raise RuntimeError("v2 target group must be non-empty")
    seen: set[str] = set()
    for request in requests:
        call_id = request.get("call_id")
        if call_id in seen or call_id not in base.REQUEST_HASHES:
            raise RuntimeError("v2 group contains duplicate or unauthorized call ID")
        seen.add(call_id)
        if base._payload_hash(request) != base.REQUEST_HASHES[call_id]:
            raise RuntimeError("v2 target request payload hash drifted")
        if base.hashlib.sha256(request["prompt"].encode("utf-8")).hexdigest() != request[
            "source_prompt_sha256"
        ]:
            raise RuntimeError("v2 target request prompt hash drifted")
    cache = base._load_cache_attestation(expected_cache_attestation_sha256)
    runtime = _runtime(
        expected_image_id=expected_modal_image_id,
        cache_attestation=cache,
    )
    llm = base._load_llm()
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    for request in requests:
        if request["method_id"] == "forced_choice_next_token_softmax.v1":
            raw = _forced_choice_v2(llm, request)
        elif request["method_id"] == "continuous_constrained_integer_generation.v1":
            raw = base._continuous_raw(llm, request)
        else:
            raise RuntimeError("v2 target request method is not allowlisted")
        results.append(
            {
                "call_id": request["call_id"],
                "model_id": request["model_id"],
                "request_payload_sha256": base.REQUEST_HASHES[request["call_id"]],
                "result": raw,
                "runtime_attestation": {
                    **runtime,
                    "call_id": request["call_id"],
                    "prompt_sha256": request["source_prompt_sha256"],
                    "asset_sha256": request["asset_sha256"],
                    "method_id": request["method_id"],
                },
            }
        )
    return {
        "schema_version": "intervenebench.cross_family_target_group.v1",
        "elapsed_seconds": time.monotonic() - started,
        "attempt_count": len(requests),
        "results": results,
        "model_downloaded": False,
        "human_outcomes_accessed": False,
        "participant_rows_read": 0,
        "automatic_next_stage": False,
    }


def materialized_target_image_id() -> str:
    image_id = seedfix_image.object_id
    if not image_id:
        raise RuntimeError("Modal did not hydrate the v2 seed-fix image")
    return image_id
