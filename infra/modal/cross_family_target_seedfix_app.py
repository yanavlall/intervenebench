"""Versioned overlay fixing null seeds for deterministic forced-choice calls.

The parent target image and 624 request hashes remain unchanged.  This overlay
only maps a frozen ``generation_seed: null`` to the engine-local seed ``0``
after request-hash validation and before ``SamplingParams`` construction.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

import modal


BASE_PATH = Path(__file__).with_name("cross_family_target_app.py")
_spec = importlib.util.spec_from_file_location(
    "infra.modal.cross_family_target_app", BASE_PATH
)
if _spec is None or _spec.loader is None:
    raise RuntimeError("could not load the pinned parent target app")
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)

if modal.is_local():
    ROOT = Path(__file__).resolve().parents[2]
    SEEDFIX_PATH = ROOT / "configs/simulators/cross_family_seedfix_v1.json"
    SOURCE_PATH = Path(__file__)
else:
    SEEDFIX_PATH = Path("/opt/intervenebench/cross_family_seedfix_v1.json")
    SOURCE_PATH = Path("/root/infra/modal/cross_family_target_seedfix_app.py")

REMOTE_SOURCE_PATH = Path("/root/infra/modal/cross_family_target_seedfix_app.py")
SEEDFIX_ENVELOPE = json.loads(SEEDFIX_PATH.read_text(encoding="utf-8"))
if set(SEEDFIX_ENVELOPE) != {"payload", "sha256"}:
    raise RuntimeError("embedded seed-fix envelope is malformed")
SEEDFIX = SEEDFIX_ENVELOPE["payload"]
if base._payload_hash(SEEDFIX) != SEEDFIX_ENVELOPE["sha256"]:
    raise RuntimeError("embedded seed-fix envelope hash is invalid")
if SEEDFIX.get("parent_execution_freeze_payload_sha256") != base._payload_hash(
    base.CONFIG
):
    raise RuntimeError("seed-fix overlay is bound to another parent freeze")

app = modal.App("intervenebench-cross-family-seedfix-v1", include_source=False)
seedfix_image = (
    base.target_image.add_local_file(
        SEEDFIX_PATH,
        "/opt/intervenebench/cross_family_seedfix_v1.json",
        copy=True,
    ).add_local_file(SOURCE_PATH, str(REMOTE_SOURCE_PATH), copy=True)
)


def _forced_choice_seedfix(llm: Any, request: dict[str, Any]) -> dict[str, Any]:
    fixed_request = dict(request)
    if fixed_request.get("generation_seed") is None:
        fixed_request["generation_seed"] = 0
        null_seed_normalized = True
    else:
        null_seed_normalized = False
    result = base._forced_choice(llm, fixed_request)
    return {
        **result,
        "null_seed_normalized": null_seed_normalized,
        "effective_generation_seed": int(fixed_request["generation_seed"]),
    }


def _runtime(
    *, expected_image_id: str, cache_attestation: dict[str, Any]
) -> dict[str, Any]:
    return {
        **base._runtime_attestation(
            expected_image_id=expected_image_id,
            cache_attestation=cache_attestation,
        ),
        "seedfix_payload_sha256": SEEDFIX_ENVELOPE["sha256"],
        "seedfix_policy": "forced_choice_null_seed_to_zero_after_hash_validation",
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
        or SEEDFIX_ENVELOPE["sha256"] != expected_seedfix_payload_sha256
    ):
        raise RuntimeError("seed-fix canary authorization mismatch")
    cache = base._load_cache_attestation(expected_cache_attestation_sha256)
    runtime = _runtime(
        expected_image_id=expected_modal_image_id,
        cache_attestation=cache,
    )
    request = {
        "prompt": "Choose exactly one label for this synthetic canary: A or B.",
        "modality": "text",
        "asset_path": None,
        "asset_sha256": None,
        "answer_codes": ["A", "B"],
        "generation_seed": None,
    }
    result = _forced_choice_seedfix(base._load_llm(), request)
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
        raise RuntimeError("seed-fix target group is bound to another parent freeze")
    if not requests:
        raise RuntimeError("seed-fix target group must be non-empty")
    seen: set[str] = set()
    for request in requests:
        call_id = request.get("call_id")
        if call_id in seen or call_id not in base.REQUEST_HASHES:
            raise RuntimeError("seed-fix group contains duplicate or unauthorized call ID")
        seen.add(call_id)
        if base._payload_hash(request) != base.REQUEST_HASHES[call_id]:
            raise RuntimeError("seed-fix target request payload hash drifted")
        if base.hashlib.sha256(request["prompt"].encode("utf-8")).hexdigest() != request[
            "source_prompt_sha256"
        ]:
            raise RuntimeError("seed-fix target request prompt hash drifted")
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
            raw = _forced_choice_seedfix(llm, request)
        elif request["method_id"] == "continuous_constrained_integer_generation.v1":
            raw = base._continuous_raw(llm, request)
        else:
            raise RuntimeError("seed-fix target request method is not allowlisted")
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
        raise RuntimeError("Modal did not hydrate the seed-fix target image")
    return image_id
