"""Zero-authority execution freeze and authorization checks for confirmation."""

from __future__ import annotations

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from .confirmation_calls import (
    DEFAULT_CONFIRMATION_CALL_PLAN_PATH,
    verify_confirmation_call_plan,
)
from .confirmation_preparation import (
    DEFAULT_CONFIRMATION_PREPARATION_PATH,
    verify_confirmation_preparation,
)
from .protocol import assert_blinded_payload, payload_hash, verify_envelope


DEFAULT_CONFIRMATION_EXECUTION_FREEZE_PATH = Path(
    "configs/simulators/confirmation_execution_v1.json"
)
APP_PATH = Path("infra/modal/confirmation_app.py")
LOCK_PATH = Path("infra/modal/multimodal-requirements.lock")
TEXT_MANIFEST_PATH = Path("data/manifests/simulators/model_file_manifests_v1.json")
MM_MANIFEST_PATH = Path(
    "data/manifests/simulators/multimodal_model_file_manifests_v1.json"
)
_CACHE_MODEL_ID = {
    "qwen3_8b_generic": "qwen3_8b_generic",
    "qwen3_8b_text_ablation": "qwen3_8b_generic",
    "qwen3_14b_generic": "qwen3_14b_generic",
    "qwen2_5_14b_generic": "qwen2_5_14b_generic",
    "socrates_qwen2_5_14b_sft": "socrates_qwen2_5_14b_sft",
    "qwen3_vl_8b_primary": "qwen3_vl_8b_primary",
    "qwen2_5_vl_7b_comparator": "qwen2_5_vl_7b_comparator",
}
_PLANNED_BY_CACHE_MODEL = {
    "qwen3_8b_generic": 560,
    "qwen3_14b_generic": 248,
    "qwen2_5_14b_generic": 248,
    "socrates_qwen2_5_14b_sft": 216,
    "qwen3_vl_8b_primary": 128,
    "qwen2_5_vl_7b_comparator": 64,
}
_RESERVE_BY_CACHE_MODEL = {
    "qwen3_8b_generic": 236,
    "qwen3_14b_generic": 0,
    "qwen2_5_14b_generic": 0,
    "socrates_qwen2_5_14b_sft": 0,
    "qwen3_vl_8b_primary": 0,
    "qwen2_5_vl_7b_comparator": 0,
}


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _planned_call_ids(plan: Mapping[str, Any]) -> list[str]:
    return [
        str(call["call_id"])
        for call in plan["calls"]
        if call["stage"] != "outcome_free_adaptive_reserve"
    ]


def build_confirmation_execution_freeze(root: Path) -> dict[str, Any]:
    preparation = verify_confirmation_preparation(
        root, root / DEFAULT_CONFIRMATION_PREPARATION_PATH
    )
    plan = verify_confirmation_call_plan(
        root, root / DEFAULT_CONFIRMATION_CALL_PLAN_PATH
    )
    if plan["preparation_payload_sha256"] != payload_hash(preparation):
        raise ValueError("confirmation execution inputs do not share a preparation")
    models: list[dict[str, Any]] = []
    for model_id, raw in preparation["model_catalog"].items():
        model = dict(raw)
        model["cache_model_id"] = _CACHE_MODEL_ID[model_id]
        model["modality"] = (
            "exact_png_vision" if "_vl_" in model_id else "text"
        )
        model["maximum_context_tokens"] = int(
            model.get("maximum_context_tokens", 16384)
        )
        models.append(model)
    models.sort(key=lambda row: row["model_id"])
    planned_counts = Counter(
        _CACHE_MODEL_ID[call["model_id"]]
        for call in plan["calls"]
        if call["stage"] != "outcome_free_adaptive_reserve"
    )
    reserve_counts = Counter(
        _CACHE_MODEL_ID[call["model_id"]]
        for call in plan["calls"]
        if call["stage"] == "outcome_free_adaptive_reserve"
    )
    if dict(planned_counts) != _PLANNED_BY_CACHE_MODEL:
        raise ValueError("planned checkpoint-group counts drifted")
    if {key: reserve_counts[key] for key in _RESERVE_BY_CACHE_MODEL} != (
        _RESERVE_BY_CACHE_MODEL
    ):
        raise ValueError("reserve checkpoint-group counts drifted")
    planned_ids = _planned_call_ids(plan)
    runtime = {
        "app_name": "intervenebench-confirmation-v1",
        "gpu": "NVIDIA L40S:1",
        "model_volume_name": "intervenebench-model-cache-v1",
        "dependency_lock_path": str(LOCK_PATH),
        "dependency_lock_sha256": _file_sha256(root / LOCK_PATH),
        "image_recipe": {
            "base": "modal.Image.debian_slim",
            "python": "3.11",
            "uv": "0.12.4",
            "requirements_require_hashes": True,
            "embedded_files": [
                str(DEFAULT_CONFIRMATION_EXECUTION_FREEZE_PATH),
                str(DEFAULT_CONFIRMATION_CALL_PLAN_PATH),
                str(TEXT_MANIFEST_PATH),
                str(MM_MANIFEST_PATH),
                str(LOCK_PATH),
                str(APP_PATH),
                "data/derived/stimuli/pb2rr",
            ],
            "network_during_inference": "blocked",
            "checkpoint_volume": "read_only",
        },
    }
    freeze = {
        "schema_version": "confirmation_execution_freeze.v1",
        "status": "frozen_nonexecuting_zero_authority",
        "freeze_date": "2026-08-14",
        "preparation": {
            "path": str(DEFAULT_CONFIRMATION_PREPARATION_PATH),
            "file_sha256": _file_sha256(
                root / DEFAULT_CONFIRMATION_PREPARATION_PATH
            ),
            "payload_sha256": payload_hash(preparation),
        },
        "call_plan": {
            "path": str(DEFAULT_CONFIRMATION_CALL_PLAN_PATH),
            "file_sha256": _file_sha256(root / DEFAULT_CONFIRMATION_CALL_PLAN_PATH),
            "payload_sha256": payload_hash(plan),
            "planned_call_count": 1464,
            "conditional_reserve_call_count": 236,
            "maximum_attempt_count": 1700,
            "planned_call_ids_sha256": payload_hash(planned_ids),
        },
        "models": models,
        "cache_model_ids": list(_PLANNED_BY_CACHE_MODEL),
        "planned_calls_by_cache_model": dict(_PLANNED_BY_CACHE_MODEL),
        "conditional_reserve_calls_by_cache_model": dict(
            _RESERVE_BY_CACHE_MODEL
        ),
        "model_download_policy": "reuse_verified_cache_only_no_download_function",
        "model_manifests": [
            {
                "path": str(TEXT_MANIFEST_PATH),
                "file_sha256": _file_sha256(root / TEXT_MANIFEST_PATH),
            },
            {
                "path": str(MM_MANIFEST_PATH),
                "file_sha256": _file_sha256(root / MM_MANIFEST_PATH),
            },
        ],
        "public_assets": preparation["pb2rr_modal_assets"],
        "runtime": runtime,
        "limits": {
            **preparation["compute_ceiling"],
            "automatic_retries": 0,
            "semantic_repair_allowed": False,
            "completed_call_rerun_forbidden": True,
            "maximum_wall_clock_seconds": 10800,
            "maximum_gpu_seconds_per_checkpoint_group": 9000,
        },
        "implementation_hashes": {
            str(APP_PATH): _file_sha256(root / APP_PATH),
            "scripts/run_confirmation.py": _file_sha256(
                root / "scripts/run_confirmation.py"
            ),
            "src/intervenebench/confirmation_calls.py": _file_sha256(
                root / "src/intervenebench/confirmation_calls.py"
            ),
            "src/intervenebench/confirmation_execution.py": _file_sha256(
                Path(__file__)
            ),
        },
        "authority": dict(preparation["authority"]),
        "confirmation_outcomes_accessed": False,
        "participant_rows_read": 0,
        "participant_rows_serialized": 0,
        "claim_boundary": preparation["claim_boundary"],
    }
    assert_blinded_payload(freeze)
    return json.loads(json.dumps(freeze, sort_keys=True, allow_nan=False))


def write_confirmation_execution_freeze(root: Path) -> Path:
    path = root / DEFAULT_CONFIRMATION_EXECUTION_FREEZE_PATH
    value = build_confirmation_execution_freeze(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    return path


def verify_confirmation_execution_freeze(root: Path, path: Path) -> dict[str, Any]:
    value = _read_object(path)
    if value != build_confirmation_execution_freeze(root):
        raise ValueError("confirmation execution freeze does not replay")
    if any(value["authority"].values()):
        raise PermissionError("confirmation execution freeze embeds expanded authority")
    return value


def validate_materialization_authorization(
    authorization: Mapping[str, Any],
    *,
    freeze: Mapping[str, Any],
) -> None:
    assert_blinded_payload(authorization)
    if authorization.get("schema_version") != (
        "confirmation_materialization_authorization.v1"
    ) or authorization.get("status") != "authorized_image_build_zero_inference":
        raise PermissionError("invalid confirmation materialization authority")
    required = {
        "modal_image_materialization_authorized": True,
        "paid_inference_authorized": False,
        "model_download_authorized": False,
        "confirmation_outcome_reveal_authorized": False,
        "participant_row_access_authorized": False,
        "automatic_next_stage_authorized": False,
    }
    if any(authorization.get(key) is not value for key, value in required.items()):
        raise PermissionError("confirmation materialization authority drifted")
    if authorization.get("freeze_payload_sha256") != payload_hash(freeze):
        raise PermissionError("materialization authority is bound to another freeze")
    if authorization.get("call_plan_payload_sha256") != freeze["call_plan"][
        "payload_sha256"
    ]:
        raise PermissionError("materialization authority is bound to another plan")


def validate_execution_authorization(
    authorization: Mapping[str, Any],
    *,
    freeze: Mapping[str, Any],
    modal_image_id: str,
    cache_attestation_sha256_by_model: Mapping[str, str],
) -> None:
    assert_blinded_payload(authorization)
    if authorization.get("schema_version") != "confirmation_execution_authorization.v1":
        raise PermissionError("invalid confirmation execution authority")
    if authorization.get("status") != "authorized_exact_planned_calls_only":
        raise PermissionError("confirmation execution authority status drifted")
    required = {
        "paid_inference_authorized": True,
        "modal_compute_authorized": True,
        "model_download_authorized": False,
        "adaptive_reserve_authorized": False,
        "confirmation_outcome_reveal_authorized": False,
        "participant_row_access_authorized": False,
        "automatic_next_stage_authorized": False,
    }
    if any(authorization.get(key) is not value for key, value in required.items()):
        raise PermissionError("confirmation execution authority drifted")
    expected = {
        "freeze_payload_sha256": payload_hash(freeze),
        "call_plan_payload_sha256": freeze["call_plan"]["payload_sha256"],
        "planned_call_ids_sha256": freeze["call_plan"][
            "planned_call_ids_sha256"
        ],
        "planned_call_count": 1464,
        "maximum_attempt_count": 1464,
        "hard_incremental_cost_cap_usd": 125.0,
        "modal_image_id": modal_image_id,
    }
    if any(authorization.get(key) != value for key, value in expected.items()):
        raise PermissionError("confirmation execution binding or ceiling drifted")
    expected_cache_ids = set(freeze["cache_model_ids"])
    if set(cache_attestation_sha256_by_model) != expected_cache_ids:
        raise PermissionError("confirmation cache binding set drifted")
    if authorization.get("cache_attestation_sha256_by_model") != dict(
        cache_attestation_sha256_by_model
    ):
        raise PermissionError("confirmation execution cache hashes drifted")
    if any(
        not isinstance(value, str) or len(value) != 64
        for value in cache_attestation_sha256_by_model.values()
    ):
        raise PermissionError("confirmation cache hashes are malformed")
