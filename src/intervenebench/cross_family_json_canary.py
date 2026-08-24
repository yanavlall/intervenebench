"""Authorization and verification for one target-free Mistral JSON canary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .cross_family_adjudication import DEFAULT_CACHE_ATTESTATION_PATH
from .cross_family_execution import (
    DEFAULT_EXECUTION_FREEZE_PATH,
    DEFAULT_MATERIALIZATION_PATH,
    validate_json_canary_result,
    verify_cross_family_execution_freeze,
)
from .protocol import assert_blinded_payload, payload_hash, verify_envelope


DEFAULT_JSON_CANARY_AUTHORIZATION_PATH = Path(
    "artifacts/cross_family_target/authorizations/json_canary_20260815_v1.json"
)
DEFAULT_JSON_CANARY_RESULT_PATH = Path(
    "artifacts/cross_family_target/json_canary_20260815_v1.json"
)
DEFAULT_JSON_CANARY_PROGRESS_PATH = Path(
    "artifacts/cross_family_target/progress/json_canary_20260815_v1.jsonl"
)

_AUTHORITY_FIELDS = frozenset(
    {
        "modal_image_materialization_authorized",
        "modal_compute_authorized",
        "model_download_authorized",
        "paid_inference_authorized",
        "json_canary_authorized",
        "target_inference_authorized",
        "target_call_authorized",
        "automatic_retry_authorized",
        "reserve_call_authorized",
        "human_outcome_access_authorized",
        "participant_row_access_authorized",
        "participant_row_serialization_authorized",
        "regression_scoring_authorized",
        "automatic_next_stage_authorized",
    }
)


def _authority(*, json_canary: bool) -> dict[str, bool]:
    value = {field: False for field in sorted(_AUTHORITY_FIELDS)}
    if json_canary:
        value["modal_compute_authorized"] = True
        value["paid_inference_authorized"] = True
        value["json_canary_authorized"] = True
    return value


def load_json_canary_bindings(root: Path) -> dict[str, Mapping[str, Any]]:
    freeze = verify_cross_family_execution_freeze(
        root, root / DEFAULT_EXECUTION_FREEZE_PATH
    )
    materialization = verify_envelope(
        root / DEFAULT_MATERIALIZATION_PATH, require_blinded=True
    )
    cache = verify_envelope(
        root / DEFAULT_CACHE_ATTESTATION_PATH, require_blinded=True
    )
    if materialization.get("freeze_payload_sha256") != payload_hash(freeze):
        raise ValueError("JSON canary materialization/freeze binding drifted")
    if materialization.get("modal_image_id") is None:
        raise ValueError("JSON canary materialization lacks an image ID")
    if materialization.get("inference_calls_made") != 0:
        raise ValueError("JSON canary materialization was not inference-free")
    if cache.get("checkpoint_commit") != freeze["model"]["checkpoint_commit"]:
        raise ValueError("JSON canary cache/freeze checkpoint binding drifted")
    if payload_hash(cache) != freeze["model"]["cache_attestation_payload_sha256"]:
        raise ValueError("JSON canary cache attestation binding drifted")
    return {
        "freeze": freeze,
        "materialization": materialization,
        "cache": cache,
    }


def build_json_canary_authorization(root: Path) -> dict[str, Any]:
    bindings = load_json_canary_bindings(root)
    freeze = bindings["freeze"]
    materialization = bindings["materialization"]
    cache = bindings["cache"]
    spec = freeze["required_json_canary"]
    value = {
        "schema_version": "intervenebench.cross_family_json_canary_authorization.v1",
        "status": "authorized_one_target_free_json_canary_only",
        "freeze_payload_sha256": payload_hash(freeze),
        "materialization_payload_sha256": payload_hash(materialization),
        "modal_image_id": materialization["modal_image_id"],
        "cache_attestation_payload_sha256": payload_hash(cache),
        "canary_id": spec["canary_id"],
        "canary_prompt_sha256": spec["prompt_sha256"],
        "planned_call_count": 1,
        "maximum_attempt_count": 1,
        "maximum_gpu_seconds": 3600,
        "hard_incremental_cost_cap_usd": 10.0,
        **_authority(json_canary=True),
    }
    assert_blinded_payload(value)
    return value


def validate_json_canary_authorization(
    authorization: Mapping[str, Any],
    *,
    freeze: Mapping[str, Any],
    materialization: Mapping[str, Any],
    cache: Mapping[str, Any],
) -> None:
    assert_blinded_payload(authorization)
    binding_fields = {
        "schema_version",
        "status",
        "freeze_payload_sha256",
        "materialization_payload_sha256",
        "modal_image_id",
        "cache_attestation_payload_sha256",
        "canary_id",
        "canary_prompt_sha256",
        "planned_call_count",
        "maximum_attempt_count",
        "maximum_gpu_seconds",
        "hard_incremental_cost_cap_usd",
    }
    if set(authorization) != binding_fields | _AUTHORITY_FIELDS:
        raise PermissionError("JSON canary authorization fields drifted")
    expected_authority = _authority(json_canary=True)
    if any(
        authorization.get(field) is not expected
        for field, expected in expected_authority.items()
    ):
        raise PermissionError("JSON canary authorization scope widened")
    spec = freeze["required_json_canary"]
    expected = {
        "schema_version": "intervenebench.cross_family_json_canary_authorization.v1",
        "status": "authorized_one_target_free_json_canary_only",
        "freeze_payload_sha256": payload_hash(freeze),
        "materialization_payload_sha256": payload_hash(materialization),
        "modal_image_id": materialization["modal_image_id"],
        "cache_attestation_payload_sha256": payload_hash(cache),
        "canary_id": spec["canary_id"],
        "canary_prompt_sha256": spec["prompt_sha256"],
        "planned_call_count": 1,
        "maximum_attempt_count": 1,
        "maximum_gpu_seconds": 3600,
        "hard_incremental_cost_cap_usd": 10.0,
    }
    if any(authorization.get(field) != value for field, value in expected.items()):
        raise PermissionError("JSON canary authorization binding or ceiling drifted")


def validate_json_canary_completion(
    completion: Mapping[str, Any],
    *,
    freeze: Mapping[str, Any],
    authorization: Mapping[str, Any],
    materialization: Mapping[str, Any],
) -> None:
    required = {
        "schema_version",
        "status",
        "freeze_payload_sha256",
        "authorization_payload_sha256",
        "materialization_payload_sha256",
        "modal_image_id",
        "attempt_count",
        "raw_result",
        "target_prompts_or_assets_accessed",
        "target_calls_made",
        "model_downloaded",
        "human_outcomes_accessed",
        "participant_rows_read",
        "automatic_next_stage",
    }
    if set(completion) != required:
        raise ValueError("JSON canary completion fields drifted")
    if (
        completion.get("schema_version")
        != "intervenebench.cross_family_json_canary_completion.v1"
        or completion.get("status") != "passed_target_free_json_schema_stop"
        or completion.get("freeze_payload_sha256") != payload_hash(freeze)
        or completion.get("authorization_payload_sha256")
        != payload_hash(authorization)
        or completion.get("materialization_payload_sha256")
        != payload_hash(materialization)
        or completion.get("modal_image_id") != materialization["modal_image_id"]
        or completion.get("attempt_count") != 1
        or completion.get("target_prompts_or_assets_accessed") is not False
        or completion.get("target_calls_made") != 0
        or completion.get("model_downloaded") is not False
        or completion.get("human_outcomes_accessed") is not False
        or completion.get("participant_rows_read") != 0
        or completion.get("automatic_next_stage") is not False
    ):
        raise ValueError("JSON canary completion exceeded or drifted from authority")
    validate_json_canary_result(
        completion["raw_result"],
        freeze=freeze,
        modal_image_id=str(materialization["modal_image_id"]),
    )
