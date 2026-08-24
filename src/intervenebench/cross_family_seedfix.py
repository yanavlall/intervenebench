"""Fail-closed seed-fix overlay for the Mistral continuation."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from .cross_family_continuation import (
    EXPECTED_CHUNK_COUNT,
    EXPECTED_REMAINING_CALL_COUNT,
    SOURCE_FAILURE_PATH,
    continuation_partition,
)
from .cross_family_target_run import load_target_bindings
from .protocol import assert_blinded_payload, payload_hash, verify_envelope


SEEDFIX_APP_PATH = Path("infra/modal/cross_family_target_seedfix_app.py")
SEEDFIX_FREEZE_PATH = Path("configs/simulators/cross_family_seedfix_v1.json")
SEEDFIX_MATERIALIZATION_AUTHORIZATION_PATH = Path(
    "artifacts/cross_family_target/authorizations/seedfix_materialization_20260815_v1.json"
)
SEEDFIX_MATERIALIZATION_PATH = Path(
    "artifacts/cross_family_target/seedfix_materialization_20260815_v1.json"
)
SEEDFIX_MATERIALIZATION_PROGRESS_PATH = Path(
    "artifacts/cross_family_target/progress/seedfix_materialization_20260815_v1.jsonl"
)
SEEDFIX_CANARY_AUTHORIZATION_PATH = Path(
    "artifacts/cross_family_target/authorizations/seedfix_canary_20260815_v1.json"
)
SEEDFIX_CANARY_PATH = Path(
    "artifacts/cross_family_target/seedfix_canary_20260815_v1.json"
)
SEEDFIX_CANARY_PROGRESS_PATH = Path(
    "artifacts/cross_family_target/progress/seedfix_canary_20260815_v1.jsonl"
)
FAILED_CONTINUATION_PATH = Path(
    "artifacts/cross_family_target/target_run_20260815_v1_continuation_v1/failure_manifest.json"
)
SEEDFIX_CONTINUATION_AUTHORIZATION_PATH = Path(
    "artifacts/cross_family_target/authorizations/no_duplicate_seedfix_continuation_20260815_v1.json"
)
SEEDFIX_CONTINUATION_RUN_ROOT = Path(
    "artifacts/cross_family_target/target_run_20260815_v1_continuation_seedfix_v1"
)
SEEDFIX_CONTINUATION_PROGRESS_PATH = Path(
    "artifacts/cross_family_target/progress/no_duplicate_seedfix_continuation_20260815_v1.jsonl"
)

_ZERO_AUTHORITY = {
    "modal_image_materialization_authorized": False,
    "modal_compute_authorized": False,
    "model_download_authorized": False,
    "paid_inference_authorized": False,
    "seedfix_canary_authorized": False,
    "remaining_target_calls_authorized": False,
    "tcg8p_rerun_authorized": False,
    "strict_parse_authorized": False,
    "recommendation_aggregation_authorized": False,
    "automatic_retry_authorized": False,
    "reserve_call_authorized": False,
    "human_outcome_access_authorized": False,
    "participant_row_access_authorized": False,
    "participant_row_serialization_authorized": False,
    "regression_scoring_authorized": False,
    "automatic_next_stage_authorized": False,
}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _failed_continuation(root: Path) -> Mapping[str, Any]:
    failure = verify_envelope(root / FAILED_CONTINUATION_PATH, require_blinded=True)
    if (
        failure.get("status") != "failed_no_retry_stop"
        or failure.get("error_type") != "TypeError"
        or "NoneType" not in str(failure.get("error_message"))
        or failure.get("completed_chunk_count") != 0
        or failure.get("preserved_raw_output_count") != 0
        or failure.get("strict_output_count") != 0
        or failure.get("automatic_retries") != 0
        or failure.get("tcg8p_reruns") != 0
        or failure.get("human_outcomes_accessed") is not False
    ):
        raise ValueError("seed-fix parent failure does not match the null-seed incident")
    return failure


def build_seedfix_freeze(root: Path) -> dict[str, Any]:
    bindings = load_target_bindings(root)
    parent_failure = verify_envelope(root / SOURCE_FAILURE_PATH, require_blinded=True)
    failed_continuation = _failed_continuation(root)
    partition = continuation_partition(root)
    value = {
        "schema_version": "intervenebench.cross_family_seedfix_freeze.v1",
        "status": "frozen_null_seed_runtime_correction_zero_authority",
        "parent_execution_freeze_payload_sha256": payload_hash(bindings["freeze"]),
        "parent_materialization_payload_sha256": payload_hash(
            bindings["materialization"]
        ),
        "cache_attestation_payload_sha256": payload_hash(bindings["cache"]),
        "parent_json_canary_payload_sha256": payload_hash(bindings["json_canary"]),
        "tcg8p_transport_failure_payload_sha256": payload_hash(parent_failure),
        "null_seed_failure_payload_sha256": payload_hash(failed_continuation),
        "null_seed_failure_location": (
            "forced_choice SamplingParams construction before llm.chat"
        ),
        "completed_target_inference_count_before_correction": 0,
        "preserved_target_output_count_before_correction": 0,
        "correction": {
            "request_payloads_changed": False,
            "request_hashes_changed": False,
            "prompt_or_asset_changed": False,
            "affected_method_id": "forced_choice_next_token_softmax.v1",
            "source_generation_seed": None,
            "effective_engine_seed": 0,
            "application_point": "after_request_hash_validation_before_sampling_params",
            "continuous_interface_changed": False,
            "semantic_repair": False,
        },
        "unavailable_experiment_id": "tcg8p",
        "unavailable_call_count": len(partition["unavailable_requests"]),
        "remaining_call_count": len(partition["remaining_requests"]),
        "remaining_call_ids_sha256": payload_hash(
            [row["call_id"] for row in partition["remaining_requests"]]
        ),
        "chunk_count": EXPECTED_CHUNK_COUNT,
        "chunk_size": 8,
        "implementation_hashes": {
            "infra/modal/cross_family_target_app.py": _file_sha256(
                root / "infra/modal/cross_family_target_app.py"
            ),
            str(SEEDFIX_APP_PATH): _file_sha256(root / SEEDFIX_APP_PATH),
        },
        "runtime": {
            "app_name": "intervenebench-cross-family-seedfix-v1",
            "parent_modal_image_id": bindings["materialization"]["modal_image_id"],
            "gpu": "A100-80GB:1",
            "automatic_retries": 0,
            "model_downloads": 0,
        },
        **_ZERO_AUTHORITY,
    }
    assert_blinded_payload(value)
    return value


def verify_seedfix_freeze(root: Path) -> Mapping[str, Any]:
    actual = verify_envelope(root / SEEDFIX_FREEZE_PATH, require_blinded=True)
    if actual != build_seedfix_freeze(root):
        raise ValueError("seed-fix freeze does not replay")
    return actual


def build_seedfix_materialization_authorization(root: Path) -> dict[str, Any]:
    freeze = verify_seedfix_freeze(root)
    authority = dict(_ZERO_AUTHORITY)
    authority["modal_image_materialization_authorized"] = True
    value = {
        "schema_version": "intervenebench.cross_family_seedfix_materialization_authorization.v1",
        "status": "authorized_seedfix_image_materialization_zero_inference",
        "seedfix_freeze_payload_sha256": payload_hash(freeze),
        "planned_remote_function_calls": 0,
        "planned_inference_calls": 0,
        **authority,
    }
    assert_blinded_payload(value)
    return value


def validate_seedfix_materialization_authorization(
    authorization: Mapping[str, Any], *, root: Path
) -> None:
    if authorization != build_seedfix_materialization_authorization(root):
        raise PermissionError("seed-fix materialization authorization drifted")


def load_seedfix_materialization(root: Path) -> Mapping[str, Any]:
    freeze = verify_seedfix_freeze(root)
    value = verify_envelope(root / SEEDFIX_MATERIALIZATION_PATH, require_blinded=True)
    if (
        value.get("status") != "materialized_seedfix_image_zero_inference_stop"
        or value.get("seedfix_freeze_payload_sha256") != payload_hash(freeze)
        or not str(value.get("modal_image_id", "")).startswith("im-")
        or value.get("remote_function_calls_made") != 0
        or value.get("inference_calls_made") != 0
        or value.get("human_outcomes_accessed") is not False
    ):
        raise ValueError("seed-fix materialization artifact drifted")
    return value


def build_seedfix_canary_authorization(root: Path) -> dict[str, Any]:
    freeze = verify_seedfix_freeze(root)
    materialization = load_seedfix_materialization(root)
    authority = dict(_ZERO_AUTHORITY)
    for field in ("modal_compute_authorized", "paid_inference_authorized", "seedfix_canary_authorized"):
        authority[field] = True
    value = {
        "schema_version": "intervenebench.cross_family_seedfix_canary_authorization.v1",
        "status": "authorized_one_target_free_null_seed_forced_choice_canary",
        "seedfix_freeze_payload_sha256": payload_hash(freeze),
        "materialization_payload_sha256": payload_hash(materialization),
        "modal_image_id": materialization["modal_image_id"],
        "planned_call_count": 1,
        "maximum_attempt_count": 1,
        "maximum_gpu_seconds": 3600,
        "hard_incremental_cost_cap_usd": 10.0,
        **authority,
    }
    assert_blinded_payload(value)
    return value


def validate_seedfix_canary_authorization(
    authorization: Mapping[str, Any], *, root: Path
) -> None:
    if authorization != build_seedfix_canary_authorization(root):
        raise PermissionError("seed-fix canary authorization drifted")


def validate_seedfix_canary_completion(
    completion: Mapping[str, Any], *, root: Path
) -> None:
    freeze = verify_seedfix_freeze(root)
    materialization = load_seedfix_materialization(root)
    authorization = verify_envelope(
        root / SEEDFIX_CANARY_AUTHORIZATION_PATH, require_blinded=True
    )
    validate_seedfix_canary_authorization(authorization, root=root)
    result = completion.get("raw_result")
    if (
        completion.get("status") != "passed_target_free_null_seed_forced_choice_stop"
        or completion.get("seedfix_freeze_payload_sha256") != payload_hash(freeze)
        or completion.get("authorization_payload_sha256") != payload_hash(authorization)
        or completion.get("materialization_payload_sha256")
        != payload_hash(materialization)
        or completion.get("attempt_count") != 1
        or not isinstance(result, Mapping)
        or result.get("status")
        != "passed_target_free_null_seed_forced_choice_stop"
        or result.get("result", {}).get("null_seed_normalized") is not True
        or result.get("result", {}).get("effective_generation_seed") != 0
        or result.get("target_calls_made") != 0
        or result.get("human_outcomes_accessed") is not False
        or completion.get("automatic_next_stage") is not False
    ):
        raise ValueError("seed-fix canary completion drifted")


def load_seedfix_canary(root: Path) -> Mapping[str, Any]:
    value = verify_envelope(root / SEEDFIX_CANARY_PATH, require_blinded=True)
    validate_seedfix_canary_completion(value, root=root)
    return value


def build_seedfix_continuation_authorization(root: Path) -> dict[str, Any]:
    freeze = verify_seedfix_freeze(root)
    materialization = load_seedfix_materialization(root)
    canary = load_seedfix_canary(root)
    failed = _failed_continuation(root)
    partition = continuation_partition(root)
    authority = dict(_ZERO_AUTHORITY)
    for field in (
        "modal_compute_authorized",
        "paid_inference_authorized",
        "remaining_target_calls_authorized",
        "strict_parse_authorized",
        "recommendation_aggregation_authorized",
    ):
        authority[field] = True
    chunks = partition["chunks"]
    value = {
        "schema_version": "intervenebench.cross_family_seedfix_continuation_authorization.v1",
        "status": "authorized_exact_504_no_duplicate_calls_tcg8p_unavailable",
        "seedfix_freeze_payload_sha256": payload_hash(freeze),
        "materialization_payload_sha256": payload_hash(materialization),
        "seedfix_canary_payload_sha256": payload_hash(canary),
        "prior_null_seed_failure_payload_sha256": payload_hash(failed),
        "prior_completed_target_inference_count": 0,
        "prior_preserved_target_output_count": 0,
        "duplicated_target_inference_count_authorized": 0,
        "modal_image_id": materialization["modal_image_id"],
        "unavailable_experiment_id": "tcg8p",
        "unavailable_call_count": 120,
        "remaining_call_count": EXPECTED_REMAINING_CALL_COUNT,
        "maximum_attempt_count": EXPECTED_REMAINING_CALL_COUNT,
        "remaining_call_ids_sha256": payload_hash(
            [row["call_id"] for row in partition["remaining_requests"]]
        ),
        "chunk_size": 8,
        "chunk_count": len(chunks),
        "chunk_plan": chunks,
        "chunk_plan_sha256": payload_hash(chunks),
        "new_chunk_failure_policy": "fail_stop_no_retry_preserve_completed_chunks",
        "maximum_wall_clock_seconds": 10_800,
        "total_run_hard_incremental_cost_cap_usd": 90.0,
        **authority,
    }
    assert_blinded_payload(value)
    return value


def validate_seedfix_continuation_authorization(
    authorization: Mapping[str, Any], *, root: Path
) -> None:
    if authorization != build_seedfix_continuation_authorization(root):
        raise PermissionError("seed-fix continuation authorization drifted")
    if (
        authorization.get("tcg8p_rerun_authorized") is not False
        or authorization.get("automatic_retry_authorized") is not False
        or authorization.get("human_outcome_access_authorized") is not False
        or authorization.get("regression_scoring_authorized") is not False
    ):
        raise PermissionError("seed-fix continuation scope widened")


# V2 preserves the complete V1 overlay and adds only a wider logprob request.
SEEDFIX_V2_APP_PATH = Path("infra/modal/cross_family_target_seedfix_v2_app.py")
SEEDFIX_V2_FREEZE_PATH = Path("configs/simulators/cross_family_seedfix_v2.json")
SEEDFIX_V1_CANARY_FAILURE_PATH = Path(
    "artifacts/cross_family_target/seedfix_canary_20260815_v1_failure.json"
)
SEEDFIX_V2_MATERIALIZATION_AUTHORIZATION_PATH = Path(
    "artifacts/cross_family_target/authorizations/seedfix_v2_materialization_20260815_v1.json"
)
SEEDFIX_V2_MATERIALIZATION_PATH = Path(
    "artifacts/cross_family_target/seedfix_v2_materialization_20260815_v1.json"
)
SEEDFIX_V2_MATERIALIZATION_PROGRESS_PATH = Path(
    "artifacts/cross_family_target/progress/seedfix_v2_materialization_20260815_v1.jsonl"
)
SEEDFIX_V2_CANARY_AUTHORIZATION_PATH = Path(
    "artifacts/cross_family_target/authorizations/seedfix_v2_canary_20260815_v1.json"
)
SEEDFIX_V2_CANARY_PATH = Path(
    "artifacts/cross_family_target/seedfix_v2_canary_20260815_v1.json"
)
SEEDFIX_V2_CANARY_PROGRESS_PATH = Path(
    "artifacts/cross_family_target/progress/seedfix_v2_canary_20260815_v1.jsonl"
)
SEEDFIX_V2_CONTINUATION_AUTHORIZATION_PATH = Path(
    "artifacts/cross_family_target/authorizations/no_duplicate_seedfix_v2_continuation_20260815_v1.json"
)
SEEDFIX_V2_CONTINUATION_RUN_ROOT = Path(
    "artifacts/cross_family_target/target_run_20260815_v1_continuation_seedfix_v2"
)
SEEDFIX_V2_CONTINUATION_PROGRESS_PATH = Path(
    "artifacts/cross_family_target/progress/no_duplicate_seedfix_v2_continuation_20260815_v1.jsonl"
)


def build_seedfix_v2_freeze(root: Path) -> dict[str, Any]:
    v1 = verify_seedfix_freeze(root)
    v1_materialization = load_seedfix_materialization(root)
    canary_failure = verify_envelope(
        root / SEEDFIX_V1_CANARY_FAILURE_PATH, require_blinded=True
    )
    if (
        canary_failure.get("status") != "failed_target_free_interface_stop_no_retry"
        or canary_failure.get("target_calls_made") != 0
        or canary_failure.get("automatic_retries") != 0
        or canary_failure.get("human_outcomes_accessed") is not False
    ):
        raise ValueError("v1 canary failure does not match the frozen interface incident")
    value = {
        "schema_version": "intervenebench.cross_family_seedfix_v2_freeze.v1",
        "status": "frozen_wider_logprob_interface_correction_zero_authority",
        "parent_seedfix_v1_payload_sha256": payload_hash(v1),
        "parent_seedfix_v1_materialization_payload_sha256": payload_hash(
            v1_materialization
        ),
        "parent_seedfix_v1_canary_failure_payload_sha256": payload_hash(
            canary_failure
        ),
        "completed_target_inference_count_before_v2": 0,
        "preserved_target_output_count_before_v2": 0,
        "correction": {
            "request_payloads_changed": False,
            "request_hashes_changed": False,
            "prompt_or_asset_changed": False,
            "affected_method_id": "forced_choice_next_token_softmax.v1",
            "effective_engine_seed": 0,
            "requested_logprob_count": 20,
            "maximum_target_answer_code_count": 11,
            "probability_extraction": "original_allowed_token_logits_only",
            "semantic_repair": False,
        },
        "remaining_call_count": EXPECTED_REMAINING_CALL_COUNT,
        "unavailable_experiment_id": "tcg8p",
        "unavailable_call_count": 120,
        "implementation_hashes": {
            str(SEEDFIX_APP_PATH): _file_sha256(root / SEEDFIX_APP_PATH),
            str(SEEDFIX_V2_APP_PATH): _file_sha256(root / SEEDFIX_V2_APP_PATH),
        },
        "runtime": {
            "app_name": "intervenebench-cross-family-seedfix-v2",
            "parent_seedfix_v1_modal_image_id": v1_materialization[
                "modal_image_id"
            ],
            "gpu": "A100-80GB:1",
            "automatic_retries": 0,
            "model_downloads": 0,
        },
        **_ZERO_AUTHORITY,
    }
    assert_blinded_payload(value)
    return value


def verify_seedfix_v2_freeze(root: Path) -> Mapping[str, Any]:
    actual = verify_envelope(root / SEEDFIX_V2_FREEZE_PATH, require_blinded=True)
    if actual != build_seedfix_v2_freeze(root):
        raise ValueError("seed-fix v2 freeze does not replay")
    return actual


def build_seedfix_v2_materialization_authorization(root: Path) -> dict[str, Any]:
    freeze = verify_seedfix_v2_freeze(root)
    authority = dict(_ZERO_AUTHORITY)
    authority["modal_image_materialization_authorized"] = True
    value = {
        "schema_version": "intervenebench.cross_family_seedfix_v2_materialization_authorization.v1",
        "status": "authorized_seedfix_v2_image_materialization_zero_inference",
        "seedfix_v2_freeze_payload_sha256": payload_hash(freeze),
        "planned_remote_function_calls": 0,
        "planned_inference_calls": 0,
        **authority,
    }
    assert_blinded_payload(value)
    return value


def validate_seedfix_v2_materialization_authorization(
    authorization: Mapping[str, Any], *, root: Path
) -> None:
    if authorization != build_seedfix_v2_materialization_authorization(root):
        raise PermissionError("seed-fix v2 materialization authorization drifted")


def load_seedfix_v2_materialization(root: Path) -> Mapping[str, Any]:
    freeze = verify_seedfix_v2_freeze(root)
    value = verify_envelope(
        root / SEEDFIX_V2_MATERIALIZATION_PATH, require_blinded=True
    )
    if (
        value.get("status") != "materialized_seedfix_v2_image_zero_inference_stop"
        or value.get("seedfix_v2_freeze_payload_sha256") != payload_hash(freeze)
        or not str(value.get("modal_image_id", "")).startswith("im-")
        or value.get("remote_function_calls_made") != 0
        or value.get("inference_calls_made") != 0
        or value.get("human_outcomes_accessed") is not False
    ):
        raise ValueError("seed-fix v2 materialization artifact drifted")
    return value


def build_seedfix_v2_canary_authorization(root: Path) -> dict[str, Any]:
    freeze = verify_seedfix_v2_freeze(root)
    materialization = load_seedfix_v2_materialization(root)
    authority = dict(_ZERO_AUTHORITY)
    for field in (
        "modal_compute_authorized",
        "paid_inference_authorized",
        "seedfix_canary_authorized",
    ):
        authority[field] = True
    value = {
        "schema_version": "intervenebench.cross_family_seedfix_v2_canary_authorization.v1",
        "status": "authorized_one_target_free_v2_forced_choice_canary",
        "seedfix_v2_freeze_payload_sha256": payload_hash(freeze),
        "materialization_payload_sha256": payload_hash(materialization),
        "modal_image_id": materialization["modal_image_id"],
        "planned_call_count": 1,
        "maximum_attempt_count": 1,
        "maximum_gpu_seconds": 3600,
        "hard_incremental_cost_cap_usd": 10.0,
        **authority,
    }
    assert_blinded_payload(value)
    return value


def validate_seedfix_v2_canary_authorization(
    authorization: Mapping[str, Any], *, root: Path
) -> None:
    if authorization != build_seedfix_v2_canary_authorization(root):
        raise PermissionError("seed-fix v2 canary authorization drifted")


def load_seedfix_v2_canary(root: Path) -> Mapping[str, Any]:
    value = verify_envelope(root / SEEDFIX_V2_CANARY_PATH, require_blinded=True)
    freeze = verify_seedfix_v2_freeze(root)
    materialization = load_seedfix_v2_materialization(root)
    authorization = verify_envelope(
        root / SEEDFIX_V2_CANARY_AUTHORIZATION_PATH, require_blinded=True
    )
    validate_seedfix_v2_canary_authorization(authorization, root=root)
    result = value.get("raw_result")
    if (
        value.get("status") != "passed_target_free_null_seed_forced_choice_stop"
        or value.get("seedfix_freeze_payload_sha256") != payload_hash(freeze)
        or value.get("authorization_payload_sha256") != payload_hash(authorization)
        or value.get("materialization_payload_sha256")
        != payload_hash(materialization)
        or value.get("attempt_count") != 1
        or not isinstance(result, Mapping)
        or result.get("result", {}).get("requested_logprob_count") != 20
        or result.get("result", {}).get("null_seed_normalized") is not True
        or result.get("target_calls_made") != 0
        or value.get("human_outcomes_accessed") is not False
    ):
        raise ValueError("seed-fix v2 canary completion drifted")
    return value


def build_seedfix_v2_continuation_authorization(root: Path) -> dict[str, Any]:
    freeze = verify_seedfix_v2_freeze(root)
    materialization = load_seedfix_v2_materialization(root)
    canary = load_seedfix_v2_canary(root)
    partition = continuation_partition(root)
    authority = dict(_ZERO_AUTHORITY)
    for field in (
        "modal_compute_authorized",
        "paid_inference_authorized",
        "remaining_target_calls_authorized",
        "strict_parse_authorized",
        "recommendation_aggregation_authorized",
    ):
        authority[field] = True
    chunks = partition["chunks"]
    value = {
        "schema_version": "intervenebench.cross_family_seedfix_v2_continuation_authorization.v1",
        "status": "authorized_exact_504_no_duplicate_calls_tcg8p_unavailable",
        "seedfix_v2_freeze_payload_sha256": payload_hash(freeze),
        "materialization_payload_sha256": payload_hash(materialization),
        "seedfix_canary_payload_sha256": payload_hash(canary),
        "prior_completed_target_inference_count": 0,
        "prior_preserved_target_output_count": 0,
        "duplicated_target_inference_count_authorized": 0,
        "modal_image_id": materialization["modal_image_id"],
        "unavailable_experiment_id": "tcg8p",
        "unavailable_call_count": 120,
        "remaining_call_count": EXPECTED_REMAINING_CALL_COUNT,
        "maximum_attempt_count": EXPECTED_REMAINING_CALL_COUNT,
        "remaining_call_ids_sha256": payload_hash(
            [row["call_id"] for row in partition["remaining_requests"]]
        ),
        "chunk_size": 8,
        "chunk_count": len(chunks),
        "chunk_plan": chunks,
        "chunk_plan_sha256": payload_hash(chunks),
        "new_chunk_failure_policy": "fail_stop_no_retry_preserve_completed_chunks",
        "maximum_wall_clock_seconds": 10_800,
        "total_run_hard_incremental_cost_cap_usd": 90.0,
        **authority,
    }
    assert_blinded_payload(value)
    return value


def validate_seedfix_v2_continuation_authorization(
    authorization: Mapping[str, Any], *, root: Path
) -> None:
    if authorization != build_seedfix_v2_continuation_authorization(root):
        raise PermissionError("seed-fix v2 continuation authorization drifted")
    if (
        authorization.get("tcg8p_rerun_authorized") is not False
        or authorization.get("automatic_retry_authorized") is not False
        or authorization.get("human_outcome_access_authorized") is not False
        or authorization.get("regression_scoring_authorized") is not False
    ):
        raise PermissionError("seed-fix v2 continuation scope widened")
