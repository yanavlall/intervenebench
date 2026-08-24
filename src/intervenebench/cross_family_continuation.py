"""No-rerun continuation after the tcg8p result-transport failure."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from .cross_family_execution import prepare_cross_family_target_requests
from .cross_family_target_run import (
    DEFAULT_TARGET_AUTHORIZATION_PATH,
    load_target_bindings,
)
from .protocol import assert_blinded_payload, payload_hash, verify_envelope


SOURCE_FAILURE_PATH = Path(
    "artifacts/cross_family_target/target_run_20260815_v1/failure_manifest.json"
)
SOURCE_PROGRESS_PATH = Path(
    "artifacts/cross_family_target/progress/target_run_20260815_v1.jsonl"
)
DEFAULT_CONTINUATION_AUTHORIZATION_PATH = Path(
    "artifacts/cross_family_target/authorizations/no_rerun_continuation_20260815_v1.json"
)
DEFAULT_CONTINUATION_RUN_ROOT = Path(
    "artifacts/cross_family_target/target_run_20260815_v1_continuation_v1"
)
DEFAULT_CONTINUATION_PROGRESS_PATH = Path(
    "artifacts/cross_family_target/progress/no_rerun_continuation_20260815_v1.jsonl"
)
UNAVAILABLE_EXPERIMENT_ID = "tcg8p"
CHUNK_SIZE = 8
EXPECTED_REMAINING_CALL_COUNT = 504
EXPECTED_CHUNK_COUNT = 63

_AUTHORITY_FIELDS = frozenset(
    {
        "modal_compute_authorized",
        "model_download_authorized",
        "paid_inference_authorized",
        "remaining_target_calls_authorized",
        "tcg8p_rerun_authorized",
        "strict_parse_authorized",
        "recommendation_aggregation_authorized",
        "automatic_retry_authorized",
        "reserve_call_authorized",
        "human_outcome_access_authorized",
        "participant_row_access_authorized",
        "participant_row_serialization_authorized",
        "regression_scoring_authorized",
        "automatic_next_stage_authorized",
    }
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def continuation_partition(root: Path) -> dict[str, Any]:
    requests = prepare_cross_family_target_requests(root)
    unavailable = [
        row for row in requests if row["experiment_id"] == UNAVAILABLE_EXPERIMENT_ID
    ]
    remaining = [
        row for row in requests if row["experiment_id"] != UNAVAILABLE_EXPERIMENT_ID
    ]
    if len(unavailable) != 120 or len(remaining) != EXPECTED_REMAINING_CALL_COUNT:
        raise ValueError("no-rerun continuation partition counts drifted")
    chunks: list[dict[str, Any]] = []
    chunk_index = 0
    for experiment_id in ("pb2rr", "z358z", "ShannonS2", "Blair1131", "KlarS44"):
        experiment_calls = [
            row for row in remaining if row["experiment_id"] == experiment_id
        ]
        if len(experiment_calls) % CHUNK_SIZE:
            raise ValueError("continuation experiment grid is not chunk divisible")
        for offset in range(0, len(experiment_calls), CHUNK_SIZE):
            chunk_index += 1
            chunk_calls = experiment_calls[offset : offset + CHUNK_SIZE]
            chunks.append(
                {
                    "chunk_id": f"continuation--{chunk_index:03d}",
                    "experiment_id": experiment_id,
                    "call_ids": [row["call_id"] for row in chunk_calls],
                    "call_count": len(chunk_calls),
                    "request_payload_hashes_sha256": payload_hash(
                        {
                            row["call_id"]: payload_hash(row) for row in chunk_calls
                        }
                    ),
                }
            )
    if len(chunks) != EXPECTED_CHUNK_COUNT:
        raise ValueError("no-rerun continuation chunk count drifted")
    flattened = [call_id for chunk in chunks for call_id in chunk["call_ids"]]
    if flattened != [row["call_id"] for row in remaining]:
        raise ValueError("continuation chunks do not preserve frozen call order")
    if set(flattened).intersection(row["call_id"] for row in unavailable):
        raise ValueError("continuation accidentally contains an unavailable tcg8p call")
    return {
        "unavailable_requests": unavailable,
        "remaining_requests": remaining,
        "chunks": chunks,
    }


def unavailable_transport_rows(
    unavailable_requests: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Represent the original tcg8p transport loss without rerunning it."""

    rows = [
        {
            "call_id": str(request["call_id"]),
            "experiment_id": str(request["experiment_id"]),
            "method_id": str(request["method_id"]),
            "stage": str(request["stage"]),
            "error_type": "TransportOutputUnavailableNoRerun",
            "error_message": (
                "original remote group result transport failed after execution; "
                "raw output unavailable and rerun forbidden"
            ),
        }
        for request in unavailable_requests
    ]
    if (
        len(rows) != 120
        or {row["experiment_id"] for row in rows} != {UNAVAILABLE_EXPERIMENT_ID}
        or len({row["call_id"] for row in rows}) != len(rows)
    ):
        raise ValueError("tcg8p unavailable-call ledger drifted")
    assert_blinded_payload(rows)
    return sorted(rows, key=lambda row: row["call_id"])


def _validate_source_failure(root: Path) -> Mapping[str, Any]:
    failure = verify_envelope(root / SOURCE_FAILURE_PATH, require_blinded=True)
    if (
        failure.get("schema_version")
        != "intervenebench.cross_family_target_run_failure.v1"
        or failure.get("status") != "failed_no_retry_stop"
        or failure.get("error_type") != "AuthError"
        or failure.get("error_message") != "Received :status = '401'"
        or failure.get("completed_experiment_groups") != []
        or failure.get("preserved_raw_output_count") != 0
        or failure.get("automatic_retries") != 0
        or failure.get("human_outcomes_accessed") is not False
        or failure.get("automatic_next_stage") is not False
    ):
        raise ValueError("source target-run failure does not match the frozen incident")
    progress = (root / SOURCE_PROGRESS_PATH).read_text(encoding="utf-8")
    if "experiment_group_submitted" not in progress or "tcg8p" not in progress:
        raise ValueError("source progress does not attest the submitted tcg8p group")
    if "experiment_group_completed" in progress:
        raise ValueError("source progress unexpectedly recorded a delivered tcg8p group")
    return failure


def build_continuation_authorization(root: Path) -> dict[str, Any]:
    bindings = load_target_bindings(root)
    source_authorization = verify_envelope(
        root / DEFAULT_TARGET_AUTHORIZATION_PATH, require_blinded=True
    )
    failure = _validate_source_failure(root)
    partition = continuation_partition(root)
    remaining = partition["remaining_requests"]
    unavailable = partition["unavailable_requests"]
    chunks = partition["chunks"]
    authority = {field: False for field in sorted(_AUTHORITY_FIELDS)}
    for field in (
        "modal_compute_authorized",
        "paid_inference_authorized",
        "remaining_target_calls_authorized",
        "strict_parse_authorized",
        "recommendation_aggregation_authorized",
    ):
        authority[field] = True
    value = {
        "schema_version": "intervenebench.cross_family_no_rerun_continuation_authorization.v1",
        "status": "authorized_remaining_504_calls_tcg8p_permanently_unavailable",
        "freeze_payload_sha256": payload_hash(bindings["freeze"]),
        "source_target_authorization_payload_sha256": payload_hash(source_authorization),
        "source_failure_manifest_payload_sha256": payload_hash(failure),
        "source_progress_file_sha256": _file_sha256(root / SOURCE_PROGRESS_PATH),
        "materialization_payload_sha256": payload_hash(bindings["materialization"]),
        "modal_image_id": bindings["materialization"]["modal_image_id"],
        "cache_attestation_payload_sha256": payload_hash(bindings["cache"]),
        "json_canary_completion_payload_sha256": payload_hash(bindings["json_canary"]),
        "unavailable_experiment_id": UNAVAILABLE_EXPERIMENT_ID,
        "unavailable_call_count": len(unavailable),
        "unavailable_call_ids_sha256": payload_hash(
            [row["call_id"] for row in unavailable]
        ),
        "unavailable_disposition": "transport_output_unavailable_no_rerun",
        "remaining_call_count": len(remaining),
        "maximum_attempt_count": len(remaining),
        "remaining_call_ids_sha256": payload_hash(
            [row["call_id"] for row in remaining]
        ),
        "chunk_size": CHUNK_SIZE,
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


def validate_continuation_authorization(
    authorization: Mapping[str, Any], *, root: Path
) -> None:
    assert_blinded_payload(authorization)
    expected = build_continuation_authorization(root)
    if authorization != expected:
        raise PermissionError("no-rerun continuation authorization drifted")
    if set(field for field in _AUTHORITY_FIELDS if field in authorization) != _AUTHORITY_FIELDS:
        raise PermissionError("no-rerun continuation authority fields drifted")
    if authorization.get("tcg8p_rerun_authorized") is not False:
        raise PermissionError("tcg8p rerun must remain forbidden")
    if authorization.get("automatic_retry_authorized") is not False:
        raise PermissionError("continuation retries must remain forbidden")
    if authorization.get("human_outcome_access_authorized") is not False:
        raise PermissionError("continuation human outcomes must remain inaccessible")
