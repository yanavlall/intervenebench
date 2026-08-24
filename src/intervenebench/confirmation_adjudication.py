"""Outcome-blind, no-rerun adjudication for the frozen confirmation run."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .protocol import assert_blinded_payload


UNAVAILABLE_MODEL_ID = "socrates_qwen2_5_14b_sft"
UNAVAILABLE_EXPERIMENT_ID = "tcg8p"
UNAVAILABLE_METHOD_ID = "continuous_constrained_integer_generation.v1"
UNAVAILABLE_STAGE = "base"
EXPECTED_FAILURE_TYPE = "ValueError"
EXPECTED_FAILURE_MESSAGE = "simulator output must contain exactly predicted_value"


def expected_unavailable_call_ids(
    requests: Iterable[Mapping[str, Any]],
) -> list[str]:
    """Return the frozen Socrates × tcg8p continuous-call IDs only."""

    selected = [
        str(request["call_id"])
        for request in requests
        if request.get("model_id") == UNAVAILABLE_MODEL_ID
        and request.get("experiment_id") == UNAVAILABLE_EXPERIMENT_ID
        and request.get("method_id") == UNAVAILABLE_METHOD_ID
        and request.get("stage") == UNAVAILABLE_STAGE
    ]
    if len(selected) != len(set(selected)):
        raise ValueError("unavailable call IDs are not unique")
    return sorted(selected)


def validate_unavailable_partition(
    failures: Sequence[Mapping[str, Any]],
    *,
    expected_call_ids: Sequence[str],
) -> None:
    """Require failures to equal the prespecified model-task cell exactly."""

    actual_ids = sorted(str(failure.get("call_id")) for failure in failures)
    if actual_ids != sorted(str(call_id) for call_id in expected_call_ids):
        raise ValueError("strict failure partition does not match expected call IDs")
    expected_scope = {
        "model_id": UNAVAILABLE_MODEL_ID,
        "experiment_id": UNAVAILABLE_EXPERIMENT_ID,
        "method_id": UNAVAILABLE_METHOD_ID,
        "stage": UNAVAILABLE_STAGE,
        "error_type": EXPECTED_FAILURE_TYPE,
        "error_message": EXPECTED_FAILURE_MESSAGE,
    }
    for failure in failures:
        if any(failure.get(key) != value for key, value in expected_scope.items()):
            raise ValueError("strict failure escaped the frozen unavailable scope")


def validate_no_rerun_adjudication_authorization(
    authorization: Mapping[str, Any],
    *,
    run_id: str,
    failure_manifest_payload_sha256: str,
    strict_parse_audit_payload_sha256: str,
    call_plan_payload_sha256: str,
) -> None:
    """Validate the narrow authority to materialize preserved valid outputs."""

    assert_blinded_payload(authorization)
    if authorization.get("schema_version") != (
        "confirmation_no_rerun_adjudication_authorization.v1"
    ) or authorization.get("status") != (
        "authorized_materialize_strict_valid_outputs_only"
    ):
        raise PermissionError("invalid confirmation adjudication authority")
    forbidden_expansions = {
        "model_calls_authorized": False,
        "modal_compute_authorized": False,
        "model_download_authorized": False,
        "semantic_repair_authorized": False,
        "confirmation_outcome_reveal_authorized": False,
        "participant_row_access_authorized": False,
        "scoring_authorized": False,
        "automatic_next_stage_authorized": False,
    }
    if any(
        authorization.get(key) is not value
        for key, value in forbidden_expansions.items()
    ):
        raise PermissionError("confirmation adjudication authority expanded")
    expected_bindings = {
        "run_id": run_id,
        "failure_manifest_payload_sha256": failure_manifest_payload_sha256,
        "strict_parse_audit_payload_sha256": strict_parse_audit_payload_sha256,
        "call_plan_payload_sha256": call_plan_payload_sha256,
        "expected_raw_call_count": 1464,
        "expected_strict_parseable_call_count": 1404,
        "expected_unavailable_call_count": 60,
    }
    if any(authorization.get(key) != value for key, value in expected_bindings.items()):
        raise PermissionError("confirmation adjudication binding drifted")

