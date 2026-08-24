from __future__ import annotations

from copy import deepcopy

import pytest

from intervenebench.confirmation_adjudication import (
    EXPECTED_FAILURE_MESSAGE,
    expected_unavailable_call_ids,
    validate_no_rerun_adjudication_authorization,
    validate_unavailable_partition,
)


def _request(call_id: str, *, model_id: str = "socrates_qwen2_5_14b_sft") -> dict:
    return {
        "call_id": call_id,
        "model_id": model_id,
        "experiment_id": "tcg8p",
        "method_id": "continuous_constrained_integer_generation.v1",
        "stage": "base",
    }


def _failure(call_id: str) -> dict:
    return {
        **_request(call_id),
        "error_type": "ValueError",
        "error_message": EXPECTED_FAILURE_MESSAGE,
    }


def _authorization() -> dict:
    return {
        "schema_version": "confirmation_no_rerun_adjudication_authorization.v1",
        "status": "authorized_materialize_strict_valid_outputs_only",
        "run_id": "confirmation_20260814_v1",
        "failure_manifest_payload_sha256": "a" * 64,
        "strict_parse_audit_payload_sha256": "b" * 64,
        "call_plan_payload_sha256": "c" * 64,
        "expected_raw_call_count": 1464,
        "expected_strict_parseable_call_count": 1404,
        "expected_unavailable_call_count": 60,
        "model_calls_authorized": False,
        "modal_compute_authorized": False,
        "model_download_authorized": False,
        "semantic_repair_authorized": False,
        "confirmation_outcome_reveal_authorized": False,
        "participant_row_access_authorized": False,
        "scoring_authorized": False,
        "automatic_next_stage_authorized": False,
    }


def test_expected_unavailable_ids_select_only_frozen_model_task_cell() -> None:
    requests = [
        _request("bad-1"),
        _request("bad-2"),
        _request("other-model", model_id="qwen3_8b_generic"),
        {**_request("other-task"), "experiment_id": "5vm8g"},
    ]
    assert expected_unavailable_call_ids(requests) == ["bad-1", "bad-2"]


def test_unavailable_partition_must_match_exact_ids_and_failure_contract() -> None:
    failures = [_failure("bad-1"), _failure("bad-2")]
    validate_unavailable_partition(failures, expected_call_ids=["bad-1", "bad-2"])

    wrong = deepcopy(failures)
    wrong[0]["model_id"] = "qwen3_8b_generic"
    with pytest.raises(ValueError, match="scope"):
        validate_unavailable_partition(wrong, expected_call_ids=["bad-1", "bad-2"])

    with pytest.raises(ValueError, match="partition"):
        validate_unavailable_partition(failures[:1], expected_call_ids=["bad-1", "bad-2"])


def test_adjudication_authority_is_exact_and_cannot_expand_scope() -> None:
    authorization = _authorization()
    validate_no_rerun_adjudication_authorization(
        authorization,
        run_id="confirmation_20260814_v1",
        failure_manifest_payload_sha256="a" * 64,
        strict_parse_audit_payload_sha256="b" * 64,
        call_plan_payload_sha256="c" * 64,
    )

    expanded = deepcopy(authorization)
    expanded["model_calls_authorized"] = True
    with pytest.raises(PermissionError, match="authority"):
        validate_no_rerun_adjudication_authorization(
            expanded,
            run_id="confirmation_20260814_v1",
            failure_manifest_payload_sha256="a" * 64,
            strict_parse_audit_payload_sha256="b" * 64,
            call_plan_payload_sha256="c" * 64,
        )

    drifted = deepcopy(authorization)
    drifted["expected_strict_parseable_call_count"] = 1405
    with pytest.raises(PermissionError, match="binding"):
        validate_no_rerun_adjudication_authorization(
            drifted,
            run_id="confirmation_20260814_v1",
            failure_manifest_payload_sha256="a" * 64,
            strict_parse_audit_payload_sha256="b" * 64,
            call_plan_payload_sha256="c" * 64,
        )
