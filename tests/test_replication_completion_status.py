from __future__ import annotations

from copy import deepcopy

import pytest

from intervenebench.replication_completion import (
    COMPLETION_GATE_NAMES,
    validate_replication_completion_status,
)


def _queue() -> dict:
    return {
        "wave_1": [
            {
                "queue_order": 1,
                "candidate_id": "alpha",
                "source_stratum": "socsci210",
            }
        ],
        "wave_2": [
            {
                "queue_order": 2,
                "candidate_id": "beta",
                "source_stratum": "external_tess",
            }
        ],
        "panel_gate": {
            "minimum_runnable_tasks": 12,
            "strong_target_tasks": 16,
        },
    }


def _candidate(
    candidate_id: str,
    queue_order: int,
    *,
    gate_state: str = "passed",
    overall_status: str = "runnable",
) -> dict:
    return {
        "queue_order": queue_order,
        "candidate_id": candidate_id,
        "overall_status": overall_status,
        "gates": {name: gate_state for name in COMPLETION_GATE_NAMES},
        "outcome_access": "sealed",
        "result_text_exposed": False,
    }


def _payload() -> dict:
    return {
        "schema_version": "intervenebench.replication_completion_status.v1",
        "status": "outcome_blind_completion_status",
        "queue_sha256": "a" * 64,
        "candidates": [
            _candidate("alpha", 1),
            _candidate(
                "beta",
                2,
                gate_state="pending",
                overall_status="pending",
            ),
        ],
        "authority": {
            "authorized_spend_usd": 0,
            "model_calls_authorized": False,
            "human_outcome_reveal_authorized": False,
            "participant_row_access_authorized": False,
        },
    }


def test_only_all_passed_candidates_count_as_runnable() -> None:
    result = validate_replication_completion_status(_payload(), _queue())
    assert result == {
        "candidate_count": 2,
        "runnable_count": 1,
        "blocked_count": 0,
        "failed_count": 0,
        "pending_count": 1,
        "runnable_socsci210_count": 1,
        "minimum_panel_ready": False,
        "strong_panel_ready": False,
    }


@pytest.mark.parametrize(
    ("gate_state", "overall_status"),
    [
        ("blocked", "runnable"),
        ("failed", "runnable"),
        ("pending", "runnable"),
        ("passed", "blocked"),
    ],
)
def test_overall_status_must_follow_gate_states(
    gate_state: str, overall_status: str
) -> None:
    payload = _payload()
    payload["candidates"][0] = _candidate(
        "alpha", 1, gate_state=gate_state, overall_status=overall_status
    )
    with pytest.raises(ValueError, match="overall_status"):
        validate_replication_completion_status(payload, _queue())


def test_status_must_cover_the_frozen_queue_exactly_in_order() -> None:
    payload = _payload()
    payload["candidates"].reverse()
    with pytest.raises(ValueError, match="frozen queue"):
        validate_replication_completion_status(payload, _queue())


def test_result_fields_exposure_and_authority_fail_closed() -> None:
    for mutation, match in (
        (("result", 0.5), "result-bearing"),
        (("result_text_exposed", True), "sealed"),
        (("outcome_access", "revealed"), "sealed"),
    ):
        payload = _payload()
        key, value = mutation
        payload["candidates"][0][key] = value
        with pytest.raises(ValueError, match=match):
            validate_replication_completion_status(payload, _queue())

    payload = deepcopy(_payload())
    payload["authority"]["model_calls_authorized"] = True
    with pytest.raises(ValueError, match="zero authority"):
        validate_replication_completion_status(payload, _queue())


def test_gate_names_are_exact_and_unknown_states_are_rejected() -> None:
    payload = _payload()
    payload["candidates"][0]["gates"].pop(COMPLETION_GATE_NAMES[-1])
    with pytest.raises(ValueError, match="gate names"):
        validate_replication_completion_status(payload, _queue())

    payload = _payload()
    payload["candidates"][0]["gates"][COMPLETION_GATE_NAMES[0]] = "maybe"
    with pytest.raises(ValueError, match="gate state"):
        validate_replication_completion_status(payload, _queue())
