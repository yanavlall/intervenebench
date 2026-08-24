from __future__ import annotations

from copy import deepcopy

import pytest

from intervenebench.independent_replication import validate_replication_intake


def _valid_intake() -> dict:
    candidates = []
    for index in range(16):
        candidates.append(
            {
                "candidate_id": f"new_{index:02d}",
                "paradigm_group": f"paradigm_{index:02d}",
                "fielding_cluster_id": f"fielding_{index:02d}",
                "outcome_access": "sealed",
                "result_text_exposed": False,
                "audit_status": "pending_contract_audit",
            }
        )
    return {
        "schema_version": "intervenebench.independent_replication_intake.v1",
        "status": "outcome_blind_intake_frozen",
        "panel_size": {"minimum_analyzable": 12, "target": 16},
        "prior_revealed_experiment_ids": ["old_a", "old_b"],
        "candidate_pool": candidates,
        "stage_gates": [
            "source_identity_and_dedup",
            "deployable_action_set",
            "stable_bounded_utility",
            "source_faithful_stimulus_and_sequence",
            "outcome_blind_human_mapping",
            "runnable_simulator_adapter",
        ],
        "compute_boundary": {
            "authorized_spend_usd": 0,
            "paid_execution_requires_separate_authorization": True,
        },
        "reveal_boundary": {
            "human_outcome_reveal_authorized": False,
            "participant_row_access_authorized": False,
        },
    }


def test_valid_replication_intake_preserves_independent_sealed_units() -> None:
    validated = validate_replication_intake(_valid_intake())
    assert validated["candidate_count"] == 16
    assert validated["minimum_analyzable"] == 12
    assert validated["target"] == 16
    assert validated["authorized_spend_usd"] == 0
    assert validated["human_outcome_reveal_authorized"] is False


def test_replication_intake_rejects_any_previously_revealed_experiment() -> None:
    payload = _valid_intake()
    payload["candidate_pool"][0]["candidate_id"] = "old_a"
    with pytest.raises(ValueError, match="previously revealed"):
        validate_replication_intake(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("outcome_access", "development_only_result_exposure"),
        ("result_text_exposed", True),
    ],
)
def test_replication_intake_rejects_outcome_exposure(field: str, value: object) -> None:
    payload = _valid_intake()
    payload["candidate_pool"][0][field] = value
    with pytest.raises(ValueError, match="outcome sealed"):
        validate_replication_intake(payload)


@pytest.mark.parametrize("field", ["paradigm_group", "fielding_cluster_id"])
def test_replication_intake_rejects_dependent_units(field: str) -> None:
    payload = _valid_intake()
    payload["candidate_pool"][1][field] = payload["candidate_pool"][0][field]
    with pytest.raises(ValueError, match="independent"):
        validate_replication_intake(payload)


def test_replication_intake_rejects_result_bearing_fields_recursively() -> None:
    payload = _valid_intake()
    payload["candidate_pool"][0]["notes"] = {"decision_regret": 0.0}
    with pytest.raises(ValueError, match="result-bearing"):
        validate_replication_intake(payload)


def test_replication_intake_cannot_authorize_spend_or_reveal() -> None:
    spend = deepcopy(_valid_intake())
    spend["compute_boundary"]["authorized_spend_usd"] = 1
    with pytest.raises(ValueError, match="authorize spending"):
        validate_replication_intake(spend)

    reveal = deepcopy(_valid_intake())
    reveal["reveal_boundary"]["human_outcome_reveal_authorized"] = True
    with pytest.raises(ValueError, match="authorize outcome reveal"):
        validate_replication_intake(reveal)
