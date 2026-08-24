from pathlib import Path

from intervenebench.cross_family_adjudication import (
    DEFAULT_ADJUDICATION_PATH,
    build_cross_family_canary_adjudication,
)
from intervenebench.protocol import verify_envelope


ROOT = Path(__file__).resolve().parents[1]


def test_cross_family_canaries_pass_only_interface_claim() -> None:
    value = build_cross_family_canary_adjudication(ROOT)

    assert value["status"] == (
        "passed_three_of_three_canaries_target_json_gap_stop"
    )
    assert len(value["interface_checks"]) == 3
    assert {row["status"] for row in value["interface_checks"]} == {
        "passed_schema_and_execution"
    }
    assert value["target_package_ready_for_separate_freeze"] is True
    assert value["target_inference_ready"] is False
    assert value["required_followup_canary"]["planned_call_count"] == 1
    assert value["required_followup_canary"][
        "must_pass_before_target_execution_authorization"
    ] is True
    assert value["target_execution_authorized"] is False
    assert value["human_outcome_access_authorized"] is False
    assert value["participant_row_access_authorized"] is False
    assert "semantic accuracy" in " ".join(
        value["claim_boundary"]["not_established"]
    )


def test_frozen_cross_family_canary_adjudication_replays() -> None:
    actual = verify_envelope(ROOT / DEFAULT_ADJUDICATION_PATH, require_blinded=True)
    assert actual == build_cross_family_canary_adjudication(ROOT)
