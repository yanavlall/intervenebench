from __future__ import annotations

from pathlib import Path

from intervenebench.evaluation_lifecycle import (
    evaluate_confirmation_lifecycle,
    render_confirmation_lifecycle,
)


ROOT = Path(__file__).resolve().parents[1]


def test_completed_confirmation_lifecycle_verifies_in_order() -> None:
    lifecycle = evaluate_confirmation_lifecycle(ROOT)

    assert lifecycle["schema_version"] == "intervenebench.evaluation_lifecycle.v1"
    assert lifecycle["overall_status"] == "complete_verified"
    assert [stage["stage"] for stage in lifecycle["stages"]] == [
        "prepare",
        "freeze_call_plan",
        "freeze_execution",
        "adjudicate_outputs",
        "aggregate_recommendations",
        "reveal_and_score",
        "audit_value",
        "release_gate",
    ]
    assert all(stage["verification"] == "passed" for stage in lifecycle["stages"])
    assert lifecycle["integrity"]["hash_chain_verified"] is True
    assert lifecycle["integrity"]["participant_rows_serialized"] == 0


def test_lifecycle_renderer_is_clear_about_authority_and_release() -> None:
    rendered = render_confirmation_lifecycle(evaluate_confirmation_lifecycle(ROOT))

    assert "PREPARE" in rendered
    assert "RELEASE GATE" in rendered
    assert "8/8 stages verified" in rendered
    assert "Candidate screening: LIMITED RESEARCH USE" in rendered
    assert "Autonomous intervention selection: HOLD" in rendered
    assert "No execution, reveal, or model-call authority is granted" in rendered
