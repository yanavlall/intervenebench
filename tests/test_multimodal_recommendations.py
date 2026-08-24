from __future__ import annotations

from pathlib import Path

import pytest

from intervenebench.multimodal_recommendations import (
    balanced_arm_summary,
    build_multimodal_recommendations,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "artifacts/prospective_multimodal/prospective_multimodal_20260813_v4"


def test_balanced_arm_summary_inverse_mapped_metrics_are_exact() -> None:
    summary = balanced_arm_summary(
        {1: 0.8, 2: 0.2},
        {1: 0.6, 2: 0.4},
        normalized_utility={1: 0.0, 2: 1.0},
    )
    assert summary["balanced_probabilities"] == pytest.approx({1: 0.7, 2: 0.3})
    assert summary["balanced_expected_normalized_utility"] == pytest.approx(0.3)
    assert summary["source_reverse_total_variation"] == pytest.approx(0.2)
    assert summary["source_reverse_modal_response_stable"] is True


def test_real_multimodal_run_builds_complete_outcome_blind_artifact() -> None:
    result = build_multimodal_recommendations(ROOT, run_root=RUN_ROOT)
    assert result["outcome_access"] == "not_accessed"
    assert result["balanced_arm_prediction_count"] == 27
    assert result["model_decision_count"] == 9
    assert result["experiment_diagnostic_count"] == 3
    assert len(result["component_call_output_sha256"]) == 54
    assert result["human_outcome_reveal_authorized"] is False
    assert result["automatic_next_stage_authorized"] is False
    assert {row["experiment_id"] for row in result["outcome_free_experiment_diagnostics"]} == {
        "nj5dx",
        "es4xw",
        "e2pyb",
    }
    assert all(
        row["target_human_outcomes_used"] is False
        for row in result["balanced_arm_predictions"]
    )
