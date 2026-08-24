from __future__ import annotations

from pathlib import Path

import pytest

from intervenebench.full_action_diagnostics import (
    build_full_action_diagnostics,
    normalized_entropy,
    verify_diagnostics_freeze,
)


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "configs/diagnostics/balanced_full_action_v2.json"


def test_normalized_entropy_has_exact_boundary_values() -> None:
    assert normalized_entropy({1: 1.0, 2: 0.0}) == 0.0
    assert normalized_entropy({1: 0.5, 2: 0.5}) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        normalized_entropy({1: 0.8, 2: 0.8})


def test_diagnostic_freeze_is_zero_authority_and_direction_frozen() -> None:
    summary = verify_diagnostics_freeze(ROOT, freeze_path=FREEZE)
    assert summary["feature_family_count"] == 6
    assert summary["decision_row_count"] == 20
    assert summary["experiment_row_count"] == 5


def test_full_action_diagnostics_are_complete_deterministic_and_outcome_free() -> None:
    first = build_full_action_diagnostics(ROOT, freeze_path=FREEZE)
    second = build_full_action_diagnostics(ROOT, freeze_path=FREEZE)
    assert first == second
    assert first["arm_diagnostic_count"] == 68
    assert first["decision_diagnostic_count"] == 20
    assert first["experiment_diagnostic_count"] == 5
    assert first["outcome_access_during_diagnostic_build"] == "not_accessed"
    assert first["target_experiment_outcome_status"] == (
        "previously_revealed_development_tasks"
    )
    assert first["prospective_validation_eligible"] is False
    assert first["primary_simulator_status"] == "not_selected_by_this_artifact"
    assert first["trust_threshold_status"] == "not_fit_or_selected"
    assert all(
        row["target_human_outcomes_used"] is False
        for row in first["decision_diagnostics"]
    )
    assert all(
        0.0 <= row["balanced_normalized_entropy"] <= 1.0
        and 0.0 <= row["order_total_variation"] <= 1.0
        for row in first["arm_diagnostics"]
    )
    assert first["summary"]["unanimous_all_model_experiments"] == 1
