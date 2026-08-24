from __future__ import annotations

import pytest

from intervenebench.evaluation import choose_best_arm, decision_regret, evaluate_decision


def test_lexicographic_tie_rule() -> None:
    assert choose_best_arm({"z": 0.7, "a": 0.7, "m": 0.2}) == "a"


def test_declared_tie_preference_can_select_control() -> None:
    assert (
        choose_best_arm(
            {"treatment": 0.5, "control": 0.5},
            tie_preferred_arm_id="control",
        )
        == "control"
    )


def test_wrong_choice_has_exact_regret() -> None:
    result = evaluate_decision(
        human_means={"control": 0.5, "a": 0.8, "b": 0.6},
        synthetic_means={"control": 0.4, "a": 0.5, "b": 0.7},
        control_arm_id="control",
        practical_regret_tolerance=0.25,
    )
    assert result.selected_arm_id == "b"
    assert result.human_best_arm_id == "a"
    assert result.correct_choice is False
    assert result.regret == pytest.approx(0.2)
    assert result.practically_reliable is True
    assert result.human_treatment_effects == pytest.approx({"a": 0.3, "b": 0.1})
    assert result.synthetic_treatment_effects == pytest.approx({"a": 0.1, "b": 0.3})


def test_correct_choice_has_zero_regret() -> None:
    assert decision_regret({"control": 0.2, "a": 0.9}, "a") == 0.0


def test_arm_mismatch_fails() -> None:
    with pytest.raises(ValueError, match="same arms"):
        evaluate_decision(
            human_means={"control": 0.5, "a": 0.8},
            synthetic_means={"control": 0.5, "b": 0.8},
            control_arm_id="control",
            practical_regret_tolerance=0.05,
        )
