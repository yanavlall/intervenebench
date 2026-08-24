from __future__ import annotations

import pytest

from intervenebench.continuous import (
    ContinuousObservation,
    continuous_arm_locations,
    evaluate_continuous_decision,
)
from intervenebench.schemas import OutcomeDirection


def test_continuous_median_excludes_only_declared_missing_codes() -> None:
    result = continuous_arm_locations(
        [
            ContinuousObservation("p1", "control", 0),
            ContinuousObservation("p2", "control", 20),
            ContinuousObservation("p3", "control", 99998),
            ContinuousObservation("p4", "treatment", 5),
            ContinuousObservation("p5", "treatment", 15),
            ContinuousObservation("p6", "treatment", None),
        ],
        arm_ids=("control", "treatment"),
        missing_codes=(77777.0, 99998.0, 99999.0),
        valid_lower_bound=0.0,
        valid_upper_bound=None,
        integer_only=True,
        estimator="median",
    )

    assert result.arm_locations == pytest.approx({"control": 10.0, "treatment": 10.0})
    assert result.valid_counts == {"control": 2, "treatment": 2}
    assert result.missing_counts == {"control": 1, "treatment": 1}


def test_continuous_mean_matches_the_source_analysis_estimand() -> None:
    result = continuous_arm_locations(
        [
            ContinuousObservation("p1", "control", 0),
            ContinuousObservation("p2", "control", 30),
            ContinuousObservation("p3", "treatment", 3),
            ContinuousObservation("p4", "treatment", 6),
            ContinuousObservation("p5", "treatment", 99999),
        ],
        arm_ids=("control", "treatment"),
        missing_codes=(77777.0, 99998.0, 99999.0),
        valid_lower_bound=0.0,
        valid_upper_bound=None,
        integer_only=True,
        estimator="mean",
    )
    assert result.arm_locations == pytest.approx({"control": 15.0, "treatment": 4.5})


def test_continuous_parser_rejects_negative_or_noninteger_values() -> None:
    common = {
        "arm_ids": ("control", "treatment"),
        "missing_codes": (77777.0, 99998.0, 99999.0),
        "valid_lower_bound": 0.0,
        "valid_upper_bound": None,
        "integer_only": True,
        "estimator": "median",
    }
    with pytest.raises(ValueError, match="declared lower bound"):
        continuous_arm_locations(
            [
                ContinuousObservation("p1", "control", -1),
                ContinuousObservation("p2", "treatment", 2),
            ],
            **common,
        )
    with pytest.raises(ValueError, match="integer"):
        continuous_arm_locations(
            [
                ContinuousObservation("p1", "control", 1.5),
                ContinuousObservation("p2", "treatment", 2),
            ],
            **common,
        )


def test_lower_is_better_continuous_effects_and_regret_use_raw_units() -> None:
    result = evaluate_continuous_decision(
        human_locations={"no_notice": 40.0, "day_notice": 25.0, "week_notice": 10.0},
        synthetic_locations={"no_notice": 30.0, "day_notice": 5.0, "week_notice": 20.0},
        control_arm_id="no_notice",
        direction=OutcomeDirection.LOWER_IS_BETTER,
        practical_regret_tolerance=10.0,
        outcome_unit="usd_per_month",
    )

    assert result.selected_arm_id == "day_notice"
    assert result.human_best_arm_id == "week_notice"
    assert result.correct_choice is False
    assert result.regret == pytest.approx(15.0)
    assert result.practically_reliable is False
    assert result.human_treatment_effects == pytest.approx(
        {"day_notice": 15.0, "week_notice": 30.0}
    )
    assert result.synthetic_treatment_effects == pytest.approx(
        {"day_notice": 25.0, "week_notice": 10.0}
    )
    assert result.regret_unit == "usd_per_month"
    assert result.normalized is False


def test_continuous_estimator_fails_closed_on_unsupported_method() -> None:
    with pytest.raises(ValueError, match="unsupported continuous estimator"):
        continuous_arm_locations(
            [
                ContinuousObservation("p1", "control", 1),
                ContinuousObservation("p2", "treatment", 2),
            ],
            arm_ids=("control", "treatment"),
            missing_codes=(),
            valid_lower_bound=0.0,
            valid_upper_bound=None,
            integer_only=True,
            estimator="winsorized_mean",
        )
