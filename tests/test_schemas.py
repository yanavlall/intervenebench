from __future__ import annotations

import pytest

from intervenebench.schemas import (
    Arm,
    ContinuousDecisionTask,
    DecisionTask,
    DesignType,
    OutcomeDirection,
    OutcomeFamily,
)


def valid_task(**overrides: object) -> DecisionTask:
    values: dict[str, object] = {
        "task_id": "exp-a:outcome-1",
        "experiment_id": "exp-a",
        "source_id": "osf-a",
        "paradigm_group": "message-framing",
        "design_type": DesignType.BETWEEN_SUBJECT,
        "randomization_unit": "participant",
        "arms": (
            Arm("control", "No message"),
            Arm("treatment", "Receive the message"),
        ),
        "control_arm_id": "control",
        "primary_outcome_id": "support",
        "outcome_family": OutcomeFamily.ORDINAL,
        "response_options": (1.0, 2.0, 3.0, 4.0, 5.0),
        "scale_lower": 1.0,
        "scale_upper": 5.0,
        "direction": OutcomeDirection.HIGHER_IS_BETTER,
        "observations_per_arm": (("control", 150), ("treatment", 151)),
    }
    values.update(overrides)
    return DecisionTask(**values)  # type: ignore[arg-type]


def test_invalid_scale_bounds_fail() -> None:
    with pytest.raises(ValueError, match="scale_lower"):
        valid_task(scale_lower=5.0, scale_upper=1.0)


def test_control_must_be_an_admissible_arm() -> None:
    with pytest.raises(ValueError, match="control arm"):
        valid_task(control_arm_id="missing")


def test_duplicate_arm_ids_fail() -> None:
    with pytest.raises(ValueError, match="unique"):
        valid_task(arms=(Arm("same", "A"), Arm("same", "B")), control_arm_id="same")


def test_phase1_rejects_unsupported_design() -> None:
    task = valid_task(design_type=DesignType.WITHIN_SUBJECT)
    with pytest.raises(ValueError, match="between-subject"):
        task.validate_phase1()


def test_phase1_rejects_small_arms() -> None:
    task = valid_task(observations_per_arm=(("control", 99), ("treatment", 150)))
    with pytest.raises(ValueError, match="minimum observations"):
        task.validate_phase1()


def test_valid_phase1_task_passes() -> None:
    valid_task().validate_phase1()


def test_valid_uncapped_continuous_extension_passes() -> None:
    task = ContinuousDecisionTask(
        task_id="tcg8p:task-0",
        experiment_id="tcg8p",
        source_id="TESS-101-Gorman",
        paradigm_group="power_outage_notice_policy",
        design_type=DesignType.BETWEEN_SUBJECT,
        randomization_unit="participant",
        arms=(
            Arm("no_notice", "No advance notice"),
            Arm("day_notice", "24 hours advance notice"),
            Arm("week_notice", "One week advance notice"),
        ),
        control_arm_id="no_notice",
        primary_outcome_id="Q11",
        outcome_unit="usd_per_month",
        direction=OutcomeDirection.LOWER_IS_BETTER,
        released_rows_per_arm=(
            ("no_notice", 669),
            ("day_notice", 678),
            ("week_notice", 718),
        ),
        valid_lower_bound=0.0,
        valid_upper_bound=None,
        missing_codes=(77777.0, 99998.0, 99999.0),
        integer_only=True,
        location_estimand="mean",
        robustness_estimands=("median",),
        practical_regret_tolerance=0.0,
    )
    task.validate_continuous_extension()


def test_continuous_extension_requires_median_robustness_check() -> None:
    with pytest.raises(ValueError, match="robustness"):
        ContinuousDecisionTask(
            task_id="exp:task-0",
            experiment_id="exp",
            source_id="source",
            paradigm_group="policy",
            design_type=DesignType.BETWEEN_SUBJECT,
            randomization_unit="participant",
            arms=(Arm("control", "Control"), Arm("treatment", "Treatment")),
            control_arm_id="control",
            primary_outcome_id="Q1",
            outcome_unit="usd",
            direction=OutcomeDirection.LOWER_IS_BETTER,
            released_rows_per_arm=(("control", 100), ("treatment", 100)),
            valid_lower_bound=0.0,
            valid_upper_bound=None,
            missing_codes=(99999.0,),
            integer_only=True,
            location_estimand="mean",
            robustness_estimands=(),
            practical_regret_tolerance=0.0,
        )
