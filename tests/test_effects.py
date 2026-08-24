from __future__ import annotations

import pytest

from intervenebench.evaluation import (
    CategoricalChoiceIndicator,
    Observation,
    StratifiedWeightedObservation,
    WeightedObservation,
    arm_means,
    normalize_utility,
    treatment_effects,
    weighted_arm_means,
    weighted_categorical_choice_arm_means,
    weighted_standardized_arm_means,
)
from intervenebench.schemas import OutcomeDirection


def test_utility_orientation_reverses_exactly() -> None:
    higher = normalize_utility(
        2.0, lower=1.0, upper=5.0, direction=OutcomeDirection.HIGHER_IS_BETTER
    )
    lower = normalize_utility(
        2.0, lower=1.0, upper=5.0, direction=OutcomeDirection.LOWER_IS_BETTER
    )
    assert higher == pytest.approx(0.25)
    assert lower == pytest.approx(0.75)


def test_known_arm_means_and_effects() -> None:
    observations = [
        Observation("p1", "control", 0.2),
        Observation("p2", "control", 0.4),
        Observation("p3", "a", 0.7),
        Observation("p4", "a", 0.9),
        Observation("p5", "b", 0.1),
        Observation("p6", "b", 0.3),
    ]
    means = arm_means(observations)
    assert means == pytest.approx({"control": 0.3, "a": 0.8, "b": 0.2})
    assert treatment_effects(means, control_arm_id="control") == pytest.approx(
        {"a": 0.5, "b": -0.1}
    )


def test_value_outside_questionnaire_bounds_fails() -> None:
    with pytest.raises(ValueError, match="outside"):
        normalize_utility(
            6.0,
            lower=1.0,
            upper=5.0,
            direction=OutcomeDirection.HIGHER_IS_BETTER,
        )


def test_hajek_arm_means_use_positive_source_weights() -> None:
    observations = [
        WeightedObservation("p1", "control", 0.0, 1.0),
        WeightedObservation("p2", "control", 1.0, 3.0),
        WeightedObservation("p3", "message", 0.5, 2.0),
        WeightedObservation("p4", "message", 1.0, 2.0),
    ]
    assert weighted_arm_means(observations) == pytest.approx(
        {"control": 0.75, "message": 0.75}
    )
    with pytest.raises(ValueError, match="positive"):
        weighted_arm_means([WeightedObservation("p", "control", 1.0, 0.0)])


def test_weighted_standardization_uses_frozen_nuisance_distribution() -> None:
    observations = [
        StratifiedWeightedObservation("p1", "control", "a", 0.0, 1.0),
        StratifiedWeightedObservation("p2", "control", "b", 1.0, 3.0),
        StratifiedWeightedObservation("p3", "message", "a", 1.0, 2.0),
        StratifiedWeightedObservation("p4", "message", "b", 0.5, 2.0),
    ]
    assert weighted_standardized_arm_means(
        observations, stratum_weights={"a": 0.5, "b": 0.5}
    ) == pytest.approx({"control": 0.5, "message": 0.75})
    with pytest.raises(ValueError, match="every arm"):
        weighted_standardized_arm_means(
            observations[:-1], stratum_weights={"a": 0.5, "b": 0.5}
        )


def test_flattened_categorical_choice_collapses_at_participant_level() -> None:
    rows = [
        CategoricalChoiceIndicator("p1", "control", "a", True, 1.0),
        CategoricalChoiceIndicator("p1", "control", "b", False, 1.0),
        CategoricalChoiceIndicator("p2", "control", "a", False, 3.0),
        CategoricalChoiceIndicator("p2", "control", "b", True, 3.0),
        CategoricalChoiceIndicator("p3", "label", "a", True, 2.0),
        CategoricalChoiceIndicator("p3", "label", "b", False, 2.0),
    ]
    assert weighted_categorical_choice_arm_means(
        rows, option_utilities={"a": 0.0, "b": 1.0}
    ) == pytest.approx({"control": 0.75, "label": 0.0})

    with pytest.raises(ValueError, match="every option"):
        weighted_categorical_choice_arm_means(
            rows[:-1], option_utilities={"a": 0.0, "b": 1.0}
        )


def test_flattened_categorical_choice_rejects_multiple_selections() -> None:
    rows = [
        CategoricalChoiceIndicator("p1", "control", "a", True),
        CategoricalChoiceIndicator("p1", "control", "b", True),
        CategoricalChoiceIndicator("p2", "label", "a", True),
        CategoricalChoiceIndicator("p2", "label", "b", False),
    ]
    with pytest.raises(ValueError, match="one selected"):
        weighted_categorical_choice_arm_means(
            rows, option_utilities={"a": 0.0, "b": 1.0}
        )
