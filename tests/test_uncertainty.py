from __future__ import annotations

from intervenebench.schemas import OutcomeDirection
from intervenebench.uncertainty import (
    bootstrap_arm_location_optimality,
    bootstrap_arm_optimality,
    bootstrap_weighted_arm_optimality,
    bootstrap_weighted_standardized_arm_optimality,
)


def test_bootstrap_is_reproducible_and_uses_lexicographic_ties() -> None:
    values = {"a": [0.0, 1.0], "b": [1.0, 1.0]}
    first = bootstrap_arm_optimality(values, replicates=200, seed=19)
    second = bootstrap_arm_optimality(values, replicates=200, seed=19)
    assert first == second
    assert sum(first.optimal_probability.values()) == 1.0
    assert first.optimal_probability["b"] > first.optimal_probability["a"]


def test_continuous_median_bootstrap_is_reproducible() -> None:
    values = {"control": [10.0, 20.0, 30.0], "notice": [0.0, 5.0, 10.0]}
    first = bootstrap_arm_location_optimality(
        values,
        estimator="median",
        direction=OutcomeDirection.LOWER_IS_BETTER,
        replicates=200,
        seed=29,
    )
    second = bootstrap_arm_location_optimality(
        values,
        estimator="median",
        direction=OutcomeDirection.LOWER_IS_BETTER,
        replicates=200,
        seed=29,
    )
    assert first == second
    assert sum(first.optimal_probability.values()) == 1.0
    assert first.optimal_probability["notice"] > first.optimal_probability["control"]


def test_continuous_mean_bootstrap_is_supported() -> None:
    result = bootstrap_arm_location_optimality(
        {"control": [10.0, 20.0, 30.0], "notice": [0.0, 5.0, 10.0]},
        estimator="mean",
        direction=OutcomeDirection.LOWER_IS_BETTER,
        replicates=100,
        seed=31,
    )
    assert sum(result.optimal_probability.values()) == 1.0
    assert result.optimal_probability["notice"] > result.optimal_probability["control"]


def test_weighted_bootstrap_is_reproducible_and_keeps_pairs_together() -> None:
    values = {
        "control": [(0.0, 1.0), (1.0, 1.0)],
        "message": [(0.75, 4.0), (1.0, 1.0)],
    }
    first = bootstrap_weighted_arm_optimality(values, replicates=200, seed=41)
    second = bootstrap_weighted_arm_optimality(values, replicates=200, seed=41)
    assert first == second
    assert sum(first.optimal_probability.values()) == 1.0
    assert first.optimal_probability["message"] > first.optimal_probability["control"]


def test_weighted_standardized_bootstrap_resamples_within_cells() -> None:
    values = {
        "control": {
            "a": [(0.0, 1.0), (0.25, 1.0)],
            "b": [(0.25, 1.0), (0.5, 1.0)],
        },
        "message": {
            "a": [(0.75, 1.0), (1.0, 1.0)],
            "b": [(0.5, 1.0), (0.75, 1.0)],
        },
    }
    first = bootstrap_weighted_standardized_arm_optimality(
        values,
        stratum_weights={"a": 0.5, "b": 0.5},
        replicates=200,
        seed=51,
    )
    second = bootstrap_weighted_standardized_arm_optimality(
        values,
        stratum_weights={"a": 0.5, "b": 0.5},
        replicates=200,
        seed=51,
    )
    assert first == second
    assert first.optimal_probability["message"] > first.optimal_probability["control"]
