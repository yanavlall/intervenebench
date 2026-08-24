from __future__ import annotations

import pytest

from intervenebench.experiment_statistics import (
    experiment_cluster_bootstrap,
    paired_experiment_cluster_bootstrap,
)


def test_experiment_cluster_bootstrap_is_reproducible() -> None:
    values = {"exp-a": 0.0, "exp-b": 0.2, "exp-c": 0.4}
    first = experiment_cluster_bootstrap(
        values, replicates=500, seed=29, confidence_level=0.90
    )
    second = experiment_cluster_bootstrap(
        values, replicates=500, seed=29, confidence_level=0.90
    )
    assert first == second
    assert first.experiment_count == 3
    assert first.point_estimate == pytest.approx(0.2)
    assert first.confidence_interval[0] <= first.point_estimate
    assert first.confidence_interval[1] >= first.point_estimate


def test_paired_bootstrap_resamples_experiments_in_pairs() -> None:
    reference = {"exp-a": 0.01, "exp-b": 0.20, "exp-c": 0.40}
    candidate = {experiment_id: value - 0.05 for experiment_id, value in reference.items()}
    result = paired_experiment_cluster_bootstrap(
        candidate,
        reference,
        replicates=500,
        seed=31,
        confidence_level=0.95,
        lower_is_better=True,
    )
    assert result.experiment_count == 3
    assert result.mean_difference == pytest.approx(-0.05)
    assert result.difference_confidence_interval == pytest.approx((-0.05, -0.05))
    assert result.probability_candidate_better == 1.0


def test_paired_bootstrap_rejects_unpaired_or_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="same experiment IDs"):
        paired_experiment_cluster_bootstrap(
            {"exp-a": 0.1, "exp-b": 0.2},
            {"exp-a": 0.1, "exp-c": 0.2},
            replicates=10,
            seed=1,
        )
    with pytest.raises(ValueError, match="finite"):
        experiment_cluster_bootstrap(
            {"exp-a": 0.1, "exp-b": float("nan")},
            replicates=10,
            seed=1,
        )


def test_cluster_bootstrap_requires_multiple_experiment_units() -> None:
    with pytest.raises(ValueError, match="at least two experiments"):
        experiment_cluster_bootstrap({"exp-a": 0.1}, replicates=10, seed=1)

