"""Experiment-level uncertainty for benchmark summaries and paired policies.

These utilities accept exactly one aggregate value per experiment. Resampling is
performed over experiment identifiers, so outcomes, arms, simulator draws, or
acquisition replicates cannot silently become independent benchmark units.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
from random import Random
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ExperimentClusterBootstrapResult:
    experiment_count: int
    replicates: int
    seed: int
    confidence_level: float
    point_estimate: float
    confidence_interval: tuple[float, float]


@dataclass(frozen=True, slots=True)
class PairedExperimentClusterBootstrapResult:
    experiment_count: int
    replicates: int
    seed: int
    confidence_level: float
    candidate_mean: float
    reference_mean: float
    mean_difference: float
    difference_confidence_interval: tuple[float, float]
    lower_is_better: bool
    probability_candidate_better: float


def _validated_values(values: Mapping[str, float]) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError("experiment bootstrap requires at least two experiments")
    validated: dict[str, float] = {}
    for experiment_id, raw_value in values.items():
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise ValueError("experiment IDs must be non-empty strings")
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
            or not isfinite(float(raw_value))
        ):
            raise ValueError("experiment aggregate values must be finite numbers")
        validated[experiment_id] = float(raw_value)
    return validated


def _validate_bootstrap_request(
    *, replicates: int, confidence_level: float
) -> None:
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("bootstrap replicates must be a positive integer")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not isfinite(float(confidence_level))
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise ValueError("confidence_level must lie strictly between zero and one")


def _mean(values: list[float] | tuple[float, ...]) -> float:
    return fsum(values) / len(values)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _interval(
    bootstrap_values: list[float], confidence_level: float
) -> tuple[float, float]:
    tail = (1.0 - confidence_level) / 2.0
    return (
        _quantile(bootstrap_values, tail),
        _quantile(bootstrap_values, 1.0 - tail),
    )


def experiment_cluster_bootstrap(
    values_by_experiment: Mapping[str, float],
    *,
    replicates: int,
    seed: int,
    confidence_level: float = 0.95,
) -> ExperimentClusterBootstrapResult:
    """Bootstrap the mean by resampling whole experiments with replacement."""

    values = _validated_values(values_by_experiment)
    _validate_bootstrap_request(
        replicates=replicates, confidence_level=confidence_level
    )
    experiment_ids = tuple(sorted(values))
    rng = Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled_ids = [
            experiment_ids[rng.randrange(len(experiment_ids))]
            for _ in experiment_ids
        ]
        estimates.append(_mean([values[experiment_id] for experiment_id in sampled_ids]))
    return ExperimentClusterBootstrapResult(
        experiment_count=len(experiment_ids),
        replicates=replicates,
        seed=seed,
        confidence_level=float(confidence_level),
        point_estimate=_mean(list(values.values())),
        confidence_interval=_interval(estimates, float(confidence_level)),
    )


def paired_experiment_cluster_bootstrap(
    candidate_by_experiment: Mapping[str, float],
    reference_by_experiment: Mapping[str, float],
    *,
    replicates: int,
    seed: int,
    confidence_level: float = 0.95,
    lower_is_better: bool = True,
) -> PairedExperimentClusterBootstrapResult:
    """Bootstrap a paired policy contrast using the same sampled experiments.

    The reported difference is always ``candidate - reference``. The direction
    flag affects only ``probability_candidate_better``.
    """

    candidate = _validated_values(candidate_by_experiment)
    reference = _validated_values(reference_by_experiment)
    if set(candidate) != set(reference):
        raise ValueError("paired bootstrap inputs must contain the same experiment IDs")
    if not isinstance(lower_is_better, bool):
        raise ValueError("lower_is_better must be boolean")
    _validate_bootstrap_request(
        replicates=replicates, confidence_level=confidence_level
    )
    experiment_ids = tuple(sorted(candidate))
    differences = {
        experiment_id: candidate[experiment_id] - reference[experiment_id]
        for experiment_id in experiment_ids
    }
    rng = Random(seed)
    bootstrap_differences: list[float] = []
    better = 0
    for _ in range(replicates):
        sampled_ids = [
            experiment_ids[rng.randrange(len(experiment_ids))]
            for _ in experiment_ids
        ]
        estimate = _mean(
            [differences[experiment_id] for experiment_id in sampled_ids]
        )
        bootstrap_differences.append(estimate)
        if (lower_is_better and estimate < 0.0) or (
            not lower_is_better and estimate > 0.0
        ):
            better += 1
    return PairedExperimentClusterBootstrapResult(
        experiment_count=len(experiment_ids),
        replicates=replicates,
        seed=seed,
        confidence_level=float(confidence_level),
        candidate_mean=_mean(list(candidate.values())),
        reference_mean=_mean(list(reference.values())),
        mean_difference=_mean(list(differences.values())),
        difference_confidence_interval=_interval(
            bootstrap_differences, float(confidence_level)
        ),
        lower_is_better=lower_is_better,
        probability_candidate_better=better / replicates,
    )

