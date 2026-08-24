"""Reproducible within-experiment uncertainty for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
from random import Random
from statistics import median
from typing import Mapping, Sequence

from .schemas import OutcomeDirection


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    replicates: int
    seed: int
    optimal_probability: dict[str, float]


def bootstrap_arm_optimality(
    arm_values: Mapping[str, Sequence[float]], *, replicates: int, seed: int
) -> BootstrapResult:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if len(arm_values) < 2 or any(not values for values in arm_values.values()):
        raise ValueError("at least two non-empty arms are required")
    rng = Random(seed)
    wins = {arm_id: 0 for arm_id in arm_values}
    for _ in range(replicates):
        means = {}
        for arm_id, values in arm_values.items():
            sampled = [values[rng.randrange(len(values))] for _ in values]
            means[arm_id] = fsum(sampled) / len(sampled)
        best = max(means.values())
        winner = min(arm_id for arm_id, mean in means.items() if mean == best)
        wins[winner] += 1
    return BootstrapResult(
        replicates=replicates,
        seed=seed,
        optimal_probability={
            arm_id: count / replicates for arm_id, count in wins.items()
        },
    )


def bootstrap_weighted_arm_optimality(
    arm_value_weights: Mapping[str, Sequence[tuple[float, float]]],
    *,
    replicates: int,
    seed: int,
) -> BootstrapResult:
    """Bootstrap Hajek arm means by resampling participant value-weight pairs."""

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if len(arm_value_weights) < 2 or any(
        not pairs for pairs in arm_value_weights.values()
    ):
        raise ValueError("at least two non-empty arms are required")
    for pairs in arm_value_weights.values():
        if any(
            not isfinite(value)
            or not isfinite(weight)
            or weight <= 0.0
            for value, weight in pairs
        ):
            raise ValueError("values must be finite and weights positive and finite")

    rng = Random(seed)
    wins = {arm_id: 0 for arm_id in arm_value_weights}
    for _ in range(replicates):
        means: dict[str, float] = {}
        for arm_id, pairs in arm_value_weights.items():
            sampled = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            numerator = fsum(value * weight for value, weight in sampled)
            denominator = fsum(weight for _, weight in sampled)
            means[arm_id] = numerator / denominator
        best = max(means.values())
        winner = min(arm_id for arm_id, mean in means.items() if mean == best)
        wins[winner] += 1
    return BootstrapResult(
        replicates=replicates,
        seed=seed,
        optimal_probability={
            arm_id: count / replicates for arm_id, count in wins.items()
        },
    )


def bootstrap_weighted_standardized_arm_optimality(
    arm_stratum_value_weights: Mapping[
        str, Mapping[str, Sequence[tuple[float, float]]]
    ],
    *,
    stratum_weights: Mapping[str, float],
    replicates: int,
    seed: int,
) -> BootstrapResult:
    """Bootstrap within randomized cells and reapply frozen standardization."""

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if len(arm_stratum_value_weights) < 2 or len(stratum_weights) < 2:
        raise ValueError("at least two arms and two nuisance strata are required")
    if any(
        not isfinite(weight) or weight <= 0.0 for weight in stratum_weights.values()
    ) or abs(fsum(stratum_weights.values()) - 1.0) > 1e-9:
        raise ValueError("stratum weights must be positive, finite, and sum to one")
    expected_strata = set(stratum_weights)
    for cells in arm_stratum_value_weights.values():
        if set(cells) != expected_strata or any(not pairs for pairs in cells.values()):
            raise ValueError("every arm must contain every non-empty nuisance stratum")
        for pairs in cells.values():
            if any(
                not isfinite(value)
                or not isfinite(weight)
                or weight <= 0.0
                for value, weight in pairs
            ):
                raise ValueError("values must be finite and weights positive and finite")

    rng = Random(seed)
    wins = {arm_id: 0 for arm_id in arm_stratum_value_weights}
    for _ in range(replicates):
        means: dict[str, float] = {}
        for arm_id, cells in arm_stratum_value_weights.items():
            standardized = 0.0
            for stratum_id, pairs in cells.items():
                sampled = [pairs[rng.randrange(len(pairs))] for _ in pairs]
                numerator = fsum(value * weight for value, weight in sampled)
                denominator = fsum(weight for _, weight in sampled)
                standardized += stratum_weights[stratum_id] * numerator / denominator
            means[arm_id] = standardized
        best = max(means.values())
        winner = min(arm_id for arm_id, mean in means.items() if mean == best)
        wins[winner] += 1
    return BootstrapResult(
        replicates=replicates,
        seed=seed,
        optimal_probability={
            arm_id: count / replicates for arm_id, count in wins.items()
        },
    )


def bootstrap_arm_location_optimality(
    arm_values: Mapping[str, Sequence[float]],
    *,
    estimator: str,
    direction: OutcomeDirection,
    replicates: int,
    seed: int,
) -> BootstrapResult:
    """Bootstrap robust arm locations under the frozen utility orientation."""

    if estimator not in {"mean", "median"}:
        raise ValueError("unsupported continuous bootstrap estimator")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if len(arm_values) < 2 or any(not values for values in arm_values.values()):
        raise ValueError("at least two non-empty arms are required")
    rng = Random(seed)
    wins = {arm_id: 0 for arm_id in arm_values}
    multiplier = 1.0 if direction is OutcomeDirection.HIGHER_IS_BETTER else -1.0
    for _ in range(replicates):
        utilities = {}
        for arm_id, values in arm_values.items():
            sampled = [values[rng.randrange(len(values))] for _ in values]
            location = (
                fsum(sampled) / len(sampled)
                if estimator == "mean"
                else float(median(sampled))
            )
            utilities[arm_id] = multiplier * location
        best = max(utilities.values())
        winner = min(
            arm_id for arm_id, utility in utilities.items() if utility == best
        )
        wins[winner] += 1
    return BootstrapResult(
        replicates=replicates,
        seed=seed,
        optimal_probability={
            arm_id: count / replicates for arm_id, count in wins.items()
        },
    )
