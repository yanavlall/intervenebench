"""Versioned empirical-Bayes extension of the frozen fallback evaluator."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from math import fsum, isfinite
from statistics import fmean
from typing import Any, Mapping, Sequence

from .human_fallback import (
    FallbackObservation,
    _choose,
    _fusion_means,
    _regret,
    _validate_observations,
    _weighted_means,
    balanced_allocation,
    hedged_allocation,
    stratified_fold_assignments,
)


@dataclass(frozen=True, slots=True)
class EffectCalibrationTask:
    experiment_id: str
    synthetic_effects: Mapping[str, float]
    human_effects: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class EffectPrior:
    alpha: float
    residual_variance: float
    training_experiment_ids: tuple[str, ...]
    contrast_count: int
    minimum_variance: float


def fit_effect_prior(
    tasks: Sequence[EffectCalibrationTask],
    *,
    excluded_experiment_id: str | None = None,
    minimum_variance: float = 1e-6,
) -> EffectPrior:
    """Fit equal-experiment-weight attenuation and residual heterogeneity."""

    if not isfinite(minimum_variance) or minimum_variance <= 0.0:
        raise ValueError("minimum prior variance must be positive and finite")
    retained: list[EffectCalibrationTask] = []
    seen: set[str] = set()
    for task in tasks:
        if not task.experiment_id or task.experiment_id in seen:
            raise ValueError("calibration experiment IDs must be unique")
        seen.add(task.experiment_id)
        if task.experiment_id == excluded_experiment_id:
            continue
        if (
            not task.synthetic_effects
            or set(task.synthetic_effects) != set(task.human_effects)
        ):
            raise ValueError("calibration treatment-effect keys must match")
        for value in (*task.synthetic_effects.values(), *task.human_effects.values()):
            if not isfinite(float(value)):
                raise ValueError("calibration effects must be finite")
        retained.append(task)
    if len(retained) < 2:
        raise ValueError("effect calibration requires two training experiments")
    numerator = 0.0
    denominator = 0.0
    for task in retained:
        contrast_weight = 1.0 / len(task.synthetic_effects)
        for contrast in task.synthetic_effects:
            synthetic = float(task.synthetic_effects[contrast])
            human = float(task.human_effects[contrast])
            numerator += contrast_weight * synthetic * human
            denominator += contrast_weight * synthetic * synthetic
    unconstrained = numerator / denominator if denominator > 0.0 else 0.0
    alpha = min(1.0, max(0.0, unconstrained))
    experiment_residual_mse = [
        fmean(
            (
                float(task.human_effects[contrast])
                - alpha * float(task.synthetic_effects[contrast])
            )
            ** 2
            for contrast in task.synthetic_effects
        )
        for task in retained
    ]
    return EffectPrior(
        alpha=alpha,
        residual_variance=max(minimum_variance, fmean(experiment_residual_mse)),
        training_experiment_ids=tuple(task.experiment_id for task in retained),
        contrast_count=sum(len(task.synthetic_effects) for task in retained),
        minimum_variance=minimum_variance,
    )


def _weighted_mean_variance_of_mean(
    rows: Sequence[FallbackObservation],
) -> tuple[float, float]:
    if not rows:
        raise ValueError("weighted moments require observations")
    total_weight = fsum(row.weight for row in rows)
    squared_weight = fsum(row.weight * row.weight for row in rows)
    mean = fsum(row.weight * row.utility for row in rows) / total_weight
    effective_n = total_weight * total_weight / squared_weight
    variance_denominator = total_weight - squared_weight / total_weight
    if variance_denominator <= 0.0 or effective_n <= 1.0:
        return mean, 1e-6
    sample_variance = (
        fsum(row.weight * (row.utility - mean) ** 2 for row in rows)
        / variance_denominator
    )
    return mean, max(1e-6, sample_variance / effective_n)


def _solve_linear_system(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> list[float]:
    size = len(vector)
    if size == 0 or len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError("linear system dimensions are invalid")
    augmented = [
        [float(value) for value in row] + [float(vector[index])]
        for index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-15:
            raise ValueError("linear system is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return [augmented[index][-1] for index in range(size)]


def _eb_effect_scores(
    synthetic_means: Mapping[str, float],
    pilot_rows: Sequence[FallbackObservation],
    *,
    arm_ids: Sequence[str],
    control_arm_id: str,
    prior: EffectPrior,
) -> dict[str, float]:
    arms = tuple(arm_ids)
    if control_arm_id not in arms:
        raise ValueError("EB control arm must be in the action set")
    non_reference = tuple(arm_id for arm_id in arms if arm_id != control_arm_id)
    grouped: dict[str, list[FallbackObservation]] = defaultdict(list)
    for row in pilot_rows:
        grouped[row.arm_id].append(row)
    if any(not grouped[arm_id] for arm_id in arms):
        raise ValueError("EB pilot requires at least one observation per arm")
    moments = {
        arm_id: _weighted_mean_variance_of_mean(grouped[arm_id])
        for arm_id in arms
    }
    human_effects = [
        moments[arm_id][0] - moments[control_arm_id][0]
        for arm_id in non_reference
    ]
    prior_effects = [
        prior.alpha
        * (
            float(synthetic_means[arm_id])
            - float(synthetic_means[control_arm_id])
        )
        for arm_id in non_reference
    ]
    control_variance = moments[control_arm_id][1]
    system = [
        [control_variance for _ in non_reference] for _ in non_reference
    ]
    prior_variance = max(prior.minimum_variance, prior.residual_variance)
    for index, arm_id in enumerate(non_reference):
        system[index][index] += moments[arm_id][1] + prior_variance
    adjustment = _solve_linear_system(
        system,
        [
            human_effect - prior_effect
            for human_effect, prior_effect in zip(
                human_effects, prior_effects, strict=True
            )
        ],
    )
    posterior = [
        prior_effect + prior_variance * delta
        for prior_effect, delta in zip(prior_effects, adjustment, strict=True)
    ]
    return {
        control_arm_id: 0.0,
        **{
            arm_id: float(value)
            for arm_id, value in zip(non_reference, posterior, strict=True)
        },
    }


def evaluate_eb_human_fallback(
    observations: Sequence[FallbackObservation],
    *,
    arm_ids: Sequence[str],
    control_arm_id: str,
    synthetic_means: Mapping[str, float],
    winner_votes: Mapping[str, int],
    budgets: Sequence[int],
    partitions: int,
    fold_count: int,
    seed: int,
    pseudocount: int,
    practical_tolerance: float,
    effect_prior: EffectPrior,
) -> dict[str, Any]:
    """Run baseline and EB policies on identical disjoint folds and pilot draws."""

    rows = _validate_observations(observations, arm_ids)
    arms = tuple(arm_ids)
    if control_arm_id not in arms:
        raise ValueError("EB fallback requires a declared control arm")
    if set(synthetic_means) != set(arms) or any(
        not isfinite(float(value)) for value in synthetic_means.values()
    ):
        raise ValueError("synthetic fallback means must cover every arm")
    if (
        tuple(budgets) != tuple(sorted(set(budgets)))
        or not budgets
        or budgets[0] != 0
        or partitions <= 0
        or pseudocount <= 0
        or not 0.0 <= practical_tolerance <= 1.0
    ):
        raise ValueError("fallback protocol request is invalid")
    policies = (
        "synthetic_only",
        "human_only_balanced",
        "synthetic_plus_balanced_fixed10",
        "synthetic_plus_hedged_fixed10",
        "synthetic_plus_balanced_eb",
        "synthetic_plus_hedged_eb",
    )
    records: dict[int, dict[str, list[dict[str, float]]]] = {
        budget: {policy: [] for policy in policies} for budget in budgets
    }
    synthetic_choice = _choose(synthetic_means, arms)
    for partition in range(partitions):
        assignments = stratified_fold_assignments(
            rows,
            arm_ids=arms,
            fold_count=fold_count,
            seed=seed + partition * 1009,
        )
        for evaluation_fold in range(fold_count):
            evaluation = [
                row
                for row in rows
                if assignments[row.participant_id] == evaluation_fold
            ]
            pool_by_arm: dict[str, list[FallbackObservation]] = defaultdict(list)
            for row in rows:
                if assignments[row.participant_id] != evaluation_fold:
                    pool_by_arm[row.arm_id].append(row)
            order_rng = random.Random(
                seed + partition * 1009 + evaluation_fold * 9173 + 41
            )
            for arm_id in arms:
                pool_by_arm[arm_id].sort(key=lambda row: row.participant_id)
                order_rng.shuffle(pool_by_arm[arm_id])
            evaluation_means = _weighted_means(evaluation, arms)
            evaluation_best = _choose(evaluation_means, arms)
            synthetic_regret = _regret(evaluation_means, synthetic_choice)
            for budget in budgets:
                baseline_record = {
                    "regret": synthetic_regret,
                    "exact": float(synthetic_choice == evaluation_best),
                    "practical": float(synthetic_regret <= practical_tolerance),
                    "delta_vs_synthetic": 0.0,
                }
                records[budget]["synthetic_only"].append(baseline_record)
                if budget == 0:
                    for policy in (
                        "synthetic_plus_balanced_fixed10",
                        "synthetic_plus_hedged_fixed10",
                        "synthetic_plus_balanced_eb",
                        "synthetic_plus_hedged_eb",
                    ):
                        records[budget][policy].append(dict(baseline_record))
                    continue
                allocations = {
                    "balanced": balanced_allocation(arms, budget),
                    "hedged": hedged_allocation(
                        arms, budget, winner_votes=winner_votes
                    ),
                }
                for allocation in allocations.values():
                    if any(
                        allocation[arm_id] > len(pool_by_arm[arm_id])
                        for arm_id in arms
                    ):
                        raise ValueError("fallback budget exceeds no-replacement pool")
                pilot = {
                    name: [
                        row
                        for arm_id in arms
                        for row in pool_by_arm[arm_id][: allocation[arm_id]]
                    ]
                    for name, allocation in allocations.items()
                }
                choices = {
                    "human_only_balanced": _choose(
                        _weighted_means(pilot["balanced"], arms), arms
                    ),
                    "synthetic_plus_balanced_fixed10": _choose(
                        _fusion_means(
                            synthetic_means,
                            pilot["balanced"],
                            arm_ids=arms,
                            pseudocount=pseudocount,
                        ),
                        arms,
                    ),
                    "synthetic_plus_hedged_fixed10": _choose(
                        _fusion_means(
                            synthetic_means,
                            pilot["hedged"],
                            arm_ids=arms,
                            pseudocount=pseudocount,
                        ),
                        arms,
                    ),
                    "synthetic_plus_balanced_eb": _choose(
                        _eb_effect_scores(
                            synthetic_means,
                            pilot["balanced"],
                            arm_ids=arms,
                            control_arm_id=control_arm_id,
                            prior=effect_prior,
                        ),
                        arms,
                    ),
                    "synthetic_plus_hedged_eb": _choose(
                        _eb_effect_scores(
                            synthetic_means,
                            pilot["hedged"],
                            arm_ids=arms,
                            control_arm_id=control_arm_id,
                            prior=effect_prior,
                        ),
                        arms,
                    ),
                }
                for policy, selected in choices.items():
                    regret = _regret(evaluation_means, selected)
                    records[budget][policy].append(
                        {
                            "regret": regret,
                            "exact": float(selected == evaluation_best),
                            "practical": float(regret <= practical_tolerance),
                            "delta_vs_synthetic": regret - synthetic_regret,
                        }
                    )
    result: dict[str, Any] = {}
    for budget in budgets:
        result[str(budget)] = {}
        for policy in policies:
            policy_rows = records[budget][policy]
            if not policy_rows:
                result[str(budget)][policy] = {
                    "status": "not_estimable_at_zero_humans",
                    "human_observations": budget,
                }
                continue
            result[str(budget)][policy] = {
                "status": "estimated",
                "human_observations": budget,
                "acquisition_evaluation_replicates": len(policy_rows),
                "mean_regret": fmean(row["regret"] for row in policy_rows),
                "exact_choice_rate": fmean(row["exact"] for row in policy_rows),
                "practical_reliability_rate": fmean(
                    row["practical"] for row in policy_rows
                ),
                "paired_mean_regret_change_vs_synthetic": fmean(
                    row["delta_vs_synthetic"] for row in policy_rows
                ),
                "negative_value_rate_vs_synthetic": fmean(
                    float(row["delta_vs_synthetic"] > 1e-12)
                    for row in policy_rows
                ),
            }
    for policy in policies:
        previous_budget: int | None = None
        previous_regret: float | None = None
        for budget in budgets:
            row = result[str(budget)][policy]
            if row["status"] != "estimated":
                continue
            row["marginal_regret_reduction_per_human"] = (
                None
                if previous_budget is None
                else (previous_regret - row["mean_regret"])
                / (budget - previous_budget)
            )
            previous_budget = budget
            previous_regret = row["mean_regret"]
    return {
        "budgets": list(budgets),
        "partitions": partitions,
        "fold_count": fold_count,
        "seed": seed,
        "pseudocount_per_arm": pseudocount,
        "effect_prior": {
            "alpha": effect_prior.alpha,
            "residual_variance": effect_prior.residual_variance,
            "training_experiment_ids": list(effect_prior.training_experiment_ids),
            "contrast_count": effect_prior.contrast_count,
            "minimum_variance": effect_prior.minimum_variance,
        },
        "pilot_evaluation_people_disjoint": True,
        "sampling_without_replacement": True,
        "nested_arm_prefixes_within_policy": True,
        "participant_rows_serialized": 0,
        "by_budget": result,
    }
