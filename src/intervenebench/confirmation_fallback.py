"""Confirmation-only fallback evaluator with nuisance-stratified folds.

The earlier development evaluator is hash-frozen as research evidence. This
version adds source-design strata without mutating that historical module.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from math import fsum, isfinite
from statistics import fmean
from typing import Any, Mapping, Sequence

from .eb_fallback import EffectPrior, _solve_linear_system
from .human_fallback import (
    FallbackObservation,
    _choose,
    _regret,
    _validate_observations,
    balanced_allocation,
    hedged_allocation,
)


@dataclass(frozen=True, slots=True)
class ConfirmationFallbackObservation(FallbackObservation):
    fold_stratum_id: str = ""
    standardization_cell_id: str = ""


def _validate_confirmation_observations(
    observations: Sequence[ConfirmationFallbackObservation],
    arm_ids: Sequence[str],
) -> tuple[ConfirmationFallbackObservation, ...]:
    rows = _validate_observations(observations, arm_ids)
    if any(
        not isinstance(row, ConfirmationFallbackObservation)
        or not isinstance(row.fold_stratum_id, str)
        or not isinstance(row.standardization_cell_id, str)
        for row in rows
    ):
        raise ValueError("confirmation fallback strata are invalid")
    return tuple(rows)


def confirmation_stratified_fold_assignments(
    observations: Sequence[ConfirmationFallbackObservation],
    *,
    arm_ids: Sequence[str],
    fold_count: int,
    seed: int,
) -> dict[str, int]:
    rows = _validate_confirmation_observations(observations, arm_ids)
    if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count < 2:
        raise ValueError("fold_count must be an integer of at least two")
    grouped: dict[tuple[str, str], list[ConfirmationFallbackObservation]] = (
        defaultdict(list)
    )
    for row in rows:
        grouped[(row.arm_id, row.fold_stratum_id)].append(row)
    if any(len(group) < fold_count for group in grouped.values()):
        raise ValueError("every arm/stratum must support every evaluation fold")
    rng = random.Random(seed)
    assignments: dict[str, int] = {}
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda row: row.participant_id)
        rng.shuffle(group)
        for index, row in enumerate(group):
            assignments[row.participant_id] = index % fold_count
    return assignments


def _confirmation_weighted_means(
    rows: Sequence[ConfirmationFallbackObservation], arm_ids: Sequence[str]
) -> dict[str, float]:
    grouped: dict[str, list[ConfirmationFallbackObservation]] = defaultdict(list)
    for row in rows:
        grouped[row.arm_id].append(row)
    means: dict[str, float] = {}
    for arm_id in arm_ids:
        arm_rows = grouped[arm_id]
        if not arm_rows:
            raise ValueError("every decision estimate requires every arm")
        cells = {
            row.standardization_cell_id
            for row in arm_rows
            if row.standardization_cell_id
        }
        if cells and any(not row.standardization_cell_id for row in arm_rows):
            raise ValueError("standardization cell IDs must be complete within an arm")
        cell_ids = sorted(cells) if cells else [""]
        cell_means = []
        for cell_id in cell_ids:
            cell_rows = [
                row for row in arm_rows if row.standardization_cell_id == cell_id
            ]
            denominator = fsum(row.weight for row in cell_rows)
            cell_means.append(
                fsum(row.weight * row.utility for row in cell_rows) / denominator
            )
        means[arm_id] = fmean(cell_means)
    return means


def _confirmation_fusion_means(
    synthetic_means: Mapping[str, float],
    pilot_rows: Sequence[ConfirmationFallbackObservation],
    *,
    arm_ids: Sequence[str],
    pseudocount: int,
) -> dict[str, float]:
    pilot_means = _confirmation_weighted_means(pilot_rows, arm_ids)
    counts = {arm_id: 0 for arm_id in arm_ids}
    for row in pilot_rows:
        counts[row.arm_id] += 1
    return {
        arm_id: (
            pseudocount * synthetic_means[arm_id]
            + counts[arm_id] * pilot_means[arm_id]
        )
        / (pseudocount + counts[arm_id])
        for arm_id in arm_ids
    }


def _weighted_mean_variance_of_mean(
    rows: Sequence[ConfirmationFallbackObservation],
) -> tuple[float, float]:
    if not rows:
        raise ValueError("weighted moments require observations")
    cells = {
        row.standardization_cell_id for row in rows if row.standardization_cell_id
    }
    if cells:
        if any(not row.standardization_cell_id for row in rows):
            raise ValueError("standardization cell IDs must be complete")
        moments = [
            _unstandardized_weighted_mean_variance_of_mean(
                [row for row in rows if row.standardization_cell_id == cell_id]
            )
            for cell_id in sorted(cells)
        ]
        count = len(moments)
        return (
            fmean(mean for mean, _ in moments),
            fsum(variance for _, variance in moments) / (count * count),
        )
    return _unstandardized_weighted_mean_variance_of_mean(rows)


def _unstandardized_weighted_mean_variance_of_mean(
    rows: Sequence[ConfirmationFallbackObservation],
) -> tuple[float, float]:
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


def _eb_effect_scores(
    synthetic_means: Mapping[str, float],
    pilot_rows: Sequence[ConfirmationFallbackObservation],
    *,
    arm_ids: Sequence[str],
    control_arm_id: str,
    prior: EffectPrior,
) -> dict[str, float]:
    arms = tuple(arm_ids)
    if control_arm_id not in arms:
        raise ValueError("EB control arm must be in the action set")
    non_reference = tuple(arm for arm in arms if arm != control_arm_id)
    grouped: dict[str, list[ConfirmationFallbackObservation]] = defaultdict(list)
    for row in pilot_rows:
        grouped[row.arm_id].append(row)
    if any(not grouped[arm] for arm in arms):
        raise ValueError("EB pilot requires at least one observation per arm")
    moments = {
        arm: _weighted_mean_variance_of_mean(grouped[arm]) for arm in arms
    }
    human_effects = [
        moments[arm][0] - moments[control_arm_id][0] for arm in non_reference
    ]
    prior_effects = [
        prior.alpha
        * (
            float(synthetic_means[arm])
            - float(synthetic_means[control_arm_id])
        )
        for arm in non_reference
    ]
    control_variance = moments[control_arm_id][1]
    system = [[control_variance for _ in non_reference] for _ in non_reference]
    prior_variance = max(prior.minimum_variance, prior.residual_variance)
    for index, arm in enumerate(non_reference):
        system[index][index] += moments[arm][1] + prior_variance
    adjustment = _solve_linear_system(
        system,
        [
            human - prior_effect
            for human, prior_effect in zip(human_effects, prior_effects, strict=True)
        ],
    )
    posterior = [
        prior_effect + prior_variance * delta
        for prior_effect, delta in zip(prior_effects, adjustment, strict=True)
    ]
    return {
        control_arm_id: 0.0,
        **dict(zip(non_reference, posterior, strict=True)),
    }


def evaluate_confirmation_eb_human_fallback(
    observations: Sequence[ConfirmationFallbackObservation],
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
    """Run the frozen policy set with confirmation-specific design strata."""

    rows = _validate_confirmation_observations(observations, arm_ids)
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
    records = {
        budget: {policy: [] for policy in policies} for budget in budgets
    }
    synthetic_choice = _choose(synthetic_means, arms)
    for partition in range(partitions):
        assignments = confirmation_stratified_fold_assignments(
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
            pool: dict[str, list[ConfirmationFallbackObservation]] = defaultdict(list)
            for row in rows:
                if assignments[row.participant_id] != evaluation_fold:
                    pool[row.arm_id].append(row)
            rng = random.Random(seed + partition * 1009 + evaluation_fold * 9173 + 41)
            for arm in arms:
                pool[arm].sort(key=lambda row: row.participant_id)
                rng.shuffle(pool[arm])
            evaluation_means = _confirmation_weighted_means(evaluation, arms)
            evaluation_best = _choose(evaluation_means, arms)
            synthetic_regret = _regret(evaluation_means, synthetic_choice)
            for budget in budgets:
                baseline = {
                    "regret": synthetic_regret,
                    "exact": float(synthetic_choice == evaluation_best),
                    "practical": float(synthetic_regret <= practical_tolerance),
                    "delta_vs_synthetic": 0.0,
                }
                records[budget]["synthetic_only"].append(baseline)
                if budget == 0:
                    for policy in policies[2:]:
                        records[budget][policy].append(dict(baseline))
                    continue
                allocations = {
                    "balanced": balanced_allocation(arms, budget),
                    "hedged": hedged_allocation(arms, budget, winner_votes=winner_votes),
                }
                if any(
                    allocation[arm] > len(pool[arm])
                    for allocation in allocations.values()
                    for arm in arms
                ):
                    raise ValueError("fallback budget exceeds no-replacement pool")
                pilot = {
                    name: [
                        row
                        for arm in arms
                        for row in pool[arm][: allocation[arm]]
                    ]
                    for name, allocation in allocations.items()
                }
                choices = {
                    "human_only_balanced": _choose(
                        _confirmation_weighted_means(pilot["balanced"], arms), arms
                    ),
                    "synthetic_plus_balanced_fixed10": _choose(
                        _confirmation_fusion_means(
                            synthetic_means,
                            pilot["balanced"],
                            arm_ids=arms,
                            pseudocount=pseudocount,
                        ),
                        arms,
                    ),
                    "synthetic_plus_hedged_fixed10": _choose(
                        _confirmation_fusion_means(
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
        previous_budget = None
        previous_regret = None
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
