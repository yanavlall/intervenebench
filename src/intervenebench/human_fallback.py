"""Design-aware, aggregate-only simulations of limited human evidence."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from math import floor, fsum, isfinite
from statistics import fmean
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class FallbackObservation:
    participant_id: str
    arm_id: str
    utility: float
    weight: float = 1.0


def _validate_observations(
    observations: Sequence[FallbackObservation], arm_ids: Sequence[str]
) -> tuple[FallbackObservation, ...]:
    arms = tuple(arm_ids)
    if len(arms) < 2 or len(set(arms)) != len(arms):
        raise ValueError("fallback requires distinct source-ordered arms")
    if not observations:
        raise ValueError("fallback requires human observations")
    seen: set[str] = set()
    counts = {arm_id: 0 for arm_id in arms}
    validated: list[FallbackObservation] = []
    for row in observations:
        if not isinstance(row, FallbackObservation):
            raise ValueError("fallback rows must be FallbackObservation objects")
        if not row.participant_id or row.participant_id in seen:
            raise ValueError("fallback participant IDs must be unique")
        if row.arm_id not in counts:
            raise ValueError("fallback row contains an unknown arm")
        if (
            not isfinite(row.utility)
            or not 0.0 <= row.utility <= 1.0
            or not isfinite(row.weight)
            or row.weight <= 0.0
        ):
            raise ValueError("fallback utilities and weights are invalid")
        seen.add(row.participant_id)
        counts[row.arm_id] += 1
        validated.append(row)
    if any(count == 0 for count in counts.values()):
        raise ValueError("every fallback arm needs observations")
    return tuple(validated)


def balanced_allocation(arm_ids: Sequence[str], budget: int) -> dict[str, int]:
    """Allocate an exact integer budget by source arm order."""

    arms = tuple(arm_ids)
    if not arms or isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("balanced allocation request is invalid")
    base, remainder = divmod(budget, len(arms))
    return {
        arm_id: base + int(index < remainder)
        for index, arm_id in enumerate(arms)
    }


def _largest_remainder(
    arm_ids: Sequence[str], total: int, weights: Mapping[str, float]
) -> dict[str, int]:
    arms = tuple(arm_ids)
    if set(weights) != set(arms) or total < 0:
        raise ValueError("largest-remainder inputs are invalid")
    denominator = fsum(float(weights[arm_id]) for arm_id in arms)
    if denominator <= 0.0:
        raise ValueError("allocation weights must have positive mass")
    quotas = {
        arm_id: total * float(weights[arm_id]) / denominator for arm_id in arms
    }
    result = {arm_id: floor(quotas[arm_id]) for arm_id in arms}
    remaining = total - sum(result.values())
    order = sorted(
        arms,
        key=lambda arm_id: (
            -(quotas[arm_id] - result[arm_id]),
            arms.index(arm_id),
        ),
    )
    for arm_id in order[:remaining]:
        result[arm_id] += 1
    return result


def hedged_allocation(
    arm_ids: Sequence[str],
    budget: int,
    *,
    winner_votes: Mapping[str, int],
    exploration_fraction: float = 0.25,
) -> dict[str, int]:
    """Allocate with a uniform floor plus outcome-free synthetic winner votes."""

    arms = tuple(arm_ids)
    if set(winner_votes) != set(arms) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in winner_votes.values()
    ):
        raise ValueError("winner votes must be non-negative integers by arm")
    if (
        isinstance(budget, bool)
        or not isinstance(budget, int)
        or budget < 0
        or not 0.0 <= exploration_fraction <= 1.0
    ):
        raise ValueError("hedged allocation request is invalid")
    if budget == 0:
        return {arm_id: 0 for arm_id in arms}
    exploration_budget = min(
        budget,
        max(len(arms), floor(exploration_fraction * budget)),
    )
    exploration = balanced_allocation(arms, exploration_budget)
    exploitation_budget = budget - exploration_budget
    # One Laplace vote per arm prevents a small synthetic ensemble from starving
    # an arm even beyond the explicit uniform exploration floor.
    exploitation = _largest_remainder(
        arms,
        exploitation_budget,
        {arm_id: winner_votes[arm_id] + 1.0 for arm_id in arms},
    )
    return {
        arm_id: exploration[arm_id] + exploitation[arm_id] for arm_id in arms
    }


def stratified_fold_assignments(
    observations: Sequence[FallbackObservation],
    *,
    arm_ids: Sequence[str],
    fold_count: int,
    seed: int,
) -> dict[str, int]:
    """Assign each person to one arm-stratified evaluation fold."""

    rows = _validate_observations(observations, arm_ids)
    if (
        isinstance(fold_count, bool)
        or not isinstance(fold_count, int)
        or fold_count < 2
    ):
        raise ValueError("fold_count must be an integer of at least two")
    grouped: dict[str, list[FallbackObservation]] = defaultdict(list)
    for row in rows:
        grouped[row.arm_id].append(row)
    if any(len(grouped[arm_id]) < fold_count for arm_id in arm_ids):
        raise ValueError("every arm must support every evaluation fold")
    rng = random.Random(seed)
    assignments: dict[str, int] = {}
    for arm_id in arm_ids:
        arm_rows = sorted(grouped[arm_id], key=lambda row: row.participant_id)
        rng.shuffle(arm_rows)
        for index, row in enumerate(arm_rows):
            assignments[row.participant_id] = index % fold_count
    return assignments


def _weighted_means(
    rows: Sequence[FallbackObservation], arm_ids: Sequence[str]
) -> dict[str, float]:
    grouped: dict[str, list[FallbackObservation]] = defaultdict(list)
    for row in rows:
        grouped[row.arm_id].append(row)
    means: dict[str, float] = {}
    for arm_id in arm_ids:
        arm_rows = grouped[arm_id]
        if not arm_rows:
            raise ValueError("every decision estimate requires every arm")
        total = fsum(row.weight for row in arm_rows)
        means[arm_id] = fsum(row.weight * row.utility for row in arm_rows) / total
    return means


def _choose(means: Mapping[str, float], arm_ids: Sequence[str]) -> str:
    arms = tuple(arm_ids)
    return max(arms, key=lambda arm_id: (means[arm_id], -arms.index(arm_id)))


def _regret(means: Mapping[str, float], selected: str) -> float:
    return max(means.values()) - means[selected]


def _fusion_means(
    synthetic_means: Mapping[str, float],
    pilot_rows: Sequence[FallbackObservation],
    *,
    arm_ids: Sequence[str],
    pseudocount: int,
) -> dict[str, float]:
    pilot_means = _weighted_means(pilot_rows, arm_ids)
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


def evaluate_human_fallback(
    observations: Sequence[FallbackObservation],
    *,
    arm_ids: Sequence[str],
    synthetic_means: Mapping[str, float],
    winner_votes: Mapping[str, int],
    budgets: Sequence[int],
    partitions: int,
    fold_count: int,
    seed: int,
    pseudocount: int,
    practical_tolerance: float,
) -> dict[str, Any]:
    """Evaluate nested human acquisition on disjoint repeated folds."""

    rows = _validate_observations(observations, arm_ids)
    arms = tuple(arm_ids)
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
                records[budget]["synthetic_only"].append(
                    {
                        "regret": synthetic_regret,
                        "exact": float(synthetic_choice == evaluation_best),
                        "practical": float(synthetic_regret <= practical_tolerance),
                        "delta_vs_synthetic": 0.0,
                    }
                )
                if budget == 0:
                    for policy in (
                        "synthetic_plus_balanced_fixed10",
                        "synthetic_plus_hedged_fixed10",
                    ):
                        records[budget][policy].append(
                            {
                                "regret": synthetic_regret,
                                "exact": float(synthetic_choice == evaluation_best),
                                "practical": float(
                                    synthetic_regret <= practical_tolerance
                                ),
                                "delta_vs_synthetic": 0.0,
                            }
                        )
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
                balanced_rows = [
                    row
                    for arm_id in arms
                    for row in pool_by_arm[arm_id][
                        : allocations["balanced"][arm_id]
                    ]
                ]
                hedged_rows = [
                    row
                    for arm_id in arms
                    for row in pool_by_arm[arm_id][
                        : allocations["hedged"][arm_id]
                    ]
                ]
                choices = {
                    "human_only_balanced": _choose(
                        _weighted_means(balanced_rows, arms), arms
                    ),
                    "synthetic_plus_balanced_fixed10": _choose(
                        _fusion_means(
                            synthetic_means,
                            balanced_rows,
                            arm_ids=arms,
                            pseudocount=pseudocount,
                        ),
                        arms,
                    ),
                    "synthetic_plus_hedged_fixed10": _choose(
                        _fusion_means(
                            synthetic_means,
                            hedged_rows,
                            arm_ids=arms,
                            pseudocount=pseudocount,
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
            if previous_budget is None:
                row["marginal_regret_reduction_per_human"] = None
            else:
                row["marginal_regret_reduction_per_human"] = (
                    previous_regret - row["mean_regret"]
                ) / (budget - previous_budget)
            previous_budget = budget
            previous_regret = row["mean_regret"]
    return {
        "budgets": list(budgets),
        "partitions": partitions,
        "fold_count": fold_count,
        "seed": seed,
        "pseudocount_per_arm": pseudocount,
        "pilot_evaluation_people_disjoint": True,
        "sampling_without_replacement": True,
        "nested_arm_prefixes_within_policy": True,
        "participant_rows_serialized": 0,
        "by_budget": result,
    }
