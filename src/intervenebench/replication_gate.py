"""Aggregate-only completion gates for the independent replication panel."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import ceil, fsum, isfinite, prod, sqrt
from random import Random
from statistics import fmean
from typing import Any, Mapping, Sequence

from .experiment_statistics import paired_experiment_cluster_bootstrap


FLOAT_TOLERANCE = 1e-12
PRACTICAL_IMPROVEMENT = 0.01


@dataclass(frozen=True, slots=True)
class ReplicationTaskScore:
    experiment_id: str
    fielding_cluster_id: str
    paradigm_group: str
    source_stratum: str
    outcome_family: str
    primary_regret: float
    default_regret: float
    classical_regret: float
    arm_regrets: Mapping[str, float]


def _finite_normalized(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) < -FLOAT_TOLERANCE
        or float(value) > 1.0 + FLOAT_TOLERANCE
    ):
        raise ValueError(f"{name} must be a finite normalized regret in [0, 1]")
    return min(1.0, max(0.0, float(value)))


def _nonempty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validated_tasks(
    tasks: Sequence[ReplicationTaskScore],
) -> list[ReplicationTaskScore]:
    if len(tasks) < 12:
        raise ValueError("replication scoring requires at least 12 experiments")
    identifiers: set[str] = set()
    fieldings: set[str] = set()
    paradigms: set[str] = set()
    result: list[ReplicationTaskScore] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, ReplicationTaskScore):
            raise ValueError("replication task scores must use the typed schema")
        experiment_id = _nonempty(task.experiment_id, name="experiment_id")
        fielding = _nonempty(task.fielding_cluster_id, name="fielding_cluster_id")
        paradigm = _nonempty(task.paradigm_group, name="paradigm_group")
        _nonempty(task.source_stratum, name="source_stratum")
        _nonempty(task.outcome_family, name="outcome_family")
        if experiment_id in identifiers:
            raise ValueError("experiment IDs must be unique")
        if fielding in fieldings:
            raise ValueError("fielding clusters must be unique")
        if paradigm in paradigms:
            raise ValueError("paradigm groups must be unique")
        identifiers.add(experiment_id)
        fieldings.add(fielding)
        paradigms.add(paradigm)

        primary = _finite_normalized(task.primary_regret, name="primary_regret")
        default = _finite_normalized(task.default_regret, name="default_regret")
        classical = _finite_normalized(task.classical_regret, name="classical_regret")
        if not isinstance(task.arm_regrets, Mapping) or len(task.arm_regrets) < 2:
            raise ValueError("arm_regrets must contain at least two admissible arms")
        arms: dict[str, float] = {}
        for arm_id, raw_regret in task.arm_regrets.items():
            arm = _nonempty(arm_id, name=f"task[{index}].arm_id")
            arms[arm] = _finite_normalized(
                raw_regret, name=f"task[{index}].arm_regrets[{arm}]"
            )
        if not any(abs(primary - regret) <= FLOAT_TOLERANCE for regret in arms.values()):
            raise ValueError("primary_regret must correspond to an admissible arm")
        result.append(
            ReplicationTaskScore(
                experiment_id=experiment_id,
                fielding_cluster_id=fielding,
                paradigm_group=paradigm,
                source_stratum=task.source_stratum,
                outcome_family=task.outcome_family,
                primary_regret=primary,
                default_regret=default,
                classical_regret=classical,
                arm_regrets=arms,
            )
        )

    required_socsci = max(8, ceil(0.625 * len(result)))
    observed_socsci = sum(row.source_stratum == "socsci210" for row in result)
    if observed_socsci < required_socsci:
        raise ValueError(
            "SocSci210 must remain primary in the independent replication panel"
        )
    return result


def _wilson_upper(successes: int, trials: int, *, z: float = 1.6448536269514722) -> float:
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = proportion + z * z / (2.0 * trials)
    radius = z * sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    )
    return min(1.0, (centre + radius) / denominator)


def uniform_random_action_tail(
    arm_regrets_by_experiment: Mapping[str, Mapping[str, float]],
    *,
    observed_mean_regret: float,
    seed: int,
    monte_carlo_replicates: int = 500_000,
    exact_combination_limit: int = 2_000_000,
) -> dict[str, Any]:
    """P(uniform random actions have mean regret no larger than observed)."""

    observed = _finite_normalized(
        observed_mean_regret, name="observed_mean_regret"
    )
    if not arm_regrets_by_experiment:
        raise ValueError("uniform random-action test requires experiments")
    validated: list[tuple[str, tuple[float, ...]]] = []
    for experiment_id in sorted(arm_regrets_by_experiment):
        _nonempty(experiment_id, name="experiment_id")
        raw_arms = arm_regrets_by_experiment[experiment_id]
        if not isinstance(raw_arms, Mapping) or len(raw_arms) < 2:
            raise ValueError("each random-action experiment requires at least two arms")
        values = tuple(
            _finite_normalized(value, name="arm_regret")
            for _, value in sorted(raw_arms.items())
        )
        validated.append((experiment_id, values))
    combination_count = prod(len(values) for _, values in validated)
    favorable = 0
    threshold = observed + FLOAT_TOLERANCE
    if combination_count <= exact_combination_limit:
        for regrets in product(*(values for _, values in validated)):
            favorable += fmean(regrets) <= threshold
        probability = favorable / combination_count
        return {
            "method": "exact_enumeration",
            "experiment_count": len(validated),
            "combination_count": combination_count,
            "enumeration_or_replicate_count": combination_count,
            "favorable_count": favorable,
            "tail_probability": probability,
            "conservative_tail_probability": probability,
            "seed": None,
        }
    if (
        isinstance(monte_carlo_replicates, bool)
        or not isinstance(monte_carlo_replicates, int)
        or monte_carlo_replicates <= 0
    ):
        raise ValueError("monte_carlo_replicates must be a positive integer")
    rng = Random(seed)
    for _ in range(monte_carlo_replicates):
        total = fsum(values[rng.randrange(len(values))] for _, values in validated)
        favorable += total / len(validated) <= threshold
    probability = favorable / monte_carlo_replicates
    return {
        "method": "deterministic_monte_carlo",
        "experiment_count": len(validated),
        "combination_count": combination_count,
        "enumeration_or_replicate_count": monte_carlo_replicates,
        "favorable_count": favorable,
        "tail_probability": probability,
        "conservative_tail_probability": _wilson_upper(
            favorable, monte_carlo_replicates
        ),
        "seed": seed,
    }


def _exact_sign_flip(differences: Mapping[str, float]) -> dict[str, Any]:
    identifiers = tuple(sorted(differences))
    observed = fmean(differences.values())
    nonzero = [
        abs(differences[identifier])
        for identifier in identifiers
        if abs(differences[identifier]) > FLOAT_TOLERANCE
    ]
    if not nonzero:
        return {
            "non_tied_count": 0,
            "enumeration_count": 1,
            "one_sided_probability": 1.0,
        }
    favorable = 0
    enumeration_count = 2 ** len(nonzero)
    for signs in product((-1.0, 1.0), repeat=len(nonzero)):
        randomized = fsum(
            sign * magnitude
            for sign, magnitude in zip(signs, nonzero, strict=True)
        ) / len(identifiers)
        favorable += randomized <= observed + FLOAT_TOLERANCE
    return {
        "non_tied_count": len(nonzero),
        "enumeration_count": enumeration_count,
        "one_sided_probability": favorable / enumeration_count,
    }


def _leave_one_out(differences: Mapping[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for omitted in sorted(differences):
        result[omitted] = fmean(
            value
            for experiment_id, value in differences.items()
            if experiment_id != omitted
        )
    return result


def _trimmed_mean(values: Sequence[float], proportion: float = 0.10) -> float:
    ordered = sorted(values)
    trim = int(len(ordered) * proportion)
    retained = ordered[trim : len(ordered) - trim] if trim else ordered
    return fmean(retained)


def _comparison(
    candidate: Mapping[str, float],
    reference: Mapping[str, float],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    minimum = paired_experiment_cluster_bootstrap(
        candidate,
        reference,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        confidence_level=0.90,
    )
    strong = paired_experiment_cluster_bootstrap(
        candidate,
        reference,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        confidence_level=0.95,
    )
    return {
        "candidate_mean": minimum.candidate_mean,
        "reference_mean": minimum.reference_mean,
        "mean_difference": minimum.mean_difference,
        "one_sided_95_interval_via_central_90": list(
            minimum.difference_confidence_interval
        ),
        "two_sided_95_interval": list(strong.difference_confidence_interval),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
    }


def evaluate_replication_gate(
    tasks: Sequence[ReplicationTaskScore],
    *,
    bootstrap_replicates: int = 50_000,
    bootstrap_seed: int,
    uniform_monte_carlo_replicates: int = 500_000,
    uniform_seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate all frozen replication gates from one aggregate row per task."""

    rows = _validated_tasks(tasks)
    identifiers = [row.experiment_id for row in rows]
    primary = {row.experiment_id: row.primary_regret for row in rows}
    uniform = {
        row.experiment_id: fmean(row.arm_regrets.values()) for row in rows
    }
    default = {row.experiment_id: row.default_regret for row in rows}
    classical = {row.experiment_id: row.classical_regret for row in rows}
    differences = {
        experiment_id: primary[experiment_id] - uniform[experiment_id]
        for experiment_id in identifiers
    }
    uniform_comparison = _comparison(
        primary,
        uniform,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    default_comparison = _comparison(
        primary,
        default,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed + 1,
    )
    classical_comparison = _comparison(
        primary,
        classical,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed + 2,
    )
    sign_flip = _exact_sign_flip(differences)
    loo = _leave_one_out(differences)
    random_tail = uniform_random_action_tail(
        {row.experiment_id: row.arm_regrets for row in rows},
        observed_mean_regret=fmean(primary.values()),
        seed=uniform_seed if uniform_seed is not None else bootstrap_seed + 3,
        monte_carlo_replicates=uniform_monte_carlo_replicates,
    )
    min_ci = uniform_comparison["one_sided_95_interval_via_central_90"]
    strong_ci = uniform_comparison["two_sided_95_interval"]
    default_min_ci = default_comparison["one_sided_95_interval_via_central_90"]
    classical_min_ci = classical_comparison[
        "one_sided_95_interval_via_central_90"
    ]
    minimum_criteria = {
        "experiment_count_at_least_12": len(rows) >= 12,
        "mean_improvement_at_least_0_01": uniform_comparison["mean_difference"]
        <= -PRACTICAL_IMPROVEMENT + FLOAT_TOLERANCE,
        "one_sided_95_upper_below_zero": min_ci[1] < 0.0,
        "uniform_random_action_tail_at_most_0_05": random_tail[
            "conservative_tail_probability"
        ]
        <= 0.05,
        "all_leave_one_experiment_means_negative": all(
            value < 0.0 for value in loo.values()
        ),
        "all_leave_one_paradigm_means_negative": all(
            value < 0.0 for value in loo.values()
        ),
        "sign_flip_at_most_0_10": sign_flip["one_sided_probability"] <= 0.10,
        "default_noninferiority_point": default_comparison["mean_difference"]
        <= 0.0,
        "default_noninferiority_upper_below_0_01": default_min_ci[1] < 0.01,
        "classical_noninferiority_point": classical_comparison["mean_difference"]
        <= 0.0,
        "classical_noninferiority_upper_below_0_01": classical_min_ci[1] < 0.01,
    }
    minimum_passed = all(minimum_criteria.values())
    socsci_differences = [
        differences[row.experiment_id]
        for row in rows
        if row.source_stratum == "socsci210"
    ]
    strong_criteria = {
        "minimum_gate_passed": minimum_passed,
        "experiment_count_at_least_16": len(rows) >= 16,
        "two_sided_95_upper_below_zero": strong_ci[1] < 0.0,
        "sign_flip_at_most_0_05": sign_flip["one_sided_probability"] <= 0.05,
        "ten_percent_trimmed_mean_at_most_minus_0_01": _trimmed_mean(
            list(differences.values())
        )
        <= -PRACTICAL_IMPROVEMENT + FLOAT_TOLERANCE,
        "socsci_only_mean_negative": fmean(socsci_differences) < 0.0,
        "default_superiority_upper_below_zero": default_comparison[
            "two_sided_95_interval"
        ][1]
        < 0.0,
        "classical_superiority_upper_below_zero": classical_comparison[
            "two_sided_95_interval"
        ][1]
        < 0.0,
    }
    strong_passed = all(strong_criteria.values())

    if strong_passed:
        classification = "strong_positive_replication"
    elif minimum_passed:
        classification = "bounded_positive_replication"
    elif min_ci[0] > 0.0:
        classification = "evidence_of_harm"
    elif min_ci[0] > -PRACTICAL_IMPROVEMENT:
        classification = "definitive_no_practically_meaningful_uniform_benefit"
    elif uniform_comparison["mean_difference"] >= 0.0 or any(
        value >= 0.0 for value in loo.values()
    ):
        classification = "non_replication"
    else:
        classification = "inconclusive"

    return {
        "schema_version": "intervenebench.independent_replication_gate_result.v1",
        "panel": {
            "experiment_count": len(rows),
            "fielding_count": len({row.fielding_cluster_id for row in rows}),
            "paradigm_count": len({row.paradigm_group for row in rows}),
            "socsci210_count": len(socsci_differences),
            "behavioral_or_consequential_count": sum(
                row.outcome_family in {"behavioral", "consequential"} for row in rows
            ),
            "claim_scope": (
                "cross_outcome_decision_replication"
                if sum(
                    row.outcome_family in {"behavioral", "consequential"}
                    for row in rows
                )
                >= 4
                else "attitudinal_messaging_decision_replication"
            ),
        },
        "primary_uniform_comparison": {
            **uniform_comparison,
            "difference_by_experiment": differences,
            "leave_one_experiment_out_mean": loo,
            "leave_one_paradigm_out_mean": loo,
            "ten_percent_trimmed_mean_difference": _trimmed_mean(
                list(differences.values())
            ),
            "socsci_only_mean_difference": fmean(socsci_differences),
            "sign_flip": sign_flip,
            "uniform_random_action_tail": random_tail,
        },
        "default_comparison": default_comparison,
        "classical_comparison": classical_comparison,
        "minimum_replication": {
            "passed": minimum_passed,
            "criteria": minimum_criteria,
        },
        "strong_replication": {
            "passed": strong_passed,
            "criteria": strong_criteria,
        },
        "completion_classification": classification,
    }
