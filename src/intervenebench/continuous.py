"""Robust estimands and decision scoring for uncapped continuous outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
from statistics import median
from typing import Iterable, Mapping

from .evaluation import choose_best_arm, decision_regret, treatment_effects
from .schemas import OutcomeDirection


@dataclass(frozen=True, slots=True)
class ContinuousObservation:
    participant_id: str
    arm_id: str
    value: float | None


@dataclass(frozen=True, slots=True)
class ContinuousArmLocations:
    arm_locations: dict[str, float]
    valid_counts: dict[str, int]
    missing_counts: dict[str, int]
    arm_values: dict[str, tuple[float, ...]]


@dataclass(frozen=True, slots=True)
class ContinuousDecisionEvaluation:
    selected_arm_id: str
    human_best_arm_id: str
    correct_choice: bool
    regret: float
    practically_reliable: bool
    regret_unit: str
    normalized: bool
    human_arm_locations: dict[str, float]
    synthetic_arm_locations: dict[str, float]
    human_treatment_effects: dict[str, float]
    synthetic_treatment_effects: dict[str, float]


def continuous_arm_locations(
    observations: Iterable[ContinuousObservation],
    *,
    arm_ids: tuple[str, ...],
    missing_codes: tuple[float, ...],
    valid_lower_bound: float | None,
    valid_upper_bound: float | None,
    integer_only: bool,
    estimator: str,
) -> ContinuousArmLocations:
    """Apply a frozen numeric contract and estimate one robust location per arm."""

    if estimator not in {"mean", "median"}:
        raise ValueError("unsupported continuous estimator; expected mean or median")
    if len(arm_ids) < 2 or len(arm_ids) != len(set(arm_ids)):
        raise ValueError("arm_ids must contain at least two unique arms")
    if (
        valid_lower_bound is not None
        and valid_upper_bound is not None
        and valid_lower_bound >= valid_upper_bound
    ):
        raise ValueError("declared lower bound must be smaller than upper bound")
    missing = set(missing_codes)
    if len(missing) != len(missing_codes) or any(not isfinite(code) for code in missing):
        raise ValueError("missing codes must be unique finite values")

    values: dict[str, list[float]] = {arm_id: [] for arm_id in arm_ids}
    missing_counts = {arm_id: 0 for arm_id in arm_ids}
    participant_ids: set[str] = set()
    for observation in observations:
        if not observation.participant_id.strip():
            raise ValueError("participant_id must be non-empty")
        if observation.participant_id in participant_ids:
            raise ValueError("participants must appear exactly once in a between-subject task")
        participant_ids.add(observation.participant_id)
        if observation.arm_id not in values:
            raise ValueError("observation arm is absent from the declared action set")
        if observation.value is None:
            missing_counts[observation.arm_id] += 1
            continue
        if isinstance(observation.value, bool) or not isinstance(
            observation.value, (int, float)
        ):
            raise ValueError("continuous values must be numeric or null")
        value = float(observation.value)
        if not isfinite(value):
            raise ValueError("continuous values must be finite")
        if value in missing:
            missing_counts[observation.arm_id] += 1
            continue
        if integer_only and not value.is_integer():
            raise ValueError("continuous value violates the declared integer contract")
        if valid_lower_bound is not None and value < valid_lower_bound:
            raise ValueError("continuous value lies below the declared lower bound")
        if valid_upper_bound is not None and value > valid_upper_bound:
            raise ValueError("continuous value lies above the declared upper bound")
        values[observation.arm_id].append(value)

    if not participant_ids:
        raise ValueError("at least one continuous observation is required")
    empty = sorted(arm_id for arm_id, arm_values in values.items() if not arm_values)
    if empty:
        raise ValueError(f"no valid continuous observations for arms: {empty}")
    frozen_values = {arm_id: tuple(arm_values) for arm_id, arm_values in values.items()}
    if estimator == "mean":
        arm_locations = {
            arm_id: fsum(arm_values) / len(arm_values)
            for arm_id, arm_values in frozen_values.items()
        }
    else:
        arm_locations = {
            arm_id: float(median(arm_values))
            for arm_id, arm_values in frozen_values.items()
        }
    return ContinuousArmLocations(
        arm_locations=arm_locations,
        valid_counts={arm_id: len(arm_values) for arm_id, arm_values in values.items()},
        missing_counts=missing_counts,
        arm_values=frozen_values,
    )


def orient_continuous_locations(
    locations: Mapping[str, float], *, direction: OutcomeDirection
) -> dict[str, float]:
    if len(locations) < 2 or any(not isfinite(value) for value in locations.values()):
        raise ValueError("at least two finite arm locations are required")
    multiplier = 1.0 if direction is OutcomeDirection.HIGHER_IS_BETTER else -1.0
    return {arm_id: multiplier * float(value) for arm_id, value in locations.items()}


def evaluate_continuous_decision(
    *,
    human_locations: Mapping[str, float],
    synthetic_locations: Mapping[str, float],
    control_arm_id: str,
    direction: OutcomeDirection,
    practical_regret_tolerance: float,
    outcome_unit: str,
) -> ContinuousDecisionEvaluation:
    if set(human_locations) != set(synthetic_locations):
        raise ValueError("human and synthetic locations must cover the same arms")
    if not outcome_unit.strip():
        raise ValueError("outcome_unit must be non-empty")
    if not isfinite(practical_regret_tolerance) or practical_regret_tolerance < 0:
        raise ValueError("practical regret tolerance must be finite and non-negative")
    human_utility = orient_continuous_locations(human_locations, direction=direction)
    synthetic_utility = orient_continuous_locations(
        synthetic_locations, direction=direction
    )
    selected = choose_best_arm(synthetic_utility)
    human_best = choose_best_arm(human_utility)
    regret = decision_regret(human_utility, selected)
    return ContinuousDecisionEvaluation(
        selected_arm_id=selected,
        human_best_arm_id=human_best,
        correct_choice=selected == human_best,
        regret=regret,
        practically_reliable=regret <= practical_regret_tolerance,
        regret_unit=outcome_unit,
        normalized=False,
        human_arm_locations=dict(human_locations),
        synthetic_arm_locations=dict(synthetic_locations),
        human_treatment_effects=treatment_effects(
            human_utility, control_arm_id=control_arm_id
        ),
        synthetic_treatment_effects=treatment_effects(
            synthetic_utility, control_arm_id=control_arm_id
        ),
    )
