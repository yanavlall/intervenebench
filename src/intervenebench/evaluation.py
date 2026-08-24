"""Treatment-effect, recommendation, and regret calculations."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
from typing import Iterable, Mapping

from .schemas import OutcomeDirection


@dataclass(frozen=True, slots=True)
class Observation:
    participant_id: str
    arm_id: str
    value: float


@dataclass(frozen=True, slots=True)
class WeightedObservation:
    participant_id: str
    arm_id: str
    value: float
    weight: float


@dataclass(frozen=True, slots=True)
class StratifiedWeightedObservation:
    participant_id: str
    arm_id: str
    stratum_id: str
    value: float
    weight: float


@dataclass(frozen=True, slots=True)
class CategoricalChoiceIndicator:
    """One option row from a flattened single-choice outcome."""

    participant_id: str
    arm_id: str
    option_id: str
    selected: bool
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class DecisionEvaluation:
    selected_arm_id: str
    human_best_arm_id: str
    correct_choice: bool
    regret: float
    practically_reliable: bool
    human_arm_means: dict[str, float]
    synthetic_arm_means: dict[str, float]
    human_treatment_effects: dict[str, float]
    synthetic_treatment_effects: dict[str, float]


def normalize_utility(
    value: float,
    *,
    lower: float,
    upper: float,
    direction: OutcomeDirection,
) -> float:
    if not all(isfinite(number) for number in (value, lower, upper)):
        raise ValueError("value and bounds must be finite")
    if lower >= upper:
        raise ValueError("lower must be smaller than upper")
    if value < lower or value > upper:
        raise ValueError("value lies outside declared questionnaire bounds")
    if direction is OutcomeDirection.HIGHER_IS_BETTER:
        return (value - lower) / (upper - lower)
    if direction is OutcomeDirection.LOWER_IS_BETTER:
        return (upper - value) / (upper - lower)
    raise ValueError(f"unsupported direction: {direction}")


def arm_means(observations: Iterable[Observation]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for observation in observations:
        if not isfinite(observation.value):
            raise ValueError("observation values must be finite")
        values.setdefault(observation.arm_id, []).append(observation.value)
    if not values:
        raise ValueError("at least one observation is required")
    return {
        arm_id: fsum(arm_values) / len(arm_values)
        for arm_id, arm_values in values.items()
    }


def weighted_arm_means(
    observations: Iterable[WeightedObservation],
) -> dict[str, float]:
    """Compute arm-specific Hajek means from positive finite source weights."""

    weighted_values: dict[str, list[float]] = {}
    weights: dict[str, list[float]] = {}
    for observation in observations:
        if not isfinite(observation.value):
            raise ValueError("observation values must be finite")
        if not isfinite(observation.weight) or observation.weight <= 0.0:
            raise ValueError("observation weights must be positive and finite")
        weighted_values.setdefault(observation.arm_id, []).append(
            observation.value * observation.weight
        )
        weights.setdefault(observation.arm_id, []).append(observation.weight)
    if not weighted_values:
        raise ValueError("at least one weighted observation is required")
    return {
        arm_id: fsum(weighted_values[arm_id]) / fsum(weights[arm_id])
        for arm_id in weighted_values
    }


def weighted_standardized_arm_means(
    observations: Iterable[StratifiedWeightedObservation],
    *,
    stratum_weights: Mapping[str, float],
) -> dict[str, float]:
    """Compute Hajek cell means, then standardize each arm to frozen strata."""

    if len(stratum_weights) < 2:
        raise ValueError("at least two nuisance strata are required")
    if any(
        not isinstance(stratum_id, str)
        or not stratum_id.strip()
        or not isfinite(weight)
        or weight <= 0.0
        for stratum_id, weight in stratum_weights.items()
    ):
        raise ValueError("stratum IDs and weights must be valid")
    if abs(fsum(stratum_weights.values()) - 1.0) > 1e-9:
        raise ValueError("stratum weights must sum to one")

    numerators: dict[tuple[str, str], list[float]] = {}
    denominators: dict[tuple[str, str], list[float]] = {}
    arms: set[str] = set()
    for observation in observations:
        if observation.stratum_id not in stratum_weights:
            raise ValueError("observation uses an undeclared nuisance stratum")
        if not isfinite(observation.value):
            raise ValueError("observation values must be finite")
        if not isfinite(observation.weight) or observation.weight <= 0.0:
            raise ValueError("observation weights must be positive and finite")
        key = (observation.arm_id, observation.stratum_id)
        arms.add(observation.arm_id)
        numerators.setdefault(key, []).append(observation.value * observation.weight)
        denominators.setdefault(key, []).append(observation.weight)
    if len(arms) < 2:
        raise ValueError("at least two arms are required")
    expected = {(arm_id, stratum_id) for arm_id in arms for stratum_id in stratum_weights}
    if set(numerators) != expected:
        raise ValueError("every arm must contain every declared nuisance stratum")
    cell_means = {
        key: fsum(numerators[key]) / fsum(denominators[key]) for key in expected
    }
    return {
        arm_id: fsum(
            stratum_weights[stratum_id] * cell_means[(arm_id, stratum_id)]
            for stratum_id in stratum_weights
        )
        for arm_id in sorted(arms)
    }


def weighted_categorical_choice_arm_means(
    indicators: Iterable[CategoricalChoiceIndicator],
    *,
    option_utilities: Mapping[str, float],
) -> dict[str, float]:
    """Collapse a complete one-hot choice grid and compute Hajek arm utility.

    Every participant must have every declared option exactly once, one and only
    one option selected, a single arm, and one consistent positive weight.
    """

    if len(option_utilities) < 2 or any(
        not isinstance(option_id, str)
        or not option_id.strip()
        or not isfinite(utility)
        or not 0.0 <= utility <= 1.0
        for option_id, utility in option_utilities.items()
    ):
        raise ValueError("option utilities must define at least two valid options")
    by_participant: dict[str, list[CategoricalChoiceIndicator]] = {}
    for indicator in indicators:
        if not indicator.participant_id.strip() or not indicator.arm_id.strip():
            raise ValueError("participant and arm IDs must be non-empty")
        if indicator.option_id not in option_utilities:
            raise ValueError("indicator uses an undeclared option")
        if not isinstance(indicator.selected, bool):
            raise ValueError("selected must be boolean")
        if not isfinite(indicator.weight) or indicator.weight <= 0.0:
            raise ValueError("choice weights must be positive and finite")
        by_participant.setdefault(indicator.participant_id, []).append(indicator)
    if not by_participant:
        raise ValueError("at least one participant choice is required")

    observations: list[WeightedObservation] = []
    expected_options = set(option_utilities)
    for participant_id, rows in by_participant.items():
        if {row.option_id for row in rows} != expected_options or len(rows) != len(
            expected_options
        ):
            raise ValueError("each participant must contain every option exactly once")
        arms = {row.arm_id for row in rows}
        weights = {row.weight for row in rows}
        selected = [row for row in rows if row.selected]
        if len(arms) != 1 or len(weights) != 1 or len(selected) != 1:
            raise ValueError(
                "each participant must have one arm, one weight, and one selected option"
            )
        observations.append(
            WeightedObservation(
                participant_id=participant_id,
                arm_id=next(iter(arms)),
                value=option_utilities[selected[0].option_id],
                weight=next(iter(weights)),
            )
        )
    if len({observation.arm_id for observation in observations}) < 2:
        raise ValueError("at least two choice arms are required")
    return weighted_arm_means(observations)


def treatment_effects(
    means: Mapping[str, float], *, control_arm_id: str
) -> dict[str, float]:
    if control_arm_id not in means:
        raise ValueError("control arm is missing from arm means")
    if len(means) < 2:
        raise ValueError("at least two arm means are required")
    control_mean = means[control_arm_id]
    return {
        arm_id: mean - control_mean
        for arm_id, mean in means.items()
        if arm_id != control_arm_id
    }


def choose_best_arm(
    means: Mapping[str, float], *, tie_preferred_arm_id: str | None = None
) -> str:
    """Choose maximum mean with a frozen lexicographic arm-ID tie rule."""

    if not means:
        raise ValueError("at least one arm mean is required")
    if any(not isfinite(value) for value in means.values()):
        raise ValueError("arm means must be finite")
    best_value = max(means.values())
    tied = tuple(arm_id for arm_id, value in means.items() if value == best_value)
    if tie_preferred_arm_id is not None:
        if tie_preferred_arm_id not in means:
            raise ValueError("tie-preferred arm is absent from arm means")
        if tie_preferred_arm_id in tied:
            return tie_preferred_arm_id
    return min(tied)


def decision_regret(human_means: Mapping[str, float], selected_arm_id: str) -> float:
    if selected_arm_id not in human_means:
        raise ValueError("selected arm is absent from human arm means")
    regret = max(human_means.values()) - human_means[selected_arm_id]
    if regret < -1e-12:
        raise AssertionError("decision regret cannot be negative")
    return max(0.0, regret)


def evaluate_decision(
    *,
    human_means: Mapping[str, float],
    synthetic_means: Mapping[str, float],
    control_arm_id: str,
    practical_regret_tolerance: float,
) -> DecisionEvaluation:
    if set(human_means) != set(synthetic_means):
        raise ValueError("human and synthetic means must cover the same arms")
    if not 0.0 <= practical_regret_tolerance <= 1.0:
        raise ValueError("practical regret tolerance must be in [0, 1]")
    selected = choose_best_arm(synthetic_means)
    human_best = choose_best_arm(human_means)
    regret = decision_regret(human_means, selected)
    return DecisionEvaluation(
        selected_arm_id=selected,
        human_best_arm_id=human_best,
        correct_choice=selected == human_best,
        regret=regret,
        practically_reliable=regret <= practical_regret_tolerance,
        human_arm_means=dict(human_means),
        synthetic_arm_means=dict(synthetic_means),
        human_treatment_effects=treatment_effects(
            human_means, control_arm_id=control_arm_id
        ),
        synthetic_treatment_effects=treatment_effects(
            synthetic_means, control_arm_id=control_arm_id
        ),
    )
