"""Scoped release decisions for behavioral-simulator evaluation evidence.

The gate intentionally separates autonomous intervention choice, research-stage
candidate screening, confidence-based abstention, and limited-human fallback.
Passing one scope never silently authorizes another.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class BehavioralEvaluationSummary:
    prospective_experiment_count: int
    normalized_experiment_count: int
    exact_choice_count: int
    exact_choice_random_tail_probability: float
    practically_reliable_count: int
    mean_normalized_regret: float
    worst_normalized_regret: float
    uniform_mean_normalized_regret: float
    uniform_regret_tail_probability: float
    control_mean_normalized_regret: float
    control_difference_interval: tuple[float, float]
    classical_mean_normalized_regret: float
    trust_ranking_better_than_random: bool
    validated_trust_threshold: bool
    human_fallback_improved: bool
    schema_valid_output_count: int
    planned_output_count: int


@dataclass(frozen=True, slots=True)
class ReleaseThresholds:
    autonomous_min_prospective_experiments: int = 12
    autonomous_min_normalized_experiments: int = 10
    screening_min_prospective_experiments: int = 5
    screening_min_normalized_experiments: int = 5
    maximum_mean_normalized_regret: float = 0.01
    maximum_worst_normalized_regret: float = 0.05
    maximum_random_tail_probability: float = 0.05
    control_noninferiority_margin: float = 0.01
    autonomous_minimum_schema_validity: float = 0.98
    screening_minimum_schema_validity: float = 0.95
    minimum_practical_reliability_rate: float = 0.90


def _probability(value: float, *, name: str) -> float:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return float(value)


def _nonnegative(value: float, *, name: str) -> float:
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _validate(summary: BehavioralEvaluationSummary) -> dict[str, float]:
    counts = (
        summary.prospective_experiment_count,
        summary.normalized_experiment_count,
        summary.exact_choice_count,
        summary.practically_reliable_count,
        summary.schema_valid_output_count,
        summary.planned_output_count,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
        raise ValueError("evaluation counts must be non-negative integers")
    if summary.prospective_experiment_count == 0:
        raise ValueError("release evaluation requires prospective experiments")
    if summary.normalized_experiment_count > summary.prospective_experiment_count:
        raise ValueError("normalized experiment count exceeds prospective count")
    if summary.exact_choice_count > summary.prospective_experiment_count:
        raise ValueError("exact-choice count exceeds prospective count")
    if summary.practically_reliable_count > summary.prospective_experiment_count:
        raise ValueError("practical-reliability count exceeds prospective count")
    if summary.planned_output_count <= 0:
        raise ValueError("planned output count must be positive")
    if summary.schema_valid_output_count > summary.planned_output_count:
        raise ValueError("schema-valid output count exceeds planned outputs")
    _probability(
        summary.exact_choice_random_tail_probability,
        name="exact_choice_random_tail_probability",
    )
    _probability(
        summary.uniform_regret_tail_probability,
        name="uniform_regret_tail_probability",
    )
    for name in (
        "mean_normalized_regret",
        "worst_normalized_regret",
        "uniform_mean_normalized_regret",
        "control_mean_normalized_regret",
        "classical_mean_normalized_regret",
    ):
        _nonnegative(float(getattr(summary, name)), name=name)
    interval = summary.control_difference_interval
    if (
        not isinstance(interval, tuple)
        or len(interval) != 2
        or not all(isfinite(value) for value in interval)
        or interval[0] > interval[1]
    ):
        raise ValueError("control_difference_interval is invalid")
    return {
        "schema_validity": summary.schema_valid_output_count
        / summary.planned_output_count,
        "practical_reliability_rate": summary.practically_reliable_count
        / summary.prospective_experiment_count,
    }


def _decision(passed: bool, reasons: list[str], *, success: str) -> dict[str, Any]:
    return {
        "decision": success if passed else "hold",
        "reasons": reasons,
    }


def evaluate_release_decision(
    summary: BehavioralEvaluationSummary,
    thresholds: ReleaseThresholds,
) -> dict[str, dict[str, Any]]:
    """Return independent release decisions for four operational scopes."""

    derived = _validate(summary)

    autonomous_failures: list[str] = []
    if summary.prospective_experiment_count < thresholds.autonomous_min_prospective_experiments:
        autonomous_failures.append("small prospective panel")
    if summary.normalized_experiment_count < thresholds.autonomous_min_normalized_experiments:
        autonomous_failures.append("insufficient normalized decision tasks")
    if summary.mean_normalized_regret > thresholds.maximum_mean_normalized_regret:
        autonomous_failures.append("mean regret exceeds release tolerance")
    if summary.worst_normalized_regret > thresholds.maximum_worst_normalized_regret:
        autonomous_failures.append("worst regret exceeds release tolerance")
    if summary.exact_choice_random_tail_probability > thresholds.maximum_random_tail_probability:
        autonomous_failures.append("exact choice remains chance-compatible")
    if summary.uniform_regret_tail_probability > thresholds.maximum_random_tail_probability:
        autonomous_failures.append("uniform-action regret gate is not established")
    if summary.control_difference_interval[1] >= thresholds.control_noninferiority_margin:
        autonomous_failures.append("control noninferiority is not established")
    if derived["practical_reliability_rate"] < thresholds.minimum_practical_reliability_rate:
        autonomous_failures.append("practical reliability is below threshold")
    if derived["schema_validity"] < thresholds.autonomous_minimum_schema_validity:
        autonomous_failures.append("schema-validity rate is below autonomous threshold")

    screening_failures: list[str] = []
    if summary.prospective_experiment_count < thresholds.screening_min_prospective_experiments:
        screening_failures.append("too few prospective experiments for screening")
    if summary.normalized_experiment_count < thresholds.screening_min_normalized_experiments:
        screening_failures.append("too few normalized tasks for screening")
    if summary.mean_normalized_regret > thresholds.maximum_mean_normalized_regret:
        screening_failures.append("mean regret exceeds screening tolerance")
    if summary.worst_normalized_regret > thresholds.maximum_worst_normalized_regret:
        screening_failures.append("worst regret exceeds screening tolerance")
    if summary.mean_normalized_regret >= summary.uniform_mean_normalized_regret:
        screening_failures.append("simulator does not improve on uniform action")
    if derived["practical_reliability_rate"] < thresholds.minimum_practical_reliability_rate:
        screening_failures.append("practical reliability is below threshold")
    if derived["schema_validity"] < thresholds.screening_minimum_schema_validity:
        screening_failures.append("schema-validity rate is below screening threshold")

    trust_failures = []
    if not summary.trust_ranking_better_than_random:
        trust_failures.append("trust ranking did not beat random abstention")
    if not summary.validated_trust_threshold:
        trust_failures.append("no prospectively validated trust threshold")

    fallback_failures = []
    if not summary.human_fallback_improved:
        fallback_failures.append("tested limited-human fallback did not improve decisions")

    return {
        "autonomous_intervention_selection": _decision(
            not autonomous_failures,
            autonomous_failures or ["all autonomous release gates passed"],
            success="pass",
        ),
        "candidate_screening": _decision(
            not screening_failures,
            screening_failures or ["low-regret screening criteria passed"],
            success="limited_research_use",
        ),
        "confidence_based_abstention": _decision(
            not trust_failures,
            trust_failures or ["trust ranking and threshold validated"],
            success="pass",
        ),
        "small_sample_human_fallback": _decision(
            not fallback_failures,
            fallback_failures or ["fallback improved decisions under the frozen evaluator"],
            success="pass",
        ),
    }
