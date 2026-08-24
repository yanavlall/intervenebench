"""Experiment-level selective-decision and diagnostic-ranking metrics.

This module consumes scored experiment aggregates. It does not fit a trust model
or inspect participant rows. Confidence scores must already have been computed
and frozen without target human outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class SelectiveDecisionRecord:
    experiment_id: str
    confidence: float
    regret: float
    exact_correct: bool
    practically_reliable: bool


@dataclass(frozen=True, slots=True)
class RiskCoveragePoint:
    covered_experiment_count: int
    coverage: float
    covered_experiment_ids: tuple[str, ...]
    mean_regret: float
    exact_choice_rate: float
    practical_reliability_rate: float


@dataclass(frozen=True, slots=True)
class BinaryRankingMetrics:
    status: str
    positive_count: int
    negative_count: int
    minimum_class_count: int
    auroc: float | None
    average_precision: float | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class SelectiveDecisionSummary:
    experiment_count: int
    risk_coverage_curve: tuple[RiskCoveragePoint, ...]
    discrete_aurc: float
    random_abstention_expected_aurc: float
    excess_aurc_vs_random: float
    exact_choice_ranking: BinaryRankingMetrics
    practical_reliability_ranking: BinaryRankingMetrics


def _validate_minimum_class_count(minimum_class_count: int) -> None:
    if (
        isinstance(minimum_class_count, bool)
        or not isinstance(minimum_class_count, int)
        or minimum_class_count <= 0
    ):
        raise ValueError("minimum_class_count must be a positive integer")


def _validated_binary_inputs(
    confidence_by_experiment: Mapping[str, float],
    label_by_experiment: Mapping[str, bool],
) -> tuple[dict[str, float], dict[str, bool]]:
    if not confidence_by_experiment:
        raise ValueError("binary ranking metrics require at least one experiment")
    if set(confidence_by_experiment) != set(label_by_experiment):
        raise ValueError("confidence scores and labels must cover the same experiments")
    confidence: dict[str, float] = {}
    labels: dict[str, bool] = {}
    for experiment_id, raw_score in confidence_by_experiment.items():
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise ValueError("experiment IDs must be non-empty strings")
        if (
            isinstance(raw_score, bool)
            or not isinstance(raw_score, (int, float))
            or not isfinite(float(raw_score))
        ):
            raise ValueError("confidence scores must be finite numbers")
        label = label_by_experiment[experiment_id]
        if not isinstance(label, bool):
            raise ValueError("binary reliability labels must be boolean")
        confidence[experiment_id] = float(raw_score)
        labels[experiment_id] = label
    return confidence, labels


def _pairwise_auroc(
    confidence: Mapping[str, float], labels: Mapping[str, bool]
) -> float:
    positives = [key for key, label in labels.items() if label]
    negatives = [key for key, label in labels.items() if not label]
    concordance = 0.0
    for positive in positives:
        for negative in negatives:
            if confidence[positive] > confidence[negative]:
                concordance += 1.0
            elif confidence[positive] == confidence[negative]:
                concordance += 0.5
    return concordance / (len(positives) * len(negatives))


def _threshold_average_precision(
    confidence: Mapping[str, float], labels: Mapping[str, bool]
) -> float:
    positive_count = sum(labels.values())
    true_positives = 0
    selected = 0
    average_precision = 0.0
    for score in sorted(set(confidence.values()), reverse=True):
        group = [key for key, value in confidence.items() if value == score]
        group_positives = sum(labels[key] for key in group)
        true_positives += group_positives
        selected += len(group)
        recall_increment = group_positives / positive_count
        average_precision += recall_increment * (true_positives / selected)
    return average_precision


def gated_binary_ranking_metrics(
    confidence_by_experiment: Mapping[str, float],
    label_by_experiment: Mapping[str, bool],
    *,
    minimum_class_count: int = 3,
) -> BinaryRankingMetrics:
    """Compute AUROC/AP only when both experiment-level classes are supported."""

    _validate_minimum_class_count(minimum_class_count)
    confidence, labels = _validated_binary_inputs(
        confidence_by_experiment, label_by_experiment
    )
    positive_count = sum(labels.values())
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return BinaryRankingMetrics(
            status="not_estimable",
            positive_count=positive_count,
            negative_count=negative_count,
            minimum_class_count=minimum_class_count,
            auroc=None,
            average_precision=None,
            reason="binary ranking requires both classes at the experiment level",
        )
    if min(positive_count, negative_count) < minimum_class_count:
        return BinaryRankingMetrics(
            status="not_estimable",
            positive_count=positive_count,
            negative_count=negative_count,
            minimum_class_count=minimum_class_count,
            auroc=None,
            average_precision=None,
            reason=(
                "experiment-level label support is below "
                f"minimum_class_count={minimum_class_count}"
            ),
        )
    return BinaryRankingMetrics(
        status="estimable",
        positive_count=positive_count,
        negative_count=negative_count,
        minimum_class_count=minimum_class_count,
        auroc=_pairwise_auroc(confidence, labels),
        average_precision=_threshold_average_precision(confidence, labels),
        reason=None,
    )


def _validated_records(
    records: Sequence[SelectiveDecisionRecord],
) -> tuple[SelectiveDecisionRecord, ...]:
    if not records:
        raise ValueError("selective-decision analysis requires at least one experiment")
    experiment_ids: set[str] = set()
    validated: list[SelectiveDecisionRecord] = []
    for record in records:
        if not isinstance(record, SelectiveDecisionRecord):
            raise ValueError("records must be SelectiveDecisionRecord objects")
        if not record.experiment_id.strip():
            raise ValueError("experiment IDs must be non-empty strings")
        if record.experiment_id in experiment_ids:
            raise ValueError("experiment IDs must be unique")
        if not isfinite(record.confidence):
            raise ValueError("confidence scores must be finite")
        if not isfinite(record.regret) or record.regret < 0.0:
            raise ValueError("decision regret must be finite and non-negative")
        if not isinstance(record.exact_correct, bool) or not isinstance(
            record.practically_reliable, bool
        ):
            raise ValueError("reliability labels must be boolean")
        experiment_ids.add(record.experiment_id)
        validated.append(record)
    return tuple(validated)


def selective_decision_summary(
    records: Sequence[SelectiveDecisionRecord],
    *,
    minimum_class_count: int = 3,
) -> SelectiveDecisionSummary:
    """Build a discrete risk-coverage curve over independent experiments.

    Confidence ties are ordered by experiment ID only to make finite-coverage
    artifacts deterministic. Ranking metrics treat equal confidence as a tie.
    The discrete AURC is the arithmetic mean of prefix risks at coverages
    ``1/n, ..., n/n``. Under random abstention its expectation is the full-set
    mean regret at every coverage.
    """

    _validate_minimum_class_count(minimum_class_count)
    validated = _validated_records(records)
    ordered = sorted(
        validated, key=lambda record: (-record.confidence, record.experiment_id)
    )
    curve: list[RiskCoveragePoint] = []
    for count in range(1, len(ordered) + 1):
        covered = ordered[:count]
        curve.append(
            RiskCoveragePoint(
                covered_experiment_count=count,
                coverage=count / len(ordered),
                covered_experiment_ids=tuple(
                    record.experiment_id for record in covered
                ),
                mean_regret=fsum(record.regret for record in covered) / count,
                exact_choice_rate=(
                    fsum(float(record.exact_correct) for record in covered) / count
                ),
                practical_reliability_rate=(
                    fsum(float(record.practically_reliable) for record in covered)
                    / count
                ),
            )
        )
    discrete_aurc = fsum(point.mean_regret for point in curve) / len(curve)
    random_aurc = fsum(record.regret for record in validated) / len(validated)
    confidence = {
        record.experiment_id: record.confidence for record in validated
    }
    exact = {
        record.experiment_id: record.exact_correct for record in validated
    }
    practical = {
        record.experiment_id: record.practically_reliable for record in validated
    }
    return SelectiveDecisionSummary(
        experiment_count=len(validated),
        risk_coverage_curve=tuple(curve),
        discrete_aurc=discrete_aurc,
        random_abstention_expected_aurc=random_aurc,
        excess_aurc_vs_random=discrete_aurc - random_aurc,
        exact_choice_ranking=gated_binary_ranking_metrics(
            confidence, exact, minimum_class_count=minimum_class_count
        ),
        practical_reliability_ranking=gated_binary_ranking_metrics(
            confidence, practical, minimum_class_count=minimum_class_count
        ),
    )

