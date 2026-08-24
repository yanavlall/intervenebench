from __future__ import annotations

import pytest

from intervenebench.selective_decision import (
    SelectiveDecisionRecord,
    gated_binary_ranking_metrics,
    selective_decision_summary,
)


def _records() -> tuple[SelectiveDecisionRecord, ...]:
    return (
        SelectiveDecisionRecord("exp-a", 0.9, 0.00, True, True),
        SelectiveDecisionRecord("exp-b", 0.8, 0.10, False, False),
        SelectiveDecisionRecord("exp-c", 0.7, 0.20, False, False),
        SelectiveDecisionRecord("exp-d", 0.6, 0.00, True, True),
    )


def test_risk_coverage_is_experiment_level_deterministic_and_exact() -> None:
    result = selective_decision_summary(_records(), minimum_class_count=1)
    assert result.experiment_count == 4
    assert [point.coverage for point in result.risk_coverage_curve] == pytest.approx(
        [0.25, 0.5, 0.75, 1.0]
    )
    assert [point.mean_regret for point in result.risk_coverage_curve] == pytest.approx(
        [0.0, 0.05, 0.1, 0.075]
    )
    assert result.discrete_aurc == pytest.approx(0.05625)
    assert result.random_abstention_expected_aurc == pytest.approx(0.075)
    assert result.excess_aurc_vs_random == pytest.approx(-0.01875)
    assert result.exact_choice_ranking.status == "estimable"
    assert result.exact_choice_ranking.auroc == pytest.approx(0.5)


def test_confidence_ties_use_experiment_id_for_coverage_but_auc_treats_ties_equally() -> None:
    records = (
        SelectiveDecisionRecord("exp-b", 0.5, 0.2, False, False),
        SelectiveDecisionRecord("exp-a", 0.5, 0.0, True, True),
    )
    result = selective_decision_summary(records, minimum_class_count=1)
    assert result.risk_coverage_curve[0].covered_experiment_ids == ("exp-a",)
    assert result.exact_choice_ranking.auroc == pytest.approx(0.5)


def test_degenerate_or_underpowered_labels_are_not_estimable() -> None:
    degenerate = gated_binary_ranking_metrics(
        {"a": 0.9, "b": 0.2},
        {"a": True, "b": True},
        minimum_class_count=1,
    )
    assert degenerate.status == "not_estimable"
    assert degenerate.positive_count == 2
    assert degenerate.negative_count == 0
    assert degenerate.auroc is None
    assert degenerate.average_precision is None
    assert "both classes" in degenerate.reason

    underpowered = gated_binary_ranking_metrics(
        {"a": 0.9, "b": 0.8, "c": 0.2, "d": 0.1},
        {"a": True, "b": True, "c": False, "d": False},
        minimum_class_count=3,
    )
    assert underpowered.status == "not_estimable"
    assert "minimum_class_count=3" in underpowered.reason


def test_average_precision_uses_score_thresholds_not_arbitrary_tie_order() -> None:
    result = gated_binary_ranking_metrics(
        {"a": 0.8, "b": 0.8, "c": 0.1, "d": 0.1},
        {"a": True, "b": False, "c": True, "d": False},
        minimum_class_count=1,
    )
    assert result.status == "estimable"
    assert result.auroc == pytest.approx(0.5)
    assert result.average_precision == pytest.approx(0.5)


def test_selective_records_fail_closed_on_duplicates_and_invalid_regret() -> None:
    duplicate = (
        SelectiveDecisionRecord("exp-a", 0.9, 0.0, True, True),
        SelectiveDecisionRecord("exp-a", 0.8, 0.1, False, False),
    )
    with pytest.raises(ValueError, match="unique"):
        selective_decision_summary(duplicate)
    with pytest.raises(ValueError, match="non-negative"):
        selective_decision_summary(
            (SelectiveDecisionRecord("exp-a", 0.9, -0.1, True, True),)
        )

