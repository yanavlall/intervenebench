from __future__ import annotations

import pytest

from intervenebench.eb_fallback import (
    EffectCalibrationTask,
    evaluate_eb_human_fallback,
    fit_effect_prior,
)
from intervenebench.confirmation_fallback import (
    ConfirmationFallbackObservation,
    confirmation_stratified_fold_assignments,
    evaluate_confirmation_eb_human_fallback,
)
from intervenebench.human_fallback import (
    FallbackObservation,
    balanced_allocation,
    evaluate_human_fallback,
    hedged_allocation,
    stratified_fold_assignments,
)


def _rows() -> list[FallbackObservation]:
    rows = []
    for arm_index, arm_id in enumerate(("a", "b", "c")):
        for person in range(30):
            rows.append(
                FallbackObservation(
                    participant_id=f"{arm_id}-{person}",
                    arm_id=arm_id,
                    utility=(arm_index + (person % 3) / 10) / 3,
                )
            )
    return rows


def test_allocations_consume_exact_budget_and_keep_exploration() -> None:
    arms = ("a", "b", "c")
    assert balanced_allocation(arms, 10) == {"a": 4, "b": 3, "c": 3}
    hedged = hedged_allocation(
        arms, 10, winner_votes={"a": 6, "b": 0, "c": 0}
    )
    assert sum(hedged.values()) == 10
    assert all(hedged[arm_id] >= 1 for arm_id in arms)
    assert hedged["a"] > hedged["b"]


def test_stratified_folds_are_deterministic_and_disjoint() -> None:
    rows = _rows()
    first = stratified_fold_assignments(
        rows, arm_ids=("a", "b", "c"), fold_count=10, seed=17
    )
    second = stratified_fold_assignments(
        rows, arm_ids=("a", "b", "c"), fold_count=10, seed=17
    )
    assert first == second
    assert set(first) == {row.participant_id for row in rows}
    assert set(first.values()) == set(range(10))


def test_fold_strata_and_equal_cell_standardization_are_preserved() -> None:
    rows = [
        ConfirmationFallbackObservation(
            participant_id=f"{arm}-{cell}-{index}",
            arm_id=arm,
            utility=float(cell == "high"),
            weight=100.0 if cell == "high" else 1.0,
            fold_stratum_id=cell,
            standardization_cell_id=cell,
        )
        for arm in ("a", "b")
        for cell in ("low", "high")
        for index in range(10)
    ]
    assignments = confirmation_stratified_fold_assignments(
        rows, arm_ids=("a", "b"), fold_count=5, seed=17
    )
    for arm in ("a", "b"):
        for cell in ("low", "high"):
            assert {
                assignments[row.participant_id]
                for row in rows
                if row.arm_id == arm and row.fold_stratum_id == cell
            } == {0, 1, 2, 3, 4}

    prior = fit_effect_prior(
        [
            EffectCalibrationTask("d1", {"b": 0.1}, {"b": 0.1}),
            EffectCalibrationTask("d2", {"b": 0.2}, {"b": 0.2}),
        ]
    )
    result = evaluate_confirmation_eb_human_fallback(
        rows,
        arm_ids=("a", "b"),
        control_arm_id="a",
        synthetic_means={"a": 0.4, "b": 0.6},
        winner_votes={"a": 1, "b": 1},
        budgets=(0, 10),
        partitions=1,
        fold_count=5,
        seed=17,
        pseudocount=10,
        practical_tolerance=0.05,
        effect_prior=prior,
    )
    # Both arms are 0.5 under equal-cell standardization despite the 100x
    # survey weight in the high cell, so either choice has zero regret.
    assert result["by_budget"]["0"]["synthetic_only"]["mean_regret"] == 0.0


def test_fallback_is_replayable_aggregate_only_and_budget_nested() -> None:
    kwargs = dict(
        arm_ids=("a", "b", "c"),
        synthetic_means={"a": 0.2, "b": 0.5, "c": 0.7},
        winner_votes={"a": 0, "b": 1, "c": 5},
        budgets=(0, 10, 25),
        partitions=2,
        fold_count=5,
        seed=101,
        pseudocount=10,
        practical_tolerance=0.05,
    )
    first = evaluate_human_fallback(_rows(), **kwargs)
    second = evaluate_human_fallback(_rows(), **kwargs)
    assert first == second
    assert first["participant_rows_serialized"] == 0
    assert first["by_budget"]["0"]["human_only_balanced"]["status"] == (
        "not_estimable_at_zero_humans"
    )
    for policy in (
        "synthetic_plus_balanced_fixed10",
        "synthetic_plus_hedged_fixed10",
    ):
        assert first["by_budget"]["0"][policy]["status"] == "estimated"
        assert (
            first["by_budget"]["0"][policy]["mean_regret"]
            == first["by_budget"]["0"]["synthetic_only"]["mean_regret"]
        )
    assert first["by_budget"]["25"]["synthetic_only"][
        "acquisition_evaluation_replicates"
    ] == 10


def test_effect_prior_is_target_excluded_and_constrained() -> None:
    tasks = [
        EffectCalibrationTask("a", {"x": 0.2}, {"x": 0.1}),
        EffectCalibrationTask("b", {"x": 0.4}, {"x": 0.2}),
        EffectCalibrationTask("target", {"x": 0.3}, {"x": -0.9}),
    ]
    first = fit_effect_prior(tasks, excluded_experiment_id="target")
    changed = [*tasks[:-1], EffectCalibrationTask("target", {"x": 0.3}, {"x": 0.9})]
    second = fit_effect_prior(changed, excluded_experiment_id="target")
    assert first == second
    assert first.alpha == pytest.approx(0.5)
    assert 0.0 <= first.alpha <= 1.0
    assert first.training_experiment_ids == ("a", "b")


def test_eb_fallback_reduces_to_synthetic_at_zero_budget() -> None:
    prior = fit_effect_prior(
        [
            EffectCalibrationTask("a", {"b": 0.2, "c": 0.1}, {"b": 0.1, "c": 0.0}),
            EffectCalibrationTask("b", {"b": 0.3}, {"b": 0.2}),
        ]
    )
    result = evaluate_eb_human_fallback(
        _rows(),
        arm_ids=("a", "b", "c"),
        control_arm_id="a",
        synthetic_means={"a": 0.2, "b": 0.5, "c": 0.7},
        winner_votes={"a": 0, "b": 1, "c": 5},
        budgets=(0, 10),
        partitions=1,
        fold_count=5,
        seed=101,
        pseudocount=10,
        practical_tolerance=0.05,
        effect_prior=prior,
    )
    baseline = result["by_budget"]["0"]["synthetic_only"]
    for policy in ("synthetic_plus_balanced_eb", "synthetic_plus_hedged_eb"):
        assert result["by_budget"]["0"][policy]["mean_regret"] == (
            baseline["mean_regret"]
        )
        assert result["by_budget"]["10"][policy]["status"] == "estimated"
