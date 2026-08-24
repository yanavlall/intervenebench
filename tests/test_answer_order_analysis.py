from __future__ import annotations

from pathlib import Path

import pytest

from intervenebench.answer_order_analysis import (
    analyze_answer_order,
    jensen_shannon,
    nearest_rank,
    total_variation,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = ROOT / "artifacts/forced_choice_screen/discovery_screen_20260813_v1"
REVERSE_RUN = (
    ROOT / "artifacts/answer_order_canary/answer_order_canary_20260813_v1"
)


def test_distribution_distances_are_exact_on_toy_inputs() -> None:
    first = {1: 1.0, 2: 0.0}
    second = {1: 0.0, 2: 1.0}
    assert total_variation(first, second) == 1.0
    assert jensen_shannon(first, first) == 0.0
    assert jensen_shannon(first, second) == pytest.approx(0.6931471805599453)


def test_nearest_rank_uses_frozen_definition() -> None:
    assert nearest_rank(range(1, 41), 0.90) == 36.0
    with pytest.raises(ValueError):
        nearest_rank([], 0.90)


def test_real_answer_order_analysis_is_paired_complete_and_outcome_blind() -> None:
    result = analyze_answer_order(
        ROOT, source_run_root=SOURCE_RUN, reverse_run_root=REVERSE_RUN
    )
    assert result["paired_call_count"] == 40
    assert result["screened_pair_count"] == 20
    assert len(result["model_summaries"]) == 4
    assert result["outcome_access"] == "not_accessed"
    assert all(
        row["full_action_set_recommendation"] is False
        for row in result["screened_pair_results"]
    )
    assert result["single_order_scaling_gate_passed"] == all(
        result["checks"].values()
    )
