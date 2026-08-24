from __future__ import annotations

from pathlib import Path

import pytest

from intervenebench.prospective_development_score import (
    DEFAULT_SCORE_PATH,
    distribution_metrics,
    verify_prospective_development_score,
)


ROOT = Path(__file__).resolve().parents[1]


def test_distribution_metrics_are_exact_on_two_point_support() -> None:
    result = distribution_metrics({1: 0.75, 2: 0.25}, {1: 0.25, 2: 0.75})
    assert result["total_variation"] == pytest.approx(0.5)
    assert result["ordinal_wasserstein_normalized"] == pytest.approx(0.5)
    assert result["jensen_shannon_divergence_bits"] > 0.0


def test_distribution_metrics_reject_mismatched_support() -> None:
    with pytest.raises(ValueError, match="shared ordered support"):
        distribution_metrics({1: 1.0, 2: 0.0}, {1: 1.0, 3: 0.0})


def test_frozen_score_is_aggregate_only_and_scope_bound() -> None:
    score = verify_prospective_development_score(ROOT, ROOT / DEFAULT_SCORE_PATH)
    assert score["participant_rows_serialized"] == 0
    assert score["experiment_ids"] == ["nj5dx", "es4xw", "e2pyb"]
    assert score["other_experiments_opened"] == []
    assert score["modal_used_for_outcome_scoring"] is False
    assert score["canonical_test_claim"] is False
