from __future__ import annotations

from pathlib import Path

from intervenebench.modal_discovery_scoring import score_modal_discovery


ROOT = Path(__file__).resolve().parents[1]


def test_modal_discovery_score_is_complete_and_explicitly_retrospective() -> None:
    result = score_modal_discovery(ROOT)
    assert result["experiment_count"] == 5
    assert result["model_count"] == 4
    assert len(result["task_scores"]) == 20
    assert len(result["model_summaries"]) == 4
    assert result["development_only"] is True
    assert result["prospective_validation"] is False
    assert result["canonical_test_claim"] is False
    assert result["selection_uses_only_revealed_discovery_set"] is True
    assert result["selected_primary_model_id_for_future_freeze"] in {
        "qwen3_8b_generic",
        "qwen3_14b_generic",
        "qwen2_5_14b_generic",
    }
    assert all(row["decision_regret"] >= 0.0 for row in result["task_scores"])
    assert all(row["development_only"] is True for row in result["task_scores"])


def test_specialist_cannot_be_selected_primary_even_if_it_scores_best() -> None:
    result = score_modal_discovery(ROOT)
    specialist = next(
        row
        for row in result["model_summaries"]
        if row["model_id"] == "socrates_qwen2_5_14b_sft"
    )
    assert specialist["primary_eligibility"] == (
        "diagnostic_only_specialist_with_known_exposure"
    )
    assert result["selected_primary_model_id_for_future_freeze"] != specialist[
        "model_id"
    ]
