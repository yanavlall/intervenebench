from __future__ import annotations

from pathlib import Path

from intervenebench.forced_choice_screen_analysis import analyze_screen
from intervenebench.modal_forced_choice import MODEL_IDS
from intervenebench.protocol import assert_blinded_payload


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "artifacts/forced_choice_screen/discovery_screen_20260813_v1"


def test_real_screen_analysis_is_complete_aggregate_and_outcome_blind() -> None:
    result = analyze_screen(ROOT, run_root=RUN_ROOT)
    assert result["call_count"] == 40
    assert result["outcome_access"] == "not_accessed"
    assert len(result["model_summaries"]) == 4
    assert {row["model_id"] for row in result["model_summaries"]} == set(MODEL_IDS)
    assert all(row["call_count"] == 10 for row in result["model_summaries"])
    assert len(result["prompt_agreement"]) == 10
    assert len(result["screened_pair_results"]) == 20
    assert len(result["experiment_summaries"]) == 5
    assert all(
        row["full_action_set_recommendation"] is False
        for row in result["screened_pair_results"]
    )
    assert_blinded_payload(result)


def test_screen_analysis_records_observed_disagreement_without_accuracy_claim() -> None:
    result = analyze_screen(ROOT, run_root=RUN_ROOT)
    assert result["summary"]["unanimous_modal_response_prompts"] == 1
    assert result["summary"]["all_models_different_prompts"] == 2
    assert "not accuracy claims" in result["interpretation_boundary"]
