from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from intervenebench.development_analysis import (
    build_development_analysis_payload,
    fit_leave_one_experiment_out_attenuation,
    load_frozen_recommendations,
    verify_development_analysis,
)
from intervenebench.portfolio_development import verify_development_score


ROOT = Path(__file__).resolve().parents[1]
SCORE_PATH = ROOT / "artifacts/portfolio_pilot/development_score_v2.json"
ANALYSIS_PATH = ROOT / "artifacts/portfolio_pilot/development_analysis_v4.json"


def _payload() -> dict:
    score = verify_development_score(ROOT, SCORE_PATH)
    return build_development_analysis_payload(
        score, load_frozen_recommendations(ROOT)
    )


def test_random_choice_baseline_is_analytic_and_arm_count_aware() -> None:
    analysis = _payload()
    random_policy = analysis["random_choice_policy"]
    by_experiment = random_policy["by_experiment"]
    assert by_experiment["5vm8g"]["expected_exact_choice_probability"] == 0.5
    assert by_experiment["wallaceS12"]["expected_exact_choice_probability"] == 1 / 6
    summary = random_policy["portfolio_summary"]
    assert summary["expected_correct_intervention_rate"] == pytest.approx(1 / 3)
    assert summary["mean_expected_decision_regret"] == pytest.approx(
        0.029294988923215238
    )
    assert summary[
        "probability_random_policy_matches_or_exceeds_observed_count"
    ] == pytest.approx(0.20370370370370372)


def test_target_human_effect_cannot_change_its_fitted_coefficient() -> None:
    score = verify_development_score(ROOT, SCORE_PATH)
    tasks = score["tasks"]
    original = fit_leave_one_experiment_out_attenuation(tasks, "turagaS11")
    changed = deepcopy(tasks)
    for arm_id in changed["turagaS11"]["human_treatment_effects"]:
        changed["turagaS11"]["human_treatment_effects"][arm_id] += 1000.0
    assert fit_leave_one_experiment_out_attenuation(
        changed, "turagaS11"
    ) == pytest.approx(original)


def test_cross_fitted_attenuation_improves_but_does_not_beat_no_effect_mae() -> None:
    analysis = _payload()
    summary = analysis["cross_fitted_effect_attenuation"]["portfolio_summary"]
    assert summary["all_choices_preserved"] is True
    assert summary["calibrated_treatment_effect_mae"] < summary[
        "raw_treatment_effect_mae"
    ]
    assert summary["calibrated_treatment_effect_mae"] > summary[
        "no_effect_treatment_effect_mae"
    ]


def test_policy_comparison_treats_experiments_as_the_independent_units() -> None:
    analysis = _payload()
    comparison = analysis["experiment_level_policy_comparison"]
    assert comparison["experiments_local_lower_regret"] == 5
    assert comparison["experiments_no_effect_lower_regret"] == 0
    assert comparison["exact_one_sided_sign_test_p"] == pytest.approx(1 / 32)
    assert comparison["exact_two_sided_sign_test_p"] == pytest.approx(1 / 16)
    assert comparison["local_practically_reliable_count"] == 5
    assert comparison["no_effect_practically_reliable_count"] == 4


def test_outcome_free_diagnostics_fail_to_rank_risk_in_the_small_portfolio() -> None:
    analysis = _payload()
    diagnostics = analysis["outcome_free_diagnostic_evaluation"]
    random_risk = diagnostics["random_coverage_expected_mean_regret"]
    assert diagnostics["diagnostics_frozen_before_outcome_reveal"] is True
    assert diagnostics["target_human_outcomes_used_as_diagnostic_inputs"] is False
    for result in diagnostics["by_diagnostic"].values():
        assert result["correct_choice_ranking_auc"] < 0.5
        assert result["mean_risk_across_coverage_levels"] > random_risk
        assert result["risk_coverage_curve"][0]["covered_experiment_ids"] == [
            "turagaS11"
        ]


def test_frozen_development_analysis_replays_from_aggregate_score_only() -> None:
    analysis = verify_development_analysis(
        ROOT,
        score_path=SCORE_PATH,
        analysis_path=ANALYSIS_PATH,
    )
    assert analysis["participant_rows_read"] == 0
    assert analysis["participant_rows_written"] == 0
    assert analysis["canonical_test_claim"] is False
