from __future__ import annotations

from dataclasses import replace

from intervenebench.release_decision import (
    BehavioralEvaluationSummary,
    ReleaseThresholds,
    evaluate_release_decision,
)


def _summary() -> BehavioralEvaluationSummary:
    return BehavioralEvaluationSummary(
        prospective_experiment_count=6,
        normalized_experiment_count=5,
        exact_choice_count=3,
        exact_choice_random_tail_probability=0.373,
        practically_reliable_count=6,
        mean_normalized_regret=0.00352,
        worst_normalized_regret=0.00854,
        uniform_mean_normalized_regret=0.04096,
        uniform_regret_tail_probability=0.0556,
        control_mean_normalized_regret=0.02738,
        control_difference_interval=(-0.07447, 0.00544),
        classical_mean_normalized_regret=0.07967,
        trust_ranking_better_than_random=False,
        validated_trust_threshold=False,
        human_fallback_improved=False,
        schema_valid_output_count=1404,
        planned_output_count=1464,
    )


def test_current_case_study_is_limited_screening_not_autonomous_release() -> None:
    decision = evaluate_release_decision(_summary(), ReleaseThresholds())
    assert decision["autonomous_intervention_selection"]["decision"] == "hold"
    assert decision["candidate_screening"]["decision"] == "limited_research_use"
    assert decision["confidence_based_abstention"]["decision"] == "hold"
    assert decision["small_sample_human_fallback"]["decision"] == "hold"
    assert "small prospective panel" in decision[
        "autonomous_intervention_selection"
    ]["reasons"]


def test_autonomous_release_requires_all_frozen_quality_gates() -> None:
    strong = replace(
        _summary(),
        prospective_experiment_count=16,
        normalized_experiment_count=15,
        exact_choice_count=12,
        exact_choice_random_tail_probability=0.01,
        practically_reliable_count=16,
        uniform_regret_tail_probability=0.01,
        control_difference_interval=(-0.04, -0.002),
        trust_ranking_better_than_random=True,
        validated_trust_threshold=True,
        schema_valid_output_count=1464,
    )
    decision = evaluate_release_decision(strong, ReleaseThresholds())
    assert decision["autonomous_intervention_selection"]["decision"] == "pass"
    assert decision["confidence_based_abstention"]["decision"] == "pass"


def test_one_bad_gate_blocks_autonomous_release() -> None:
    base = replace(
        _summary(),
        prospective_experiment_count=16,
        normalized_experiment_count=15,
        exact_choice_count=12,
        exact_choice_random_tail_probability=0.01,
        practically_reliable_count=16,
        uniform_regret_tail_probability=0.01,
        control_difference_interval=(-0.04, -0.002),
        schema_valid_output_count=1464,
    )
    for changed in (
        {"mean_normalized_regret": 0.02},
        {"worst_normalized_regret": 0.08},
        {"uniform_regret_tail_probability": 0.08},
        {"control_difference_interval": (-0.04, 0.02)},
        {"schema_valid_output_count": 1300},
    ):
        decision = evaluate_release_decision(
            replace(base, **changed), ReleaseThresholds()
        )
        assert decision["autonomous_intervention_selection"]["decision"] == "hold"


def test_human_fallback_is_independently_scoped() -> None:
    decision = evaluate_release_decision(
        replace(_summary(), human_fallback_improved=True), ReleaseThresholds()
    )
    assert decision["small_sample_human_fallback"]["decision"] == "pass"
    assert decision["autonomous_intervention_selection"]["decision"] == "hold"


def test_invalid_summary_fails_closed() -> None:
    invalid = replace(_summary(), schema_valid_output_count=1465)
    try:
        evaluate_release_decision(invalid, ReleaseThresholds())
    except ValueError as error:
        assert "schema-valid" in str(error)
    else:
        raise AssertionError("invalid evaluation summary was accepted")
