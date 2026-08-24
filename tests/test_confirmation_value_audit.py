from __future__ import annotations

from pathlib import Path

import pytest

from intervenebench.confirmation_value_audit import (
    _exact_sign_flip_test,
    _exact_uniform_combination_summary,
    _uniform_choice_task,
    build_confirmation_value_audit_payload,
    build_confirmation_value_audit_spec,
)
from intervenebench.protocol import payload_hash, verify_envelope


ROOT = Path(__file__).resolve().parents[1]
SCORE_PATH = ROOT / "artifacts/confirmation/confirmation_20260814_v1/score_v1.json"


def test_uniform_choice_baseline_is_arm_count_and_regret_aware() -> None:
    result = _uniform_choice_task(
        {"a": 0.2, "b": 0.5, "c": 0.4}, practical_tolerance=0.05
    )
    assert result["arm_count"] == 3
    assert result["expected_exact_choice_probability"] == pytest.approx(1 / 3)
    assert result["expected_decision_regret"] == pytest.approx(0.4 / 3)
    assert result["expected_practical_reliability_probability"] == pytest.approx(
        1 / 3
    )


def test_exact_uniform_enumeration_uses_whole_experiment_choices() -> None:
    tasks = {
        "one": {
            "regret_by_arm": {"a": 0.0, "b": 0.2},
            "primary_selected_arm_id": "a",
        },
        "two": {
            "regret_by_arm": {"a": 0.0, "b": 0.4},
            "primary_selected_arm_id": "b",
        },
    }
    result = _exact_uniform_combination_summary(tasks)
    assert result["combination_count"] == 4
    assert result["observed_primary_exact_count"] == 1
    assert result["probability_uniform_exact_count_at_least_observed"] == pytest.approx(
        0.75
    )
    assert result["observed_primary_mean_regret"] == pytest.approx(0.2)
    assert result["probability_uniform_mean_regret_at_most_observed"] == pytest.approx(
        0.75
    )


def test_exact_sign_flip_comparison_treats_experiments_as_units() -> None:
    result = _exact_sign_flip_test(
        {"one": 0.0, "two": 0.1}, {"one": 0.2, "two": 0.2}
    )
    assert result["non_tied_experiment_count"] == 2
    assert result["mean_candidate_minus_reference"] == pytest.approx(-0.15)
    assert result["one_sided_probability_under_symmetric_null"] == pytest.approx(
        0.25
    )


def test_value_audit_spec_is_explicitly_post_reveal_and_non_tuning() -> None:
    score = verify_envelope(SCORE_PATH, require_blinded=False)
    spec = build_confirmation_value_audit_spec(ROOT, score=score)
    assert spec["outcomes_known_when_specified"] is True
    assert spec["simulator_or_threshold_tuning_authorized"] is False
    assert spec["new_model_calls_authorized"] is False
    assert spec["primary_comparators"] == [
        "uniform_random_action",
        "no_effect_control_tie",
        "frozen_classical_baseline",
    ]


def test_real_value_audit_is_aggregate_only_and_keeps_units_separate() -> None:
    score = verify_envelope(SCORE_PATH, require_blinded=False)
    spec = build_confirmation_value_audit_spec(ROOT, score=score)
    audit = build_confirmation_value_audit_payload(ROOT, score=score, spec=spec)
    assert audit["participant_rows_read"] == 0
    assert audit["participant_rows_serialized"] == 0
    assert audit["prospective_experiment_count"] == 6
    assert audit["normalized_experiment_count"] == 5
    assert audit["tcg8p_reported_separately"] is True
    assert audit["new_model_calls_made"] == 0


def test_frozen_value_audit_replays_to_the_same_payload_hash() -> None:
    score = verify_envelope(SCORE_PATH, require_blinded=False)
    spec = verify_envelope(
        ROOT / "data/manifests/research/confirmation_value_audit_spec_v1.json",
        require_blinded=False,
    )
    stored = verify_envelope(
        ROOT
        / "artifacts/confirmation/confirmation_20260814_v1/value_audit_v1.json",
        require_blinded=False,
    )
    rebuilt = build_confirmation_value_audit_payload(ROOT, score=score, spec=spec)
    assert payload_hash(rebuilt) == payload_hash(stored)
