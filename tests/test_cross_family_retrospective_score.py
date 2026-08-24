from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from intervenebench.cross_family_retrospective_score import (
    DEFAULT_DIAGNOSTIC_SPEC_PATH,
    DEFAULT_RETROSPECTIVE_SCORE_PATH,
    build_future_cross_family_diagnostic_spec,
    build_retrospective_score_authorization,
    build_retrospective_cross_family_score,
    freeze_future_cross_family_diagnostic_spec,
    freeze_retrospective_cross_family_score,
    validate_retrospective_score_authorization,
)
from intervenebench.protocol import payload_hash, verify_envelope


ROOT = Path(__file__).resolve().parents[1]


def test_retrospective_score_authority_is_exact_and_cannot_expand() -> None:
    authorization = build_retrospective_score_authorization(ROOT)
    validate_retrospective_score_authorization(authorization, root=ROOT)
    assert authorization["aggregate_human_outcome_access_authorized"] is True
    assert authorization["participant_row_access_authorized"] is False
    assert authorization["model_calls_authorized"] is False
    assert authorization["recommendation_changes_authorized"] is False
    assert authorization["trust_threshold_tuning_authorized"] is False
    assert authorization["model_switching_rule_authorized"] is False
    assert authorization["automatic_next_stage_authorized"] is False

    expanded = deepcopy(authorization)
    expanded["model_switching_rule_authorized"] = True
    with pytest.raises(PermissionError, match="expanded"):
        validate_retrospective_score_authorization(expanded, root=ROOT)


def test_real_retrospective_score_preserves_negative_and_mixed_result() -> None:
    authorization = build_retrospective_score_authorization(ROOT)
    scored = build_retrospective_cross_family_score(ROOT, authorization=authorization)

    assert scored["study_role"] == "retrospective_development_only_cross_family_score"
    assert scored["independent_experiment_n"] == 5
    assert scored["candidate_unavailable_experiments"] == ["tcg8p"]
    assert scored["primary_policy_changed"] is False
    assert scored["candidate_exact_choice_rate"] == pytest.approx(0.4)
    assert scored["primary_exact_choice_rate"] == pytest.approx(0.4)
    assert scored["candidate_mean_decision_regret"] == pytest.approx(
        0.004295147420206114
    )
    assert scored["primary_mean_decision_regret"] == pytest.approx(
        0.003520577798391078
    )
    assert scored["paired_mean_regret_delta_candidate_minus_primary"] == pytest.approx(
        0.0007745696218150356
    )
    assert scored["candidate_mean_treatment_effect_mae"] == pytest.approx(
        0.050255692268200404
    )
    assert scored["primary_mean_treatment_effect_mae"] == pytest.approx(
        0.06370846108314762
    )
    assert scored["candidate_mean_effect_sign_accuracy"] == pytest.approx(0.5)
    assert scored["primary_mean_effect_sign_accuracy"] == pytest.approx(0.42)
    assert scored["decision_transitions"] == {
        "fixed_by_candidate": ["ShannonS2"],
        "harmed_by_candidate": ["KlarS44"],
        "unchanged_incorrect": ["pb2rr", "z358z"],
        "unchanged_correct": ["Blair1131"],
    }
    assert scored["diagnostic_evaluation"]["winner_disagreement"][
        "primary_error_rate_when_disagree"
    ] == pytest.approx(0.5)
    assert scored["diagnostic_evaluation"]["winner_disagreement"][
        "primary_error_rate_when_agree"
    ] == pytest.approx(2 / 3)
    assert scored["diagnostic_evaluation"]["conclusion"] == (
        "no_positive_retrospective_signal_do_not_deploy_or_tune_threshold"
    )
    assert scored["model_calls_made"] == 0
    assert scored["participant_rows_accessed"] == 0
    assert scored["participant_rows_serialized"] == 0
    assert scored["automatic_next_stage"] is False


def test_future_diagnostic_is_frozen_secondary_without_threshold_or_switching() -> None:
    score = build_retrospective_cross_family_score(
        ROOT, authorization=build_retrospective_score_authorization(ROOT)
    )
    spec = build_future_cross_family_diagnostic_spec(score)
    assert spec["status"] == "frozen_for_future_untouched_replication_only"
    assert spec["development_evidence_payload_sha256"] == payload_hash(score)
    assert spec["diagnostic_role"] == "secondary_unvalidated_failure_diagnostic"
    assert spec["hypothesized_direction"] == "more_disagreement_means_lower_trust"
    assert spec["validated_on_development_panel"] is False
    assert spec["trust_threshold"] is None
    assert spec["accept_abstain_policy"] == "not_validated_not_deployed"
    assert spec["model_switching_rule"] == "forbidden"
    assert spec["experiment_is_independent_unit"] is True
    assert spec["target_human_outcomes_must_be_hidden_until_diagnostic_freeze"] is True


def test_create_only_score_and_diagnostic_spec_replay(tmp_path: Path) -> None:
    authorization = build_retrospective_score_authorization(ROOT)
    score_path = tmp_path / "score.json"
    score_digest = freeze_retrospective_cross_family_score(
        ROOT, authorization=authorization, destination=score_path
    )
    score = verify_envelope(score_path)
    assert score_digest == payload_hash(score)

    spec_path = tmp_path / "spec.json"
    spec_digest = freeze_future_cross_family_diagnostic_spec(
        score, destination=spec_path
    )
    spec = verify_envelope(spec_path)
    assert spec_digest == payload_hash(spec)
    with pytest.raises(FileExistsError):
        freeze_future_cross_family_diagnostic_spec(score, destination=spec_path)


def test_repository_artifacts_replay_if_present() -> None:
    score_path = ROOT / DEFAULT_RETROSPECTIVE_SCORE_PATH
    spec_path = ROOT / DEFAULT_DIAGNOSTIC_SPEC_PATH
    if not score_path.exists() and not spec_path.exists():
        pytest.skip("retrospective score artifacts have not been materialized")
    assert score_path.exists() and spec_path.exists()
    authorization = build_retrospective_score_authorization(ROOT)
    score = verify_envelope(score_path)
    rebuilt = build_retrospective_cross_family_score(ROOT, authorization=authorization)
    assert score == rebuilt
    assert verify_envelope(spec_path) == build_future_cross_family_diagnostic_spec(score)
