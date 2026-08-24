from __future__ import annotations

from copy import deepcopy

import pytest

from intervenebench.confirmation_scoring import (
    _decode_public_csv,
    HumanArmSummary,
    RawFallbackObservation,
    build_confirmation_scoring_protocol,
    evaluate_raw_human_fallback,
    score_synthetic_recommendation,
    validate_confirmation_reveal_authorization,
)


def test_public_csv_decoder_has_narrow_deterministic_fallback() -> None:
    assert _decode_public_csv(b"label,value\nplain,1\n") == (
        "label,value\nplain,1\n"
    )
    assert _decode_public_csv(b"label,value\nsmart\x92quote,1\n") == (
        "label,value\nsmart\N{RIGHT SINGLE QUOTATION MARK}quote,1\n"
    )


def test_scoring_protocol_freezes_development_prior_and_source_allowlists() -> None:
    from pathlib import Path

    protocol = build_confirmation_scoring_protocol(Path("."))
    prior = protocol["human_fallback"][
        "effect_prior_frozen_on_all_development_experiments"
    ]
    assert prior["training_experiment_ids"] == [
        "jf46x",
        "5vm8g",
        "xc4yq",
        "de5hx",
        "turagaS11",
        "wallaceS12",
        "nj5dx",
        "es4xw",
        "e2pyb",
    ]
    assert prior["contrast_count"] == 19
    assert protocol["source_projection"]["pb2rr"]["columns"] == [
        "XTESS187",
        "DOV_INSERT_NAME",
        "Q4",
        "weight",
    ]
    assert protocol["participant_rows_may_be_serialized"] is False
    assert protocol["recommendations_may_change"] is False


def test_synthetic_recommendation_score_uses_human_effects_and_regret() -> None:
    human = HumanArmSummary(
        arm_means={"a": 0.4, "b": 0.6, "c": 0.5},
        complete_case_count_by_arm={"a": 10, "b": 10, "c": 10},
        outcome_unit="normalized_utility",
    )
    result = score_synthetic_recommendation(
        arm_ids=("a", "b", "c"),
        control_arm_id="a",
        human=human,
        synthetic_arm_scores={"a": 0.2, "b": 0.7, "c": 0.4},
        selected_arm_id="b",
        practical_tolerance=0.05,
    )
    assert result["human_selected_arm_id"] == "b"
    assert result["synthetic_selected_arm_id"] == "b"
    assert result["exact_choice"] is True
    assert result["decision_regret"] == 0.0
    assert result["human_treatment_effects"] == {
        "a": 0.0,
        "b": pytest.approx(0.2),
        "c": pytest.approx(0.1),
    }
    assert result["mean_absolute_treatment_effect_error"] == pytest.approx(0.2)
    assert result["treatment_effect_sign_accuracy"] == 1.0


def test_raw_fallback_is_deterministic_and_aggregate_only() -> None:
    rows = [
        RawFallbackObservation(f"a-{i}", "a", 10.0 + i % 2)
        for i in range(30)
    ] + [
        RawFallbackObservation(f"b-{i}", "b", 5.0 + i % 2)
        for i in range(30)
    ]
    kwargs = dict(
        arm_ids=("a", "b"),
        synthetic_locations={"a": 7.0, "b": 9.0},
        budgets=(0, 10, 25),
        partitions=2,
        fold_count=5,
        seed=11,
        practical_tolerance=0.0,
    )
    first = evaluate_raw_human_fallback(rows, **kwargs)
    second = evaluate_raw_human_fallback(rows, **kwargs)
    assert first == second
    assert first["participant_rows_serialized"] == 0
    assert first["by_budget"]["0"]["synthetic_only"]["mean_regret"] > 0
    assert first["by_budget"]["25"]["human_only_balanced"]["mean_regret"] == 0


def _authorization() -> dict:
    return {
        "schema_version": "confirmation_reveal_authorization.v1",
        "status": "authorized_frozen_confirmation_outcome_scoring",
        "aggregation_payload_sha256": "a" * 64,
        "scoring_protocol_payload_sha256": "b" * 64,
        "development_evidence_payload_sha256": "c" * 64,
        "authorized_experiment_ids": [
            "tcg8p",
            "pb2rr",
            "z358z",
            "ShannonS2",
            "Blair1131",
            "KlarS44",
        ],
        "outcome_reveal_authorized": True,
        "aggregate_scoring_authorized": True,
        "human_fallback_authorized": True,
        "model_calls_authorized": False,
        "modal_compute_authorized": False,
        "model_download_authorized": False,
        "recommendation_changes_authorized": False,
        "diagnostic_changes_authorized": False,
        "threshold_tuning_authorized": False,
        "participant_row_serialization_authorized": False,
        "automatic_followup_authorized": False,
    }


def test_confirmation_reveal_authority_is_exact_and_cannot_expand() -> None:
    validate_confirmation_reveal_authorization(
        _authorization(),
        aggregation_payload_sha256="a" * 64,
        scoring_protocol_payload_sha256="b" * 64,
        development_evidence_payload_sha256="c" * 64,
    )
    expanded = deepcopy(_authorization())
    expanded["threshold_tuning_authorized"] = True
    with pytest.raises(PermissionError, match="expanded"):
        validate_confirmation_reveal_authorization(
            expanded,
            aggregation_payload_sha256="a" * 64,
            scoring_protocol_payload_sha256="b" * 64,
            development_evidence_payload_sha256="c" * 64,
        )
