from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import pytest

from intervenebench.classical_development import (
    ArmEffectRecord,
    DEFAULT_CLASSICAL_DEVELOPMENT_PATH,
    DEFAULT_CLASSICAL_MODEL_PATH,
    build_contrast_features,
    cross_fit_arm_effect_predictions,
    verify_classical_development,
    verify_classical_model,
)


ROOT = Path(__file__).resolve().parents[1]


def _candidate(experiment_id: str = "example") -> dict[str, object]:
    return {
        "schema_version": "toy.v1",
        "experiment_id": experiment_id,
        "outcome_access": "sealed",
        "outcome_family": "ordinal",
        "design_type": "between_subject",
        "direction": "higher_is_better",
        "outcome_question": "How willing are you to help?",
        "response_options": [
            {"raw_value": 1, "label": "Not willing"},
            {"raw_value": 2, "label": "Willing"},
        ],
        "arms": [
            {"arm_id": "control", "description": "Show a neutral message"},
            {"arm_id": "treatment", "description": "Show a helpful message"},
        ],
        "control_arm_id": "control",
    }


def _bundle(experiment_id: str = "example") -> dict[str, object]:
    return {
        "schema_version": "toy_bundle.v1",
        "experiment_id": experiment_id,
        "outcome_access": "sealed",
        "common_context": "A community program needs volunteers.",
        "outcome_question": "How willing are you to help?",
        "arms": [
            {"arm_id": "control", "message": "The program exists."},
            {"arm_id": "treatment", "message": "The program helps families."},
        ],
    }


def test_contrast_features_are_identifier_free_and_response_blind() -> None:
    first = build_contrast_features(
        _candidate("first"),
        _bundle("first"),
        arm_id="treatment",
        control_arm_id="control",
    )
    second = build_contrast_features(
        _candidate("renamed"),
        _bundle("renamed"),
        arm_id="treatment",
        control_arm_id="control",
    )
    assert first == second
    assert not any("first" in key or "renamed" in key for key in first)
    contaminated = _bundle()
    contaminated["human_outcomes"] = [1, 2]
    with pytest.raises(ValueError, match="forbidden outcome-derived"):
        build_contrast_features(
            _candidate(), contaminated, arm_id="treatment", control_arm_id="control"
        )


def test_cross_fit_prediction_does_not_use_target_experiment_effects() -> None:
    records = {
        "a": (
            ArmEffectRecord("a", "a1", {"signal": -1.0}, -0.5),
            ArmEffectRecord("a", "a2", {"signal": -0.5}, -0.2),
        ),
        "b": (ArmEffectRecord("b", "b1", {"signal": 1.0}, 0.6),),
        "c": (ArmEffectRecord("c", "c1", {"signal": 0.8}, 0.4),),
    }
    original = cross_fit_arm_effect_predictions(
        records, feature_dimension=16, l2_penalty=0.2
    )
    mutated = dict(records)
    mutated["a"] = tuple(
        replace(record, normalized_effect=-record.normalized_effect)
        for record in records["a"]
    )
    replay = cross_fit_arm_effect_predictions(
        mutated, feature_dimension=16, l2_penalty=0.2
    )
    assert original["a"] == replay["a"]
    assert set(original) == {"a", "b", "c"}


def test_frozen_classical_development_artifacts_replay() -> None:
    result = verify_classical_development(
        ROOT, ROOT / DEFAULT_CLASSICAL_DEVELOPMENT_PATH
    )
    model = verify_classical_model(ROOT, ROOT / DEFAULT_CLASSICAL_MODEL_PATH)
    assert result["experiment_count"] == 9
    assert result["participant_rows_read"] == 0
    assert result["participant_rows_serialized"] == 0
    assert result["confirmation_outcomes_accessed"] is False
    assert model["training_experiment_count"] == 9
    assert model["confirmation_experiment_ids"] == [
        "tcg8p",
        "pb2rr",
        "z358z",
        "ShannonS2",
        "Blair1131",
        "KlarS44",
    ]


def test_lora_gate_denies_spend_training_and_confirmation_access() -> None:
    gate = json.loads(
        (ROOT / "data/manifests/research/lora_development_gate_v1.json").read_text()
    )
    assert gate["status"] == "not_authorized_not_scientifically_justified"
    assert gate["decision"] == "do_not_run_before_confirmation"
    assert gate["fine_tuning_authorized"] is False
    assert gate["modal_spend_authorized"] is False
    assert gate["confirmation_outcomes_accessed"] is False
