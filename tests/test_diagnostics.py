from __future__ import annotations

from copy import deepcopy

import pytest

from intervenebench.diagnostics import build_outcome_free_diagnostics
from intervenebench.protocol import payload_hash


def _inputs() -> dict:
    return {
        "schema_version": "outcome_free_diagnostic_inputs.v1",
        "experiment_id": "toy",
        "primary_model_id": "model-a",
        "control_arm_id": "control",
        "utility_bounds": [0.0, 1.0],
        "decision_task_sha256": "a" * 64,
        "blinded_bundle_sha256": "b" * 64,
        "models": [
            {
                "model_id": "model-a",
                "model_revision": "rev-a",
                "recommendation_sha256": "c" * 64,
                "outputs_sha256": "d" * 64,
                "arm_means": {"control": 0.4, "message": 0.7},
                "draw_arm_means": [
                    {"control": 0.4, "message": 0.7},
                    {"control": 0.6, "message": 0.5},
                    {"control": 0.3, "message": 0.8},
                ],
                "prompt_variants": [
                    {
                        "variant_id": "answer_order_reversed",
                        "inverse_mapping_applied": True,
                        "arm_means": {"control": 0.42, "message": 0.68},
                    },
                    {
                        "variant_id": "format_paraphrase",
                        "inverse_mapping_applied": True,
                        "arm_means": {"control": 0.75, "message": 0.6},
                    },
                ],
            },
            {
                "model_id": "model-b",
                "model_revision": "rev-b",
                "recommendation_sha256": "e" * 64,
                "outputs_sha256": "f" * 64,
                "arm_means": {"control": 0.55, "message": 0.65},
                "draw_arm_means": [
                    {"control": 0.55, "message": 0.65},
                    {"control": 0.52, "message": 0.68},
                ],
                "prompt_variants": [],
            },
            {
                "model_id": "model-c",
                "model_revision": "rev-c",
                "recommendation_sha256": "1" * 64,
                "outputs_sha256": "2" * 64,
                "arm_means": {"control": 0.8, "message": 0.2},
                "draw_arm_means": [
                    {"control": 0.8, "message": 0.2},
                    {"control": 0.7, "message": 0.3},
                ],
                "prompt_variants": [],
            },
        ],
    }


def test_outcome_free_diagnostics_are_exact_and_hash_bound() -> None:
    inputs = _inputs()
    artifact = build_outcome_free_diagnostics(inputs)
    features = artifact["features"]
    assert artifact["diagnostic_inputs_sha256"] == payload_hash(inputs)
    assert artifact["decision_task_sha256"] == "a" * 64
    assert artifact["blinded_bundle_sha256"] == "b" * 64
    assert artifact["model_recommendation_sha256s"] == {
        "model-a": "c" * 64,
        "model-b": "e" * 64,
        "model-c": "1" * 64,
    }
    assert features["normalized_winner_margin"] == pytest.approx(0.3)
    assert features["winner_stability"] == pytest.approx(2 / 3)
    assert features["winner_rank_entropy"] == pytest.approx(0.9182958340544896)
    assert features["cross_model_winner_agreement"] == pytest.approx(2 / 3)
    assert features["prompt_winner_robustness"] == pytest.approx(0.5)
    assert features["prompt_max_arm_mean_shift"] == pytest.approx(0.35)
    assert features["cross_model_effect_dispersion"] > 0.0
    assert artifact["target_human_outcomes_used"] is False


def test_diagnostic_ties_use_lexicographic_arm_id() -> None:
    inputs = _inputs()
    primary = inputs["models"][0]
    primary["arm_means"] = {"z": 0.5, "a": 0.5}
    primary["draw_arm_means"] = [{"z": 0.5, "a": 0.5}]
    primary["prompt_variants"] = []
    inputs["control_arm_id"] = "a"
    for model in inputs["models"][1:]:
        model["arm_means"] = {"z": 0.5, "a": 0.5}
        model["draw_arm_means"] = [{"z": 0.5, "a": 0.5}]
    artifact = build_outcome_free_diagnostics(inputs)
    assert artifact["primary_selected_arm_id"] == "a"
    assert artifact["features"]["winner_stability"] == 1.0


@pytest.mark.parametrize(
    "forbidden",
    [
        {"human_outcomes": [1, 2]},
        {"nested": {"human_arm_means": {"control": 0.2}}},
        {"models": [{"response": "secret"}]},
        {"models": [{"reasoning": "secret"}]},
        {"regret": 0.1},
    ],
)
def test_diagnostics_reject_human_or_result_bearing_fields(forbidden: dict) -> None:
    inputs = _inputs()
    if "models" in forbidden:
        inputs["models"][0].update(forbidden["models"][0])
    else:
        inputs.update(forbidden)
    with pytest.raises(ValueError, match="forbidden outcome-derived"):
        build_outcome_free_diagnostics(inputs)


def test_diagnostics_reject_incomplete_or_unmapped_perturbations() -> None:
    inputs = _inputs()
    inputs["models"][0]["prompt_variants"][0]["inverse_mapping_applied"] = False
    with pytest.raises(ValueError, match="inverse mapped"):
        build_outcome_free_diagnostics(inputs)

    inputs = _inputs()
    del inputs["models"][0]["prompt_variants"][0]["arm_means"]["message"]
    with pytest.raises(ValueError, match="same arms|at least two arms"):
        build_outcome_free_diagnostics(inputs)


def test_mutating_any_synthetic_input_changes_diagnostic_binding() -> None:
    inputs = _inputs()
    first = build_outcome_free_diagnostics(inputs)
    mutated = deepcopy(inputs)
    mutated["models"][0]["draw_arm_means"][0]["message"] = 0.6
    second = build_outcome_free_diagnostics(mutated)
    assert first["diagnostic_inputs_sha256"] != second["diagnostic_inputs_sha256"]
