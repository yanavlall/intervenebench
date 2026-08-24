"""Retrospective discovery scoring for the four-model full-action run."""

from __future__ import annotations

from math import fsum, sqrt
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from .balanced_forced_choice import EXPERIMENT_IDS, MODEL_IDS, read_json_object
from .protocol import payload_hash, verify_envelope


GENERIC_MODEL_IDS = MODEL_IDS[:3]


def _sign(value: float) -> int:
    return (value > 0.0) - (value < 0.0)


def _pearson(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("correlation vectors must have equal length at least two")
    first_center = fmean(first)
    second_center = fmean(second)
    first_deviation = [value - first_center for value in first]
    second_deviation = [value - second_center for value in second]
    denominator = sqrt(
        fsum(value * value for value in first_deviation)
        * fsum(value * value for value in second_deviation)
    )
    if denominator == 0.0:
        return None
    return fsum(
        first_value * second_value
        for first_value, second_value in zip(first_deviation, second_deviation)
    ) / denominator


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position]] = average_rank
        start = end
    return ranks


def _spearman(first: Sequence[float], second: Sequence[float]) -> float | None:
    return _pearson(_average_ranks(first), _average_ranks(second))


def score_modal_discovery(root: Path) -> dict[str, Any]:
    """Score only the already revealed five-experiment development set."""

    recommendation_path = (
        root
        / "artifacts/balanced_full_action/balanced_full_action_20260813_v1/"
        "full_action_recommendations.json"
    )
    human_score_path = root / "artifacts/portfolio_pilot/development_score_v2.json"
    recommendations = verify_envelope(recommendation_path, require_blinded=True)
    human_score = verify_envelope(human_score_path, require_blinded=False)
    if human_score.get("development_only") is not True or human_score.get(
        "human_outcomes_opened"
    ) is not True:
        raise ValueError("discovery score requires the declared revealed development set")
    if set(human_score["experiment_ids"]) != set(EXPERIMENT_IDS):
        raise ValueError("revealed development set differs from simulator set")
    arm_predictions = {
        (row["model_id"], row["experiment_id"], row["arm_id"]): float(
            row["balanced_expected_normalized_utility"]
        )
        for row in recommendations["balanced_arm_predictions"]
    }
    recommendation_rows = {
        (row["model_id"], row["experiment_id"]): row
        for row in recommendations["full_action_recommendations"]
    }
    task_rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        for experiment_id in EXPERIMENT_IDS:
            human_task = human_score["tasks"][experiment_id]
            contract = read_json_object(
                root
                / f"data/manifests/contracts/{experiment_id}_decision_task_candidate.json"
            )
            control_arm_id = contract["control_arm_id"]
            arm_order = [arm["arm_id"] for arm in contract["arms"]]
            human_arm_means = {
                arm_id: float(human_task["human_arm_means"][arm_id])
                for arm_id in arm_order
            }
            synthetic_arm_means = {
                arm_id: arm_predictions[(model_id, experiment_id, arm_id)]
                for arm_id in arm_order
            }
            selected_arm_id = recommendation_rows[(model_id, experiment_id)][
                "chosen_arm_id"
            ]
            human_best_arm_id = human_task["human_best_arm_id"]
            regret = human_arm_means[human_best_arm_id] - human_arm_means[
                selected_arm_id
            ]
            if regret < -1e-12:
                raise ValueError("decision regret cannot be negative")
            regret = max(0.0, regret)
            human_effects = {
                arm_id: human_arm_means[arm_id]
                - human_arm_means[control_arm_id]
                for arm_id in arm_order
                if arm_id != control_arm_id
            }
            synthetic_effects = {
                arm_id: synthetic_arm_means[arm_id]
                - synthetic_arm_means[control_arm_id]
                for arm_id in arm_order
                if arm_id != control_arm_id
            }
            errors = {
                arm_id: synthetic_effects[arm_id] - human_effects[arm_id]
                for arm_id in human_effects
            }
            sign_correct = {
                arm_id: _sign(synthetic_effects[arm_id])
                == _sign(human_effects[arm_id])
                for arm_id in human_effects
            }
            task_rows.append(
                {
                    "model_id": model_id,
                    "experiment_id": experiment_id,
                    "model_exposure": recommendation_rows[
                        (model_id, experiment_id)
                    ]["model_exposure"],
                    "control_arm_id": control_arm_id,
                    "human_best_arm_id": human_best_arm_id,
                    "selected_arm_id": selected_arm_id,
                    "correct_intervention_choice": selected_arm_id
                    == human_best_arm_id,
                    "decision_regret": regret,
                    "practically_reliable_at_0_05": regret
                    <= float(contract["practical_regret_tolerance"]),
                    "human_arm_means": human_arm_means,
                    "synthetic_arm_means": synthetic_arm_means,
                    "human_treatment_effects": human_effects,
                    "synthetic_treatment_effects": synthetic_effects,
                    "treatment_effect_errors": errors,
                    "treatment_effect_mae": fmean(abs(value) for value in errors.values()),
                    "effect_sign_correct": sign_correct,
                    "effect_sign_accuracy": fmean(
                        float(value) for value in sign_correct.values()
                    ),
                    "development_only": True,
                    "prospective_validation": False,
                }
            )
    model_summaries = []
    for model_id in MODEL_IDS:
        rows = [row for row in task_rows if row["model_id"] == model_id]
        human_effect_vector = [
            effect
            for row in rows
            for effect in row["human_treatment_effects"].values()
        ]
        synthetic_effect_vector = [
            effect
            for row in rows
            for effect in row["synthetic_treatment_effects"].values()
        ]
        sign_values = [
            value for row in rows for value in row["effect_sign_correct"].values()
        ]
        model_summaries.append(
            {
                "model_id": model_id,
                "primary_eligibility": (
                    "eligible_generic"
                    if model_id in GENERIC_MODEL_IDS
                    else "diagnostic_only_specialist_with_known_exposure"
                ),
                "experiment_count": len(rows),
                "treatment_contrast_count": len(human_effect_vector),
                "correct_intervention_count": sum(
                    row["correct_intervention_choice"] for row in rows
                ),
                "correct_intervention_rate": fmean(
                    float(row["correct_intervention_choice"]) for row in rows
                ),
                "practically_reliable_count": sum(
                    row["practically_reliable_at_0_05"] for row in rows
                ),
                "mean_decision_regret": fmean(row["decision_regret"] for row in rows),
                "worst_case_decision_regret": max(
                    row["decision_regret"] for row in rows
                ),
                "mean_experiment_treatment_effect_mae": fmean(
                    row["treatment_effect_mae"] for row in rows
                ),
                "pooled_effect_sign_accuracy": fmean(
                    float(value) for value in sign_values
                ),
                "treatment_effect_pearson_correlation": _pearson(
                    human_effect_vector, synthetic_effect_vector
                ),
                "treatment_effect_spearman_correlation": _spearman(
                    human_effect_vector, synthetic_effect_vector
                ),
            }
        )
    eligible = [
        row for row in model_summaries if row["primary_eligibility"] == "eligible_generic"
    ]
    selected = min(
        eligible,
        key=lambda row: (
            row["mean_decision_regret"],
            -row["correct_intervention_rate"],
            row["mean_experiment_treatment_effect_mae"],
            row["model_id"],
        ),
    )
    result = {
        "schema_version": "modal_full_action_discovery_score.v1",
        "full_action_recommendations_payload_sha256": payload_hash(recommendations),
        "revealed_development_score_payload_sha256": payload_hash(human_score),
        "development_only": True,
        "prospective_validation": False,
        "canonical_test_claim": False,
        "experiment_count": len(EXPERIMENT_IDS),
        "model_count": len(MODEL_IDS),
        "task_scores": task_rows,
        "model_summaries": model_summaries,
        "primary_model_selection_rule": (
            "among generic eligible models: minimize mean decision regret; then "
            "maximize exact choice; then minimize mean experiment treatment-effect "
            "MAE; then lexicographic model ID"
        ),
        "selected_primary_model_id_for_future_freeze": selected["model_id"],
        "selection_uses_only_revealed_discovery_set": True,
        "interpretation_boundary": (
            "This retrospective score uses five previously revealed development "
            "experiments. It may select a method for future sealed tasks but is not "
            "prospective validation or a general trust-model result."
        ),
        "status": "complete_retrospective_modal_discovery_score",
    }
    return result
