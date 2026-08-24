"""Outcome-free recommendations and diagnostics for the prospective image run."""

from __future__ import annotations

from collections import defaultdict
from math import fsum, isfinite, log, sqrt
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

from .answer_order_analysis import total_variation
from .balanced_forced_choice import read_json_object, weighted_distribution_average
from .multimodal_freeze import verify_prospective_multimodal_freeze
from .multimodal_prospective import EXPERIMENT_IDS, MODEL_IDS, VISION_MODEL_IDS
from .protocol import assert_blinded_payload, payload_hash, verify_envelope


PRIMARY_MODEL_ID = "qwen3_vl_8b_primary"
TEXT_ABLATION_MODEL_ID = "qwen3_8b_text_ablation"


def _probabilities(values: Mapping[Any, Any]) -> dict[int, float]:
    parsed = {int(key): float(value) for key, value in values.items()}
    if (
        len(parsed) < 2
        or any(not isfinite(value) or value < 0.0 for value in parsed.values())
        or abs(fsum(parsed.values()) - 1.0) > 1e-6
    ):
        raise ValueError("multimodal probabilities are not finite and normalized")
    return parsed


def _normalized_entropy(probabilities: Mapping[int, float]) -> float:
    return -fsum(
        value * log(value) for value in probabilities.values() if value > 0.0
    ) / log(len(probabilities))


def balanced_arm_summary(
    source: Mapping[Any, Any],
    reverse: Mapping[Any, Any],
    *,
    normalized_utility: Mapping[int, float],
) -> dict[str, Any]:
    """Summarize one arm after inverse-mapped source/reverse averaging."""

    source_probabilities = _probabilities(source)
    reverse_probabilities = _probabilities(reverse)
    if set(source_probabilities) != set(reverse_probabilities) or set(
        source_probabilities
    ) != set(normalized_utility):
        raise ValueError("multimodal response support does not match utility")
    balanced = weighted_distribution_average(
        source_probabilities, reverse_probabilities
    )
    source_values = tuple(normalized_utility)

    def expected(probabilities: Mapping[int, float]) -> float:
        return fsum(
            probabilities[value] * normalized_utility[value]
            for value in source_values
        )

    def modal(probabilities: Mapping[int, float]) -> int:
        return max(
            source_values,
            key=lambda value: (
                probabilities[value],
                -source_values.index(value),
            ),
        )

    return {
        "source_probabilities": source_probabilities,
        "reverse_probabilities": reverse_probabilities,
        "balanced_probabilities": balanced,
        "source_expected_normalized_utility": expected(source_probabilities),
        "reverse_expected_normalized_utility": expected(reverse_probabilities),
        "balanced_expected_normalized_utility": expected(balanced),
        "source_reverse_total_variation": total_variation(
            source_probabilities, reverse_probabilities
        ),
        "balanced_normalized_response_entropy": _normalized_entropy(balanced),
        "source_modal_response_value": modal(source_probabilities),
        "reverse_modal_response_value": modal(reverse_probabilities),
        "source_reverse_modal_response_stable": modal(source_probabilities)
        == modal(reverse_probabilities),
    }


def _load_verified_outputs(
    root: Path, *, run_root: Path, plan: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    final = verify_envelope(run_root / "final_manifest.json", require_blinded=True)
    expected = {call["call_id"] for call in plan["calls"]}
    if (
        final.get("status") != "prospective_multimodal_passed_54_of_54_stop"
        or final.get("strict_result_count") != 54
        or set(final.get("call_output_sha256", {})) != expected
    ):
        raise ValueError("prospective multimodal run is incomplete")
    outputs: dict[str, dict[str, Any]] = {}
    for call in plan["calls"]:
        path = run_root / call["artifact_relative_path"]
        output = verify_envelope(path, require_blinded=True)
        if (
            payload_hash(output) != final["call_output_sha256"][call["call_id"]]
            or output.get("call_id") != call["call_id"]
            or output.get("outcome_access") != "not_accessed"
        ):
            raise ValueError("prospective multimodal call binding is invalid")
        outputs[call["call_id"]] = output
    return final, outputs


def build_multimodal_recommendations(
    root: Path, *, run_root: Path
) -> dict[str, Any]:
    """Build all-arm recommendations and frozen diagnostics without outcomes."""

    freeze = read_json_object(
        root / "configs/simulators/prospective_multimodal_v4.json"
    )
    verify_prospective_multimodal_freeze(root, freeze)
    plan = read_json_object(
        root / "data/manifests/simulators/prospective_multimodal_plan_v1.json"
    )
    final, outputs = _load_verified_outputs(root, run_root=run_root, plan=plan)
    if (
        final["freeze_payload_sha256"] != payload_hash(freeze)
        or final["plan_payload_sha256"] != payload_hash(plan)
    ):
        raise ValueError("prospective run does not bind the active freeze and plan")
    bundles = {
        experiment_id: read_json_object(
            root
            / f"data/manifests/contracts/{experiment_id}_blinded_bundle.json"
        )
        for experiment_id in EXPERIMENT_IDS
    }
    paired: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for call in plan["calls"]:
        key = (call["model_id"], call["experiment_id"], call["arm_id"])
        paired[key][call["option_order"]] = outputs[call["call_id"]]

    arm_rows: list[dict[str, Any]] = []
    by_arm: dict[tuple[str, str, str], dict[str, Any]] = {}
    for model_id in MODEL_IDS:
        for experiment_id in EXPERIMENT_IDS:
            bundle = bundles[experiment_id]
            utility = {
                int(option["value"]): float(option["normalized_utility"])
                for option in bundle["response_options"]
            }
            for arm in bundle["arms"]:
                key = (model_id, experiment_id, arm["arm_id"])
                ordered = paired[key]
                if set(ordered) != {"source", "reverse"}:
                    raise ValueError("multimodal arm lacks a source/reverse pair")
                summary = balanced_arm_summary(
                    ordered["source"]["probabilities"],
                    ordered["reverse"]["probabilities"],
                    normalized_utility=utility,
                )
                row = {
                    "model_id": model_id,
                    "experiment_id": experiment_id,
                    "arm_id": arm["arm_id"],
                    **summary,
                    "source_order_call_id": ordered["source"]["call_id"],
                    "reverse_order_call_id": ordered["reverse"]["call_id"],
                    "target_human_outcomes_used": False,
                }
                arm_rows.append(row)
                by_arm[key] = row

    decision_rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        for experiment_id in EXPERIMENT_IDS:
            arm_order = [arm["arm_id"] for arm in bundles[experiment_id]["arms"]]

            def choose(field: str) -> str:
                return max(
                    arm_order,
                    key=lambda arm_id: (
                        by_arm[(model_id, experiment_id, arm_id)][field],
                        -arm_order.index(arm_id),
                    ),
                )

            chosen = choose("balanced_expected_normalized_utility")
            source_choice = choose("source_expected_normalized_utility")
            reverse_choice = choose("reverse_expected_normalized_utility")
            utilities = sorted(
                (
                    by_arm[(model_id, experiment_id, arm_id)][
                        "balanced_expected_normalized_utility"
                    ]
                    for arm_id in arm_order
                ),
                reverse=True,
            )
            rows = [by_arm[(model_id, experiment_id, arm_id)] for arm_id in arm_order]
            decision_rows.append(
                {
                    "model_id": model_id,
                    "experiment_id": experiment_id,
                    "admissible_arm_ids": arm_order,
                    "balanced_chosen_arm_id": chosen,
                    "balanced_winner_margin": utilities[0] - utilities[1],
                    "source_order_choice": source_choice,
                    "reverse_order_choice": reverse_choice,
                    "source_reverse_full_action_choice_stable": source_choice
                    == reverse_choice,
                    "mean_arm_source_reverse_total_variation": fmean(
                        row["source_reverse_total_variation"] for row in rows
                    ),
                    "chosen_arm_normalized_response_entropy": by_arm[
                        (model_id, experiment_id, chosen)
                    ]["balanced_normalized_response_entropy"],
                    "target_human_outcomes_used": False,
                }
            )

    decisions = {
        (row["model_id"], row["experiment_id"]): row for row in decision_rows
    }
    experiment_rows: list[dict[str, Any]] = []
    for experiment_id in EXPERIMENT_IDS:
        primary = decisions[(PRIMARY_MODEL_ID, experiment_id)]
        vision_choices = [
            decisions[(model_id, experiment_id)]["balanced_chosen_arm_id"]
            for model_id in VISION_MODEL_IDS
        ]
        text_choice = decisions[(TEXT_ABLATION_MODEL_ID, experiment_id)][
            "balanced_chosen_arm_id"
        ]
        arm_order = primary["admissible_arm_ids"]
        dispersion_by_arm: dict[str, float] = {}
        for arm_id in arm_order:
            values = [
                by_arm[(model_id, experiment_id, arm_id)][
                    "balanced_expected_normalized_utility"
                ]
                for model_id in VISION_MODEL_IDS
            ]
            center = fmean(values)
            dispersion_by_arm[arm_id] = sqrt(
                fmean((value - center) ** 2 for value in values)
            )
        experiment_rows.append(
            {
                "experiment_id": experiment_id,
                "primary_model_chosen_arm_id": primary["balanced_chosen_arm_id"],
                "primary_model_balanced_winner_margin": primary[
                    "balanced_winner_margin"
                ],
                "primary_model_source_reverse_choice_stability": primary[
                    "source_reverse_full_action_choice_stable"
                ],
                "primary_model_mean_arm_source_reverse_total_variation": primary[
                    "mean_arm_source_reverse_total_variation"
                ],
                "two_vlm_complete_action_choice_agreement": len(set(vision_choices))
                == 1,
                "vision_vs_accessible_text_choice_agreement": vision_choices[0]
                == text_choice,
                "primary_model_chosen_arm_normalized_response_entropy": primary[
                    "chosen_arm_normalized_response_entropy"
                ],
                "per_arm_two_vlm_expected_utility_population_sd": dispersion_by_arm,
                "mean_two_vlm_expected_utility_population_sd": fmean(
                    dispersion_by_arm.values()
                ),
                "target_human_outcomes_used": False,
            }
        )

    result = {
        "schema_version": "prospective_multimodal_recommendations.v1",
        "freeze_payload_sha256": payload_hash(freeze),
        "plan_payload_sha256": payload_hash(plan),
        "run_manifest_payload_sha256": payload_hash(final),
        "component_call_output_sha256": final["call_output_sha256"],
        "outcome_access": "not_accessed",
        "prospective_development_experiment_count": 3,
        "balanced_arm_prediction_count": len(arm_rows),
        "model_decision_count": len(decision_rows),
        "experiment_diagnostic_count": len(experiment_rows),
        "balanced_arm_predictions": arm_rows,
        "model_decisions": decision_rows,
        "outcome_free_experiment_diagnostics": experiment_rows,
        "primary_model_id": PRIMARY_MODEL_ID,
        "diagnostic_directions": freeze["diagnostics"]["directions"],
        "trust_threshold_status": "not_fit_or_selected",
        "human_outcome_reveal_authorized": False,
        "automatic_next_stage_authorized": False,
        "interpretation_boundary": (
            "These are outcome-blind prospective-development recommendations and "
            "diagnostics, not evidence of human accuracy, treatment-effect fidelity, "
            "decision regret, calibration, or canonical held-out performance."
        ),
        "status": "complete_outcome_blind_multimodal_recommendations_stop",
    }
    assert_blinded_payload(result)
    return result
