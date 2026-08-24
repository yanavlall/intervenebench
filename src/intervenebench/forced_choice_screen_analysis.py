"""Outcome-blind discovery diagnostics for the parser-free screen."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import fsum, log, sqrt
from pathlib import Path
from statistics import fmean
from typing import Any

from .forced_choice_screen import EXPERIMENT_IDS, read_json_object
from .modal_forced_choice import MODEL_IDS
from .protocol import assert_blinded_payload, payload_hash, verify_envelope


def _entropy(probabilities: list[float]) -> float:
    return -fsum(value * log(value) for value in probabilities if value > 0.0)


def _population_sd(values: list[float]) -> float:
    mean = fmean(values)
    return sqrt(fmean([(value - mean) ** 2 for value in values]))


def analyze_screen(root: Path, *, run_root: Path) -> dict[str, Any]:
    """Create aggregate synthetic diagnostics without opening human outcomes."""

    final = verify_envelope(run_root / "final_manifest.json", require_blinded=True)
    if final.get("status") != "forced_choice_screen_passed_40_of_40_stop":
        raise ValueError("forced-choice screen is not a complete passed run")
    plan = read_json_object(
        root / "data/manifests/simulators/forced_choice_screen_plan_v1.json"
    )
    by_call = {call["call_id"]: call for call in plan["calls"]}
    if set(by_call) != set(final["call_output_sha256"]):
        raise ValueError("screen final manifest does not cover the exact plan")
    bundles = {
        experiment_id: read_json_object(
            root / f"data/manifests/contracts/{experiment_id}_blinded_bundle.json"
        )
        for experiment_id in EXPERIMENT_IDS
    }
    calls: list[dict[str, Any]] = []
    by_model_experiment: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    modal_by_prompt: dict[tuple[str, str], list[int]] = defaultdict(list)
    for call_id, call in by_call.items():
        path = run_root / call["artifact_relative_path"]
        if not path.is_file():
            raise ValueError(f"screen call artifact is absent: {call_id}")
        envelope = read_json_object(path)
        if payload_hash(envelope["payload"]) != final["call_output_sha256"][call_id]:
            raise ValueError("screen call artifact hash differs from final manifest")
        output = verify_envelope(path, require_blinded=True)
        probabilities = {
            int(value): float(probability)
            for value, probability in output["probabilities"].items()
        }
        bundle = bundles[call["experiment_id"]]
        utility = {
            int(option["value"]): float(option["normalized_utility"])
            for option in bundle["response_options"]
        }
        ordered = [probabilities[int(option["value"])] for option in bundle["response_options"]]
        entropy = _entropy(ordered)
        modal_value = max(
            (int(option["value"]) for option in bundle["response_options"]),
            key=lambda value: (probabilities[value], -list(probabilities).index(value)),
        )
        record = {
            "call_id": call_id,
            "model_id": call["model_id"],
            "experiment_id": call["experiment_id"],
            "arm_id": call["arm_id"],
            "option_count": len(ordered),
            "entropy": entropy,
            "normalized_entropy": entropy / log(len(ordered)),
            "top_probability": max(ordered),
            "modal_response_value": modal_value,
            "expected_normalized_utility": fsum(
                probabilities[value] * utility[value] for value in probabilities
            ),
        }
        calls.append(record)
        by_model_experiment[(call["model_id"], call["experiment_id"])].append(record)
        modal_by_prompt[(call["experiment_id"], call["arm_id"])].append(modal_value)

    model_summaries = []
    for model_id in MODEL_IDS:
        rows = [row for row in calls if row["model_id"] == model_id]
        model_summaries.append(
            {
                "model_id": model_id,
                "call_count": len(rows),
                "mean_normalized_entropy": fmean(row["normalized_entropy"] for row in rows),
                "mean_top_probability": fmean(row["top_probability"] for row in rows),
            }
        )

    prompt_agreement = []
    for (experiment_id, arm_id), modal_values in sorted(modal_by_prompt.items()):
        counts = Counter(modal_values)
        prompt_agreement.append(
            {
                "experiment_id": experiment_id,
                "arm_id": arm_id,
                "unique_modal_response_count": len(counts),
                "maximum_modal_agreement_fraction": max(counts.values()) / len(MODEL_IDS),
                "unanimous": len(counts) == 1,
                "all_models_different": len(counts) == len(MODEL_IDS),
            }
        )

    pair_rows = []
    for (model_id, experiment_id), rows in sorted(by_model_experiment.items()):
        arm_order = [
            call["arm_id"]
            for call in plan["calls"]
            if call["model_id"] == model_id and call["experiment_id"] == experiment_id
        ]
        if len(rows) != 2 or len(arm_order) != 2:
            raise ValueError("screen pair does not have exactly two arms")
        indexed = {row["arm_id"]: row for row in rows}
        first, last = (indexed[arm_id] for arm_id in arm_order)
        effect = last["expected_normalized_utility"] - first["expected_normalized_utility"]
        chosen_arm = (
            last["arm_id"] if last["expected_normalized_utility"] > first["expected_normalized_utility"]
            else first["arm_id"]
        )
        pair_rows.append(
            {
                "model_id": model_id,
                "experiment_id": experiment_id,
                "screened_first_arm_id": first["arm_id"],
                "screened_last_arm_id": last["arm_id"],
                "synthetic_effect_last_minus_first": effect,
                "absolute_screened_pair_margin": abs(effect),
                "screened_pair_choice": chosen_arm,
                "full_action_set_recommendation": False,
            }
        )

    experiment_summaries = []
    for experiment_id in EXPERIMENT_IDS:
        rows = [row for row in pair_rows if row["experiment_id"] == experiment_id]
        choices = Counter(row["screened_pair_choice"] for row in rows)
        effects = [row["synthetic_effect_last_minus_first"] for row in rows]
        experiment_summaries.append(
            {
                "experiment_id": experiment_id,
                "model_count": len(rows),
                "maximum_screened_pair_choice_agreement_fraction": max(choices.values()) / len(rows),
                "screened_pair_choice_unanimous": len(choices) == 1,
                "mean_synthetic_effect_last_minus_first": fmean(effects),
                "synthetic_effect_population_sd": _population_sd(effects),
                "minimum_synthetic_effect": min(effects),
                "maximum_synthetic_effect": max(effects),
                "full_action_set_recommendation": False,
            }
        )

    result = {
        "schema_version": "forced_choice_screen_discovery_diagnostics.v1",
        "source_run_manifest_payload_sha256": payload_hash(final),
        "source_call_output_sha256": final["call_output_sha256"],
        "outcome_access": "not_accessed",
        "scope": "development_discovery_only_no_human_scoring",
        "call_count": len(calls),
        "model_summaries": model_summaries,
        "prompt_agreement": prompt_agreement,
        "screened_pair_results": pair_rows,
        "experiment_summaries": experiment_summaries,
        "summary": {
            "unanimous_modal_response_prompts": sum(row["unanimous"] for row in prompt_agreement),
            "all_models_different_prompts": sum(row["all_models_different"] for row in prompt_agreement),
            "prompt_count": len(prompt_agreement),
            "unanimous_screened_pair_choices": sum(
                row["screened_pair_choice_unanimous"] for row in experiment_summaries
            ),
            "experiment_count": len(experiment_summaries),
        },
        "interpretation_boundary": (
            "These are synthetic, outcome-blind discovery diagnostics. Screened-pair "
            "choices are not full-action-set recommendations and are not accuracy claims."
        ),
        "status": "complete_outcome_blind_discovery_diagnostics",
    }
    assert_blinded_payload(result)
    return result
