"""Paired outcome-blind analysis of source and reversed answer orders."""

from __future__ import annotations

from collections import defaultdict
from math import ceil, fsum, log
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable, Mapping

from .answer_order_canary import read_json_object
from .forced_choice_screen import EXPERIMENT_IDS
from .modal_forced_choice import MODEL_IDS
from .protocol import assert_blinded_payload, payload_hash, verify_envelope


def total_variation(
    first: Mapping[int, float], second: Mapping[int, float]
) -> float:
    if set(first) != set(second):
        raise ValueError("total-variation distributions have different support")
    return 0.5 * fsum(abs(float(first[key]) - float(second[key])) for key in first)


def jensen_shannon(
    first: Mapping[int, float], second: Mapping[int, float]
) -> float:
    if set(first) != set(second):
        raise ValueError("Jensen-Shannon distributions have different support")
    midpoint = {
        key: 0.5 * (float(first[key]) + float(second[key])) for key in first
    }

    def divergence(values: Mapping[int, float]) -> float:
        return fsum(
            float(value) * log(float(value) / midpoint[key])
            for key, value in values.items()
            if float(value) > 0.0
        )

    return 0.5 * divergence(first) + 0.5 * divergence(second)


def nearest_rank(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or not 0.0 < probability <= 1.0:
        raise ValueError("nearest-rank input is invalid")
    return ordered[ceil(probability * len(ordered)) - 1]


def _probabilities(output: Mapping[str, Any]) -> dict[int, float]:
    return {
        int(value): float(probability)
        for value, probability in output["probabilities"].items()
    }


def _modal_value(
    probabilities: Mapping[int, float], source_values: list[int]
) -> int:
    return max(source_values, key=lambda value: (probabilities[value], -source_values.index(value)))


def _expected_utility(
    probabilities: Mapping[int, float], utility: Mapping[int, float]
) -> float:
    return fsum(probabilities[value] * utility[value] for value in probabilities)


def _verified_output(
    *, run_root: Path, relative_path: str, call_id: str, expected_hash: str
) -> dict[str, Any]:
    path = run_root / relative_path
    envelope = read_json_object(path)
    if payload_hash(envelope["payload"]) != expected_hash:
        raise ValueError(f"call output hash mismatch: {call_id}")
    output = verify_envelope(path, require_blinded=True)
    if output["call_id"] != call_id:
        raise ValueError("call identity differs from its plan")
    return output


def analyze_answer_order(
    root: Path, *, source_run_root: Path, reverse_run_root: Path
) -> dict[str, Any]:
    """Compare paired synthetic distributions without reading human outcomes."""

    source_final = verify_envelope(
        source_run_root / "final_manifest.json", require_blinded=True
    )
    reverse_final = verify_envelope(
        reverse_run_root / "final_manifest.json", require_blinded=True
    )
    if source_final.get("status") != "forced_choice_screen_passed_40_of_40_stop":
        raise ValueError("source-order screen is not a complete passed run")
    if reverse_final.get("status") != "answer_order_canary_passed_40_of_40_stop":
        raise ValueError("reverse-order canary is not a complete passed run")
    source_plan = read_json_object(
        root / "data/manifests/simulators/forced_choice_screen_plan_v1.json"
    )
    reverse_plan = read_json_object(
        root / "data/manifests/simulators/answer_order_canary_plan_v1.json"
    )
    freeze = read_json_object(
        root / "configs/simulators/answer_order_canary_v1.json"
    )
    source_calls = {call["call_id"]: call for call in source_plan["calls"]}
    reverse_calls = {call["call_id"]: call for call in reverse_plan["calls"]}
    if set(source_calls) != set(source_final["call_output_sha256"]):
        raise ValueError("source final manifest does not cover its exact plan")
    if set(reverse_calls) != set(reverse_final["call_output_sha256"]):
        raise ValueError("reverse final manifest does not cover its exact plan")

    bundles = {
        experiment_id: read_json_object(
            root / f"data/manifests/contracts/{experiment_id}_blinded_bundle.json"
        )
        for experiment_id in EXPERIMENT_IDS
    }
    call_rows: list[dict[str, Any]] = []
    utilities: dict[tuple[str, str, str], tuple[float, float]] = {}
    for reverse_id, reverse_call in reverse_calls.items():
        source_id = reverse_call["source_order_call_id"]
        source_call = source_calls[source_id]
        for field in ("model_id", "experiment_id", "arm_id", "variant_id"):
            if source_call[field] != reverse_call[field]:
                raise ValueError(f"paired call field mismatch: {field}")
        source_output = _verified_output(
            run_root=source_run_root,
            relative_path=source_call["artifact_relative_path"],
            call_id=source_id,
            expected_hash=source_final["call_output_sha256"][source_id],
        )
        reverse_output = _verified_output(
            run_root=reverse_run_root,
            relative_path=reverse_call["artifact_relative_path"],
            call_id=reverse_id,
            expected_hash=reverse_final["call_output_sha256"][reverse_id],
        )
        source_probabilities = _probabilities(source_output)
        reverse_probabilities = _probabilities(reverse_output)
        source_values = [int(value) for value in reverse_call["source_option_values"]]
        if set(source_probabilities) != set(source_values) or set(
            reverse_probabilities
        ) != set(source_values):
            raise ValueError("paired probability support differs from source values")
        bundle = bundles[reverse_call["experiment_id"]]
        utility = {
            int(option["value"]): float(option["normalized_utility"])
            for option in bundle["response_options"]
        }
        source_expected = _expected_utility(source_probabilities, utility)
        reverse_expected = _expected_utility(reverse_probabilities, utility)
        source_modal = _modal_value(source_probabilities, source_values)
        reverse_modal = _modal_value(reverse_probabilities, source_values)
        tv = total_variation(source_probabilities, reverse_probabilities)
        call_rows.append(
            {
                "source_order_call_id": source_id,
                "reverse_order_call_id": reverse_id,
                "model_id": reverse_call["model_id"],
                "experiment_id": reverse_call["experiment_id"],
                "arm_id": reverse_call["arm_id"],
                "total_variation": tv,
                "jensen_shannon_divergence_nats": jensen_shannon(
                    source_probabilities, reverse_probabilities
                ),
                "source_order_modal_response": source_modal,
                "reverse_order_modal_response": reverse_modal,
                "modal_response_stable": source_modal == reverse_modal,
                "source_order_expected_normalized_utility": source_expected,
                "reverse_order_expected_normalized_utility": reverse_expected,
                "absolute_expected_utility_change": abs(
                    source_expected - reverse_expected
                ),
            }
        )
        utilities[
            (
                reverse_call["model_id"],
                reverse_call["experiment_id"],
                reverse_call["arm_id"],
            )
        ] = (source_expected, reverse_expected)

    pair_rows = []
    for model_id in MODEL_IDS:
        for experiment_id in EXPERIMENT_IDS:
            arm_order = [
                call["arm_id"]
                for call in source_plan["calls"]
                if call["model_id"] == model_id
                and call["experiment_id"] == experiment_id
            ]
            if len(arm_order) != 2:
                raise ValueError("screened pair must contain exactly two arms")
            first_arm, last_arm = arm_order
            source_first, reverse_first = utilities[
                (model_id, experiment_id, first_arm)
            ]
            source_last, reverse_last = utilities[
                (model_id, experiment_id, last_arm)
            ]
            source_choice = last_arm if source_last > source_first else first_arm
            reverse_choice = last_arm if reverse_last > reverse_first else first_arm
            pair_rows.append(
                {
                    "model_id": model_id,
                    "experiment_id": experiment_id,
                    "screened_first_arm_id": first_arm,
                    "screened_last_arm_id": last_arm,
                    "source_order_choice": source_choice,
                    "reverse_order_choice": reverse_choice,
                    "screened_pair_choice_stable": source_choice == reverse_choice,
                    "source_order_margin": abs(source_last - source_first),
                    "reverse_order_margin": abs(reverse_last - reverse_first),
                    "full_action_set_recommendation": False,
                }
            )

    tv_values = [row["total_variation"] for row in call_rows]
    modal_stability = fmean(
        float(row["modal_response_stable"]) for row in call_rows
    )
    choice_stability = fmean(
        float(row["screened_pair_choice_stable"]) for row in pair_rows
    )
    observed = {
        "median_total_variation": median(tv_values),
        "nearest_rank_p90_total_variation": nearest_rank(tv_values, 0.90),
        "modal_response_stability": modal_stability,
        "screened_pair_choice_stability": choice_stability,
    }
    thresholds = freeze["robustness_gate"]
    checks = {
        "median_total_variation_pass": observed["median_total_variation"]
        <= thresholds["maximum_median_total_variation"],
        "nearest_rank_p90_total_variation_pass": observed[
            "nearest_rank_p90_total_variation"
        ]
        <= thresholds["maximum_nearest_rank_p90_total_variation"],
        "modal_response_stability_pass": observed["modal_response_stability"]
        >= thresholds["minimum_modal_response_stability"],
        "screened_pair_choice_stability_pass": observed[
            "screened_pair_choice_stability"
        ]
        >= thresholds["minimum_screened_pair_choice_stability"],
    }
    gate_passed = all(checks.values())
    model_summaries = []
    for model_id in MODEL_IDS:
        model_calls = [row for row in call_rows if row["model_id"] == model_id]
        model_pairs = [row for row in pair_rows if row["model_id"] == model_id]
        model_summaries.append(
            {
                "model_id": model_id,
                "call_count": len(model_calls),
                "mean_total_variation": fmean(
                    row["total_variation"] for row in model_calls
                ),
                "median_total_variation": median(
                    row["total_variation"] for row in model_calls
                ),
                "modal_response_stability": fmean(
                    float(row["modal_response_stable"]) for row in model_calls
                ),
                "screened_pair_choice_stability": fmean(
                    float(row["screened_pair_choice_stable"])
                    for row in model_pairs
                ),
            }
        )

    result = {
        "schema_version": "answer_order_paired_robustness_diagnostics.v1",
        "source_order_run_manifest_payload_sha256": payload_hash(source_final),
        "reverse_order_run_manifest_payload_sha256": payload_hash(reverse_final),
        "source_call_output_sha256": source_final["call_output_sha256"],
        "reverse_call_output_sha256": reverse_final["call_output_sha256"],
        "freeze_payload_sha256": payload_hash(freeze),
        "outcome_access": "not_accessed",
        "scope": "development_discovery_only_no_human_scoring",
        "paired_call_count": len(call_rows),
        "screened_pair_count": len(pair_rows),
        "observed": observed,
        "thresholds": thresholds,
        "checks": checks,
        "single_order_scaling_gate_passed": gate_passed,
        "method_disposition": (
            "single_order_method_may_scale"
            if gate_passed
            else thresholds["failure_pivot"]
        ),
        "model_summaries": model_summaries,
        "paired_call_results": call_rows,
        "screened_pair_results": pair_rows,
        "interpretation_boundary": (
            "This is a synthetic, outcome-blind prompt-robustness test. It is not a "
            "human-accuracy result, and screened-pair choices are not full-action-set "
            "recommendations."
        ),
        "status": "complete_outcome_blind_answer_order_robustness_diagnostics",
    }
    assert_blinded_payload(result)
    return result
