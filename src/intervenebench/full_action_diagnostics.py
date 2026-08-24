"""Outcome-free diagnostics for balanced full-action simulator decisions."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from math import fsum, isfinite, log, sqrt
from pathlib import Path, PurePosixPath
from statistics import fmean, median
from typing import Any, Mapping, Sequence

from .answer_order_analysis import jensen_shannon, total_variation
from .balanced_forced_choice import (
    EXPERIMENT_IDS,
    MODEL_IDS,
    build_completed_full_action_artifact,
    read_json_object,
    sha256_file,
)
from .protocol import assert_blinded_payload, payload_hash, verify_envelope


FULL_RUN_RELATIVE = Path(
    "artifacts/balanced_full_action/balanced_full_action_20260813_v1"
)
GENERIC_MODEL_IDS = MODEL_IDS[:3]


def normalized_entropy(probabilities: Mapping[int, float]) -> float:
    values = [float(value) for value in probabilities.values()]
    if (
        len(values) < 2
        or any(not isfinite(value) or value < 0.0 for value in values)
        or abs(fsum(values) - 1.0) > 1e-6
    ):
        raise ValueError("entropy requires a normalized finite distribution")
    return -fsum(value * log(value) for value in values if value > 0.0) / log(
        len(values)
    )


def _population_sd(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("dispersion requires at least one value")
    center = fmean(values)
    return sqrt(fmean((value - center) ** 2 for value in values))


def _choice_entropy(choices: Sequence[str], *, arm_count: int) -> float:
    counts = Counter(choices)
    if arm_count <= 1 or not choices:
        return 0.0
    return -fsum(
        (count / len(choices)) * log(count / len(choices))
        for count in counts.values()
    ) / log(arm_count)


def _pairwise_agreement(choices: Mapping[str, str]) -> float:
    pairs = list(combinations(choices.values(), 2))
    if not pairs:
        return 1.0
    return fmean(float(first == second) for first, second in pairs)


def verify_diagnostics_freeze(root: Path, *, freeze_path: Path) -> dict[str, Any]:
    freeze = read_json_object(freeze_path)
    if freeze.get("schema_version") != "balanced_full_action_diagnostics_freeze.v1":
        raise ValueError("unsupported full-action diagnostics freeze")
    if freeze.get("status") != "frozen_outcome_free_feature_definitions":
        raise ValueError("diagnostics freeze status is invalid")
    if any(freeze["authority"].values()):
        raise PermissionError("diagnostics freeze embeds expanded authority")
    for entry in freeze["implementation_hashes"]:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("diagnostic implementation path escapes repository")
        if sha256_file(root / relative) != entry["file_sha256"]:
            raise ValueError(f"diagnostic implementation hash mismatch: {entry['path']}")
    for entry in freeze["input_artifacts"]:
        path = root / entry["path"]
        if sha256_file(path) != entry["file_sha256"]:
            raise ValueError(f"diagnostic input file hash mismatch: {entry['path']}")
        if entry.get("envelope_payload_sha256") is not None:
            payload = verify_envelope(path, require_blinded=True)
            if payload_hash(payload) != entry["envelope_payload_sha256"]:
                raise ValueError("diagnostic input payload hash mismatch")
        else:
            payload = read_json_object(path)
            if payload_hash(payload) != entry["json_payload_sha256"]:
                raise ValueError("diagnostic JSON input hash mismatch")
    if freeze["feature_directions"] != {
        "winner_margin": "larger_hypothesized_more_reliable",
        "response_entropy": "smaller_hypothesized_more_reliable",
        "order_total_variation": "smaller_hypothesized_more_reliable",
        "order_choice_stability": "stable_hypothesized_more_reliable",
        "cross_model_choice_agreement": "larger_hypothesized_more_reliable",
        "cross_model_utility_dispersion": "smaller_hypothesized_more_reliable",
    }:
        raise ValueError("diagnostic feature directions drifted")
    assert_blinded_payload(freeze)
    return {
        "feature_family_count": 6,
        "decision_row_count": 20,
        "experiment_row_count": 5,
        "freeze_payload_sha256": payload_hash(freeze),
    }


def _component_output(
    root: Path, run_root: Path, call: Mapping[str, Any],
    new_hashes: Mapping[str, str]
) -> dict[str, Any]:
    if call["acquisition"] == "reuse_verified_existing":
        path = root / call["repository_artifact_path"]
        expected_hash = call["artifact_payload_sha256"]
    else:
        path = run_root / call["artifact_relative_path"]
        expected_hash = new_hashes[call["call_id"]]
    envelope = read_json_object(path)
    if payload_hash(envelope["payload"]) != expected_hash:
        raise ValueError("diagnostic component output hash mismatch")
    output = verify_envelope(path, require_blinded=True)
    if output["call_id"] != call["call_id"]:
        raise ValueError("diagnostic component output identity mismatch")
    return output


def build_full_action_diagnostics(
    root: Path, *, freeze_path: Path
) -> dict[str, Any]:
    """Derive diagnostics from synthetic outputs and design bundles only."""

    freeze_summary = verify_diagnostics_freeze(root, freeze_path=freeze_path)
    freeze = read_json_object(freeze_path)
    run_root = root / FULL_RUN_RELATIVE
    completion = verify_envelope(
        run_root / "final_manifest.json", require_blinded=True
    )
    recommendations = verify_envelope(
        run_root / "full_action_recommendations.json", require_blinded=True
    )
    rebuilt = build_completed_full_action_artifact(root, new_run_root=run_root)
    if payload_hash(recommendations) != payload_hash(rebuilt):
        raise ValueError("full-action recommendation artifact does not replay")
    plan = read_json_object(
        root / "data/manifests/simulators/balanced_full_action_plan_v1.json"
    )
    bundles = {
        experiment_id: read_json_object(
            root / f"data/manifests/contracts/{experiment_id}_blinded_bundle.json"
        )
        for experiment_id in EXPERIMENT_IDS
    }
    calls_by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for call in plan["logical_calls"]:
        calls_by_pair[call["full_action_pair_key"]][call["order_variant"]] = {
            "call": dict(call),
            "output": _component_output(
                root, run_root, call, completion["call_output_sha256"]
            ),
        }
    balanced_rows = {
        (row["model_id"], row["experiment_id"], row["arm_id"]): row
        for row in recommendations["balanced_arm_predictions"]
    }

    arm_rows: list[dict[str, Any]] = []
    source_utility: dict[tuple[str, str, str], float] = {}
    reverse_utility: dict[tuple[str, str, str], float] = {}
    for pair_key, ordered in sorted(calls_by_pair.items()):
        if set(ordered) != {"source", "reverse"}:
            raise ValueError("diagnostic arm lacks source/reverse components")
        model_id, experiment_id, arm_id = pair_key.split("--", 2)
        bundle = bundles[experiment_id]
        source_values = [int(item["value"]) for item in bundle["response_options"]]
        utility = {
            int(item["value"]): float(item["normalized_utility"])
            for item in bundle["response_options"]
        }
        source_probabilities = {
            int(key): float(value)
            for key, value in ordered["source"]["output"]["probabilities"].items()
        }
        reverse_probabilities = {
            int(key): float(value)
            for key, value in ordered["reverse"]["output"]["probabilities"].items()
        }
        balanced = {
            int(key): float(value)
            for key, value in balanced_rows[
                (model_id, experiment_id, arm_id)
            ]["balanced_probabilities"].items()
        }
        source_expected = fsum(
            source_probabilities[value] * utility[value]
            for value in source_probabilities
        )
        reverse_expected = fsum(
            reverse_probabilities[value] * utility[value]
            for value in reverse_probabilities
        )
        source_modal = max(
            source_values,
            key=lambda value: (
                source_probabilities[value], -source_values.index(value)
            ),
        )
        reverse_modal = max(
            source_values,
            key=lambda value: (
                reverse_probabilities[value], -source_values.index(value)
            ),
        )
        source_utility[(model_id, experiment_id, arm_id)] = source_expected
        reverse_utility[(model_id, experiment_id, arm_id)] = reverse_expected
        arm_rows.append(
            {
                "model_id": model_id,
                "experiment_id": experiment_id,
                "arm_id": arm_id,
                "model_exposure": balanced_rows[
                    (model_id, experiment_id, arm_id)
                ]["model_exposure"],
                "balanced_normalized_entropy": normalized_entropy(balanced),
                "balanced_top_response_probability": max(balanced.values()),
                "order_total_variation": total_variation(
                    source_probabilities, reverse_probabilities
                ),
                "order_jensen_shannon_divergence_nats": jensen_shannon(
                    source_probabilities, reverse_probabilities
                ),
                "order_modal_response_stable": source_modal == reverse_modal,
                "source_order_expected_normalized_utility": source_expected,
                "reverse_order_expected_normalized_utility": reverse_expected,
                "absolute_order_expected_utility_shift": abs(
                    source_expected - reverse_expected
                ),
            }
        )

    decision_rows: list[dict[str, Any]] = []
    full_recommendations = {
        (row["model_id"], row["experiment_id"]): row
        for row in recommendations["full_action_recommendations"]
    }
    for model_id in MODEL_IDS:
        for experiment_id in EXPERIMENT_IDS:
            bundle = bundles[experiment_id]
            arm_order = [arm["arm_id"] for arm in bundle["arms"]]
            balanced_utilities = {
                arm_id: float(
                    balanced_rows[(model_id, experiment_id, arm_id)][
                        "balanced_expected_normalized_utility"
                    ]
                )
                for arm_id in arm_order
            }
            source_utilities = {
                arm_id: source_utility[(model_id, experiment_id, arm_id)]
                for arm_id in arm_order
            }
            reverse_utilities = {
                arm_id: reverse_utility[(model_id, experiment_id, arm_id)]
                for arm_id in arm_order
            }
            source_choice = max(
                arm_order,
                key=lambda arm_id: (
                    source_utilities[arm_id], -arm_order.index(arm_id)
                ),
            )
            reverse_choice = max(
                arm_order,
                key=lambda arm_id: (
                    reverse_utilities[arm_id], -arm_order.index(arm_id)
                ),
            )
            ordered_utilities = sorted(balanced_utilities.values(), reverse=True)
            chosen = full_recommendations[(model_id, experiment_id)]["chosen_arm_id"]
            rows = [
                row
                for row in arm_rows
                if row["model_id"] == model_id
                and row["experiment_id"] == experiment_id
            ]
            chosen_row = next(row for row in rows if row["arm_id"] == chosen)
            decision_rows.append(
                {
                    "model_id": model_id,
                    "experiment_id": experiment_id,
                    "model_exposure": chosen_row["model_exposure"],
                    "arm_count": len(arm_order),
                    "balanced_chosen_arm_id": chosen,
                    "balanced_winner_margin": ordered_utilities[0]
                    - ordered_utilities[1],
                    "chosen_arm_response_entropy": chosen_row[
                        "balanced_normalized_entropy"
                    ],
                    "mean_arm_response_entropy": fmean(
                        row["balanced_normalized_entropy"] for row in rows
                    ),
                    "median_arm_order_total_variation": median(
                        row["order_total_variation"] for row in rows
                    ),
                    "maximum_arm_order_total_variation": max(
                        row["order_total_variation"] for row in rows
                    ),
                    "mean_arm_order_total_variation": fmean(
                        row["order_total_variation"] for row in rows
                    ),
                    "mean_arm_order_jensen_shannon_nats": fmean(
                        row["order_jensen_shannon_divergence_nats"]
                        for row in rows
                    ),
                    "arm_modal_response_stability": fmean(
                        float(row["order_modal_response_stable"]) for row in rows
                    ),
                    "source_order_choice": source_choice,
                    "reverse_order_choice": reverse_choice,
                    "source_reverse_full_action_choice_stable": source_choice
                    == reverse_choice,
                    "target_human_outcomes_used": False,
                }
            )

    experiment_rows: list[dict[str, Any]] = []
    for experiment_id in EXPERIMENT_IDS:
        bundle = bundles[experiment_id]
        arm_order = [arm["arm_id"] for arm in bundle["arms"]]
        rows = [
            row for row in decision_rows if row["experiment_id"] == experiment_id
        ]
        choices = {row["model_id"]: row["balanced_chosen_arm_id"] for row in rows}
        generic_choices = {
            model_id: choices[model_id] for model_id in GENERIC_MODEL_IDS
        }
        utility_sds = []
        for arm_id in arm_order:
            values = [
                float(
                    balanced_rows[(model_id, experiment_id, arm_id)][
                        "balanced_expected_normalized_utility"
                    ]
                )
                for model_id in MODEL_IDS
            ]
            utility_sds.append(_population_sd(values))
        counts = Counter(choices.values())
        generic_counts = Counter(generic_choices.values())
        experiment_rows.append(
            {
                "experiment_id": experiment_id,
                "arm_count": len(arm_order),
                "model_choices": choices,
                "unique_model_choice_count": len(counts),
                "maximum_model_choice_agreement": max(counts.values())
                / len(MODEL_IDS),
                "pairwise_model_choice_agreement": _pairwise_agreement(choices),
                "model_choice_entropy": _choice_entropy(
                    list(choices.values()), arm_count=len(arm_order)
                ),
                "generic_model_choices": generic_choices,
                "maximum_generic_choice_agreement": max(generic_counts.values())
                / len(GENERIC_MODEL_IDS),
                "pairwise_generic_choice_agreement": _pairwise_agreement(
                    generic_choices
                ),
                "mean_cross_model_arm_utility_population_sd": fmean(utility_sds),
                "maximum_cross_model_arm_utility_population_sd": max(utility_sds),
                "mean_model_winner_margin": fmean(
                    row["balanced_winner_margin"] for row in rows
                ),
                "mean_model_order_total_variation": fmean(
                    row["mean_arm_order_total_variation"] for row in rows
                ),
                "source_reverse_choice_stability": fmean(
                    float(row["source_reverse_full_action_choice_stable"])
                    for row in rows
                ),
                "target_human_outcomes_used": False,
            }
        )

    result = {
        "schema_version": "balanced_full_action_outcome_free_diagnostics.v1",
        "diagnostics_freeze_payload_sha256": freeze_summary[
            "freeze_payload_sha256"
        ],
        "completion_manifest_payload_sha256": payload_hash(completion),
        "full_recommendations_payload_sha256": payload_hash(recommendations),
        "component_call_output_sha256": recommendations[
            "component_call_output_sha256"
        ],
        "outcome_access_during_diagnostic_build": "not_accessed",
        "target_experiment_outcome_status": (
            "previously_revealed_development_tasks"
        ),
        "scope": "revealed_development_discovery_feature_construction_only",
        "prospective_validation_eligible": False,
        "feature_directions": freeze["feature_directions"],
        "primary_simulator_status": "not_selected_by_this_artifact",
        "trust_threshold_status": "not_fit_or_selected",
        "arm_diagnostic_count": len(arm_rows),
        "decision_diagnostic_count": len(decision_rows),
        "experiment_diagnostic_count": len(experiment_rows),
        "arm_diagnostics": arm_rows,
        "decision_diagnostics": decision_rows,
        "experiment_diagnostics": experiment_rows,
        "summary": {
            "unanimous_all_model_experiments": sum(
                row["unique_model_choice_count"] == 1 for row in experiment_rows
            ),
            "unanimous_generic_model_experiments": sum(
                row["maximum_generic_choice_agreement"] == 1.0
                for row in experiment_rows
            ),
            "source_reverse_choice_stable_decisions": sum(
                row["source_reverse_full_action_choice_stable"]
                for row in decision_rows
            ),
            "mean_decision_winner_margin": fmean(
                row["balanced_winner_margin"] for row in decision_rows
            ),
            "mean_arm_order_total_variation": fmean(
                row["order_total_variation"] for row in arm_rows
            ),
        },
        "interpretation_boundary": (
            "All features use only frozen synthetic outputs and design bundles. "
            "The five target experiments were already revealed in prior development "
            "work, so these rows can formulate diagnostics but cannot prospectively "
            "validate them. They are not validated trust scores. No threshold, primary "
            "simulator, or human-accuracy claim is selected here."
        ),
        "status": "complete_frozen_outcome_free_full_action_diagnostics",
    }
    assert_blinded_payload(result)
    return result
