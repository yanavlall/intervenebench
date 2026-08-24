"""Freeze definitions and input hashes for full-action diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from intervenebench.balanced_forced_choice import read_json_object, sha256_file
from intervenebench.protocol import payload_hash, verify_envelope


IMPLEMENTATION_PATHS = (
    "src/intervenebench/full_action_diagnostics.py",
    "scripts/build_full_action_diagnostics_freeze.py",
    "scripts/build_full_action_diagnostics.py",
)


def build(root: Path) -> dict:
    paths = (
        (
            "data/manifests/simulators/balanced_full_action_plan_v1.json",
            False,
        ),
        ("configs/simulators/balanced_full_action_v1.json", False),
        (
            "artifacts/balanced_full_action/balanced_full_action_20260813_v1/"
            "final_manifest.json",
            True,
        ),
        (
            "artifacts/balanced_full_action/balanced_full_action_20260813_v1/"
            "full_action_recommendations.json",
            True,
        ),
    )
    inputs = []
    for relative, envelope in paths:
        path = root / relative
        entry = {"path": relative, "file_sha256": sha256_file(path)}
        if envelope:
            entry["envelope_payload_sha256"] = payload_hash(
                verify_envelope(path, require_blinded=True)
            )
        else:
            entry["json_payload_sha256"] = payload_hash(read_json_object(path))
        inputs.append(entry)
    return {
        "schema_version": "balanced_full_action_diagnostics_freeze.v1",
        "freeze_id": "intervenebench-full-action-diagnostics-20260813-v2",
        "freeze_date": "2026-08-13",
        "status": "frozen_outcome_free_feature_definitions",
        "purpose": (
            "Construct candidate reliability features on the previously revealed "
            "development discovery set without reading outcomes during this build."
        ),
        "implementation_hashes": [
            {"path": path, "file_sha256": sha256_file(root / path)}
            for path in IMPLEMENTATION_PATHS
        ],
        "input_artifacts": inputs,
        "authority": {
            "human_outcome_access_authorized": False,
            "outcome_reveal_authorized": False,
            "trust_model_fit_authorized": False,
            "trust_threshold_selection_authorized": False,
            "simulator_selection_authorized": False,
            "paid_inference_authorized": False,
            "next_stage_authorized": False,
        },
        "feature_definitions": {
            "winner_margin": (
                "largest minus second-largest balanced expected normalized arm utility"
            ),
            "response_entropy": (
                "Shannon entropy of balanced response distribution divided by log option count"
            ),
            "order_total_variation": (
                "TV between source and inverse-mapped reverse response distributions"
            ),
            "order_choice_stability": (
                "whether complete source-order and reverse-order arm choices agree"
            ),
            "cross_model_choice_agreement": (
                "maximum and pairwise agreement among complete arm choices"
            ),
            "cross_model_utility_dispersion": (
                "population SD across model expected utilities, computed per arm"
            ),
        },
        "feature_directions": {
            "winner_margin": "larger_hypothesized_more_reliable",
            "response_entropy": "smaller_hypothesized_more_reliable",
            "order_total_variation": "smaller_hypothesized_more_reliable",
            "order_choice_stability": "stable_hypothesized_more_reliable",
            "cross_model_choice_agreement": "larger_hypothesized_more_reliable",
            "cross_model_utility_dispersion": "smaller_hypothesized_more_reliable",
        },
        "aggregation_rules": {
            "experiment_is_primary_unit": True,
            "model_rows_do_not_inflate_experiment_n": True,
            "generic_models_reported_separately_from_specialist": True,
            "socrates_known_exposure_labels_preserved": True,
            "ties_follow_source_arm_order": True,
        },
        "selection_boundary": {
            "primary_simulator_selected": False,
            "trust_score_selected": False,
            "trust_threshold_selected": False,
            "target_experiment_outcomes_previously_revealed": True,
            "human_reliability_labels_available_in_repository": True,
            "human_reliability_labels_used_to_construct_features": False,
            "prospective_validation_eligible": False,
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    target = root / "configs/diagnostics/balanced_full_action_v2.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(build(root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
