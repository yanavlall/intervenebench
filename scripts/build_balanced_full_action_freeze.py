"""Build the zero-authority balanced full-action completion freeze."""

from __future__ import annotations

import json
from pathlib import Path

from intervenebench.balanced_forced_choice import (
    EXPERIMENT_IDS,
    read_json_object,
    sha256_file,
)
from intervenebench.protocol import payload_hash, verify_envelope


IMPLEMENTATION_PATHS = (
    "src/intervenebench/forced_choice_screen.py",
    "src/intervenebench/answer_order_canary.py",
    "src/intervenebench/answer_order_analysis.py",
    "src/intervenebench/balanced_forced_choice.py",
    "scripts/build_balanced_discovery_artifact.py",
    "scripts/build_balanced_full_action_plan.py",
    "scripts/build_balanced_full_action_freeze.py",
    "infra/modal/balanced_full_action_app.py",
    "scripts/run_balanced_full_action.py",
    "scripts/build_balanced_full_action_results.py",
)


def build(root: Path) -> dict:
    parent = read_json_object(
        root / "configs/simulators/answer_order_canary_v1.json"
    )
    screen = read_json_object(
        root / "configs/simulators/forced_choice_screen_v1.json"
    )
    plan = read_json_object(
        root / "data/manifests/simulators/balanced_full_action_plan_v1.json"
    )
    balanced_path = (
        root
        / "artifacts/answer_order_canary/answer_order_canary_20260813_v1/"
        "balanced_discovery_predictions.json"
    )
    balanced = verify_envelope(balanced_path, require_blinded=True)
    diagnostics_path = (
        root
        / "artifacts/answer_order_canary/answer_order_canary_20260813_v1/"
        "paired_robustness_diagnostics.json"
    )
    diagnostics = verify_envelope(diagnostics_path, require_blinded=True)
    model_manifest_path = (
        root / "data/manifests/simulators/model_file_manifests_v1.json"
    )
    model_manifest = read_json_object(model_manifest_path)
    packaged_files = []
    for experiment_id in EXPERIMENT_IDS:
        path = root / f"data/manifests/contracts/{experiment_id}_blinded_bundle.json"
        bundle = read_json_object(path)
        packaged_files.append(
            {
                "experiment_id": experiment_id,
                "path": str(path.relative_to(root)),
                "file_sha256": sha256_file(path),
                "payload_sha256": payload_hash(bundle),
            }
        )
    return {
        "schema_version": "balanced_full_action_freeze.v1",
        "freeze_id": "intervenebench-balanced-full-action-20260813-v1",
        "freeze_date": "2026-08-13",
        "status": "frozen_nonexecuting_zero_authority",
        "purpose": (
            "Complete every action in the five blinded discovery bundles using an "
            "equal source/reverse answer-order estimator while reusing all verified calls."
        ),
        "parent_hashes": {
            "answer_order_canary_freeze_payload_sha256": payload_hash(parent),
            "forced_choice_screen_freeze_payload_sha256": payload_hash(screen),
            "robustness_diagnostics_file_sha256": sha256_file(diagnostics_path),
            "robustness_diagnostics_payload_sha256": payload_hash(diagnostics),
            "balanced_discovery_file_sha256": sha256_file(balanced_path),
            "balanced_discovery_payload_sha256": payload_hash(balanced),
        },
        "implementation_hashes": [
            {"path": path, "file_sha256": sha256_file(root / path)}
            for path in IMPLEMENTATION_PATHS
        ],
        "call_plan_payload_sha256": payload_hash(plan),
        "model_file_manifest": {
            "path": str(model_manifest_path.relative_to(root)),
            "file_sha256": sha256_file(model_manifest_path),
            "payload_sha256": payload_hash(model_manifest),
        },
        "dependency_lock": parent["dependency_lock"],
        "authority": {
            "image_materialization_authorized": False,
            "modal_execution_authorized": False,
            "model_download_authorized": False,
            "paid_inference_authorized": False,
            "sealed_task_inference_authorized": False,
            "outcome_access_authorized": False,
            "fine_tuning_authorized": False,
            "next_stage_authorized": False,
        },
        "task_scope": {
            "mode": "five_exact_blinded_bundles_all_admissible_actions",
            "access_regime": "DESIGN_ONLY",
            "experiment_ids": list(EXPERIMENT_IDS),
            "all_unlisted_experiments_denied": True,
            "packaged_files": packaged_files,
        },
        "models": screen["models"],
        "runtime": {
            **screen["runtime"],
            "app_name": "intervenebench-balanced-full-action-v1",
            "worker_shape": "one_model_load_then_fourteen_missing_forward_passes",
            "execution_implementation_status": (
                "hash_bound_execution_wrapper_implemented_zero_authority"
            ),
        },
        "method": {
            "method_id": "balanced_forced_choice_source_reverse.v1",
            "base_method_id": "forced_choice_next_token_softmax.v1",
            "arm_coverage": "all source-declared admissible arms",
            "nuisance_variant_rule": "first source-declared message variant",
            "orders": ["source", "full_reverse"],
            "reverse_mapping": "inverse-map codes to source response values",
            "component_normalization": (
                "renormalize each accepted softmax vector to unit mass"
            ),
            "order_aggregation": (
                "equal_weight_source_and_full_reverse_after_inverse_mapping"
            ),
            "aggregation_formula": (
                "p_balanced(y|arm)=0.5*p_source(y|arm)+0.5*p_reverse(y|arm)"
            ),
            "recommendation_rule": (
                "maximize balanced expected normalized utility over every arm; "
                "tie breaks by source arm order"
            ),
            "arbitrary_permutation_invariance_claimed": False,
            "human_accuracy_claimed": False,
            "generation_calls": 0,
            "semantic_retry_allowed": False,
            "transport_retry_allowed": False,
            "repair_allowed": False,
        },
        "reuse_contract": {
            "logical_ordered_calls": 136,
            "verified_existing_calls": 80,
            "new_calls_required": 56,
            "existing_artifact_hashes": plan["existing_artifact_hashes"],
            "duplicate_existing_calls_forbidden": True,
        },
        "limits": {
            "maximum_planned_new_calls": 56,
            "maximum_model_attempts": 56,
            "new_calls_per_model": 14,
            "maximum_wall_clock_seconds": 1200,
            "maximum_gpu_seconds_per_model_group": 600,
            "maximum_aggregate_gpu_seconds": 2400,
            "l40s_price_per_second_usd": 0.000542,
            "maximum_gpu_cost_usd": 1.3008,
            "ancillary_reserve_usd": 0.4492,
            "hard_incremental_cost_cap_usd": 1.75,
        },
        "success_gate": {
            "required_new_valid_results": 56,
            "required_new_results_per_model": 14,
            "required_logical_ordered_calls_after_reuse": 136,
            "required_balanced_arm_predictions": 68,
            "required_full_action_recommendations": 20,
            "allow_partial_pass": False,
            "next_stage_automatic": False,
            "stop_after_pass_or_fail": True,
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    target = root / "configs/simulators/balanced_full_action_v1.json"
    target.write_text(
        json.dumps(build(root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
