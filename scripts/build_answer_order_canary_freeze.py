"""Build the zero-authority reverse-answer-order canary freeze."""

from __future__ import annotations

import json
from pathlib import Path

from intervenebench.answer_order_canary import (
    EXPERIMENT_IDS,
    read_json_object,
    sha256_file,
)
from intervenebench.protocol import payload_hash, verify_envelope


IMPLEMENTATION_PATHS = (
    "src/intervenebench/modal_forced_choice.py",
    "src/intervenebench/forced_choice_screen.py",
    "src/intervenebench/answer_order_canary.py",
    "scripts/build_answer_order_canary_plan.py",
    "scripts/build_answer_order_canary_freeze.py",
    "infra/modal/answer_order_app.py",
    "scripts/run_answer_order_canary.py",
)


def build(root: Path) -> dict:
    parent = read_json_object(root / "configs/simulators/forced_choice_screen_v1.json")
    plan = read_json_object(
        root / "data/manifests/simulators/answer_order_canary_plan_v1.json"
    )
    source_plan = read_json_object(
        root / "data/manifests/simulators/forced_choice_screen_plan_v1.json"
    )
    source_final_path = (
        root
        / "artifacts/forced_choice_screen/discovery_screen_20260813_v1/final_manifest.json"
    )
    source_final = verify_envelope(source_final_path, require_blinded=True)
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
        "schema_version": "modal_answer_order_canary_freeze.v1",
        "freeze_id": "intervenebench-reverse-answer-order-canary-20260813-v1",
        "freeze_date": "2026-08-13",
        "status": "frozen_nonexecuting_zero_authority",
        "purpose": (
            "Measure answer-order sensitivity by reversing every option list in the "
            "completed 40-call outcome-blind discovery screen."
        ),
        "parent_hashes": {
            "source_screen_freeze_payload_sha256": payload_hash(parent),
            "source_screen_plan_payload_sha256": payload_hash(source_plan),
            "source_screen_final_file_sha256": sha256_file(source_final_path),
            "source_screen_final_payload_sha256": payload_hash(source_final),
            "source_call_output_sha256": source_final["call_output_sha256"],
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
            "mode": "same_five_exact_blinded_bundles_same_screened_arms",
            "access_regime": "DESIGN_ONLY",
            "experiment_ids": list(EXPERIMENT_IDS),
            "all_unlisted_experiments_denied": True,
            "packaged_files": packaged_files,
        },
        "models": parent["models"],
        "runtime": {
            **parent["runtime"],
            "app_name": "intervenebench-answer-order-canary-v1",
            "worker_shape": "one_model_load_then_ten_reverse_order_forward_passes",
        },
        "method": {
            **parent["method"],
            "answer_order": "full_reverse",
            "inverse_mapping": "map codes back to original source response values",
            "probability_rule": (
                "softmax of final-position logits restricted to reversed-order "
                "answer-code token IDs"
            ),
        },
        "limits": {
            **parent["limits"],
            "hard_incremental_cost_cap_usd": 1.75,
        },
        "robustness_gate": {
            "maximum_median_total_variation": 0.10,
            "maximum_nearest_rank_p90_total_variation": 0.25,
            "minimum_modal_response_stability": 0.75,
            "minimum_screened_pair_choice_stability": 0.80,
            "required_to_scale_single_order_method": "all_thresholds_pass",
            "failure_pivot": (
                "do_not_scale_single_order; develop_balanced_permutation_average"
            ),
        },
        "success_gate": {
            "required_valid_results": 40,
            "required_results_per_model": 10,
            "allow_partial_pass": False,
            "next_stage_automatic": False,
            "stop_after_pass_or_fail": True,
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    target = root / "configs/simulators/answer_order_canary_v1.json"
    target.write_text(
        json.dumps(build(root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
