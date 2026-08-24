"""Build the zero-authority 40-call parser-free discovery freeze."""

from __future__ import annotations

import json
from pathlib import Path

from intervenebench.forced_choice_screen import EXPERIMENT_IDS, read_json_object, sha256_file
from intervenebench.protocol import payload_hash


IMPLEMENTATION_PATHS = (
    "src/intervenebench/modal_forced_choice.py",
    "src/intervenebench/forced_choice_screen.py",
    "scripts/build_forced_choice_screen_plan.py",
    "scripts/build_forced_choice_screen_freeze.py",
    "infra/modal/forced_choice_screen_app.py",
    "scripts/run_forced_choice_screen.py",
)


def build(root: Path) -> dict:
    parent = read_json_object(root / "configs/simulators/modal_forced_choice_v1.json")
    plan = read_json_object(root / "data/manifests/simulators/forced_choice_screen_plan_v1.json")
    model_manifest_path = root / "data/manifests/simulators/model_file_manifests_v1.json"
    model_manifest = read_json_object(model_manifest_path)
    parent_result = root / "artifacts/modal_forced_choice/forced_choice_canary_20260813_v1/final_manifest.json"
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
        "schema_version": "modal_forced_choice_screen_freeze.v1",
        "freeze_id": "intervenebench-forced-choice-discovery-screen-20260813-v1",
        "freeze_date": "2026-08-13",
        "status": "frozen_nonexecuting_zero_authority",
        "purpose": "Run the passed parser-free interface on two source-order arms from each of five blinded development experiments and four frozen models.",
        "parent_hashes": {
            "forced_choice_canary_payload_sha256": payload_hash(parent),
            "forced_choice_canary_final_file_sha256": sha256_file(parent_result),
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
        "dependency_lock": {
            "input_path": "infra/modal/preflight-requirements.in",
            "input_file_sha256": sha256_file(root / "infra/modal/preflight-requirements.in"),
            "lock_path": "infra/modal/preflight-requirements.lock",
            "lock_file_sha256": sha256_file(root / "infra/modal/preflight-requirements.lock"),
            "resolver": "uv==0.12.4",
            "target": "CPython 3.11 x86_64 manylinux_2_28",
        },
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
            "mode": "five_exact_blinded_bundles",
            "access_regime": "DESIGN_ONLY",
            "experiment_ids": list(EXPERIMENT_IDS),
            "all_unlisted_experiments_denied": True,
            "packaged_files": packaged_files,
        },
        "models": parent["models"],
        "runtime": {
            "provider": "modal", "modal_sdk_version": "1.5.4",
            "expected_cuda_runtime_version": "12.8",
            "app_name": "intervenebench-forced-choice-screen-v1",
            "image_recipe": {
                "base": "debian_slim", "python_version": "3.11",
                "source_date_epoch": "2026-08-13T00:00:00Z",
                "dependency_lock": "infra/modal/preflight-requirements.lock",
            },
            "packages": parent["runtime"]["packages"],
            "inference_backend": "single_forward_pass_next_token_softmax",
            "gpu_type": "L40S", "gpu_count": 1, "gpu_fallback_allowed": False,
            "worker_shape": "one_model_load_then_ten_sequential_forward_passes",
            "maximum_total_model_containers": 4,
            "function_timeout_seconds": 600,
            "startup_timeout_seconds": 900,
            "scaledown_window_seconds": 2,
            "model_cache_policy": "reuse_existing_pinned_hash_verified_read_only_volume",
            "network_during_inference": "blocked",
        },
        "method": {
            "method_id": "forced_choice_next_token_softmax.v1",
            "answer_code_range": "A-H prefix selected by source option count",
            "supported_option_counts": [2, 3, 4, 5, 6, 7, 8],
            "token_contract": "each used code appends as exactly one distinct token and decodes exactly for every prompt",
            "temperature": 1.0,
            "generation_calls": 0,
            "semantic_retry_allowed": False,
            "transport_retry_allowed": False,
            "repair_allowed": False,
            "probability_rule": "softmax of final-position logits restricted to source-ordered answer-code token IDs",
        },
        "limits": {
            "maximum_planned_calls": 40,
            "maximum_model_attempts": 40,
            "maximum_wall_clock_seconds": 1200,
            "maximum_gpu_seconds_per_model_group": 600,
            "maximum_aggregate_gpu_seconds": 2400,
            "l40s_price_per_second_usd": 0.000542,
            "maximum_gpu_cost_usd": 1.3008,
            "ancillary_reserve_usd": 0.4492,
            "hard_incremental_cost_cap_usd": 1.75,
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
    target = root / "configs/simulators/forced_choice_screen_v1.json"
    target.write_text(
        json.dumps(build(root), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
