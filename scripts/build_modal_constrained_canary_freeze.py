"""Build the zero-authority freeze for the four-call constrained canary."""

from __future__ import annotations

import json
from pathlib import Path

from intervenebench.modal_canary import read_json_object, sha256_file
from intervenebench.protocol import payload_hash


IMPLEMENTATION_PATHS = (
    "src/intervenebench/modal_canary.py",
    "scripts/build_modal_constrained_canary_plan.py",
    "scripts/build_modal_constrained_canary_freeze.py",
    "infra/modal/canary_app.py",
    "scripts/run_modal_constrained_canary.py",
)


def build(root: Path) -> dict:
    parent = read_json_object(root / "configs/simulators/modal_discovery_preflight_v2.json")
    plan_path = root / "data/manifests/simulators/modal_constrained_canary_call_plan_v1.json"
    plan = read_json_object(plan_path)
    bundle_path = root / "data/manifests/contracts/5vm8g_blinded_bundle.json"
    bundle = read_json_object(bundle_path)
    model_manifest_path = root / "data/manifests/simulators/model_file_manifests_v1.json"
    model_manifest = read_json_object(model_manifest_path)
    failure_path = root / "artifacts/modal_discovery_preflight/discovery_preflight_20260813_v2/failure_manifest.json"
    return {
        "schema_version": "modal_constrained_canary_freeze.v1",
        "freeze_id": "intervenebench-modal-constrained-canary-20260813-v1",
        "freeze_date": "2026-08-13",
        "status": "frozen_nonexecuting_zero_authority",
        "purpose": "Test exact constrained decoding compatibility with one identical blinded call per frozen simulator model; stop after four calls.",
        "parent_hashes": {
            "preflight_v2_payload_sha256": payload_hash(parent),
            "preflight_v2_failure_file_sha256": sha256_file(failure_path),
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
            "input_path": "infra/modal/canary-requirements.in",
            "input_file_sha256": sha256_file(root / "infra/modal/canary-requirements.in"),
            "lock_path": "infra/modal/canary-requirements.lock",
            "lock_file_sha256": sha256_file(root / "infra/modal/canary-requirements.lock"),
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
            "mode": "single_exact_blinded_bundle",
            "access_regime": "DESIGN_ONLY",
            "experiment_ids": ["5vm8g"],
            "all_unlisted_experiments_denied": True,
            "packaged_file": {
                "experiment_id": "5vm8g",
                "path": str(bundle_path.relative_to(root)),
                "file_sha256": sha256_file(bundle_path),
                "payload_sha256": payload_hash(bundle),
            },
        },
        "models": parent["models"],
        "runtime": {
            "provider": "modal",
            "modal_sdk_version": "1.5.4",
            "expected_cuda_runtime_version": "12.8",
            "app_name": "intervenebench-constrained-canary-v1",
            "image_recipe": {
                "base": "debian_slim",
                "python_version": "3.11",
                "source_date_epoch": "2026-08-13T00:00:00Z",
                "dependency_lock": "infra/modal/canary-requirements.lock",
            },
            "packages": [
                "modal==1.5.4", "torch==2.9.1", "transformers==4.57.6",
                "accelerate==1.14.0", "safetensors==0.8.0",
                "huggingface-hub==0.36.2", "hf-xet==1.6.0",
                "tokenizers==0.22.2", "numpy==2.3.5",
                "outlines==1.2.11", "outlines-core==0.2.14"
            ],
            "inference_backend": "transformers_generate_with_outlines_core_json_schema",
            "gpu_type": "L40S",
            "gpu_count": 1,
            "gpu_fallback_allowed": False,
            "worker_shape": "one_call_one_model_one_single_use_container",
            "maximum_total_model_containers": 4,
            "function_timeout_seconds": 420,
            "startup_timeout_seconds": 900,
            "scaledown_window_seconds": 2,
            "model_cache_policy": "reuse_existing_pinned_hash_verified_read_only_volume",
            "network_during_inference": "blocked",
        },
        "generation": {
            "parser_id": "parse_ordinal_relative_weights.v1",
            "constraint_backend": "outlines_core_json_schema",
            "json_schema": plan["json_schema"],
            "json_schema_sha256": payload_hash(plan["json_schema"]),
            "temperature": 0.2,
            "top_p": 0.95,
            "maximum_output_tokens": 128,
            "semantic_retry_allowed": False,
            "transport_retry_allowed": False,
            "repair_allowed": False,
            "normalization_rule": "positive_integer_relative_weights_divided_by_their_sum",
        },
        "limits": {
            "maximum_planned_calls": 4,
            "maximum_model_attempts": 4,
            "maximum_wall_clock_seconds": 1200,
            "maximum_gpu_seconds_per_call": 420,
            "maximum_aggregate_gpu_seconds": 1680,
            "l40s_price_per_second_usd": 0.000542,
            "maximum_gpu_cost_usd": 0.91056,
            "ancillary_reserve_usd": 0.33944,
            "hard_incremental_cost_cap_usd": 1.25,
        },
        "success_gate": {
            "required_strict_parses": 4,
            "required_calls_per_model": 1,
            "allow_partial_pass": False,
            "next_stage_automatic": False,
            "stop_after_pass_or_fail": True,
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    target = root / "configs/simulators/modal_constrained_canary_v1.json"
    target.write_text(
        json.dumps(build(root), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
