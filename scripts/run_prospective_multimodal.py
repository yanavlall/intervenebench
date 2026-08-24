"""Authority wrapper for the 54-call prospective multimodal run."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from intervenebench.balanced_forced_choice import read_json_object
from intervenebench.multimodal_freeze import (
    build_cache_authorization,
    build_execution_authorization,
    prepare_multimodal_requests,
    validate_cache_authorization,
    validate_execution_authorization,
    validate_materialization_authorization,
    validate_runtime_attestation,
    verify_multimodal_raw_result,
    verify_prospective_multimodal_freeze,
)
from intervenebench.protocol import freeze_envelope, payload_hash, verify_envelope


ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / "configs/simulators/prospective_multimodal_v4.json"
PLAN_PATH = ROOT / "data/manifests/simulators/prospective_multimodal_plan_v1.json"
APP_PATH = ROOT / "infra/modal/prospective_multimodal_app.py"
ARTIFACT_ROOT = ROOT / "artifacts/prospective_multimodal"
PRIOR_CACHE_PATH = (
    ROOT / "artifacts/modal_discovery_preflight/cache_manifest_20260813_v3.json"
)


def _load_app() -> Any:
    spec = importlib.util.spec_from_file_location(
        "prospective_multimodal_app", APP_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load prospective multimodal app")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _common() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = read_json_object(FREEZE_PATH)
    plan = read_json_object(PLAN_PATH)
    verify_prospective_multimodal_freeze(ROOT, freeze)
    return freeze, plan


def materialize(authorization_path: Path, output_path: Path) -> None:
    freeze, plan = _common()
    authorization = verify_envelope(authorization_path, require_blinded=True)
    validate_materialization_authorization(
        authorization, freeze=freeze, plan=plan
    )
    modal_app = _load_app()
    with modal_app.app.run(
        name="intervenebench-prospective-multimodal-v4",
        environment_name="main",
        detach=False,
        interactive=False,
    ):
        image_id = modal_app.materialized_inference_image_id()
    freeze_envelope(
        {
            "schema_version": "prospective_multimodal_materialization.v1",
            "freeze_payload_sha256": payload_hash(freeze),
            "plan_payload_sha256": payload_hash(plan),
            "authorization_sha256": payload_hash(authorization),
            "modal_image_id": image_id,
            "status": "materialized_zero_model_calls",
        },
        output_path,
        require_blinded=True,
    )


def cache(
    authorization_path: Path,
    materialization_path: Path,
    output_path: Path,
) -> None:
    freeze, plan = _common()
    authorization = verify_envelope(authorization_path, require_blinded=True)
    materialization = verify_envelope(materialization_path, require_blinded=True)
    image_id = materialization["modal_image_id"]
    validate_cache_authorization(
        authorization,
        freeze=freeze,
        plan=plan,
        modal_image_id=image_id,
    )
    prior = verify_envelope(PRIOR_CACHE_PATH, require_blinded=True)
    prior_attestation = prior["cache_attestations"]["qwen3_8b_generic"]
    attestations: dict[str, Mapping[str, Any]] = {
        "qwen3_8b_text_ablation": prior_attestation
    }
    modal_app = _load_app()
    started = time.monotonic()
    with modal_app.app.run(
        name="intervenebench-prospective-multimodal-v4",
        environment_name="main",
        detach=False,
        interactive=False,
    ):
        hydrated_id = modal_app.materialized_inference_image_id()
        validate_cache_authorization(
            authorization,
            freeze=freeze,
            plan=plan,
            modal_image_id=hydrated_id,
        )

        def dispatch(model_id: str) -> tuple[str, Mapping[str, Any]]:
            return model_id, modal_app.cache_multimodal_checkpoint.remote(model_id)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(dispatch, model_id)
                for model_id in authorization["download_model_ids"]
            ]
            for future in as_completed(futures):
                model_id, attestation = future.result()
                attestations[model_id] = attestation
    if set(attestations) != {
        "qwen3_vl_8b_primary",
        "qwen2_5_vl_7b_comparator",
        "qwen3_8b_text_ablation",
    }:
        raise RuntimeError("multimodal cache stage did not produce all bindings")
    hashes = {
        model_id: payload_hash(attestation)
        for model_id, attestation in attestations.items()
    }
    freeze_envelope(
        {
            "schema_version": "prospective_multimodal_cache_manifest.v1",
            "freeze_payload_sha256": payload_hash(freeze),
            "plan_payload_sha256": payload_hash(plan),
            "authorization_sha256": payload_hash(authorization),
            "modal_image_id": image_id,
            "cache_attestations": dict(sorted(attestations.items())),
            "cache_attestation_sha256_by_model": dict(sorted(hashes.items())),
            "cache_wall_seconds": time.monotonic() - started,
            "status": "three_model_cache_bindings_verified",
        },
        output_path,
        require_blinded=True,
    )


def execute(
    authorization_path: Path, cache_manifest_path: Path, run_id: str
) -> None:
    freeze, plan = _common()
    authorization = verify_envelope(authorization_path, require_blinded=True)
    cache_manifest = verify_envelope(cache_manifest_path, require_blinded=True)
    cache_hashes = dict(cache_manifest["cache_attestation_sha256_by_model"])
    validate_execution_authorization(
        authorization,
        freeze=freeze,
        plan=plan,
        modal_image_id=authorization["modal_image_id"],
        cache_hashes=cache_hashes,
    )
    requests = prepare_multimodal_requests(ROOT)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for request in requests:
        grouped.setdefault(request["model_id"], []).append(request)
    if len(grouped) != 3 or any(len(group) != 18 for group in grouped.values()):
        raise ValueError("multimodal calls must form three eighteen-call groups")
    run_root = ARTIFACT_ROOT / run_id
    if run_root.exists():
        raise FileExistsError(f"create-only multimodal run exists: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    raw_groups: dict[str, Mapping[str, Any]] = {}
    modal_app = _load_app()
    try:
        with modal_app.app.run(
            name="intervenebench-prospective-multimodal-v4",
            environment_name="main",
            detach=False,
            interactive=False,
        ):
            image_id = modal_app.materialized_inference_image_id()
            validate_execution_authorization(
                authorization,
                freeze=freeze,
                plan=plan,
                modal_image_id=image_id,
                cache_hashes=cache_hashes,
            )

            def dispatch(model_id: str) -> tuple[str, Mapping[str, Any]]:
                return model_id, modal_app.run_prospective_multimodal_group.remote(
                    model_id,
                    grouped[model_id],
                    image_id,
                    cache_hashes[model_id],
                )

            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(dispatch, model_id) for model_id in grouped]
                for future in as_completed(futures):
                    if time.monotonic() - started > 1800:
                        raise TimeoutError("multimodal wall-clock ledger expired")
                    model_id, response = future.result()
                    raw_groups[model_id] = response
        by_id = {request["call_id"]: request for request in requests}
        output_hashes: dict[str, str] = {}
        for model_id, group in raw_groups.items():
            if group.get("model_id") != model_id or len(group.get("results", [])) != 18:
                raise RuntimeError("multimodal model group is incomplete")
            for raw in group["results"]:
                request = by_id[raw["call_id"]]
                validate_runtime_attestation(
                    request,
                    raw["runtime_attestation"],
                    freeze=freeze,
                    authorization=authorization,
                )
                verified = verify_multimodal_raw_result(request, raw)
                target = run_root / request["artifact_relative_path"]
                output_hashes[request["call_id"]] = freeze_envelope(
                    verified, target, require_blinded=True
                )
        if set(output_hashes) != set(by_id):
            raise RuntimeError("multimodal run did not return all 54 calls")
        freeze_envelope(
            {
                "schema_version": "prospective_multimodal_run_result.v1",
                "run_id": run_id,
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "freeze_payload_sha256": payload_hash(freeze),
                "plan_payload_sha256": payload_hash(plan),
                "authorization_sha256": payload_hash(authorization),
                "cache_manifest_sha256": payload_hash(cache_manifest),
                "modal_image_id": authorization["modal_image_id"],
                "attempt_count": 54,
                "strict_result_count": 54,
                "results_per_model": {model_id: 18 for model_id in sorted(grouped)},
                "call_output_sha256": dict(sorted(output_hashes.items())),
                "wall_seconds": time.monotonic() - started,
                "human_outcome_access_authorized": False,
                "outcome_reveal_authorized": False,
                "automatic_next_stage_authorized": False,
                "status": "prospective_multimodal_passed_54_of_54_stop",
            },
            run_root / "final_manifest.json",
            require_blinded=True,
        )
    except Exception as error:
        freeze_envelope(
            {
                "schema_version": "prospective_multimodal_run_failure.v1",
                "run_id": run_id,
                "failed_at_utc": datetime.now(UTC).isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "received_model_groups": sorted(raw_groups),
                "attempt_count": 54,
                "automatic_next_stage_authorized": False,
                "status": "prospective_multimodal_failed_stop",
            },
            run_root / "failure_manifest.json",
            require_blinded=True,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    mat = sub.add_parser("materialize")
    mat.add_argument("--authorization", type=Path, required=True)
    mat.add_argument("--output", type=Path, required=True)
    cache_parser = sub.add_parser("cache")
    cache_parser.add_argument("--authorization", type=Path, required=True)
    cache_parser.add_argument("--materialization", type=Path, required=True)
    cache_parser.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("execute")
    run.add_argument("--authorization", type=Path, required=True)
    run.add_argument("--cache-manifest", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.command == "materialize":
        materialize(args.authorization, args.output)
    elif args.command == "cache":
        cache(args.authorization, args.materialization, args.output)
    else:
        execute(args.authorization, args.cache_manifest, args.run_id)


if __name__ == "__main__":
    main()
