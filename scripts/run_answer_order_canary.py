"""Authority wrapper for the 40-call reverse-answer-order canary."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from intervenebench.answer_order_canary import (
    prepare_requests,
    read_json_object,
    validate_execution_authorization,
    validate_materialization_authorization,
    validate_runtime_attestation,
    verify_freeze,
    verify_result,
)
from intervenebench.protocol import freeze_envelope, payload_hash, verify_envelope


ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / "configs/simulators/answer_order_canary_v1.json"
PLAN_PATH = ROOT / "data/manifests/simulators/answer_order_canary_plan_v1.json"
APP_PATH = ROOT / "infra/modal/answer_order_app.py"
ARTIFACT_ROOT = ROOT / "artifacts/answer_order_canary"


def _load_app() -> Any:
    spec = importlib.util.spec_from_file_location("answer_order_app", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load answer-order app")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _common() -> tuple[dict[str, Any], dict[str, Any]]:
    verify_freeze(ROOT, freeze_path=FREEZE_PATH, plan_path=PLAN_PATH)
    return read_json_object(FREEZE_PATH), read_json_object(PLAN_PATH)


def materialize(authorization_path: Path, output_path: Path) -> None:
    freeze, plan = _common()
    authorization = verify_envelope(authorization_path, require_blinded=True)
    validate_materialization_authorization(authorization, freeze=freeze, plan=plan)
    modal_app = _load_app()
    with modal_app.app.run(
        name="intervenebench-answer-order-canary-v1",
        environment_name="main",
        detach=False,
        interactive=False,
    ):
        image_id = modal_app.materialized_inference_image_id()
    freeze_envelope(
        {
            "schema_version": "modal_answer_order_materialization.v1",
            "freeze_payload_sha256": payload_hash(freeze),
            "call_plan_payload_sha256": payload_hash(plan),
            "authorization_sha256": payload_hash(authorization),
            "modal_image_id": image_id,
            "status": "materialized_zero_model_calls",
        },
        output_path,
        require_blinded=True,
    )


def _cache_hashes(cache_manifest: Mapping[str, Any]) -> dict[str, str]:
    hashes = cache_manifest.get("cache_attestation_sha256_by_model")
    attestations = cache_manifest.get("cache_attestations")
    if not isinstance(hashes, Mapping) or not isinstance(attestations, Mapping):
        raise ValueError("answer-order cache manifest is incomplete")
    if set(hashes) != set(attestations):
        raise ValueError("answer-order cache manifest model sets differ")
    for model_id, digest in hashes.items():
        if payload_hash(attestations[model_id]) != digest:
            raise ValueError("answer-order cache attestation hash mismatch")
    return dict(hashes)


def _group_requests(
    requests: tuple[dict[str, Any], ...]
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for request in requests:
        grouped.setdefault(request["model_id"], []).append(request)
    if len(grouped) != 4 or any(len(group) != 10 for group in grouped.values()):
        raise ValueError("answer-order calls must form four exact ten-call groups")
    return grouped


def execute(authorization_path: Path, cache_manifest_path: Path, run_id: str) -> None:
    freeze, plan = _common()
    authorization = verify_envelope(authorization_path, require_blinded=True)
    cache_manifest = verify_envelope(cache_manifest_path, require_blinded=True)
    cache_hashes = _cache_hashes(cache_manifest)
    validate_execution_authorization(
        authorization,
        freeze=freeze,
        plan=plan,
        modal_image_id=authorization["modal_image_id"],
        cache_hashes=cache_hashes,
    )
    requests = prepare_requests(ROOT, freeze_path=FREEZE_PATH, plan_path=PLAN_PATH)
    grouped = _group_requests(requests)
    run_root = ARTIFACT_ROOT / run_id
    if run_root.exists():
        raise FileExistsError(f"create-only answer-order run exists: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    raw_groups: dict[str, Mapping[str, Any]] = {}
    modal_app = _load_app()
    try:
        with modal_app.app.run(
            name="intervenebench-answer-order-canary-v1",
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
                response = modal_app.run_answer_order_group.remote(
                    model_id,
                    grouped[model_id],
                    image_id,
                    cache_hashes[model_id],
                )
                return model_id, response

            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(dispatch, model_id) for model_id in grouped]
                for future in as_completed(futures):
                    if (
                        time.monotonic() - started
                        > freeze["limits"]["maximum_wall_clock_seconds"]
                    ):
                        raise TimeoutError("answer-order wall-clock ledger expired")
                    model_id, response = future.result()
                    raw_groups[model_id] = response

        by_id = {request["call_id"]: request for request in requests}
        output_hashes: dict[str, str] = {}
        for model_id, group in raw_groups.items():
            if group.get("model_id") != model_id or len(group.get("results", [])) != 10:
                raise RuntimeError("answer-order model group is incomplete")
            for raw in group["results"]:
                request = by_id[raw["call_id"]]
                validate_runtime_attestation(
                    request,
                    raw["runtime_attestation"],
                    freeze=freeze,
                    authorization=authorization,
                )
                verified = verify_result(request, raw)
                target = run_root / request["artifact_relative_path"]
                output_hashes[request["call_id"]] = freeze_envelope(
                    verified, target, require_blinded=True
                )
        if set(output_hashes) != set(by_id):
            raise RuntimeError("answer-order canary did not return all forty calls")
        freeze_envelope(
            {
                "schema_version": "modal_answer_order_canary_result.v1",
                "run_id": run_id,
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "freeze_payload_sha256": payload_hash(freeze),
                "call_plan_payload_sha256": payload_hash(plan),
                "authorization_sha256": payload_hash(authorization),
                "cache_manifest_sha256": payload_hash(cache_manifest),
                "modal_image_id": authorization["modal_image_id"],
                "attempt_count": 40,
                "strict_result_count": 40,
                "results_per_model": {
                    model_id: 10 for model_id in sorted(grouped)
                },
                "call_output_sha256": dict(sorted(output_hashes.items())),
                "wall_seconds": time.monotonic() - started,
                "next_stage_authorized": False,
                "status": "answer_order_canary_passed_40_of_40_stop",
            },
            run_root / "final_manifest.json",
            require_blinded=True,
        )
    except Exception as error:
        freeze_envelope(
            {
                "schema_version": "modal_answer_order_canary_failure.v1",
                "run_id": run_id,
                "failed_at_utc": datetime.now(UTC).isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "received_model_groups": sorted(raw_groups),
                "attempt_count": 40,
                "next_stage_authorized": False,
                "status": "answer_order_canary_failed_stop",
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
    run = sub.add_parser("execute")
    run.add_argument("--authorization", type=Path, required=True)
    run.add_argument("--cache-manifest", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.command == "materialize":
        materialize(args.authorization, args.output)
    else:
        execute(args.authorization, args.cache_manifest, args.run_id)


if __name__ == "__main__":
    main()
