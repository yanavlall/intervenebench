"""Local authority wrapper for the three-stage Modal discovery preflight."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from intervenebench.modal_freeze import verify_modal_preflight_freeze
from intervenebench.modal_runner import (
    PreparedModalCall,
    prepare_modal_calls,
    read_json_object,
    validate_cache_authorization,
    validate_execution_authorization,
    validate_materialization_authorization,
    validate_runtime_attestation,
    verify_remote_result,
)
from intervenebench.protocol import (
    freeze_envelope,
    payload_hash,
    verify_envelope,
)


ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / "configs/simulators/modal_discovery_preflight_v2.json"
CALL_PLAN_PATH = (
    ROOT / "data/manifests/simulators/modal_preflight_call_plan_v1.json"
)
APP_PATH = ROOT / "infra/modal/preflight_app.py"
ARTIFACT_ROOT = ROOT / "artifacts/modal_discovery_preflight"


def _load_modal_app() -> Any:
    spec = importlib.util.spec_from_file_location(
        "preflight_app", APP_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the frozen Modal app module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _verify_common() -> tuple[dict[str, Any], dict[str, Any]]:
    verify_modal_preflight_freeze(
        ROOT, freeze_path=FREEZE_PATH, call_plan_path=CALL_PLAN_PATH
    )
    return read_json_object(FREEZE_PATH), read_json_object(CALL_PLAN_PATH)


def _verify_authorization(path: Path) -> dict[str, Any]:
    return verify_envelope(path, require_blinded=True)


def _modal_run_kwargs(authorization: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": "intervenebench-discovery-preflight-v1",
        "environment_name": "main",
        "detach": False,
        "interactive": False,
    }


def materialize(authorization_path: Path, output_path: Path) -> None:
    """Hydrate the locked image without downloading checkpoints or using GPUs."""

    freeze, plan = _verify_common()
    authorization = _verify_authorization(authorization_path)
    validate_materialization_authorization(
        authorization, freeze=freeze, call_plan=plan
    )
    modal_app = _load_modal_app()
    with modal_app.app.run(**_modal_run_kwargs(authorization)):
        image_id = modal_app.materialized_inference_image_id()
        smoke = modal_app.container_import_smoke.remote()
        if smoke.get("status") != "import_ok" or smoke.get(
            "freeze_payload_sha256"
        ) != payload_hash(freeze):
            raise RuntimeError("remote container import smoke test failed")
    result = {
        "schema_version": "modal_preflight_materialization_attestation.v1",
        "freeze_payload_sha256": payload_hash(freeze),
        "call_plan_payload_sha256": payload_hash(plan),
        "authorization_sha256": payload_hash(authorization),
        "modal_image_id": image_id,
        "image_recipe_sha256": payload_hash(freeze["runtime"]["image_recipe"]),
        "dependency_lock_sha256": freeze["dependency_lock"]["lock_file_sha256"],
        "container_import_smoke": smoke,
        "status": "materialized_no_model_calls",
    }
    freeze_envelope(result, output_path, require_blinded=True)


def cache(authorization_path: Path, output_path: Path) -> None:
    """Cache all four exact models sequentially; GPU inference remains impossible."""

    freeze, plan = _verify_common()
    authorization = _verify_authorization(authorization_path)
    validate_cache_authorization(
        authorization,
        freeze=freeze,
        call_plan=plan,
        modal_image_id=authorization["modal_image_id"],
    )
    modal_app = _load_modal_app()
    cache_attestations: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    with modal_app.app.run(**_modal_run_kwargs(authorization)):
        image_id = modal_app.materialized_inference_image_id()
        validate_cache_authorization(
            authorization,
            freeze=freeze,
            call_plan=plan,
            modal_image_id=image_id,
        )
        for model in freeze["models"]:
            if time.monotonic() - started > 1800:
                raise TimeoutError("cache stage exceeded the frozen wall-clock ledger")
            model_id = model["model_id"]
            result = modal_app.cache_checkpoint.remote(model_id)
            if result.get("model_id") != model_id or result.get("status") != "verified":
                raise RuntimeError(f"cache attestation failed for {model_id}")
            cache_attestations[model_id] = result
    result = {
        "schema_version": "modal_preflight_cache_manifest.v1",
        "freeze_payload_sha256": payload_hash(freeze),
        "call_plan_payload_sha256": payload_hash(plan),
        "authorization_sha256": payload_hash(authorization),
        "modal_image_id": authorization["modal_image_id"],
        "cache_attestations": cache_attestations,
        "cache_attestation_sha256_by_model": {
            model_id: payload_hash(attestation)
            for model_id, attestation in sorted(cache_attestations.items())
        },
        "cache_wall_seconds": time.monotonic() - started,
        "status": "four_exact_checkpoints_verified_no_inference",
    }
    freeze_envelope(result, output_path, require_blinded=True)


def _group_calls(calls: tuple[PreparedModalCall, ...]) -> dict[str, list[PreparedModalCall]]:
    grouped: dict[str, list[PreparedModalCall]] = {}
    for call in calls:
        grouped.setdefault(call.model_id, []).append(call)
    if any(len(group) != 10 for group in grouped.values()) or len(grouped) != 4:
        raise ValueError("preflight calls do not form four exact ten-call model groups")
    return grouped


def _write_call_result(
    *, run_root: Path, call: PreparedModalCall, verified: Mapping[str, Any]
) -> str:
    target = run_root / call.request["artifact_relative_path"]
    digest = freeze_envelope(dict(verified), target, require_blinded=True)
    return digest


def execute(
    authorization_path: Path,
    cache_manifest_path: Path,
    run_id: str,
    transport_retry_of: str | None = None,
) -> None:
    """Run exactly four model groups and require all 40 outputs to parse locally."""

    freeze, plan = _verify_common()
    authorization = _verify_authorization(authorization_path)
    cache_manifest = _verify_authorization(cache_manifest_path)
    cache_hashes = cache_manifest.get("cache_attestation_sha256_by_model")
    if not isinstance(cache_hashes, Mapping):
        raise ValueError("cache manifest is missing attestation hashes")
    attestations = cache_manifest.get("cache_attestations")
    if not isinstance(attestations, Mapping) or set(attestations) != set(cache_hashes):
        raise ValueError("cache manifest attestations are incomplete")
    if any(payload_hash(attestations[model_id]) != digest for model_id, digest in cache_hashes.items()):
        raise ValueError("cache manifest attestation hash mismatch")
    validate_execution_authorization(
        authorization,
        freeze=freeze,
        call_plan=plan,
        modal_image_id=authorization["modal_image_id"],
        cache_attestation_sha256_by_model=cache_hashes,
    )
    calls = prepare_modal_calls(
        ROOT, freeze_path=FREEZE_PATH, call_plan_path=CALL_PLAN_PATH
    )
    grouped = _group_calls(calls)
    run_root = ARTIFACT_ROOT / run_id
    if run_root.exists():
        raise FileExistsError(f"create-only run already exists: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)

    modal_app = _load_modal_app()
    started = time.monotonic()
    attempt_count = len(calls) + (4 if transport_retry_of is not None else 0)
    if attempt_count > authorization["maximum_model_attempts"]:
        raise RuntimeError("frozen model-attempt ceiling exceeded")
    raw_groups: dict[str, dict[str, Any]] = {}
    with modal_app.app.run(**_modal_run_kwargs(authorization)):
        image_id = modal_app.materialized_inference_image_id()
        validate_execution_authorization(
            authorization,
            freeze=freeze,
            call_plan=plan,
            modal_image_id=image_id,
            cache_attestation_sha256_by_model=cache_manifest[
                "cache_attestation_sha256_by_model"
            ],
        )

        def dispatch(model_id: str) -> tuple[str, dict[str, Any]]:
            model_calls = grouped[model_id]
            response = modal_app.run_model_group.remote(
                model_id,
                [call.request for call in model_calls],
                authorization["modal_image_id"],
                authorization["cache_attestation_sha256_by_model"][model_id],
            )
            return model_id, response

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(dispatch, model_id): model_id for model_id in grouped
            }
            for future in as_completed(futures):
                if time.monotonic() - started > freeze["limits"][
                    "maximum_wall_clock_seconds"
                ]:
                    raise TimeoutError("preflight wall-clock ledger expired")
                model_id, response = future.result()
                raw_groups[model_id] = response

    by_call = {call.call_id: call for call in calls}
    output_hashes: dict[str, str] = {}
    parse_counts: dict[str, int] = {model_id: 0 for model_id in grouped}
    observed_call_ids: set[str] = set()
    for model_id, group in raw_groups.items():
        if group.get("model_id") != model_id or len(group.get("results", [])) != 10:
            raise RuntimeError(f"remote model group is incomplete: {model_id}")
        for raw_result in group["results"]:
            call_id = raw_result.get("call_id")
            if call_id in observed_call_ids or call_id not in by_call:
                raise RuntimeError("remote output has duplicate or unknown call ID")
            observed_call_ids.add(call_id)
            call = by_call[call_id]
            validate_runtime_attestation(
                call,
                raw_result["runtime_attestation"],
                freeze=freeze,
                authorization=authorization,
            )
            verified = verify_remote_result(call, raw_result)
            output_hashes[call_id] = _write_call_result(
                run_root=run_root, call=call, verified=verified
            )
            parse_counts[model_id] += 1
    if observed_call_ids != set(by_call) or any(count != 10 for count in parse_counts.values()):
        raise RuntimeError("preflight did not achieve 10/10 strict parses for every model")

    final = {
        "schema_version": "modal_discovery_preflight_result.v1",
        "run_id": run_id,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "freeze_payload_sha256": payload_hash(freeze),
        "call_plan_payload_sha256": payload_hash(plan),
        "authorization_sha256": payload_hash(authorization),
        "cache_manifest_sha256": payload_hash(cache_manifest),
        "modal_image_id": authorization["modal_image_id"],
        "attempt_count": attempt_count,
        "transport_retry_of": transport_retry_of,
        "parse_counts_by_model": parse_counts,
        "call_output_sha256": dict(sorted(output_hashes.items())),
        "wall_seconds": time.monotonic() - started,
        "next_stage_authorized": False,
        "status": "preflight_complete_aggregate_stage_not_authorized",
    }
    freeze_envelope(final, run_root / "final_manifest.json", require_blinded=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("materialize", "cache"):
        child = subparsers.add_parser(command)
        child.add_argument("--authorization", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("execute")
    run.add_argument("--authorization", type=Path, required=True)
    run.add_argument("--cache-manifest", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--transport-retry-of")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "materialize":
        materialize(args.authorization, args.output)
    elif args.command == "cache":
        cache(args.authorization, args.output)
    else:
        execute(
            args.authorization,
            args.cache_manifest,
            args.run_id,
            transport_retry_of=args.transport_retry_of,
        )


if __name__ == "__main__":
    main()
