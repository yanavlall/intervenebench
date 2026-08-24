#!/usr/bin/env python3
"""Authority wrapper for outcome-blind confirmation inference on Modal."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from intervenebench.confirmation_calls import (
    DEFAULT_CONFIRMATION_CALL_PLAN_PATH,
    prepare_confirmation_requests,
    verify_confirmation_call_plan,
)
from intervenebench.confirmation_execution import (
    DEFAULT_CONFIRMATION_EXECUTION_FREEZE_PATH,
    validate_execution_authorization,
    validate_materialization_authorization,
    verify_confirmation_execution_freeze,
)
from intervenebench.protocol import (
    assert_blinded_payload,
    freeze_envelope,
    payload_hash,
    verify_envelope,
)
from intervenebench.simulators import parse_continuous_prediction


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "infra/modal/confirmation_app.py"
TEXT_CACHE_PATH = (
    ROOT / "artifacts/modal_discovery_preflight/cache_manifest_20260813_v3.json"
)
MM_CACHE_PATH = (
    ROOT / "artifacts/prospective_multimodal/cache_manifest_20260813_v4.json"
)
ARTIFACT_ROOT = ROOT / "artifacts/confirmation"


def _load_app() -> Any:
    spec = importlib.util.spec_from_file_location("confirmation_app", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load confirmation Modal app")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cache_bindings(freeze: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    text = verify_envelope(TEXT_CACHE_PATH, require_blinded=True)
    multimodal = verify_envelope(MM_CACHE_PATH, require_blinded=True)
    text_hashes = text["cache_attestation_sha256_by_model"]
    mm_hashes = multimodal["cache_attestation_sha256_by_model"]
    hashes = {
        "qwen3_8b_generic": text_hashes["qwen3_8b_generic"],
        "qwen3_14b_generic": text_hashes["qwen3_14b_generic"],
        "qwen2_5_14b_generic": text_hashes["qwen2_5_14b_generic"],
        "socrates_qwen2_5_14b_sft": text_hashes[
            "socrates_qwen2_5_14b_sft"
        ],
        "qwen3_vl_8b_primary": mm_hashes["qwen3_vl_8b_primary"],
        "qwen2_5_vl_7b_comparator": mm_hashes["qwen2_5_vl_7b_comparator"],
    }
    if text_hashes["qwen3_8b_generic"] != mm_hashes["qwen3_8b_text_ablation"]:
        raise ValueError("Qwen3-8B cache alias attestations disagree")
    if set(hashes) != set(freeze["cache_model_ids"]):
        raise ValueError("reused cache set does not cover the confirmation freeze")
    return hashes, {
        "text_cache_manifest_payload_sha256": payload_hash(text),
        "multimodal_cache_manifest_payload_sha256": payload_hash(multimodal),
    }


def materialize(authorization_path: Path, output_path: Path) -> None:
    freeze = verify_confirmation_execution_freeze(
        ROOT, ROOT / DEFAULT_CONFIRMATION_EXECUTION_FREEZE_PATH
    )
    authorization = verify_envelope(authorization_path, require_blinded=True)
    validate_materialization_authorization(authorization, freeze=freeze)
    modal_app = _load_app()
    with modal_app.app.run(
        name=freeze["runtime"]["app_name"],
        environment_name="main",
        detach=False,
        interactive=False,
    ):
        image_id = modal_app.materialized_inference_image_id()
    freeze_envelope(
        {
            "schema_version": "confirmation_materialization.v1",
            "freeze_payload_sha256": payload_hash(freeze),
            "call_plan_payload_sha256": freeze["call_plan"]["payload_sha256"],
            "authorization_payload_sha256": payload_hash(authorization),
            "modal_image_id": image_id,
            "model_calls_made": 0,
            "confirmation_outcomes_accessed": False,
            "status": "materialized_zero_inference_stop",
        },
        output_path,
        require_blinded=True,
    )


def _verify_raw_result(
    request: Mapping[str, Any], raw: Mapping[str, Any]
) -> dict[str, Any]:
    if raw.get("call_id") != request["call_id"] or raw.get("model_id") != request[
        "model_id"
    ]:
        raise ValueError("confirmation raw result identity mismatch")
    runtime = raw.get("runtime_attestation")
    if not isinstance(runtime, Mapping):
        raise ValueError("confirmation raw result lacks runtime attestation")
    expected_runtime = {
        "call_id": request["call_id"],
        "prompt_sha256": request["prompt_sha256"],
        "asset_sha256": request["asset_sha256"],
        "method_id": request["method_id"],
    }
    if any(runtime.get(key) != value for key, value in expected_runtime.items()):
        raise ValueError("confirmation per-call runtime binding mismatch")
    verified: dict[str, Any] = {
        "schema_version": "confirmation_raw_call.v1",
        "call_id": request["call_id"],
        "model_id": request["model_id"],
        "experiment_id": request["experiment_id"],
        "arm_id": request["arm_id"],
        "stage": request["stage"],
        "nuisance_id": request["nuisance_id"],
        "answer_order": request["answer_order"],
        "prompt_variant": request["prompt_variant"],
        "prompt_sha256": request["prompt_sha256"],
        "runtime_attestation": dict(runtime),
    }
    if request["method_id"] == "forced_choice_next_token_softmax.v1":
        probabilities = raw.get("probabilities_by_code")
        codes = request["answer_codes"]
        if not isinstance(probabilities, Mapping) or set(probabilities) != set(codes):
            raise ValueError("confirmation forced-choice support mismatch")
        if any(
            not isinstance(probabilities[code], (int, float))
            or isinstance(probabilities[code], bool)
            or not math.isfinite(float(probabilities[code]))
            or float(probabilities[code]) < 0.0
            for code in codes
        ):
            raise ValueError("confirmation forced-choice probabilities are invalid")
        total = sum(float(probabilities[code]) for code in codes)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("confirmation forced-choice probabilities are not normalized")
        by_source = {
            str(source_value): 0.0 for source_value in request["source_option_values"]
        }
        for code, display_value in zip(
            codes, request["display_option_values"], strict=True
        ):
            by_source[str(display_value)] = float(probabilities[code]) / total
        verified["probabilities_by_source_value"] = by_source
        verified["candidate_token_ids"] = raw["candidate_token_ids"]
    else:
        text = raw.get("raw_text")
        if not isinstance(text, str):
            raise ValueError("confirmation continuous result lacks raw text")
        prediction = parse_continuous_prediction(text, integer_only=True)
        verified["raw_text"] = text
        verified["predicted_value"] = prediction.value
        verified["generation_seed"] = request["generation_seed"]
    assert_blinded_payload(verified)
    return verified


def execute(
    authorization_path: Path,
    materialization_path: Path,
    run_id: str,
) -> None:
    freeze = verify_confirmation_execution_freeze(
        ROOT, ROOT / DEFAULT_CONFIRMATION_EXECUTION_FREEZE_PATH
    )
    plan = verify_confirmation_call_plan(
        ROOT, ROOT / DEFAULT_CONFIRMATION_CALL_PLAN_PATH
    )
    authorization = verify_envelope(authorization_path, require_blinded=True)
    materialization = verify_envelope(materialization_path, require_blinded=True)
    image_id = str(materialization["modal_image_id"])
    cache_hashes, cache_provenance = _cache_bindings(freeze)
    validate_execution_authorization(
        authorization,
        freeze=freeze,
        modal_image_id=image_id,
        cache_attestation_sha256_by_model=cache_hashes,
    )
    requests = prepare_confirmation_requests(
        ROOT, plan=plan, include_reserve=False
    )
    model_to_cache = {
        model["model_id"]: model["cache_model_id"] for model in freeze["models"]
    }
    grouped: dict[str, list[dict[str, Any]]] = {
        model_id: [] for model_id in freeze["cache_model_ids"]
    }
    for request in requests:
        grouped[model_to_cache[request["model_id"]]].append(request)
    if {key: len(value) for key, value in grouped.items()} != freeze[
        "planned_calls_by_cache_model"
    ]:
        raise ValueError("confirmation dispatch group counts drifted")

    run_root = ARTIFACT_ROOT / run_id
    if run_root.exists():
        raise FileExistsError(f"create-only confirmation run exists: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    raw_groups: dict[str, Mapping[str, Any]] = {}
    raw_output_hashes: dict[str, str] = {}
    modal_app = _load_app()

    def persist_received_raw_outputs() -> None:
        for group in raw_groups.values():
            for raw in group.get("results", []):
                call_id = str(raw["call_id"])
                if call_id in raw_output_hashes:
                    continue
                request = by_id[call_id]
                raw_output_hashes[call_id] = freeze_envelope(
                    {
                        "schema_version": "confirmation_unparsed_raw_call.v1",
                        "call_id": call_id,
                        "request_prompt_sha256": request["prompt_sha256"],
                        "raw_result": raw,
                    },
                    run_root / "raw" / request["artifact_relative_path"],
                    require_blinded=True,
                )

    by_id = {request["call_id"]: request for request in requests}
    try:
        with modal_app.app.run(
            name=freeze["runtime"]["app_name"],
            environment_name="main",
            detach=False,
            interactive=False,
        ):
            hydrated_id = modal_app.materialized_inference_image_id()
            validate_execution_authorization(
                authorization,
                freeze=freeze,
                modal_image_id=hydrated_id,
                cache_attestation_sha256_by_model=cache_hashes,
            )

            def dispatch(cache_model_id: str) -> tuple[str, Mapping[str, Any]]:
                result = modal_app.run_confirmation_checkpoint_group.remote(
                    cache_model_id,
                    grouped[cache_model_id],
                    hydrated_id,
                    cache_hashes[cache_model_id],
                )
                return cache_model_id, result

            with ThreadPoolExecutor(max_workers=6) as pool:
                futures = [
                    pool.submit(dispatch, cache_model_id)
                    for cache_model_id in grouped
                ]
                for future in as_completed(futures):
                    if time.monotonic() - started > freeze["limits"][
                        "maximum_wall_clock_seconds"
                    ]:
                        raise TimeoutError("confirmation wall-clock ledger expired")
                    cache_model_id, result = future.result()
                    raw_groups[cache_model_id] = result

        # Preserve every unparsed model return before strict local parsing. A
        # parse failure remains a failure and receives no semantic retry, but it
        # must not erase the evidence produced by an expensive outcome-blind run.
        persist_received_raw_outputs()
        output_hashes: dict[str, str] = {}
        for cache_model_id, group in raw_groups.items():
            expected_count = freeze["planned_calls_by_cache_model"][cache_model_id]
            if group.get("cache_model_id") != cache_model_id or len(
                group.get("results", [])
            ) != expected_count:
                raise RuntimeError("confirmation checkpoint group is incomplete")
            for raw in group["results"]:
                request = by_id[raw["call_id"]]
                verified = _verify_raw_result(request, raw)
                target = run_root / request["artifact_relative_path"]
                output_hashes[request["call_id"]] = freeze_envelope(
                    verified, target, require_blinded=True
                )
        if set(output_hashes) != set(by_id):
            raise RuntimeError("confirmation run did not return every planned call")
        freeze_envelope(
            {
                "schema_version": "confirmation_inference_run.v1",
                "run_id": run_id,
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "freeze_payload_sha256": payload_hash(freeze),
                "call_plan_payload_sha256": payload_hash(plan),
                "authorization_payload_sha256": payload_hash(authorization),
                "materialization_payload_sha256": payload_hash(materialization),
                **cache_provenance,
                "modal_image_id": image_id,
                "attempt_count": len(requests),
                "strict_result_count": len(output_hashes),
                "results_by_cache_model": freeze["planned_calls_by_cache_model"],
                "unparsed_raw_call_output_sha256": dict(
                    sorted(raw_output_hashes.items())
                ),
                "call_output_sha256": dict(sorted(output_hashes.items())),
                "wall_seconds": time.monotonic() - started,
                "adaptive_reserve_authorized": False,
                "confirmation_outcome_reveal_authorized": False,
                "automatic_next_stage_authorized": False,
                "status": "confirmation_outcome_blind_inference_complete_stop",
            },
            run_root / "final_manifest.json",
            require_blinded=True,
        )
    except Exception as error:
        try:
            persist_received_raw_outputs()
        except Exception:
            # Preserve the original scientific/runtime error in the failure
            # manifest; any raw-persistence failure is visible from the count.
            pass
        freeze_envelope(
            {
                "schema_version": "confirmation_inference_failure.v1",
                "run_id": run_id,
                "failed_at_utc": datetime.now(UTC).isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "received_checkpoint_groups": sorted(raw_groups),
                "persisted_unparsed_raw_call_count": len(raw_output_hashes),
                "unparsed_raw_call_output_sha256": dict(
                    sorted(raw_output_hashes.items())
                ),
                "planned_attempt_count": len(requests),
                "automatic_next_stage_authorized": False,
                "status": "confirmation_inference_failed_stop",
            },
            run_root / "failure_manifest.json",
            require_blinded=True,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    materialize_parser = sub.add_parser("materialize")
    materialize_parser.add_argument("--authorization", type=Path, required=True)
    materialize_parser.add_argument("--output", type=Path, required=True)
    execute_parser = sub.add_parser("execute")
    execute_parser.add_argument("--authorization", type=Path, required=True)
    execute_parser.add_argument("--materialization", type=Path, required=True)
    execute_parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.command == "materialize":
        materialize(args.authorization, args.output)
    else:
        execute(args.authorization, args.materialization, args.run_id)


if __name__ == "__main__":
    main()
