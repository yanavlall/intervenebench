#!/usr/bin/env python3
"""Materialize parseable confirmation outputs without reruns or outcome access."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
from typing import Any

from intervenebench.confirmation_adjudication import (
    EXPECTED_FAILURE_MESSAGE,
    expected_unavailable_call_ids,
    validate_no_rerun_adjudication_authorization,
    validate_unavailable_partition,
)
from intervenebench.confirmation_calls import (
    DEFAULT_CONFIRMATION_CALL_PLAN_PATH,
    prepare_confirmation_requests,
    verify_confirmation_call_plan,
)
from intervenebench.protocol import freeze_envelope, payload_hash, verify_envelope
from run_confirmation import _verify_raw_result


ROOT = Path(__file__).resolve().parents[1]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adjudicate(
    *,
    run_root: Path,
    authorization_path: Path,
    output_root: Path,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"create-only adjudication output exists: {output_root}")
    failure = verify_envelope(run_root / "failure_manifest.json", require_blinded=True)
    audit = verify_envelope(run_root / "strict_parse_audit.json", require_blinded=True)
    plan = verify_confirmation_call_plan(
        ROOT, ROOT / DEFAULT_CONFIRMATION_CALL_PLAN_PATH
    )
    authorization = verify_envelope(authorization_path, require_blinded=True)
    validate_no_rerun_adjudication_authorization(
        authorization,
        run_id=str(failure["run_id"]),
        failure_manifest_payload_sha256=payload_hash(failure),
        strict_parse_audit_payload_sha256=payload_hash(audit),
        call_plan_payload_sha256=payload_hash(plan),
    )
    if failure.get("status") != "confirmation_inference_failed_stop":
        raise ValueError("source run is not the frozen failed confirmation run")
    if audit.get("status") != (
        "inference_complete_strict_validation_failed_no_retry_stop"
    ):
        raise ValueError("strict parse audit status drifted")
    if failure.get("persisted_unparsed_raw_call_count") != 1464:
        raise ValueError("source failure manifest does not preserve all raw calls")

    requests = prepare_confirmation_requests(ROOT, plan=plan, include_reserve=False)
    by_id = {str(request["call_id"]): request for request in requests}
    if len(requests) != 1464 or len(by_id) != 1464:
        raise ValueError("confirmation request set drifted")
    expected_unavailable = expected_unavailable_call_ids(requests)
    if len(expected_unavailable) != 60:
        raise ValueError("frozen unavailable model-task cell is not exactly 60 calls")
    expected_raw_hashes = failure.get("unparsed_raw_call_output_sha256")
    if not isinstance(expected_raw_hashes, dict) or set(expected_raw_hashes) != set(by_id):
        raise ValueError("source raw-output hash map does not cover the call plan")

    verified_by_id: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for call_id, request in by_id.items():
        relative = Path(str(request["artifact_relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("confirmation artifact path escapes its root")
        raw_envelope = verify_envelope(
            run_root / "raw" / relative, require_blinded=True
        )
        if raw_envelope.get("call_id") != call_id:
            raise ValueError("preserved raw call identity drifted")
        if payload_hash(raw_envelope) != expected_raw_hashes[call_id]:
            raise ValueError("preserved raw call hash drifted")
        try:
            verified_by_id[call_id] = _verify_raw_result(
                request, raw_envelope["raw_result"]
            )
        except ValueError as error:
            failures.append(
                {
                    "call_id": call_id,
                    "model_id": request["model_id"],
                    "experiment_id": request["experiment_id"],
                    "method_id": request["method_id"],
                    "stage": request["stage"],
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
    validate_unavailable_partition(
        failures, expected_call_ids=expected_unavailable
    )
    if len(verified_by_id) != 1404 or set(verified_by_id).intersection(
        expected_unavailable
    ):
        raise ValueError("strictly parseable confirmation partition drifted")
    if set(verified_by_id).union(expected_unavailable) != set(by_id):
        raise ValueError("adjudicated partition does not cover the call plan")

    output_root.mkdir(parents=True, exist_ok=False)
    output_hashes: dict[str, str] = {}
    for call_id, verified in verified_by_id.items():
        target = output_root / "strict" / Path(
            str(by_id[call_id]["artifact_relative_path"])
        )
        output_hashes[call_id] = freeze_envelope(
            verified, target, require_blinded=True
        )
    freeze_envelope(
        {
            "schema_version": "confirmation_no_rerun_adjudication.v1",
            "run_id": failure["run_id"],
            "source_failure_manifest_payload_sha256": payload_hash(failure),
            "source_strict_parse_audit_payload_sha256": payload_hash(audit),
            "authorization_payload_sha256": payload_hash(authorization),
            "call_plan_payload_sha256": payload_hash(plan),
            "source_raw_output_map_sha256": payload_hash(expected_raw_hashes),
            "adjudication_script_sha256": _file_sha256(Path(__file__)),
            "adjudication_module_sha256": _file_sha256(
                ROOT / "src/intervenebench/confirmation_adjudication.py"
            ),
            "raw_call_count": len(expected_raw_hashes),
            "strict_output_count": len(output_hashes),
            "unavailable_call_count": len(expected_unavailable),
            "strict_output_sha256_by_call": dict(sorted(output_hashes.items())),
            "unavailable_call_ids": expected_unavailable,
            "unavailable_model_task_cell": {
                "model_id": "socrates_qwen2_5_14b_sft",
                "experiment_id": "tcg8p",
                "method_id": "continuous_constrained_integer_generation.v1",
                "failure_message": EXPECTED_FAILURE_MESSAGE,
                "disposition": "unavailable_schema_noncompliance_no_rerun",
            },
            "model_calls_made": 0,
            "modal_compute_used": False,
            "semantic_repairs_made": 0,
            "confirmation_outcomes_accessed": False,
            "participant_rows_accessed": 0,
            "scoring_performed": False,
            "automatic_next_stage_authorized": False,
            "status": "confirmation_no_rerun_adjudication_complete_stop",
        },
        output_root / "final_manifest.json",
        require_blinded=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    adjudicate(
        run_root=args.run_root,
        authorization_path=args.authorization,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
