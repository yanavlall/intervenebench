#!/usr/bin/env python3
"""Run only the 504 never-submitted Mistral calls, then freeze recommendations."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

from intervenebench.cross_family_continuation import (
    DEFAULT_CONTINUATION_AUTHORIZATION_PATH,
    DEFAULT_CONTINUATION_PROGRESS_PATH,
    DEFAULT_CONTINUATION_RUN_ROOT,
    EXPECTED_REMAINING_CALL_COUNT,
    continuation_partition,
    unavailable_transport_rows,
    validate_continuation_authorization,
)
from intervenebench.cross_family_target_run import (
    build_recommendations,
    load_target_bindings,
    strict_parse_target_result,
)
from intervenebench.cross_family_seedfix import (
    SEEDFIX_APP_PATH,
    SEEDFIX_CONTINUATION_AUTHORIZATION_PATH,
    SEEDFIX_CONTINUATION_PROGRESS_PATH,
    SEEDFIX_CONTINUATION_RUN_ROOT,
    load_seedfix_canary,
    load_seedfix_materialization,
    validate_seedfix_continuation_authorization,
    verify_seedfix_freeze,
    SEEDFIX_V2_APP_PATH,
    SEEDFIX_V2_CONTINUATION_AUTHORIZATION_PATH,
    SEEDFIX_V2_CONTINUATION_PROGRESS_PATH,
    SEEDFIX_V2_CONTINUATION_RUN_ROOT,
    load_seedfix_v2_canary,
    load_seedfix_v2_materialization,
    validate_seedfix_v2_continuation_authorization,
    verify_seedfix_v2_freeze,
)
from intervenebench.protocol import (
    assert_blinded_payload,
    freeze_envelope,
    payload_hash,
    verify_envelope,
)


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "infra/modal/cross_family_target_app.py"


class _Reporter:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8"):
            pass

    def emit(self, state: str, **details: Any) -> None:
        row = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "stage": "cross_family_no_rerun_504_call_continuation",
            "state": state,
            **details,
        }
        print(
            f"[{row['timestamp_utc']}] {state} {details if details else ''}",
            flush=True,
        )
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _load_app(app_path: Path = APP_PATH) -> Any:
    spec = importlib.util.spec_from_file_location(
        f"infra.modal.{app_path.stem}", app_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the cross-family target Modal app")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _await_chunk(
    function: Any,
    args: tuple[Any, ...],
    *,
    chunk_id: str,
    experiment_id: str,
    reporter: _Reporter,
    timeout_error_type: type[Exception],
    global_started: float,
    global_deadline_seconds: float,
) -> tuple[Mapping[str, Any], float]:
    chunk_started = time.monotonic()
    call = function.spawn(*args)
    modal_call_id = getattr(call, "object_id", "unavailable")
    reporter.emit(
        "chunk_submitted",
        chunk_id=chunk_id,
        experiment_id=experiment_id,
        modal_function_call_id=modal_call_id,
    )
    while True:
        total_elapsed = time.monotonic() - global_started
        remaining = global_deadline_seconds - total_elapsed
        if remaining <= 0:
            call.cancel(terminate_containers=True)
            reporter.emit(
                "global_deadline_exceeded_cancelled",
                chunk_id=chunk_id,
                experiment_id=experiment_id,
                modal_function_call_id=modal_call_id,
                total_elapsed_seconds=round(total_elapsed, 1),
            )
            raise TimeoutError("cross-family continuation exceeded its global deadline")
        try:
            result = call.get(timeout=min(30.0, remaining))
        except (TimeoutError, timeout_error_type):
            reporter.emit(
                "chunk_active",
                chunk_id=chunk_id,
                experiment_id=experiment_id,
                modal_function_call_id=modal_call_id,
                chunk_elapsed_seconds=round(time.monotonic() - chunk_started, 1),
                total_elapsed_seconds=round(time.monotonic() - global_started, 1),
            )
            continue
        elapsed = time.monotonic() - chunk_started
        reporter.emit(
            "chunk_completed",
            chunk_id=chunk_id,
            experiment_id=experiment_id,
            modal_function_call_id=modal_call_id,
            chunk_elapsed_seconds=round(elapsed, 1),
        )
        return result, elapsed


def _result_path(
    run_root: Path, request: Mapping[str, Any], *, kind: str
) -> Path:
    relative = Path(str(request["artifact_relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("continuation artifact path escapes its run root")
    if kind not in {"raw", "strict"}:
        raise ValueError("unsupported continuation artifact kind")
    return run_root / kind / relative


def _validate_remote_chunk(
    result: Mapping[str, Any], group: list[Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    returned = result.get("results")
    if (
        result.get("schema_version")
        != "intervenebench.cross_family_target_group.v1"
        or result.get("attempt_count") != len(group)
        or not isinstance(returned, list)
        or len(returned) != len(group)
        or result.get("model_downloaded") is not False
        or result.get("human_outcomes_accessed") is not False
        or result.get("participant_rows_read") != 0
        or result.get("automatic_next_stage") is not False
    ):
        raise RuntimeError("continuation remote chunk result drifted")
    returned_by_id = {str(row.get("call_id")): row for row in returned}
    expected_ids = {str(row["call_id"]) for row in group}
    if set(returned_by_id) != expected_ids:
        raise RuntimeError("continuation remote chunk does not cover its call IDs")
    return returned_by_id


def execute(
    *, authorization_path: Path, run_root: Path, progress_log_path: Path
) -> str:
    if run_root.exists():
        raise FileExistsError(f"create-only continuation run exists: {run_root}")
    bindings = load_target_bindings(ROOT)
    freeze = bindings["freeze"]
    cache = bindings["cache"]
    authorization = verify_envelope(authorization_path, require_blinded=True)
    schema_version = authorization.get("schema_version")
    seedfix_version = (
        2
        if schema_version
        == "intervenebench.cross_family_seedfix_v2_continuation_authorization.v1"
        else 1
        if schema_version
        == "intervenebench.cross_family_seedfix_continuation_authorization.v1"
        else 0
    )
    seedfix_mode = seedfix_version > 0
    if seedfix_version == 2:
        validate_seedfix_v2_continuation_authorization(authorization, root=ROOT)
        seedfix_freeze = verify_seedfix_v2_freeze(ROOT)
        materialization = load_seedfix_v2_materialization(ROOT)
        seedfix_canary = load_seedfix_v2_canary(ROOT)
        app_path = ROOT / SEEDFIX_V2_APP_PATH
        modal_app_name = str(seedfix_freeze["runtime"]["app_name"])
    elif seedfix_version == 1:
        validate_seedfix_continuation_authorization(authorization, root=ROOT)
        seedfix_freeze = verify_seedfix_freeze(ROOT)
        materialization = load_seedfix_materialization(ROOT)
        seedfix_canary = load_seedfix_canary(ROOT)
        app_path = ROOT / SEEDFIX_APP_PATH
        modal_app_name = str(seedfix_freeze["runtime"]["app_name"])
    else:
        validate_continuation_authorization(authorization, root=ROOT)
        materialization = bindings["materialization"]
        seedfix_freeze = None
        seedfix_canary = None
        app_path = APP_PATH
        modal_app_name = str(freeze["runtime"]["app_name"])
    partition = continuation_partition(ROOT)
    all_requests = [
        *partition["unavailable_requests"], *partition["remaining_requests"]
    ]
    # The frozen plan orders tcg8p first, followed by the five remaining tasks;
    # this concatenation therefore preserves the exact original 624-call order.
    remaining = partition["remaining_requests"]
    by_id = {str(request["call_id"]): request for request in remaining}
    if len(remaining) != EXPECTED_REMAINING_CALL_COUNT or len(by_id) != len(remaining):
        raise ValueError("continuation request set must contain 504 unique calls")
    chunks = authorization["chunk_plan"]

    run_root.mkdir(parents=True, exist_ok=False)
    reporter = _Reporter(progress_log_path)
    reporter.emit(
        "local_authorization_verified",
        remaining_call_count=len(remaining),
        permanently_unavailable_call_count=len(partition["unavailable_requests"]),
        chunk_count=len(chunks),
    )
    # Importing Modal is deliberately after the pure-local authority check.
    modal_app = _load_app(app_path)
    reporter.emit("modal_package_imported")
    global_started = time.monotonic()
    raw_hashes: dict[str, str] = {}
    strict_hashes: dict[str, str] = {}
    strict_outputs: dict[str, dict[str, Any]] = {}
    parse_failures: list[dict[str, Any]] = unavailable_transport_rows(
        partition["unavailable_requests"]
    )
    chunk_elapsed: dict[str, float] = {}
    completed_chunks: list[str] = []

    try:
        with modal_app.modal.enable_output():
            with modal_app.app.run(
                name=modal_app_name,
                environment_name="main",
                detach=False,
                interactive=False,
            ):
                hydrated_id = modal_app.materialized_target_image_id()
                if hydrated_id != authorization["modal_image_id"]:
                    raise RuntimeError(
                        "continuation hydrated image differs from authorization"
                    )
                reporter.emit(
                    "modal_image_binding_verified", modal_image_id=hydrated_id
                )
                for chunk_number, chunk in enumerate(chunks, start=1):
                    group = [by_id[str(call_id)] for call_id in chunk["call_ids"]]
                    if (
                        len(group) != chunk["call_count"]
                        or payload_hash(
                            {row["call_id"]: payload_hash(row) for row in group}
                        )
                        != chunk["request_payload_hashes_sha256"]
                    ):
                        raise RuntimeError("continuation chunk binding drifted")
                    result, elapsed = _await_chunk(
                        modal_app.run_cross_family_target_group,
                        (
                            group,
                            payload_hash(freeze),
                            hydrated_id,
                            payload_hash(cache),
                        ),
                        chunk_id=str(chunk["chunk_id"]),
                        experiment_id=str(chunk["experiment_id"]),
                        reporter=reporter,
                        timeout_error_type=modal_app.modal.exception.TimeoutError,
                        global_started=global_started,
                        global_deadline_seconds=float(
                            authorization["maximum_wall_clock_seconds"]
                        ),
                    )
                    returned_by_id = _validate_remote_chunk(result, group)
                    if seedfix_mode and any(
                        row.get("runtime_attestation", {}).get(
                            "seedfix_v2_payload_sha256"
                            if seedfix_version == 2
                            else "seedfix_payload_sha256"
                        )
                        != payload_hash(seedfix_freeze)
                        or row.get("result", {}).get("null_seed_normalized")
                        is not True
                        or row.get("result", {}).get("effective_generation_seed")
                        != 0
                        for row in returned_by_id.values()
                    ):
                        raise RuntimeError("seed-fix target runtime attestation drifted")
                    for request in group:
                        call_id = str(request["call_id"])
                        raw = returned_by_id[call_id]
                        raw_hashes[call_id] = freeze_envelope(
                            {
                                "schema_version": (
                                    "intervenebench.cross_family_unparsed_raw_call.v1"
                                ),
                                "call_id": call_id,
                                "request_payload_sha256": payload_hash(request),
                                "raw_result": raw,
                            },
                            _result_path(run_root, request, kind="raw"),
                            require_blinded=True,
                        )
                    for request in group:
                        call_id = str(request["call_id"])
                        try:
                            strict = strict_parse_target_result(
                                request,
                                returned_by_id[call_id],
                                freeze=freeze,
                                expected_modal_image_id=hydrated_id,
                            )
                        except ValueError as error:
                            parse_failures.append(
                                {
                                    "call_id": call_id,
                                    "experiment_id": str(request["experiment_id"]),
                                    "method_id": str(request["method_id"]),
                                    "stage": str(request["stage"]),
                                    "error_type": type(error).__name__,
                                    "error_message": str(error),
                                }
                            )
                            continue
                        strict_outputs[call_id] = strict
                        strict_hashes[call_id] = freeze_envelope(
                            strict,
                            _result_path(run_root, request, kind="strict"),
                            require_blinded=True,
                        )
                    completed_chunks.append(str(chunk["chunk_id"]))
                    chunk_elapsed[str(chunk["chunk_id"])] = elapsed
                    reporter.emit(
                        "chunk_raw_frozen_and_strict_parse_complete",
                        chunk_id=str(chunk["chunk_id"]),
                        chunk_number=chunk_number,
                        chunk_count=len(chunks),
                        completed_call_count=len(raw_hashes),
                        strict_call_count=len(strict_hashes),
                        new_parse_failure_count=(
                            len(parse_failures)
                            - len(partition["unavailable_requests"])
                        ),
                    )

        if set(raw_hashes) != set(by_id):
            raise RuntimeError("continuation did not preserve all 504 raw outputs")
        new_failures = [
            row for row in parse_failures if row["experiment_id"] != "tcg8p"
        ]
        if set(strict_hashes).union(row["call_id"] for row in new_failures) != set(
            by_id
        ):
            raise RuntimeError("continuation strict/failure partition is incomplete")
        if set(strict_hashes).intersection(row["call_id"] for row in new_failures):
            raise RuntimeError("continuation strict/failure partition overlaps")
        failure_rows = sorted(parse_failures, key=lambda row: row["call_id"])
        recommendations = build_recommendations(
            ROOT,
            requests=all_requests,
            strict_outputs=strict_outputs,
            strict_output_hashes=dict(sorted(strict_hashes.items())),
            parse_failures=failure_rows,
        )
        recommendations.update(
            {
                "continuation_authorization_payload_sha256": payload_hash(
                    authorization
                ),
                "freeze_payload_sha256": payload_hash(freeze),
                "materialization_payload_sha256": payload_hash(materialization),
                "raw_output_map_sha256": payload_hash(dict(sorted(raw_hashes.items()))),
                "tcg8p_disposition": "transport_output_unavailable_no_rerun",
                "seedfix_applied": seedfix_mode,
                "seedfix_version": seedfix_version,
            }
        )
        if seedfix_mode:
            recommendations.update(
                {
                    "seedfix_freeze_payload_sha256": payload_hash(seedfix_freeze),
                    "seedfix_canary_payload_sha256": payload_hash(seedfix_canary),
                    "duplicated_target_inference_count": 0,
                }
            )
            if seedfix_version == 1:
                recommendations["prior_null_seed_failure_payload_sha256"] = (
                    authorization["prior_null_seed_failure_payload_sha256"]
                )
        else:
            recommendations.update(
                {
                    "source_target_authorization_payload_sha256": authorization[
                        "source_target_authorization_payload_sha256"
                    ],
                    "source_failure_manifest_payload_sha256": authorization[
                        "source_failure_manifest_payload_sha256"
                    ],
                }
            )
        assert_blinded_payload(recommendations)
        recommendation_hash = freeze_envelope(
            recommendations,
            run_root / "recommendations_v1.json",
            require_blinded=True,
        )
        final = {
            "schema_version": "intervenebench.cross_family_no_rerun_continuation.v1",
            "status": (
                "completed_504_calls_tcg8p_unavailable_recommendations_frozen_stop"
            ),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "freeze_payload_sha256": payload_hash(freeze),
            "continuation_authorization_payload_sha256": payload_hash(authorization),
            "materialization_payload_sha256": payload_hash(materialization),
            "modal_image_id": hydrated_id,
            "checkpoint_commit": freeze["model"]["checkpoint_commit"],
            "source_unavailable_experiment_id": "tcg8p",
            "source_unavailable_call_count": 120,
            "source_unavailable_disposition": "transport_output_unavailable_no_rerun",
            "continuation_attempt_count": len(remaining),
            "continuation_raw_output_count": len(raw_hashes),
            "strict_output_count": len(strict_hashes),
            "strict_parse_failure_count": len(failure_rows),
            "new_strict_parse_failure_count": len(new_failures),
            "raw_output_sha256_by_call": dict(sorted(raw_hashes.items())),
            "strict_output_sha256_by_call": dict(sorted(strict_hashes.items())),
            "strict_parse_failures": failure_rows,
            "recommendations_payload_sha256": recommendation_hash,
            "completed_chunk_count": len(completed_chunks),
            "completed_chunks": completed_chunks,
            "remote_chunk_elapsed_seconds": chunk_elapsed,
            "total_wall_seconds": time.monotonic() - global_started,
            "automatic_retries": 0,
            "tcg8p_reruns": 0,
            "reserve_calls": 0,
            "semantic_repairs": 0,
            "model_downloads": 0,
            "human_outcomes_accessed": False,
            "participant_rows_accessed": 0,
            "human_outcome_scoring_performed": False,
            "automatic_next_stage": False,
            "seedfix_applied": seedfix_mode,
            "seedfix_version": seedfix_version,
        }
        if seedfix_mode:
            final.update(
                {
                    "seedfix_freeze_payload_sha256": payload_hash(seedfix_freeze),
                    "seedfix_canary_payload_sha256": payload_hash(seedfix_canary),
                    "duplicated_target_inference_count": 0,
                }
            )
            if seedfix_version == 1:
                final["prior_null_seed_failure_payload_sha256"] = authorization[
                    "prior_null_seed_failure_payload_sha256"
                ]
        else:
            final.update(
                {
                    "source_target_authorization_payload_sha256": authorization[
                        "source_target_authorization_payload_sha256"
                    ],
                    "source_failure_manifest_payload_sha256": authorization[
                        "source_failure_manifest_payload_sha256"
                    ],
                }
            )
        digest = freeze_envelope(
            final, run_root / "final_manifest.json", require_blinded=True
        )
        reporter.emit(
            "continuation_and_recommendations_frozen_stop",
            final_manifest_payload_sha256=digest,
            strict_output_count=len(strict_hashes),
            strict_parse_failure_count=len(failure_rows),
            recommendation_count=recommendations["recommendation_count"],
            unavailable_experiment_count=recommendations[
                "unavailable_experiment_count"
            ],
        )
        return digest
    except BaseException as error:
        failure = {
            "schema_version": (
                "intervenebench.cross_family_no_rerun_continuation_failure.v1"
            ),
            "status": "failed_no_retry_stop",
            "failed_at_utc": datetime.now(UTC).isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "completed_chunk_count": len(completed_chunks),
            "completed_chunks": completed_chunks,
            "preserved_raw_output_count": len(raw_hashes),
            "strict_output_count": len(strict_hashes),
            "new_strict_parse_failure_count": (
                len(parse_failures) - len(partition["unavailable_requests"])
            ),
            "raw_output_sha256_by_call": dict(sorted(raw_hashes.items())),
            "strict_output_sha256_by_call": dict(sorted(strict_hashes.items())),
            "automatic_retries": 0,
            "tcg8p_reruns": 0,
            "reserve_calls": 0,
            "semantic_repairs": 0,
            "human_outcomes_accessed": False,
            "participant_rows_accessed": 0,
            "human_outcome_scoring_performed": False,
            "automatic_next_stage": False,
        }
        freeze_envelope(
            failure, run_root / "failure_manifest.json", require_blinded=True
        )
        reporter.emit(
            "continuation_failed_frozen_no_retry_stop",
            error_type=type(error).__name__,
            error_message=str(error),
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--authorization",
        type=Path,
        default=ROOT / SEEDFIX_V2_CONTINUATION_AUTHORIZATION_PATH,
    )
    parser.add_argument(
        "--run-root", type=Path, default=ROOT / SEEDFIX_V2_CONTINUATION_RUN_ROOT
    )
    parser.add_argument(
        "--progress-log",
        type=Path,
        default=ROOT / SEEDFIX_V2_CONTINUATION_PROGRESS_PATH,
    )
    args = parser.parse_args()
    print(
        execute(
            authorization_path=args.authorization,
            run_root=args.run_root,
            progress_log_path=args.progress_log,
        )
    )


if __name__ == "__main__":
    main()
