#!/usr/bin/env python3
"""Run, strictly parse, and aggregate the exact authorized 624 Mistral calls."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

from intervenebench.cross_family_target_run import (
    DEFAULT_TARGET_RUN_ROOT,
    build_recommendations,
    load_target_bindings,
    strict_parse_target_result,
    validate_target_authorization,
)
from intervenebench.cross_family_execution import prepare_cross_family_target_requests
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
            "stage": "cross_family_624_call_target_run",
            "state": state,
            **details,
        }
        print(f"[{row['timestamp_utc']}] {state} {details if details else ''}", flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _load_app() -> Any:
    spec = importlib.util.spec_from_file_location(
        "infra.modal.cross_family_target_app", APP_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the cross-family target Modal app")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _await_group(
    function: Any,
    args: tuple[Any, ...],
    *,
    experiment_id: str,
    reporter: _Reporter,
    timeout_error_type: type[Exception],
    global_started: float,
    global_deadline_seconds: float,
) -> tuple[Mapping[str, Any], float]:
    group_started = time.monotonic()
    call = function.spawn(*args)
    modal_call_id = getattr(call, "object_id", "unavailable")
    reporter.emit(
        "experiment_group_submitted",
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
                experiment_id=experiment_id,
                modal_function_call_id=modal_call_id,
                total_elapsed_seconds=round(total_elapsed, 1),
            )
            raise TimeoutError("cross-family target run exceeded its global deadline")
        try:
            result = call.get(timeout=min(30.0, remaining))
        except (TimeoutError, timeout_error_type):
            reporter.emit(
                "experiment_group_active",
                experiment_id=experiment_id,
                modal_function_call_id=modal_call_id,
                group_elapsed_seconds=round(time.monotonic() - group_started, 1),
                total_elapsed_seconds=round(time.monotonic() - global_started, 1),
            )
            continue
        elapsed = time.monotonic() - group_started
        reporter.emit(
            "experiment_group_completed",
            experiment_id=experiment_id,
            modal_function_call_id=modal_call_id,
            group_elapsed_seconds=round(elapsed, 1),
        )
        return result, elapsed


def _raw_result_path(run_root: Path, request: Mapping[str, Any]) -> Path:
    relative = Path(str(request["artifact_relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("target artifact path escapes its run root")
    return run_root / "raw" / relative


def _strict_result_path(run_root: Path, request: Mapping[str, Any]) -> Path:
    relative = Path(str(request["artifact_relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("target artifact path escapes its run root")
    return run_root / "strict" / relative


def execute(
    *,
    authorization_path: Path,
    run_root: Path,
    progress_log_path: Path,
) -> str:
    if run_root.exists():
        raise FileExistsError(f"create-only target run exists: {run_root}")
    bindings = load_target_bindings(ROOT)
    freeze = bindings["freeze"]
    materialization = bindings["materialization"]
    cache = bindings["cache"]
    json_canary = bindings["json_canary"]
    authorization = verify_envelope(authorization_path, require_blinded=True)
    validate_target_authorization(
        authorization,
        root=ROOT,
        freeze=freeze,
        materialization=materialization,
        cache=cache,
        json_canary=json_canary,
    )
    requests = prepare_cross_family_target_requests(ROOT)
    by_id = {str(request["call_id"]): request for request in requests}
    if len(requests) != 624 or len(by_id) != 624:
        raise ValueError("target request set must contain exactly 624 unique calls")
    grouped = {
        experiment_id: [
            request for request in requests
            if request["experiment_id"] == experiment_id
        ]
        for experiment_id in authorization["experiment_group_order"]
    }
    if {key: len(value) for key, value in grouped.items()} != authorization[
        "call_count_by_experiment"
    ]:
        raise ValueError("target experiment group counts drifted")

    run_root.mkdir(parents=True, exist_ok=False)
    reporter = _Reporter(progress_log_path)
    reporter.emit("local_authorization_verified", planned_call_count=624)
    modal_app = _load_app()
    reporter.emit("modal_package_imported")
    global_started = time.monotonic()
    raw_hashes: dict[str, str] = {}
    strict_hashes: dict[str, str] = {}
    strict_outputs: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    group_elapsed: dict[str, float] = {}
    completed_groups: list[str] = []

    try:
        with modal_app.modal.enable_output():
            with modal_app.app.run(
                name=freeze["runtime"]["app_name"],
                environment_name="main",
                detach=False,
                interactive=False,
            ):
                hydrated_id = modal_app.materialized_target_image_id()
                if hydrated_id != authorization["modal_image_id"]:
                    raise RuntimeError("target hydrated image differs from authorization")
                reporter.emit("modal_image_binding_verified", modal_image_id=hydrated_id)
                for experiment_id in authorization["experiment_group_order"]:
                    group = grouped[experiment_id]
                    result, elapsed = _await_group(
                        modal_app.run_cross_family_target_group,
                        (
                            group,
                            payload_hash(freeze),
                            hydrated_id,
                            payload_hash(cache),
                        ),
                        experiment_id=experiment_id,
                        reporter=reporter,
                        timeout_error_type=modal_app.modal.exception.TimeoutError,
                        global_started=global_started,
                        global_deadline_seconds=float(
                            authorization["maximum_wall_clock_seconds"]
                        ),
                    )
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
                        raise RuntimeError("target remote group result drifted")
                    returned_by_id = {str(row.get("call_id")): row for row in returned}
                    if set(returned_by_id) != {str(row["call_id"]) for row in group}:
                        raise RuntimeError("target remote group does not cover its call IDs")
                    # Persist every exact remote return before any strict parse.
                    for request in group:
                        call_id = str(request["call_id"])
                        raw = returned_by_id[call_id]
                        raw_hashes[call_id] = freeze_envelope(
                            {
                                "schema_version": "intervenebench.cross_family_unparsed_raw_call.v1",
                                "call_id": call_id,
                                "request_payload_sha256": payload_hash(request),
                                "raw_result": raw,
                            },
                            _raw_result_path(run_root, request),
                            require_blinded=True,
                        )
                    reporter.emit(
                        "experiment_group_raw_frozen",
                        experiment_id=experiment_id,
                        raw_call_count=len(group),
                    )
                    # Strict parsing is deterministic, local, and repair-free.
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
                            failures.append(
                                {
                                    "call_id": call_id,
                                    "experiment_id": experiment_id,
                                    "method_id": request["method_id"],
                                    "stage": request["stage"],
                                    "error_type": type(error).__name__,
                                    "error_message": str(error),
                                }
                            )
                            continue
                        strict_outputs[call_id] = strict
                        strict_hashes[call_id] = freeze_envelope(
                            strict,
                            _strict_result_path(run_root, request),
                            require_blinded=True,
                        )
                    completed_groups.append(experiment_id)
                    group_elapsed[experiment_id] = elapsed
                    reporter.emit(
                        "experiment_group_strict_parse_complete",
                        experiment_id=experiment_id,
                        strict_count=sum(
                            request["call_id"] in strict_outputs for request in group
                        ),
                        failure_count=sum(
                            failure["experiment_id"] == experiment_id
                            for failure in failures
                        ),
                    )

        if set(raw_hashes) != set(by_id):
            raise RuntimeError("target run did not preserve all 624 raw outputs")
        if set(strict_hashes).union(failure["call_id"] for failure in failures) != set(
            by_id
        ):
            raise RuntimeError("target strict/failure partition does not cover all calls")
        if set(strict_hashes).intersection(failure["call_id"] for failure in failures):
            raise RuntimeError("target strict/failure partition overlaps")
        failure_rows = sorted(failures, key=lambda row: row["call_id"])
        recommendations = build_recommendations(
            ROOT,
            requests=requests,
            strict_outputs=strict_outputs,
            strict_output_hashes=dict(sorted(strict_hashes.items())),
            parse_failures=failure_rows,
        )
        recommendations.update(
            {
                "authorization_payload_sha256": payload_hash(authorization),
                "freeze_payload_sha256": payload_hash(freeze),
                "materialization_payload_sha256": payload_hash(materialization),
                "json_canary_completion_payload_sha256": payload_hash(json_canary),
                "raw_output_map_sha256": payload_hash(dict(sorted(raw_hashes.items()))),
            }
        )
        assert_blinded_payload(recommendations)
        recommendation_hash = freeze_envelope(
            recommendations,
            run_root / "recommendations_v1.json",
            require_blinded=True,
        )
        final = {
            "schema_version": "intervenebench.cross_family_target_run.v1",
            "status": "completed_624_calls_strictly_parsed_recommendations_frozen_stop",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "freeze_payload_sha256": payload_hash(freeze),
            "authorization_payload_sha256": payload_hash(authorization),
            "materialization_payload_sha256": payload_hash(materialization),
            "json_canary_completion_payload_sha256": payload_hash(json_canary),
            "modal_image_id": materialization["modal_image_id"],
            "checkpoint_commit": freeze["model"]["checkpoint_commit"],
            "attempt_count": 624,
            "raw_output_count": len(raw_hashes),
            "strict_output_count": len(strict_hashes),
            "strict_parse_failure_count": len(failure_rows),
            "raw_output_sha256_by_call": dict(sorted(raw_hashes.items())),
            "strict_output_sha256_by_call": dict(sorted(strict_hashes.items())),
            "strict_parse_failures": failure_rows,
            "recommendations_payload_sha256": recommendation_hash,
            "completed_experiment_groups": completed_groups,
            "remote_group_elapsed_seconds": group_elapsed,
            "total_wall_seconds": time.monotonic() - global_started,
            "automatic_retries": 0,
            "reserve_calls": 0,
            "semantic_repairs": 0,
            "model_downloads": 0,
            "human_outcomes_accessed": False,
            "participant_rows_accessed": 0,
            "human_outcome_scoring_performed": False,
            "automatic_next_stage": False,
        }
        digest = freeze_envelope(
            final, run_root / "final_manifest.json", require_blinded=True
        )
        reporter.emit(
            "run_and_recommendations_frozen_stop",
            final_manifest_payload_sha256=digest,
            strict_output_count=len(strict_hashes),
            strict_parse_failure_count=len(failure_rows),
        )
        return digest
    except BaseException as error:
        failure = {
            "schema_version": "intervenebench.cross_family_target_run_failure.v1",
            "status": "failed_no_retry_stop",
            "failed_at_utc": datetime.now(UTC).isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "completed_experiment_groups": completed_groups,
            "preserved_raw_output_count": len(raw_hashes),
            "strict_output_count": len(strict_hashes),
            "strict_parse_failure_count": len(failures),
            "raw_output_sha256_by_call": dict(sorted(raw_hashes.items())),
            "strict_output_sha256_by_call": dict(sorted(strict_hashes.items())),
            "automatic_retries": 0,
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
            "run_failed_frozen_no_retry_stop",
            error_type=type(error).__name__,
            error_message=str(error),
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=ROOT / DEFAULT_TARGET_RUN_ROOT)
    parser.add_argument("--progress-log", type=Path, required=True)
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
