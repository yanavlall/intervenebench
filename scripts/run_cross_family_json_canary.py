#!/usr/bin/env python3
"""Run exactly one authorized target-free Mistral JSON-schema canary."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

from intervenebench.cross_family_json_canary import (
    DEFAULT_JSON_CANARY_RESULT_PATH,
    load_json_canary_bindings,
    validate_json_canary_authorization,
    validate_json_canary_completion,
)
from intervenebench.protocol import freeze_envelope, payload_hash, verify_envelope


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
            "stage": "cross_family_target_free_json_canary",
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


def _await_one_call(
    function: Any,
    args: tuple[Any, ...],
    *,
    reporter: _Reporter,
    timeout_error_type: type[Exception],
) -> Any:
    started = time.monotonic()
    call = function.spawn(*args)
    call_id = getattr(call, "object_id", "unavailable")
    reporter.emit("one_call_submitted", modal_function_call_id=call_id)
    while True:
        elapsed = time.monotonic() - started
        if elapsed >= 3600:
            call.cancel(terminate_containers=True)
            reporter.emit(
                "deadline_exceeded_cancelled",
                modal_function_call_id=call_id,
                elapsed_seconds=round(elapsed, 1),
            )
            raise TimeoutError("JSON canary exceeded its 3600-second client deadline")
        try:
            result = call.get(timeout=min(30.0, 3600 - elapsed))
        except (TimeoutError, timeout_error_type):
            reporter.emit(
                "remote_call_active",
                modal_function_call_id=call_id,
                elapsed_seconds=round(time.monotonic() - started, 1),
            )
            continue
        reporter.emit(
            "one_call_completed",
            modal_function_call_id=call_id,
            elapsed_seconds=round(time.monotonic() - started, 1),
        )
        return result


def execute(
    authorization_path: Path,
    output_path: Path,
    progress_log_path: Path,
) -> str:
    bindings = load_json_canary_bindings(ROOT)
    freeze = bindings["freeze"]
    materialization = bindings["materialization"]
    cache = bindings["cache"]
    authorization = verify_envelope(authorization_path, require_blinded=True)
    validate_json_canary_authorization(
        authorization,
        freeze=freeze,
        materialization=materialization,
        cache=cache,
    )
    if output_path.exists():
        raise FileExistsError(f"create-only JSON canary output exists: {output_path}")
    reporter = _Reporter(progress_log_path)
    reporter.emit("local_authorization_verified")

    modal_app = _load_app()
    reporter.emit("modal_package_imported")
    with modal_app.modal.enable_output():
        with modal_app.app.run(
            name=freeze["runtime"]["app_name"],
            environment_name="main",
            detach=False,
            interactive=False,
        ):
            hydrated_id = modal_app.materialized_target_image_id()
            if hydrated_id != materialization["modal_image_id"]:
                raise RuntimeError("JSON canary hydrated image differs from authorization")
            reporter.emit("modal_image_binding_verified", modal_image_id=hydrated_id)
            raw = _await_one_call(
                modal_app.run_cross_family_json_canary,
                (
                    payload_hash(freeze),
                    hydrated_id,
                    payload_hash(cache),
                    freeze["required_json_canary"]["prompt_sha256"],
                ),
                reporter=reporter,
                timeout_error_type=modal_app.modal.exception.TimeoutError,
            )

    # Preserve the exact model return before local adjudication.  Invalid JSON
    # remains a frozen negative result and never receives a repair or retry.
    raw_path = output_path.with_name(output_path.stem + "_raw.json")
    raw_digest = freeze_envelope(raw, raw_path, require_blinded=True)
    reporter.emit("raw_response_frozen", raw_payload_sha256=raw_digest)
    if raw.get("status") != "passed_target_free_json_schema":
        failure = {
            "schema_version": "intervenebench.cross_family_json_canary_failure.v1",
            "status": "failed_target_free_json_schema_stop",
            "freeze_payload_sha256": payload_hash(freeze),
            "authorization_payload_sha256": payload_hash(authorization),
            "materialization_payload_sha256": payload_hash(materialization),
            "raw_result_payload_sha256": payload_hash(raw),
            "attempt_count": 1,
            "target_calls_made": 0,
            "human_outcomes_accessed": False,
            "participant_rows_read": 0,
            "automatic_next_stage": False,
        }
        digest = freeze_envelope(failure, output_path, require_blinded=True)
        reporter.emit("schema_failed_frozen_stop", output_payload_sha256=digest)
        return digest

    completion = {
        "schema_version": "intervenebench.cross_family_json_canary_completion.v1",
        "status": "passed_target_free_json_schema_stop",
        "freeze_payload_sha256": payload_hash(freeze),
        "authorization_payload_sha256": payload_hash(authorization),
        "materialization_payload_sha256": payload_hash(materialization),
        "modal_image_id": materialization["modal_image_id"],
        "attempt_count": 1,
        "raw_result": raw,
        "target_prompts_or_assets_accessed": False,
        "target_calls_made": 0,
        "model_downloaded": False,
        "human_outcomes_accessed": False,
        "participant_rows_read": 0,
        "automatic_next_stage": False,
    }
    validate_json_canary_completion(
        completion,
        freeze=freeze,
        authorization=authorization,
        materialization=materialization,
    )
    digest = freeze_envelope(completion, output_path, require_blinded=True)
    reporter.emit("schema_passed_frozen_stop", output_payload_sha256=digest)
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / DEFAULT_JSON_CANARY_RESULT_PATH)
    parser.add_argument("--progress-log", type=Path, required=True)
    args = parser.parse_args()
    print(execute(args.authorization, args.output, args.progress_log))


if __name__ == "__main__":
    main()
