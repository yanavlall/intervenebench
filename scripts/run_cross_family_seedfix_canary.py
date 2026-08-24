#!/usr/bin/env python3
"""Run exactly one synthetic null-seed forced-choice canary."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import importlib.util
import json
from math import fsum, isfinite
from pathlib import Path
import sys
import time
from typing import Any, Mapping

from intervenebench.cross_family_seedfix import (
    SEEDFIX_APP_PATH,
    SEEDFIX_CANARY_AUTHORIZATION_PATH,
    SEEDFIX_CANARY_PATH,
    SEEDFIX_CANARY_PROGRESS_PATH,
    load_seedfix_materialization,
    validate_seedfix_canary_authorization,
    validate_seedfix_canary_completion,
    verify_seedfix_freeze,
    SEEDFIX_V2_APP_PATH,
    SEEDFIX_V2_CANARY_AUTHORIZATION_PATH,
    SEEDFIX_V2_CANARY_PATH,
    SEEDFIX_V2_CANARY_PROGRESS_PATH,
    load_seedfix_v2_canary,
    load_seedfix_v2_materialization,
    validate_seedfix_v2_canary_authorization,
    verify_seedfix_v2_freeze,
)
from intervenebench.cross_family_target_run import load_target_bindings
from intervenebench.protocol import freeze_envelope, payload_hash, verify_envelope


ROOT = Path(__file__).resolve().parents[1]


class _Reporter:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8"):
            pass

    def emit(self, state: str, **details: Any) -> None:
        row = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "stage": "cross_family_seedfix_target_free_canary",
            "state": state,
            **details,
        }
        print(f"[{row['timestamp_utc']}] {state} {details if details else ''}", flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _load_app(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        f"infra.modal.{path.stem}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load seed-fix Modal app")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _await(call: Any, *, reporter: _Reporter, timeout_type: type[Exception]) -> Any:
    started = time.monotonic()
    while True:
        elapsed = time.monotonic() - started
        if elapsed >= 3600:
            call.cancel(terminate_containers=True)
            raise TimeoutError("seed-fix canary exceeded its deadline")
        try:
            return call.get(timeout=min(30.0, 3600 - elapsed))
        except (TimeoutError, timeout_type):
            reporter.emit("canary_active", elapsed_seconds=round(elapsed, 1))


def _validate_raw(
    raw: Mapping[str, Any], *, image_id: str, seedfix_sha256: str, v2: bool
) -> None:
    result = raw.get("result")
    runtime = raw.get("runtime_attestation")
    if (
        raw.get("schema_version")
        != "intervenebench.cross_family_seedfix_canary_result.v1"
        or raw.get("status")
        != "passed_target_free_null_seed_forced_choice_stop"
        or raw.get("attempt_count") != 1
        or raw.get("target_prompts_or_assets_accessed") is not False
        or raw.get("target_calls_made") != 0
        or raw.get("model_downloaded") is not False
        or raw.get("human_outcomes_accessed") is not False
        or raw.get("participant_rows_read") != 0
        or raw.get("automatic_next_stage") is not False
        or not isinstance(result, Mapping)
        or result.get("null_seed_normalized") is not True
        or result.get("effective_generation_seed") != 0
        or not isinstance(runtime, Mapping)
        or runtime.get("modal_image_id") != image_id
        or runtime.get(
            "seedfix_v2_payload_sha256" if v2 else "seedfix_payload_sha256"
        )
        != seedfix_sha256
    ):
        raise ValueError("seed-fix canary return drifted")
    probabilities = result.get("probabilities_by_code")
    token_ids = result.get("candidate_token_ids")
    expected_codes = {"A", "B", "C"} if v2 else {"A", "B"}
    if (
        not isinstance(probabilities, Mapping)
        or set(probabilities) != expected_codes
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(float(value))
            or float(value) < 0
            for value in probabilities.values()
        )
        or abs(fsum(float(value) for value in probabilities.values()) - 1.0) > 1e-8
        or not isinstance(token_ids, list)
        or len(token_ids) != len(expected_codes)
        or len(set(token_ids)) != len(expected_codes)
        or any(not isinstance(token_id, int) for token_id in token_ids)
    ):
        raise ValueError("seed-fix canary forced-choice distribution drifted")


def execute(authorization_path: Path, output_path: Path, progress_path: Path) -> str:
    cache = load_target_bindings(ROOT)["cache"]
    authorization = verify_envelope(authorization_path, require_blinded=True)
    v2 = (
        authorization.get("schema_version")
        == "intervenebench.cross_family_seedfix_v2_canary_authorization.v1"
    )
    parent = verify_seedfix_freeze(ROOT)
    if v2:
        freeze = verify_seedfix_v2_freeze(ROOT)
        materialization = load_seedfix_v2_materialization(ROOT)
        validate_seedfix_v2_canary_authorization(authorization, root=ROOT)
        app_path = ROOT / SEEDFIX_V2_APP_PATH
    else:
        freeze = parent
        materialization = load_seedfix_materialization(ROOT)
        validate_seedfix_canary_authorization(authorization, root=ROOT)
        app_path = ROOT / SEEDFIX_APP_PATH
    if output_path.exists():
        raise FileExistsError(f"create-only seed-fix canary exists: {output_path}")
    reporter = _Reporter(progress_path)
    reporter.emit("local_authorization_verified")
    modal_app = _load_app(app_path)
    reporter.emit("modal_package_imported")
    with modal_app.modal.enable_output():
        with modal_app.app.run(
            name=freeze["runtime"]["app_name"],
            environment_name="main",
            detach=False,
            interactive=False,
        ):
            image_id = modal_app.materialized_target_image_id()
            if image_id != materialization["modal_image_id"]:
                raise RuntimeError("seed-fix canary image differs from authorization")
            reporter.emit("modal_image_binding_verified", modal_image_id=image_id)
            call = modal_app.run_cross_family_seedfix_canary.spawn(
                parent["parent_execution_freeze_payload_sha256"],
                image_id,
                payload_hash(cache),
                payload_hash(freeze),
            )
            reporter.emit(
                "canary_submitted",
                modal_function_call_id=getattr(call, "object_id", "unavailable"),
            )
            raw = _await(
                call,
                reporter=reporter,
                timeout_type=modal_app.modal.exception.TimeoutError,
            )
    _validate_raw(
        raw, image_id=image_id, seedfix_sha256=payload_hash(freeze), v2=v2
    )
    raw_digest = freeze_envelope(
        raw, output_path.with_name(output_path.stem + "_raw.json"), require_blinded=True
    )
    completion = {
        "schema_version": "intervenebench.cross_family_seedfix_canary_completion.v1",
        "status": "passed_target_free_null_seed_forced_choice_stop",
        "seedfix_freeze_payload_sha256": payload_hash(freeze),
        "authorization_payload_sha256": payload_hash(authorization),
        "materialization_payload_sha256": payload_hash(materialization),
        "modal_image_id": image_id,
        "raw_result_payload_sha256": raw_digest,
        "raw_result": raw,
        "attempt_count": 1,
        "target_calls_made": 0,
        "human_outcomes_accessed": False,
        "participant_rows_read": 0,
        "automatic_next_stage": False,
    }
    digest = freeze_envelope(completion, output_path, require_blinded=True)
    if v2:
        load_seedfix_v2_canary(ROOT)
    else:
        validate_seedfix_canary_completion(completion, root=ROOT)
    reporter.emit("canary_passed_frozen_stop", output_payload_sha256=digest)
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--authorization", type=Path, default=ROOT / SEEDFIX_V2_CANARY_AUTHORIZATION_PATH
    )
    parser.add_argument("--output", type=Path, default=ROOT / SEEDFIX_V2_CANARY_PATH)
    parser.add_argument(
        "--progress-log", type=Path, default=ROOT / SEEDFIX_V2_CANARY_PROGRESS_PATH
    )
    args = parser.parse_args()
    print(execute(args.authorization, args.output, args.progress_log))


if __name__ == "__main__":
    main()
