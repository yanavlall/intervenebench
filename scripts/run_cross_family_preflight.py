#!/usr/bin/env python3
"""Pure-local authority wrapper for target-free cross-family Modal stages."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import time
from typing import Any

from intervenebench.cross_family_modal import (
    DEFAULT_CROSS_FAMILY_MODAL_FREEZE_PATH,
    validate_cache_authorization,
    validate_canary_authorization,
    validate_forced_choice_probe,
    validate_materialization_authorization,
    verify_cross_family_modal_freeze,
)
from intervenebench.protocol import freeze_envelope, payload_hash, verify_envelope


ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / DEFAULT_CROSS_FAMILY_MODAL_FREEZE_PATH


class _ProgressReporter:
    """Emit human-readable heartbeats and an append-only JSONL run ledger."""

    def __init__(self, path: Path | None):
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8"):
                pass

    def emit(self, stage: str, state: str, **details: Any) -> None:
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "state": state,
            **details,
        }
        print(
            f"[{row['timestamp_utc']}] {stage}: {state}"
            + (f" {details}" if details else ""),
            flush=True,
        )
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _run_remote_with_watchdog(
    function: Any,
    args: tuple[Any, ...],
    *,
    reporter: _ProgressReporter,
    label: str,
    heartbeat_seconds: float,
    maximum_seconds: float,
    timeout_error_type: type[Exception],
    clock: Any = time.monotonic,
) -> Any:
    """Run one Modal call with visible heartbeats and a hard client deadline."""

    started = clock()
    call = function.spawn(*args)
    call_id = getattr(call, "object_id", "unavailable")
    reporter.emit(label, "submitted", function_call_id=call_id)
    try:
        while True:
            elapsed = max(0.0, clock() - started)
            remaining = maximum_seconds - elapsed
            if remaining <= 0:
                call.cancel(terminate_containers=True)
                reporter.emit(
                    label,
                    "deadline_exceeded_cancelled",
                    function_call_id=call_id,
                    elapsed_seconds=round(elapsed, 1),
                )
                raise TimeoutError(
                    f"{label} exceeded its {maximum_seconds:.0f}-second client deadline"
                )
            try:
                result = call.get(timeout=min(heartbeat_seconds, remaining))
            # Modal 1.5.4's FunctionCall.get() currently raises Python's
            # built-in TimeoutError for a healthy poll timeout, while other
            # SDK paths expose modal.exception.TimeoutError.  Treat either as
            # a heartbeat; all other exceptions remain fatal.
            except (TimeoutError, timeout_error_type):
                elapsed = max(0.0, clock() - started)
                reporter.emit(
                    label,
                    "remote_call_active",
                    function_call_id=call_id,
                    elapsed_seconds=round(elapsed, 1),
                    client_deadline_seconds=maximum_seconds,
                )
                continue
            reporter.emit(
                label,
                "completed",
                function_call_id=call_id,
                elapsed_seconds=round(max(0.0, clock() - started), 1),
            )
            return result
    except BaseException as exc:
        if not isinstance(exc, TimeoutError):
            reporter.emit(
                label,
                "failed",
                function_call_id=call_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        raise


def _validate_local_authorization(
    stage: str,
    authorization_path: Path,
    cache_attestation_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    freeze = verify_cross_family_modal_freeze(ROOT, FREEZE_PATH)
    authorization = verify_envelope(authorization_path, require_blinded=True)
    cache_attestation: dict[str, Any] | None = None
    if stage == "materialize":
        validate_materialization_authorization(authorization, freeze=freeze)
    elif stage == "cache":
        modal_image_id = authorization.get("modal_image_id")
        if not isinstance(modal_image_id, str) or not modal_image_id.startswith("im-"):
            raise PermissionError("cache authority lacks a pinned Modal image ID")
        validate_cache_authorization(
            authorization, freeze=freeze, modal_image_id=modal_image_id
        )
    elif stage == "canary":
        if cache_attestation_path is None:
            raise PermissionError("canary requires a verified cache attestation")
        cache_attestation = verify_envelope(
            cache_attestation_path, require_blinded=True
        )
        modal_image_id = authorization.get("modal_image_id")
        if not isinstance(modal_image_id, str) or not modal_image_id.startswith("im-"):
            raise PermissionError("canary authority lacks a pinned Modal image ID")
        validate_canary_authorization(
            authorization,
            freeze=freeze,
            modal_image_id=modal_image_id,
            cache_attestation_sha256=payload_hash(cache_attestation),
        )
    else:
        raise ValueError("unsupported cross-family preflight stage")
    return freeze, authorization, cache_attestation


def execute(
    stage: str,
    authorization_path: Path,
    output_path: Path,
    cache_attestation_path: Path | None = None,
    progress_log_path: Path | None = None,
) -> str:
    freeze, authorization, cache_attestation = _validate_local_authorization(
        stage, authorization_path, cache_attestation_path
    )
    if output_path.exists():
        raise FileExistsError(f"create-only preflight output already exists: {output_path}")

    reporter = _ProgressReporter(progress_log_path)
    reporter.emit(stage, "local_authorization_verified")

    app_module = importlib.import_module("infra.modal.cross_family_app")
    modal_timeout = app_module.modal.exception.TimeoutError
    with app_module.modal.enable_output():
        with app_module.app.run(
            name=freeze["runtime"]["app_name"], environment_name="main"
        ):
            image_id = app_module.runtime_image.object_id
            if not isinstance(image_id, str) or not image_id.startswith("im-"):
                raise RuntimeError("hydrated Modal image ID is unavailable")
            reporter.emit(stage, "modal_app_hydrated", modal_image_id=image_id)
            if stage == "materialize":
                result = {
                    "schema_version": "intervenebench.cross_family_materialization.v1",
                    "freeze_payload_sha256": payload_hash(freeze),
                    "modal_image_id": image_id,
                    "image_recipe_sha256": freeze["runtime"]["image_recipe_sha256"],
                    "dependency_lock_sha256": freeze["runtime"][
                        "dependency_lock_sha256"
                    ],
                    "model_downloaded": False,
                    "inference_calls_made": 0,
                    "target_prompts_or_assets_uploaded": False,
                    "automatic_next_stage": False,
                }
            elif stage == "cache":
                if image_id != authorization["modal_image_id"]:
                    raise RuntimeError("hydrated Modal image differs from cache authority")
                smoke = _run_remote_with_watchdog(
                    app_module.cross_family_startup_smoke,
                    (payload_hash(freeze), image_id),
                    reporter=reporter,
                    label="cache_startup_smoke",
                    heartbeat_seconds=15.0,
                    maximum_seconds=180.0,
                    timeout_error_type=modal_timeout,
                )
                if (
                    smoke.get("status") != "passed"
                    or smoke.get("module_path")
                    != "/root/infra/modal/cross_family_app.py"
                ):
                    raise RuntimeError("remote packaging smoke did not attest success")
                result = _run_remote_with_watchdog(
                    app_module.cache_cross_family_checkpoint,
                    (payload_hash(freeze), image_id),
                    reporter=reporter,
                    label="checkpoint_cache",
                    heartbeat_seconds=60.0,
                    maximum_seconds=7200.0,
                    timeout_error_type=modal_timeout,
                )
            else:
                if image_id != authorization["modal_image_id"]:
                    raise RuntimeError("hydrated Modal image differs from canary authority")
                if cache_attestation is None:
                    raise RuntimeError("verified cache attestation was lost")
                result = _run_remote_with_watchdog(
                    app_module.run_cross_family_canary,
                    (
                        payload_hash(freeze),
                        image_id,
                        payload_hash(cache_attestation),
                        freeze["canary"]["manifest_payload_sha256"],
                    ),
                    reporter=reporter,
                    label="synthetic_canaries",
                    heartbeat_seconds=30.0,
                    maximum_seconds=3600.0,
                    timeout_error_type=modal_timeout,
                )
                for row, request in zip(
                    result["canary_results"],
                    freeze["canary"]["manifest"]["requests"],
                    strict=True,
                ):
                    if request["adapter"] == "forced_choice_next_token_softmax.v1":
                        validate_forced_choice_probe(
                            row["result"], expected_codes=request["answer_codes"]
                        )
    digest = freeze_envelope(result, output_path, require_blinded=True)
    reporter.emit(stage, "output_frozen", output_sha256=digest)
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("materialize", "cache", "canary")
    )
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-attestation", type=Path)
    parser.add_argument(
        "--progress-log",
        type=Path,
        help="create-only JSONL ledger with live stage transitions and heartbeats",
    )
    args = parser.parse_args()
    digest = execute(
        args.stage,
        args.authorization,
        args.output,
        args.cache_attestation,
        args.progress_log,
    )
    print(digest)


if __name__ == "__main__":
    main()
