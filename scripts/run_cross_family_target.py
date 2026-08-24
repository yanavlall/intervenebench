#!/usr/bin/env python3
"""Pure-local wrapper for the authorized cross-family target image build.

This version intentionally exposes only image materialization.  The packaged
JSON-canary and target workers cannot be invoked by this wrapper until a later
authorization adds an explicit command and validator.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from intervenebench.cross_family_execution import (
    DEFAULT_EXECUTION_FREEZE_PATH,
    validate_materialization_authorization,
    verify_cross_family_execution_freeze,
)
from intervenebench.protocol import freeze_envelope, payload_hash, verify_envelope


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "infra/modal/cross_family_target_app.py"


class _ProgressReporter:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8"):
            pass

    def emit(self, state: str, **details: Any) -> None:
        row = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "stage": "cross_family_target_materialization",
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


def materialize(
    authorization_path: Path,
    output_path: Path,
    progress_log_path: Path,
) -> str:
    freeze = verify_cross_family_execution_freeze(
        ROOT, ROOT / DEFAULT_EXECUTION_FREEZE_PATH
    )
    authorization = verify_envelope(authorization_path, require_blinded=True)
    validate_materialization_authorization(authorization, freeze=freeze)
    if output_path.exists():
        raise FileExistsError(f"create-only materialization output exists: {output_path}")
    reporter = _ProgressReporter(progress_log_path)
    reporter.emit("local_authorization_verified")

    # Importing Modal and hydrating/building the app happens only after the
    # local hash-bound authority check above has passed.
    modal_app = _load_app()
    reporter.emit("modal_package_imported")
    with modal_app.modal.enable_output():
        with modal_app.app.run(
            name=freeze["runtime"]["app_name"],
            environment_name="main",
            detach=False,
            interactive=False,
        ):
            image_id = modal_app.materialized_target_image_id()
            if not isinstance(image_id, str) or not image_id.startswith("im-"):
                raise RuntimeError("hydrated cross-family target image ID is invalid")
            reporter.emit("modal_image_hydrated", modal_image_id=image_id)
    result = {
        "schema_version": "intervenebench.cross_family_target_materialization.v1",
        "status": "materialized_zero_inference_stop",
        "freeze_payload_sha256": payload_hash(freeze),
        "call_plan_payload_sha256": freeze["call_plan"]["payload_sha256"],
        "authorization_payload_sha256": payload_hash(authorization),
        "modal_image_id": image_id,
        "image_recipe_sha256": freeze["runtime"]["image_recipe_sha256"],
        "dependency_lock_sha256": freeze["runtime"]["dependency_lock_sha256"],
        "planned_target_call_count": freeze["limits"]["planned_call_count"],
        "model_downloaded": False,
        "remote_function_calls_made": 0,
        "inference_calls_made": 0,
        "target_calls_made": 0,
        "human_outcomes_accessed": False,
        "participant_rows_read": 0,
        "json_canary_authorized": False,
        "target_inference_authorized": False,
        "automatic_next_stage": False,
    }
    digest = freeze_envelope(result, output_path, require_blinded=True)
    reporter.emit("materialization_frozen_stop", output_payload_sha256=digest)
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("materialize", choices=("materialize",))
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-log", type=Path, required=True)
    args = parser.parse_args()
    print(materialize(args.authorization, args.output, args.progress_log))


if __name__ == "__main__":
    main()
