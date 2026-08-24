#!/usr/bin/env python3
"""Materialize only the corrected image; make zero remote function calls."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from intervenebench.cross_family_seedfix import (
    SEEDFIX_APP_PATH,
    SEEDFIX_MATERIALIZATION_AUTHORIZATION_PATH,
    SEEDFIX_MATERIALIZATION_PATH,
    SEEDFIX_MATERIALIZATION_PROGRESS_PATH,
    validate_seedfix_materialization_authorization,
    verify_seedfix_freeze,
    SEEDFIX_V2_APP_PATH,
    SEEDFIX_V2_MATERIALIZATION_AUTHORIZATION_PATH,
    SEEDFIX_V2_MATERIALIZATION_PATH,
    SEEDFIX_V2_MATERIALIZATION_PROGRESS_PATH,
    validate_seedfix_v2_materialization_authorization,
    verify_seedfix_v2_freeze,
)
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
            "stage": "cross_family_seedfix_materialization",
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


def execute(authorization_path: Path, output_path: Path, progress_path: Path) -> str:
    authorization = verify_envelope(authorization_path, require_blinded=True)
    v2 = (
        authorization.get("schema_version")
        == "intervenebench.cross_family_seedfix_v2_materialization_authorization.v1"
    )
    if v2:
        freeze = verify_seedfix_v2_freeze(ROOT)
        validate_seedfix_v2_materialization_authorization(authorization, root=ROOT)
        app_path = ROOT / SEEDFIX_V2_APP_PATH
        freeze_key = "seedfix_v2_freeze_payload_sha256"
        status = "materialized_seedfix_v2_image_zero_inference_stop"
    else:
        freeze = verify_seedfix_freeze(ROOT)
        validate_seedfix_materialization_authorization(authorization, root=ROOT)
        app_path = ROOT / SEEDFIX_APP_PATH
        freeze_key = "seedfix_freeze_payload_sha256"
        status = "materialized_seedfix_image_zero_inference_stop"
    if output_path.exists():
        raise FileExistsError(f"create-only seed-fix materialization exists: {output_path}")
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
            if not isinstance(image_id, str) or not image_id.startswith("im-"):
                raise RuntimeError("seed-fix image ID is invalid")
            reporter.emit("modal_image_hydrated", modal_image_id=image_id)
    result = {
        "schema_version": "intervenebench.cross_family_seedfix_materialization.v1",
        "status": status,
        freeze_key: payload_hash(freeze),
        "authorization_payload_sha256": payload_hash(authorization),
        "modal_image_id": image_id,
        "remote_function_calls_made": 0,
        "inference_calls_made": 0,
        "model_downloaded": False,
        "human_outcomes_accessed": False,
        "participant_rows_read": 0,
        "automatic_next_stage": False,
    }
    digest = freeze_envelope(result, output_path, require_blinded=True)
    reporter.emit("materialization_frozen_stop", output_payload_sha256=digest)
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--authorization",
        type=Path,
        default=ROOT / SEEDFIX_V2_MATERIALIZATION_AUTHORIZATION_PATH,
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / SEEDFIX_V2_MATERIALIZATION_PATH
    )
    parser.add_argument(
        "--progress-log",
        type=Path,
        default=ROOT / SEEDFIX_V2_MATERIALIZATION_PROGRESS_PATH,
    )
    args = parser.parse_args()
    print(execute(args.authorization, args.output, args.progress_log))


if __name__ == "__main__":
    main()
