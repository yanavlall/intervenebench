#!/usr/bin/env python3
"""Create exact authority for two zero-inference remote import smokes."""

from __future__ import annotations

import argparse
from pathlib import Path

from intervenebench.evidence_report_execution import (
    validate_report_import_smoke_authorization,
)
from intervenebench.protocol import freeze_envelope, payload_hash, verify_envelope


ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / "configs/simulators/evidence_report_execution_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze = verify_envelope(FREEZE_PATH, require_blinded=True)
    materialization = verify_envelope(args.materialization, require_blinded=True)
    if materialization.get("execution_freeze_payload_sha256") != payload_hash(freeze):
        raise ValueError("materialization is bound to another execution freeze")
    image_ids = materialization.get("modal_image_ids")
    authorization = {
        "schema_version": "intervenebench.report_eval_import_smoke_authorization.v1",
        "execution_freeze_payload_sha256": payload_hash(freeze),
        "modal_image_ids": image_ids,
        "exact_import_smoke_call_count": 2,
        "import_smoke_authorized": True,
        "model_download_authorized": False,
        "inference_authorized": False,
        "participant_row_access_authorized": False,
        "experiment_level_human_score_access_authorized": False,
        "automatic_next_stage_authorized": False,
    }
    validate_report_import_smoke_authorization(
        authorization,
        freeze,
        materialized_image_ids=image_ids,
    )
    digest = freeze_envelope(authorization, args.output, require_blinded=True)
    print(
        {
            "path": str(args.output),
            "payload_sha256": digest,
            "exact_import_smoke_call_count": 2,
            "inference_authorized": False,
        }
    )


if __name__ == "__main__":
    main()
