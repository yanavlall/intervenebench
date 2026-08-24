#!/usr/bin/env python3
"""Create the one-time zero-inference evidence-report image authorization."""

from __future__ import annotations

import argparse
from pathlib import Path

from intervenebench.evidence_report_execution import (
    validate_report_materialization_authorization,
)
from intervenebench.protocol import freeze_envelope, payload_hash, verify_envelope


ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / "configs/simulators/evidence_report_execution_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze = verify_envelope(FREEZE_PATH, require_blinded=True)
    authorization = {
        "schema_version": "intervenebench.report_eval_materialization_authorization.v1",
        "execution_freeze_payload_sha256": payload_hash(freeze),
        "modal_image_materialization_authorized": True,
        "model_download_authorized": False,
        "inference_authorized": False,
        "participant_row_access_authorized": False,
        "automatic_next_stage_authorized": False,
    }
    validate_report_materialization_authorization(authorization, freeze)
    digest = freeze_envelope(authorization, args.output, require_blinded=True)
    print(
        {
            "path": str(args.output),
            "payload_sha256": digest,
            "model_download_authorized": False,
            "inference_authorized": False,
        }
    )


if __name__ == "__main__":
    main()

