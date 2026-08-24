#!/usr/bin/env python3
"""Freeze the zero-authority report-generation runtime and cache bindings."""

from __future__ import annotations

import json
from pathlib import Path

from intervenebench.evidence_report_execution import build_report_execution_freeze
from intervenebench.protocol import freeze_envelope, verify_envelope


ROOT = Path(__file__).resolve().parents[1]
PACKET_PATH = ROOT / (
    "data/manifests/qualitative_eval/"
    "intervenebench_report_evidence_packet_v1.json"
)
PROTOCOL_PATH = ROOT / "data/manifests/research/evidence_report_eval_v1.json"
PLAN_PATH = ROOT / (
    "data/manifests/qualitative_eval/report_generation_plan_v1.json"
)
OUTPUT_PATH = ROOT / "configs/simulators/evidence_report_execution_v1.json"


def main() -> None:
    packet = json.loads(PACKET_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    plan = verify_envelope(PLAN_PATH, require_blinded=True)
    digest = freeze_envelope(
        build_report_execution_freeze(ROOT, packet, protocol, plan),
        OUTPUT_PATH,
        require_blinded=True,
    )
    print(
        {
            "path": OUTPUT_PATH.relative_to(ROOT).as_posix(),
            "payload_sha256": digest,
            "planned_call_count": plan["call_count"],
            "inference_authorized": False,
        }
    )


if __name__ == "__main__":
    main()

