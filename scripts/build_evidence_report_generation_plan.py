"""Freeze the zero-authority evidence-report generation plan."""

from __future__ import annotations

import json
from pathlib import Path

from intervenebench.evidence_report_eval import (
    build_report_generation_plan,
    validate_eval_protocol,
    validate_evidence_packet,
)
from intervenebench.protocol import freeze_envelope, payload_hash, verify_envelope


ROOT = Path(__file__).resolve().parents[1]
PACKET_PATH = ROOT / (
    "data/manifests/qualitative_eval/"
    "intervenebench_report_evidence_packet_v1.json"
)
PROTOCOL_PATH = ROOT / "data/manifests/research/evidence_report_eval_v1.json"
OUTPUT_PATH = ROOT / (
    "data/manifests/qualitative_eval/report_generation_plan_v1.json"
)


def main() -> None:
    packet = json.loads(PACKET_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_evidence_packet(packet)
    validate_eval_protocol(protocol, packet)

    for source in packet["source_artifacts"]:
        source_path = ROOT / source["path"]
        source_payload = verify_envelope(source_path)
        if payload_hash(source_payload) != source["payload_sha256"]:
            raise ValueError(f"evidence source payload drifted: {source['path']}")

    digest = freeze_envelope(
        build_report_generation_plan(packet, protocol),
        OUTPUT_PATH,
        require_blinded=True,
    )
    print(
        {
            "path": OUTPUT_PATH.relative_to(ROOT).as_posix(),
            "payload_sha256": digest,
            "model_calls_authorized": False,
        }
    )


if __name__ == "__main__":
    main()
