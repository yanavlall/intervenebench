#!/usr/bin/env python3
"""Build a create-only evidence-aware reviewer for the frozen report panel."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from intervenebench.evidence_report_eval import render_evidence_aware_labeling_app
from intervenebench.protocol import freeze_envelope, payload_hash, verify_envelope


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "artifacts/evidence_report_eval/report_generation_20260820_v3"
QUEUE_PATH = RUN_ROOT / "labeling/blinded_queue.json"
PACKET_PATH = ROOT / (
    "data/manifests/qualitative_eval/"
    "intervenebench_report_evidence_packet_v1.json"
)
PROTOCOL_PATH = ROOT / "data/manifests/research/evidence_report_eval_v1.json"
FINAL_PATH = RUN_ROOT / "final_manifest.json"
OUTPUT_PATH = RUN_ROOT / "labeling/reviewer_v3.html"
MANIFEST_PATH = RUN_ROOT / "labeling/reviewer_v3_manifest.json"


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT_PATH.exists() or MANIFEST_PATH.exists():
        raise FileExistsError("evidence-aware reviewer is create-only")
    queue = verify_envelope(QUEUE_PATH, require_blinded=True)
    final = verify_envelope(FINAL_PATH, require_blinded=True)
    packet = json.loads(PACKET_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    html = render_evidence_aware_labeling_app(queue, packet, protocol)
    with OUTPUT_PATH.open("x", encoding="utf-8") as stream:
        stream.write(html)
    manifest = {
        "schema_version": "intervenebench.evidence_aware_reviewer_manifest.v1",
        "status": "frozen_before_human_labeling",
        "generation_final_manifest_payload_sha256": payload_hash(final),
        "blinded_queue_payload_sha256": payload_hash(queue),
        "evidence_packet_sha256": payload_hash(packet),
        "evaluation_protocol_sha256": payload_hash(protocol),
        "reviewer_html_sha256": file_sha256(OUTPUT_PATH),
        "item_count": queue["item_count"],
        "model_identity_visible": False,
        "human_labels_collected": False,
        "automated_judge_calls": 0,
        "automatic_next_stage": False,
    }
    digest = freeze_envelope(manifest, MANIFEST_PATH, require_blinded=True)
    print(
        {
            "reviewer": str(OUTPUT_PATH.relative_to(ROOT)),
            "manifest_payload_sha256": digest,
            "item_count": queue["item_count"],
        }
    )


if __name__ == "__main__":
    main()
