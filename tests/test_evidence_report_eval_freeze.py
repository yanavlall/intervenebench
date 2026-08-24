from __future__ import annotations

import json
from pathlib import Path

from intervenebench.evidence_report_eval import (
    validate_eval_protocol,
    validate_evidence_packet,
    verify_report_generation_plan,
)
from intervenebench.protocol import payload_hash, verify_envelope


ROOT = Path(__file__).resolve().parents[1]
PACKET_PATH = ROOT / (
    "data/manifests/qualitative_eval/"
    "intervenebench_report_evidence_packet_v1.json"
)
PROTOCOL_PATH = ROOT / "data/manifests/research/evidence_report_eval_v1.json"
PLAN_PATH = ROOT / "data/manifests/qualitative_eval/report_generation_plan_v1.json"


def test_real_evidence_report_eval_freeze_replays_with_zero_authority() -> None:
    packet = json.loads(PACKET_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    plan = verify_envelope(PLAN_PATH, require_blinded=True)

    validate_evidence_packet(packet)
    validate_eval_protocol(protocol, packet)
    verify_report_generation_plan(plan, packet, protocol)

    assert plan["call_count"] == 48
    assert sum(call["scenario_split"] == "held_out" for call in plan["calls"]) == 12
    assert plan["evidence_packet_sha256"] == payload_hash(packet)
    assert plan["evaluation_protocol_sha256"] == payload_hash(protocol)
    assert protocol["analysis"]["release_gate"] == {
        "maximum_false_pass_count": 0,
        "minimum_balanced_accuracy": 0.8,
        "maximum_dimension_mae": 0.75,
        "minimum_second_rater_items": 12,
    }
    assert all(value is False for value in plan["authority"].values())
    assert all(value is False for value in protocol["authority"].values())


def test_real_evidence_packet_is_bound_to_public_aggregate_findings() -> None:
    packet = json.loads(PACKET_PATH.read_text(encoding="utf-8"))
    source = packet["source_artifacts"]
    assert source == [
        {
            "path": "data/public/research_findings_v1.json",
            "payload_sha256": payload_hash(
                verify_envelope(ROOT / "data/public/research_findings_v1.json")
            ),
        }
    ]
    assert packet["privacy"] == {
        "contains_participant_rows": False,
        "contains_experiment_level_human_scores": False,
    }
