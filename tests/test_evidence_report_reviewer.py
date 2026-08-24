from __future__ import annotations

import json
from pathlib import Path

from intervenebench.evidence_report_eval import (
    render_evidence_aware_labeling_app,
)
from intervenebench.protocol import verify_envelope


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "artifacts/evidence_report_eval/report_generation_20260820_v3"


def test_evidence_aware_reviewer_is_blinded_complete_and_rule_derived() -> None:
    queue = verify_envelope(
        RUN_ROOT / "labeling/blinded_queue.json", require_blinded=True
    )
    packet = json.loads(
        (
            ROOT
            / "data/manifests/qualitative_eval/intervenebench_report_evidence_packet_v1.json"
        ).read_text(encoding="utf-8")
    )
    protocol = json.loads(
        (
            ROOT / "data/manifests/research/evidence_report_eval_v1.json"
        ).read_text(encoding="utf-8")
    )
    html = render_evidence_aware_labeling_app(queue, packet, protocol)

    assert "Reference decision" in html
    assert "Required facts for this scenario" in html
    assert "Show complete locked evidence" in html
    assert packet["facts"][0]["statement"] in html
    assert protocol["rubric"]["dimensions"][0]["description"] in html
    assert "Fundamentally wrong or unsafe" in html
    assert "Download labels" in html
    assert "overall_pass:computed(scores,flags)" in html
    assert 'name="overall"' not in html
    assert "Report under review" in html
    assert "renderReport(item.report)" in html
    assert "JSON.stringify(item.report" not in html
    assert "reviewed · blinded" in html
    for role in protocol["generation"]["model_roles"]:
        assert role not in html
    assert '"report_id"' not in html


def test_evidence_aware_reviewer_rejects_unblinded_queue() -> None:
    queue = verify_envelope(
        RUN_ROOT / "labeling/blinded_queue.json", require_blinded=True
    )
    packet = json.loads(
        (
            ROOT
            / "data/manifests/qualitative_eval/intervenebench_report_evidence_packet_v1.json"
        ).read_text(encoding="utf-8")
    )
    protocol = json.loads(
        (
            ROOT / "data/manifests/research/evidence_report_eval_v1.json"
        ).read_text(encoding="utf-8")
    )
    drifted = dict(queue)
    drifted["model_identity_visible"] = True
    try:
        render_evidence_aware_labeling_app(drifted, packet, protocol)
    except ValueError as error:
        assert "not blinded" in str(error)
    else:
        raise AssertionError("unblinded queue was accepted")
