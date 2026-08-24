from __future__ import annotations

import json
from pathlib import Path

import pytest

from intervenebench.protocol import freeze_envelope
from intervenebench.public_case_study import (
    DEFAULT_PUBLIC_CASE_STUDY_PATH,
    build_public_case_study_payload,
    render_public_case_study,
    verify_public_case_study,
)


ROOT = Path(__file__).resolve().parents[1]


def test_public_payload_is_aggregate_only_and_matches_frozen_confirmation() -> None:
    payload = build_public_case_study_payload(ROOT)

    assert payload["schema_version"] == "intervenebench.public_case_study.v1"
    assert payload["evidence_scope"] == {
        "development_experiment_count": 9,
        "normalized_confirmation_task_count": 5,
        "prospective_confirmation_experiment_count": 6,
    }
    assert payload["run_integrity"]["planned_model_outputs"] == 1464
    assert payload["run_integrity"]["schema_valid_model_outputs"] == 1404
    assert payload["run_integrity"]["participant_rows_serialized"] == 0
    assert payload["decision_evidence"]["exact_choice"]["count"] == 3
    assert payload["decision_evidence"]["exact_choice"]["experiment_count"] == 6
    assert payload["decision_evidence"]["normalized_regret"]["primary_mean"] == pytest.approx(
        0.003520577798391078
    )
    assert payload["decision_evidence"]["normalized_regret"]["uniform_mean"] == pytest.approx(
        0.040958979015419957
    )
    assert "experiment_scores" not in json.dumps(payload)
    encoded = json.dumps(payload)
    assert '"human_arm_means":' not in encoded
    assert '"participant_id":' not in encoded


def test_release_decisions_are_recomputed_not_trusted_from_json(tmp_path: Path) -> None:
    payload = build_public_case_study_payload(ROOT)
    path = tmp_path / "case.json"
    freeze_envelope(payload, path)

    report = verify_public_case_study(path)
    decisions = report["release_decisions"]
    assert decisions["candidate_screening"]["decision"] == "limited_research_use"
    assert decisions["autonomous_intervention_selection"]["decision"] == "hold"
    assert decisions["confidence_based_abstention"]["decision"] == "hold"
    assert decisions["small_sample_human_fallback"]["decision"] == "hold"


def test_public_bundle_hash_tampering_is_rejected(tmp_path: Path) -> None:
    payload = build_public_case_study_payload(ROOT)
    path = tmp_path / "case.json"
    freeze_envelope(payload, path)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["run_integrity"]["planned_model_outputs"] = 1
    path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_public_case_study(path)


def test_checked_in_public_bundle_verifies_and_renders() -> None:
    report = verify_public_case_study(ROOT / DEFAULT_PUBLIC_CASE_STUDY_PATH)
    rendered = render_public_case_study(report)

    assert "Candidate screening: LIMITED RESEARCH USE" in rendered
    assert "Autonomous intervention selection: HOLD" in rendered
    assert "3/6" in rendered
    assert "0.0035" in rendered
