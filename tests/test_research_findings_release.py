from __future__ import annotations

import json
from pathlib import Path

import pytest

from intervenebench.protocol import freeze_envelope
from intervenebench.release_audit import (
    build_research_release_manifest,
    verify_research_release,
)
from intervenebench.research_findings_release import (
    DEFAULT_RESEARCH_FINDINGS_PATH,
    build_research_findings_payload,
    verify_research_findings,
)


ROOT = Path(__file__).resolve().parents[1]


def test_findings_synthesis_preserves_positive_and_negative_results() -> None:
    payload = build_research_findings_payload(ROOT)

    assert payload["schema_version"] == "intervenebench.research_findings.v1"
    prospective = payload["evidence"]["prospective_confirmation"]
    assert prospective["experiment_count"] == 6
    assert prospective["normalized_experiment_count"] == 5
    assert prospective["exact_choice_count"] == 3
    assert prospective["practically_reliable_count"] == 6
    assert prospective["mean_normalized_regret"] == pytest.approx(
        0.003520577798391078
    )
    assert prospective["uniform_mean_normalized_regret"] == pytest.approx(
        0.040958979015419957
    )

    trust = payload["evidence"]["prospective_trust_diagnostics"]
    assert trust["validated_threshold"] is False
    assert trust["ranking_better_than_random_abstention"] is False

    fallback = payload["evidence"]["prospective_human_fallback"]
    assert fallback["any_tested_policy_improved_at_any_nonzero_budget"] is False
    assert fallback["balanced_eb_directionally_replicated"] is True
    assert fallback["harm_to_correction_magnitude_ratio"] == pytest.approx(
        13.59010662821557
    )

    cross_family = payload["evidence"]["retrospective_cross_family"]
    assert cross_family["experiment_count"] == 5
    assert cross_family["primary_exact_choice_rate"] == 0.4
    assert cross_family["mistral_exact_choice_rate"] == 0.4
    assert cross_family["model_disagreement_predictive_signal"] is False

    decisions = payload["release_decisions"]
    assert decisions["candidate_screening"]["decision"] == "limited_research_use"
    assert decisions["autonomous_intervention_selection"]["decision"] == "hold"
    assert decisions["confidence_based_abstention"]["decision"] == "hold"
    assert decisions["small_sample_human_fallback"]["decision"] == "hold"


def test_findings_artifact_is_aggregate_only() -> None:
    payload = build_research_findings_payload(ROOT)
    encoded = json.dumps(payload).casefold()

    for forbidden in (
        '"participant_id"',
        '"participant_rows"',
        '"experiment_scores"',
        '"human_arm_means"',
        '"human_treatment_effects"',
        '"tau_h"',
    ):
        assert forbidden not in encoded
    assert payload["privacy"]["participant_rows_accessed"] == 0
    assert payload["privacy"]["participant_rows_serialized"] == 0


def test_findings_hash_tampering_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "findings.json"
    freeze_envelope(build_research_findings_payload(ROOT), path)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["evidence"]["prospective_confirmation"][
        "exact_choice_count"
    ] = 6
    path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_research_findings(path)


def test_checked_in_findings_and_release_manifest_verify() -> None:
    payload = verify_research_findings(ROOT / DEFAULT_RESEARCH_FINDINGS_PATH)
    assert payload == build_research_findings_payload(ROOT)

    report = verify_research_release(ROOT)
    assert report["status"] == "pass"
    assert report["verified_file_count"] >= 8
    assert report["findings_payload_matches_rebuild"] is True


def test_release_manifest_detects_file_drift(tmp_path: Path) -> None:
    tracked = tmp_path / "tracked.md"
    tracked.write_text("frozen\n", encoding="utf-8")
    manifest = build_research_release_manifest(
        tmp_path, tracked_paths=(Path("tracked.md"),)
    )
    manifest_path = tmp_path / "manifest.json"
    freeze_envelope(manifest, manifest_path)
    tracked.write_text("drifted\n", encoding="utf-8")

    with pytest.raises(ValueError, match="file hash drifted"):
        verify_research_release(
            tmp_path,
            manifest_path=manifest_path,
            rebuild_findings=False,
        )
