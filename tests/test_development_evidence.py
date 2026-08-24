from __future__ import annotations

from pathlib import Path

import pytest

from intervenebench.development_evidence import (
    DEFAULT_DEVELOPMENT_EVIDENCE_PATH,
    build_development_evidence,
    equal_rank_confidence,
    verify_development_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


def test_equal_rank_confidence_respects_frozen_directions_and_ties() -> None:
    rows = [
        {"experiment_id": "a", "margin": 0.1, "sensitivity": 0.3},
        {"experiment_id": "b", "margin": 0.2, "sensitivity": 0.1},
        {"experiment_id": "c", "margin": 0.2, "sensitivity": 0.2},
    ]
    confidence = equal_rank_confidence(
        rows,
        feature_directions={"margin": "larger", "sensitivity": "smaller"},
    )
    assert confidence["b"] > confidence["c"] > confidence["a"]
    assert all(0.0 <= value <= 1.0 for value in confidence.values())


def test_development_evidence_contains_nine_tasks_without_expanding_reveal() -> None:
    evidence = build_development_evidence(ROOT)
    assert evidence["experiment_count"] == 9
    assert evidence["rich_diagnostic_experiment_count"] == 8
    assert evidence["participant_rows_read"] == 0
    assert evidence["participant_rows_serialized"] == 0
    assert evidence["primary_summary"]["correct_intervention_count"] == 8
    assert evidence["trust_screening"]["threshold_status"] == (
        "no_validated_abstention_threshold"
    )
    assert evidence["trust_screening"]["classifier_status"] == "not_estimable"
    assert evidence["sealed_confirmation_experiment_ids"] == [
        "tcg8p",
        "pb2rr",
        "z358z",
        "ShannonS2",
        "Blair1131",
        "KlarS44",
    ]


def test_frozen_development_evidence_replays_exactly() -> None:
    evidence = verify_development_evidence(
        ROOT, ROOT / DEFAULT_DEVELOPMENT_EVIDENCE_PATH
    )
    assert evidence == build_development_evidence(ROOT)

