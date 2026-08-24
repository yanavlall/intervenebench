from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from intervenebench.balanced_forced_choice import read_json_object
from intervenebench.prospective_development_protocol import (
    AUTHORIZATION_PATH,
    PROTOCOL_PATH,
    build_equal_rank_confidence,
    build_pre_reveal_protocol,
    build_reveal_authorization,
    verify_pre_reveal_protocol,
    verify_reveal_authorization,
)


ROOT = Path(__file__).resolve().parents[1]


def test_equal_rank_confidence_is_outcome_free_bounded_and_deterministic() -> None:
    diagnostics = [
        {
            "experiment_id": experiment_id,
            "primary_model_balanced_winner_margin": margin,
            "primary_model_source_reverse_choice_stability": stable,
            "primary_model_mean_arm_source_reverse_total_variation": tv,
            "two_vlm_complete_action_choice_agreement": agree,
            "vision_vs_accessible_text_choice_agreement": text,
        }
        for experiment_id, margin, stable, tv, agree, text in (
            ("nj5dx", 0.3, True, 0.1, True, False),
            ("es4xw", 0.1, True, 0.2, False, False),
            ("e2pyb", 0.2, True, 0.5, True, True),
        )
    ]
    result = build_equal_rank_confidence(diagnostics)
    assert set(result["confidence_by_experiment"]) == {"nj5dx", "es4xw", "e2pyb"}
    assert all(0.0 <= value <= 1.0 for value in result["confidence_by_experiment"].values())
    assert result == build_equal_rank_confidence(diagnostics)


def test_protocol_and_authorization_replay_exactly_after_creation() -> None:
    assert read_json_object(ROOT / PROTOCOL_PATH) == build_pre_reveal_protocol(ROOT)
    protocol = verify_pre_reveal_protocol(ROOT)
    assert not any(protocol["authority"].values())
    assert protocol["human_fallback"]["budgets_total_outcome_observations"] == [
        0,
        10,
        25,
        50,
        100,
        250,
    ]
    assert read_json_object(ROOT / AUTHORIZATION_PATH) == build_reveal_authorization(ROOT)
    authorization = verify_reveal_authorization(ROOT)
    assert authorization["human_outcome_reveal_authorized"] is True
    assert authorization["canonical_test_eligible"] is False
    assert authorization["other_experiments_must_remain_sealed"] == [
        "tcg8p",
        "pb2rr",
        "z358z",
        "ShannonS2",
        "Blair1131",
        "KlarS44",
    ]


def test_reveal_authorization_rejects_scope_expansion() -> None:
    authorization = build_reveal_authorization(ROOT)
    changed = deepcopy(authorization)
    changed["experiment_ids"].append("tcg8p")
    assert changed != build_reveal_authorization(ROOT)
