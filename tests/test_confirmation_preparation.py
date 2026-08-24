from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from intervenebench.confirmation_preparation import (
    BASE_CALLS,
    CONFIRMATION_IDS,
    DEFAULT_CONFIRMATION_PREPARATION_PATH,
    MAXIMUM_ATTEMPTS,
    PERTURBATION_CALLS,
    PLANNED_CALLS,
    RESERVE_CALLS,
    build_confirmation_preparation,
    validate_confirmation_protocol,
    verify_confirmation_preparation,
)


ROOT = Path(__file__).resolve().parents[1]


def test_confirmation_protocol_freezes_exact_scope_calls_and_authority() -> None:
    payload = build_confirmation_preparation(ROOT)
    assert tuple(payload["experiment_ids"]) == CONFIRMATION_IDS
    assert payload["call_plan"] == {
        "base_calls": BASE_CALLS,
        "primary_prompt_perturbation_calls": PERTURBATION_CALLS,
        "outcome_free_adaptive_reserve_calls": RESERVE_CALLS,
        "planned_calls": PLANNED_CALLS,
        "maximum_attempts": MAXIMUM_ATTEMPTS,
    }
    assert payload["call_plan"] == {
        "base_calls": 1152,
        "primary_prompt_perturbation_calls": 312,
        "outcome_free_adaptive_reserve_calls": 236,
        "planned_calls": 1464,
        "maximum_attempts": 1700,
    }
    assert all(value is False for value in payload["authority"].values())
    assert payload["confirmation_outcomes_accessed"] is False
    assert payload["participant_rows_read"] == 0
    assert payload["participant_rows_serialized"] == 0


def test_confirmation_task_matrix_and_checkpoint_exposure_are_exact() -> None:
    payload = build_confirmation_preparation(ROOT)
    tasks = {row["experiment_id"]: row for row in payload["tasks"]}
    assert tasks["tcg8p"]["base_calls"] == 240
    assert tasks["pb2rr"]["base_calls"] == 192
    assert tasks["z358z"]["base_calls"] == 96
    assert tasks["ShannonS2"]["base_calls"] == 384
    assert tasks["Blair1131"]["base_calls"] == 48
    assert tasks["KlarS44"]["base_calls"] == 192

    socrates = payload["socrates_checkpoint_compatibility"]
    assert socrates["pb2rr"]["primary_eligible"] is False
    assert socrates["z358z"]["primary_eligible"] is False
    assert socrates["tcg8p"]["primary_eligible"] is True
    assert socrates["KlarS44"]["checkpoint_experiment_id"] == "xtvu5"
    assert socrates["KlarS44"]["primary_eligible"] is True


def test_pb2rr_rendered_assets_are_hash_bound_to_source_pdfs() -> None:
    payload = build_confirmation_preparation(ROOT)
    assets = payload["pb2rr_modal_assets"]
    assert {row["arm_id"] for row in assets} == {
        "iphone_growth_control_article",
        "hispanic_population_growth_article",
    }
    assert all(row["source_page"] == 1 for row in assets)
    assert all(row["visual_qa"] == "passed_full_page_no_clipping" for row in assets)
    assert all(row["png_width"] == 1600 for row in assets)
    assert all(row["png_height"] == 1200 for row in assets)


def test_classical_confirmation_predictions_use_only_development_fit() -> None:
    payload = build_confirmation_preparation(ROOT)
    classical = payload["classical_baseline_predictions"]
    assert set(classical) == {
        "pb2rr",
        "z358z",
        "ShannonS2",
        "Blair1131",
        "KlarS44",
    }
    assert "tcg8p" not in classical
    assert all(
        row["human_outcome_accessed"] is False for row in classical.values()
    )
    assert payload["classical_training_experiment_ids"] == [
        "5vm8g",
        "de5hx",
        "e2pyb",
        "es4xw",
        "jf46x",
        "nj5dx",
        "turagaS11",
        "wallaceS12",
        "xc4yq",
    ]


def test_trust_and_fallback_claim_boundaries_are_frozen() -> None:
    payload = build_confirmation_preparation(ROOT)
    trust = payload["trust_evaluation"]
    assert trust["learned_threshold"] is None
    assert trust["accept_abstain_policy"] == "not_validated_not_deployed"
    assert trust["coverage_counts"] == {"50_percent": 3, "75_percent": 5, "100_percent": 6}
    assert trust["experiment_is_unit"] is True

    fallback = payload["human_fallback"]
    assert fallback["budgets"] == [0, 10, 25, 50, 100, 250]
    assert fallback["fusion_tuning"] == "stopped_after_negative_development_result"
    assert fallback["pooled_normalized_experiment_ids"] == [
        "pb2rr",
        "z358z",
        "ShannonS2",
        "Blair1131",
        "KlarS44",
    ]
    assert fallback["raw_unit_separate_experiment_ids"] == ["tcg8p"]


def test_protocol_validator_rejects_call_or_authority_drift() -> None:
    payload = build_confirmation_preparation(ROOT)
    protocol = deepcopy(payload["protocol_snapshot"])
    protocol["tasks"][0]["base_calls"] += 1
    with pytest.raises(ValueError, match="call"):
        validate_confirmation_protocol(ROOT, protocol)

    protocol = deepcopy(payload["protocol_snapshot"])
    protocol["authority"]["modal_compute_authorized"] = True
    with pytest.raises(ValueError, match="authority"):
        validate_confirmation_protocol(ROOT, protocol)


def test_frozen_confirmation_preparation_replays() -> None:
    payload = verify_confirmation_preparation(
        ROOT, ROOT / DEFAULT_CONFIRMATION_PREPARATION_PATH
    )
    assert payload == build_confirmation_preparation(ROOT)

