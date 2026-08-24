from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from intervenebench.confirmation_execution import (
    DEFAULT_CONFIRMATION_EXECUTION_FREEZE_PATH,
    build_confirmation_execution_freeze,
    validate_execution_authorization,
    validate_materialization_authorization,
    verify_confirmation_execution_freeze,
)
from intervenebench.protocol import payload_hash


ROOT = Path(__file__).resolve().parents[1]


def test_execution_freeze_has_zero_authority_and_exact_groups() -> None:
    freeze = build_confirmation_execution_freeze(ROOT)
    assert all(value is False for value in freeze["authority"].values())
    assert freeze["call_plan"]["planned_call_count"] == 1464
    assert freeze["call_plan"]["conditional_reserve_call_count"] == 236
    assert freeze["planned_calls_by_cache_model"] == {
        "qwen3_8b_generic": 560,
        "qwen3_14b_generic": 248,
        "qwen2_5_14b_generic": 248,
        "socrates_qwen2_5_14b_sft": 216,
        "qwen3_vl_8b_primary": 128,
        "qwen2_5_vl_7b_comparator": 64,
    }
    assert freeze["model_download_policy"] == (
        "reuse_verified_cache_only_no_download_function"
    )


def test_materialization_authorization_is_separate_and_narrow() -> None:
    freeze = build_confirmation_execution_freeze(ROOT)
    authorization = {
        "schema_version": "confirmation_materialization_authorization.v1",
        "status": "authorized_image_build_zero_inference",
        "freeze_payload_sha256": payload_hash(freeze),
        "call_plan_payload_sha256": freeze["call_plan"]["payload_sha256"],
        "modal_image_materialization_authorized": True,
        "paid_inference_authorized": False,
        "model_download_authorized": False,
        "confirmation_outcome_reveal_authorized": False,
        "participant_row_access_authorized": False,
        "automatic_next_stage_authorized": False,
    }
    validate_materialization_authorization(authorization, freeze=freeze)
    contaminated = deepcopy(authorization)
    contaminated["paid_inference_authorized"] = True
    with pytest.raises(PermissionError, match="authority"):
        validate_materialization_authorization(contaminated, freeze=freeze)


def test_execution_authorization_cannot_enable_reveal_or_reserve() -> None:
    freeze = build_confirmation_execution_freeze(ROOT)
    cache = {model_id: "a" * 64 for model_id in freeze["cache_model_ids"]}
    authorization = {
        "schema_version": "confirmation_execution_authorization.v1",
        "status": "authorized_exact_planned_calls_only",
        "freeze_payload_sha256": payload_hash(freeze),
        "call_plan_payload_sha256": freeze["call_plan"]["payload_sha256"],
        "planned_call_ids_sha256": freeze["call_plan"]["planned_call_ids_sha256"],
        "planned_call_count": 1464,
        "maximum_attempt_count": 1464,
        "hard_incremental_cost_cap_usd": 125.0,
        "modal_image_id": "im-test",
        "cache_attestation_sha256_by_model": cache,
        "paid_inference_authorized": True,
        "modal_compute_authorized": True,
        "model_download_authorized": False,
        "adaptive_reserve_authorized": False,
        "confirmation_outcome_reveal_authorized": False,
        "participant_row_access_authorized": False,
        "automatic_next_stage_authorized": False,
    }
    validate_execution_authorization(
        authorization,
        freeze=freeze,
        modal_image_id="im-test",
        cache_attestation_sha256_by_model=cache,
    )
    for key in ("adaptive_reserve_authorized", "confirmation_outcome_reveal_authorized"):
        widened = deepcopy(authorization)
        widened[key] = True
        with pytest.raises(PermissionError, match="authority"):
            validate_execution_authorization(
                widened,
                freeze=freeze,
                modal_image_id="im-test",
                cache_attestation_sha256_by_model=cache,
            )


def test_frozen_confirmation_execution_config_replays() -> None:
    value = verify_confirmation_execution_freeze(
        ROOT, ROOT / DEFAULT_CONFIRMATION_EXECUTION_FREEZE_PATH
    )
    assert value == build_confirmation_execution_freeze(ROOT)

