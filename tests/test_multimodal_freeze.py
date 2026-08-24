from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from intervenebench.balanced_forced_choice import read_json_object
from intervenebench.multimodal_freeze import (
    build_cache_authorization,
    build_execution_authorization,
    build_materialization_authorization,
    build_prospective_multimodal_freeze,
    prepare_multimodal_requests,
    validate_cache_authorization,
    validate_execution_authorization,
    validate_materialization_authorization,
    verify_multimodal_raw_result,
    verify_prospective_multimodal_freeze,
)


ROOT = Path(__file__).resolve().parents[1]


def test_multimodal_freeze_is_complete_zero_authority_and_replayable() -> None:
    freeze = build_prospective_multimodal_freeze(ROOT)
    assert read_json_object(
        ROOT / "configs/simulators/prospective_multimodal_v4.json"
    ) == freeze
    summary = verify_prospective_multimodal_freeze(ROOT, freeze)
    assert summary["experiment_count"] == 3
    assert summary["model_count"] == 3
    assert summary["call_count"] == 54
    assert not any(freeze["authority"].values())
    assert freeze["method"]["primary_model_id"] == "qwen3_vl_8b_primary"
    assert freeze["diagnostics"]["frozen_before_target_outcomes"] is True
    assert freeze["schema_version"] == "prospective_multimodal_freeze.v4"


def test_multimodal_freeze_binds_plan_models_assets_and_discovery_selection() -> None:
    freeze = build_prospective_multimodal_freeze(ROOT)
    plan = read_json_object(ROOT / freeze["plan"]["path"])
    assert plan["primary_model_id"] == freeze["method"]["primary_model_id"]
    assert len(freeze["assets"]) == 9
    assert freeze["discovery_selection_basis"]["selected_text_model_id"] == (
        "qwen3_8b_generic"
    )
    assert freeze["runtime"]["pillow"] == "11.3.0"
    assert freeze["runtime"]["torchvision"] == "0.24.1"


def test_multimodal_freeze_rejects_authority_and_method_mutations() -> None:
    freeze = build_prospective_multimodal_freeze(ROOT)
    authorized = deepcopy(freeze)
    authorized["authority"]["paid_inference_authorized"] = True
    with pytest.raises(ValueError, match="does not replay"):
        verify_prospective_multimodal_freeze(ROOT, authorized)
    changed = deepcopy(freeze)
    changed["method"]["primary_model_id"] = "qwen2_5_vl_7b_comparator"
    with pytest.raises(ValueError, match="does not replay"):
        verify_prospective_multimodal_freeze(ROOT, changed)


def test_multimodal_authorizations_are_exact_and_staged() -> None:
    freeze = build_prospective_multimodal_freeze(ROOT)
    plan = read_json_object(ROOT / freeze["plan"]["path"])
    materialization = build_materialization_authorization(
        freeze=freeze, plan=plan
    )
    validate_materialization_authorization(
        materialization, freeze=freeze, plan=plan
    )
    cache = build_cache_authorization(
        freeze=freeze, plan=plan, modal_image_id="im-fixture"
    )
    validate_cache_authorization(
        cache, freeze=freeze, plan=plan, modal_image_id="im-fixture"
    )
    cache_hashes = {
        "qwen3_vl_8b_primary": "a" * 64,
        "qwen2_5_vl_7b_comparator": "b" * 64,
        "qwen3_8b_text_ablation": "c" * 64,
    }
    execution = build_execution_authorization(
        freeze=freeze,
        plan=plan,
        modal_image_id="im-fixture",
        cache_hashes=cache_hashes,
    )
    validate_execution_authorization(
        execution,
        freeze=freeze,
        plan=plan,
        modal_image_id="im-fixture",
        cache_hashes=cache_hashes,
    )
    assert materialization["paid_inference_authorized"] is False
    assert cache["paid_gpu_inference_authorized"] is False
    assert execution["paid_inference_authorized"] is True
    assert execution["outcome_reveal_authorized"] is False


def test_multimodal_raw_result_inverse_maps_reverse_display_values() -> None:
    request = next(
        request
        for request in prepare_multimodal_requests(ROOT)
        if request["option_order"] == "reverse"
    )
    raw = {
        "call_id": request["call_id"],
        "model_id": request["model_id"],
        "probabilities_by_code": {
            code: float(index == 0)
            for index, code in enumerate(request["answer_codes"])
        },
        "candidate_token_ids": list(range(7)),
        "candidate_token_strings": request["answer_codes"],
        "runtime_attestation": {"fixture": True},
    }
    verified = verify_multimodal_raw_result(request, raw)
    assert verified["probabilities"][7] == 1.0
    assert verified["probabilities"][1] == 0.0
