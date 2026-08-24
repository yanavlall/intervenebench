from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from intervenebench.modal_canary import (
    MODEL_IDS,
    build_canary_call_plan,
    build_execution_authorization,
    build_materialization_authorization,
    prepare_canary_calls,
    read_json_object,
    validate_execution_authorization,
    validate_materialization_authorization,
    verify_canary_freeze,
    verify_canary_result,
)
from intervenebench.protocol import assert_blinded_payload


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "configs/simulators/modal_constrained_canary_v1.json"
PLAN = ROOT / "data/manifests/simulators/modal_constrained_canary_call_plan_v1.json"


def test_canary_freeze_is_exact_small_and_zero_authority() -> None:
    summary = verify_canary_freeze(ROOT, freeze_path=FREEZE, call_plan_path=PLAN)
    assert summary["call_count"] == 4
    assert summary["incremental_cost_cap_usd"] == 1.25
    freeze = read_json_object(FREEZE)
    assert not any(freeze["authority"].values())
    assert freeze["limits"]["maximum_model_attempts"] == 4
    assert freeze["success_gate"]["next_stage_automatic"] is False


def test_canary_plan_replays_and_uses_identical_blinded_prompt() -> None:
    assert read_json_object(PLAN) == build_canary_call_plan(ROOT)
    calls = prepare_canary_calls(ROOT, freeze_path=FREEZE, call_plan_path=PLAN)
    assert tuple(call.model_id for call in calls) == MODEL_IDS
    assert len({call.prompt for call in calls}) == 1
    assert len({call.request["prompt_sha256"] for call in calls}) == 1
    assert {call.request["experiment_id"] for call in calls} == {"5vm8g"}
    for call in calls:
        assert_blinded_payload(call.request)


def test_canary_relative_weights_normalize_without_repair() -> None:
    call = prepare_canary_calls(ROOT, freeze_path=FREEZE, call_plan_path=PLAN)[0]
    attestation = {"placeholder": True}
    verified = verify_canary_result(
        call,
        {
            "call_id": call.call_id,
            "model_id": call.model_id,
            "raw_text": '{"relative_weights":{"1":10,"2":20,"3":30,"4":20,"5":20}}',
            "runtime_attestation": attestation,
        },
    )
    assert verified["relative_weights"] == {
        "1": 10.0, "2": 20.0, "3": 30.0, "4": 20.0, "5": 20.0
    }
    assert verified["probabilities"][1] == pytest.approx(0.1)
    assert sum(verified["probabilities"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "raw",
    [
        '{"relative_weights":{"1":0,"2":20,"3":30,"4":20,"5":30}}',
        '{"relative_weights":{"1":1.5,"2":20,"3":30,"4":20,"5":30}}',
        '{"relative_weights":{"1":101,"2":20,"3":30,"4":20,"5":30}}',
        '{"relative_weights":{"1":10,"2":20,"3":30,"4":40}}',
    ],
)
def test_canary_rejects_output_outside_exact_integer_contract(raw: str) -> None:
    call = prepare_canary_calls(ROOT, freeze_path=FREEZE, call_plan_path=PLAN)[0]
    with pytest.raises(ValueError):
        verify_canary_result(
            call,
            {
                "call_id": call.call_id,
                "model_id": call.model_id,
                "raw_text": raw,
                "runtime_attestation": {},
            },
        )


def test_canary_authorities_are_hash_bound_and_do_not_expand_scope() -> None:
    freeze = read_json_object(FREEZE)
    plan = read_json_object(PLAN)
    materialization = build_materialization_authorization(freeze=freeze, plan=plan)
    validate_materialization_authorization(
        materialization, freeze=freeze, plan=plan
    )
    expanded = deepcopy(materialization)
    expanded["paid_inference_authorized"] = True
    with pytest.raises(PermissionError):
        validate_materialization_authorization(expanded, freeze=freeze, plan=plan)

    cache_hashes = {model_id: f"{index + 1:064x}" for index, model_id in enumerate(MODEL_IDS)}
    execution = build_execution_authorization(
        freeze=freeze, plan=plan, modal_image_id="im-canary",
        cache_hashes=cache_hashes
    )
    validate_execution_authorization(
        execution, freeze=freeze, plan=plan, modal_image_id="im-canary",
        cache_hashes=cache_hashes
    )
    drifted = deepcopy(execution)
    drifted["maximum_model_attempts"] = 5
    with pytest.raises(PermissionError):
        validate_execution_authorization(
            drifted, freeze=freeze, plan=plan, modal_image_id="im-canary",
            cache_hashes=cache_hashes
        )


def test_canary_freeze_detects_plan_mutation() -> None:
    plan = read_json_object(PLAN)
    plan["calls"][0]["prompt_sha256"] = "0" * 64
    assert plan != build_canary_call_plan(ROOT)
