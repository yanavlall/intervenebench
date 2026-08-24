from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from intervenebench.forced_choice_screen import (
    EXPERIMENT_IDS,
    answer_codes,
    build_call_plan,
    build_execution_authorization,
    build_materialization_authorization,
    prepare_calls,
    read_json_object,
    validate_execution_authorization,
    validate_materialization_authorization,
    verify_freeze,
    verify_result,
)
from intervenebench.modal_forced_choice import MODEL_IDS
from intervenebench.protocol import assert_blinded_payload


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "configs/simulators/forced_choice_screen_v1.json"
PLAN = ROOT / "data/manifests/simulators/forced_choice_screen_plan_v1.json"


def test_screen_freeze_is_exact_bounded_and_zero_authority() -> None:
    summary = verify_freeze(ROOT, freeze_path=FREEZE, plan_path=PLAN)
    assert summary == {
        "call_count": 40,
        "model_count": 4,
        "experiment_count": 5,
        "incremental_cost_cap_usd": 1.75,
        "freeze_payload_sha256": summary["freeze_payload_sha256"],
        "call_plan_payload_sha256": summary["call_plan_payload_sha256"],
    }
    freeze = read_json_object(FREEZE)
    assert not any(freeze["authority"].values())
    assert freeze["method"]["generation_calls"] == 0
    assert freeze["success_gate"]["next_stage_automatic"] is False


def test_screen_plan_replays_and_has_ten_identical_scope_calls_per_model() -> None:
    plan = read_json_object(PLAN)
    assert plan == build_call_plan(ROOT)
    calls = prepare_calls(ROOT, freeze_path=FREEZE, plan_path=PLAN)
    assert len(calls) == 40
    assert {call.experiment_id for call in calls} == set(EXPERIMENT_IDS)
    for model_id in MODEL_IDS:
        assert sum(call.model_id == model_id for call in calls) == 10
    for experiment_id in EXPERIMENT_IDS:
        assert sum(call.experiment_id == experiment_id for call in calls) == 8
    prompt_by_task_arm: dict[tuple[str, str], set[str]] = {}
    for call in calls:
        key = (call.experiment_id, call.request["arm_id"])
        prompt_by_task_arm.setdefault(key, set()).add(call.prompt)
        assert call.answer_codes == answer_codes(len(call.option_values))
        assert_blinded_payload(call.request)
    assert len(prompt_by_task_arm) == 10
    assert all(len(prompts) == 1 for prompts in prompt_by_task_arm.values())


@pytest.mark.parametrize("option_count", range(2, 9))
def test_answer_code_prefix_is_deterministic(option_count: int) -> None:
    assert answer_codes(option_count) == tuple("ABCDEFGH"[:option_count])


def test_screen_result_accepts_eight_code_distribution_and_maps_source_values() -> None:
    call = next(
        item
        for item in prepare_calls(ROOT, freeze_path=FREEZE, plan_path=PLAN)
        if item.experiment_id == "xc4yq"
    )
    probabilities = {code: 0.125 for code in call.answer_codes}
    verified = verify_result(
        call,
        {
            "call_id": call.call_id,
            "model_id": call.model_id,
            "probabilities_by_code": probabilities,
            "candidate_token_ids": list(range(32, 40)),
            "candidate_token_strings": list(call.answer_codes),
            "runtime_attestation": {"placeholder": True},
        },
    )
    assert len(verified["probabilities"]) == 8
    assert sum(verified["probabilities"].values()) == 1.0


def test_screen_result_rejects_non_normalized_or_duplicate_token_output() -> None:
    call = prepare_calls(ROOT, freeze_path=FREEZE, plan_path=PLAN)[0]
    with pytest.raises(ValueError):
        verify_result(
            call,
            {
                "call_id": call.call_id,
                "model_id": call.model_id,
                "probabilities_by_code": {
                    code: 0.1 for code in call.answer_codes
                },
                "candidate_token_ids": [32] * len(call.answer_codes),
                "candidate_token_strings": list(call.answer_codes),
                "runtime_attestation": {},
            },
        )


def test_screen_authorizations_are_hash_bound_and_scope_closed() -> None:
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
    cache_hashes = {
        model_id: f"{index + 1:064x}" for index, model_id in enumerate(MODEL_IDS)
    }
    execution = build_execution_authorization(
        freeze=freeze, plan=plan, modal_image_id="im-screen",
        cache_hashes=cache_hashes
    )
    validate_execution_authorization(
        execution, freeze=freeze, plan=plan, modal_image_id="im-screen",
        cache_hashes=cache_hashes
    )
    expanded = deepcopy(execution)
    expanded["next_stage_authorized"] = True
    with pytest.raises(PermissionError):
        validate_execution_authorization(
            expanded, freeze=freeze, plan=plan, modal_image_id="im-screen",
            cache_hashes=cache_hashes
        )


def test_screen_remote_worker_has_no_generation_or_parser() -> None:
    source = (ROOT / "infra/modal/forced_choice_screen_app.py").read_text(
        encoding="utf-8"
    )
    assert ".generate(" not in source
    assert "loaded_model(**inputs, use_cache=False)" in source
    assert "torch.softmax" in source
