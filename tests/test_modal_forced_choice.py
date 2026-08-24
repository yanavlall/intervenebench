from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from intervenebench.modal_forced_choice import (
    MODEL_IDS,
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
from intervenebench.protocol import assert_blinded_payload


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "configs/simulators/modal_forced_choice_v1.json"
PLAN = ROOT / "data/manifests/simulators/modal_forced_choice_call_plan_v1.json"


def test_forced_choice_freeze_is_small_zero_authority_and_replays() -> None:
    summary = verify_freeze(ROOT, freeze_path=FREEZE, call_plan_path=PLAN)
    assert summary["call_count"] == 4
    assert summary["incremental_cost_cap_usd"] == 0.9
    assert read_json_object(PLAN) == build_call_plan(ROOT)
    freeze = read_json_object(FREEZE)
    assert not any(freeze["authority"].values())
    assert freeze["method"]["generation_calls"] == 0
    assert freeze["success_gate"]["next_stage_automatic"] is False


def test_forced_choice_calls_use_identical_prompt_and_exact_codes() -> None:
    calls = prepare_calls(ROOT, freeze_path=FREEZE, call_plan_path=PLAN)
    assert tuple(call.model_id for call in calls) == MODEL_IDS
    assert len({call.prompt for call in calls}) == 1
    assert len({call.request["prompt_sha256"] for call in calls}) == 1
    for call in calls:
        assert call.answer_codes == ("A", "B", "C", "D", "E")
        assert call.option_values == (1, 2, 3, 4, 5)
        assert_blinded_payload(call.request)


def test_forced_choice_result_maps_code_probabilities_to_source_values() -> None:
    call = prepare_calls(ROOT, freeze_path=FREEZE, call_plan_path=PLAN)[0]
    result = verify_result(
        call,
        {
            "call_id": call.call_id,
            "model_id": call.model_id,
            "probabilities_by_code": {
                "A": 0.1, "B": 0.2, "C": 0.3, "D": 0.25, "E": 0.15
            },
            "candidate_token_ids": [32, 33, 34, 35, 36],
            "candidate_token_strings": ["A", "B", "C", "D", "E"],
            "runtime_attestation": {"placeholder": True},
        },
    )
    assert result["probabilities"] == {
        1: 0.1, 2: 0.2, 3: 0.3, 4: 0.25, 5: 0.15
    }


@pytest.mark.parametrize(
    "probabilities,token_ids",
    [
        ({"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.2, "E": 0.1}, [1, 2, 3, 4, 5]),
        ({"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.2, "E": 0.2}, [1, 2, 3, 4, 4]),
        ({"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.2, "F": 0.2}, [1, 2, 3, 4, 5]),
    ],
)
def test_forced_choice_result_fails_closed(
    probabilities: dict[str, float], token_ids: list[int]
) -> None:
    call = prepare_calls(ROOT, freeze_path=FREEZE, call_plan_path=PLAN)[0]
    with pytest.raises(ValueError):
        verify_result(
            call,
            {
                "call_id": call.call_id,
                "model_id": call.model_id,
                "probabilities_by_code": probabilities,
                "candidate_token_ids": token_ids,
                "candidate_token_strings": ["A", "B", "C", "D", "E"],
                "runtime_attestation": {},
            },
        )


def test_forced_choice_authorizations_are_exact_and_hash_bound() -> None:
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
        freeze=freeze, plan=plan, modal_image_id="im-forced",
        cache_hashes=cache_hashes
    )
    validate_execution_authorization(
        execution, freeze=freeze, plan=plan, modal_image_id="im-forced",
        cache_hashes=cache_hashes
    )
    drifted = deepcopy(execution)
    drifted["maximum_model_attempts"] = 5
    with pytest.raises(PermissionError):
        validate_execution_authorization(
            drifted, freeze=freeze, plan=plan, modal_image_id="im-forced",
            cache_hashes=cache_hashes
        )


def test_remote_method_has_no_generation_or_response_parser() -> None:
    source = (ROOT / "infra/modal/forced_choice_app.py").read_text(encoding="utf-8")
    assert ".generate(" not in source
    assert "loaded_model(**inputs, use_cache=False)" in source
    assert "torch.softmax" in source
