from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from intervenebench.answer_order_canary import (
    build_call_plan,
    build_execution_authorization,
    build_materialization_authorization,
    prepare_requests,
    read_json_object,
    reverse_order_prompt,
    validate_execution_authorization,
    validate_materialization_authorization,
    verify_freeze,
    verify_result,
)
from intervenebench.modal_forced_choice import MODEL_IDS
from intervenebench.protocol import assert_blinded_payload


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "configs/simulators/answer_order_canary_v1.json"
PLAN = ROOT / "data/manifests/simulators/answer_order_canary_plan_v1.json"


def test_answer_order_freeze_is_bounded_and_zero_authority() -> None:
    summary = verify_freeze(ROOT, freeze_path=FREEZE, plan_path=PLAN)
    assert summary["call_count"] == 40
    assert summary["incremental_cost_cap_usd"] == 1.75
    freeze = read_json_object(FREEZE)
    assert not any(freeze["authority"].values())
    assert freeze["method"]["generation_calls"] == 0
    assert freeze["success_gate"]["next_stage_automatic"] is False
    assert freeze["robustness_gate"] == {
        "maximum_median_total_variation": 0.10,
        "maximum_nearest_rank_p90_total_variation": 0.25,
        "minimum_modal_response_stability": 0.75,
        "minimum_screened_pair_choice_stability": 0.80,
        "required_to_scale_single_order_method": "all_thresholds_pass",
        "failure_pivot": (
            "do_not_scale_single_order; develop_balanced_permutation_average"
        ),
    }


def test_answer_order_plan_replays_and_reverses_every_option_list() -> None:
    plan = read_json_object(PLAN)
    assert plan == build_call_plan(ROOT)
    requests = prepare_requests(ROOT, freeze_path=FREEZE, plan_path=PLAN)
    assert len(requests) == 40
    for request in requests:
        assert request["display_option_values"] == list(
            reversed(request["source_option_values"])
        )
        assert request["call_id"].startswith("reverse-order--")
        assert request["source_order_call_id"].startswith("screen--")
        assert request["order_variant"] == "full_reverse"
        assert_blinded_payload(request)


def test_reverse_prompt_labels_follow_reversed_source_order() -> None:
    bundle = read_json_object(
        ROOT / "data/manifests/contracts/5vm8g_blinded_bundle.json"
    )
    arm = bundle["arms"][0]
    variant = None if "message" in arm else arm["message_variants"][0]["variant_id"]
    prompt = reverse_order_prompt(
        bundle, arm_id=arm["arm_id"], variant_id=variant
    )
    labels = [item["label"] for item in reversed(bundle["response_options"])]
    positions = [prompt.index(f"{code}: {label}") for code, label in zip("ABCDEFGH", labels)]
    assert positions == sorted(positions)


def test_answer_order_result_inverse_maps_codes_to_source_values() -> None:
    request = prepare_requests(ROOT, freeze_path=FREEZE, plan_path=PLAN)[0]
    codes = request["answer_codes"]
    probabilities = {code: 0.0 for code in codes}
    probabilities[codes[0]] = 1.0
    verified = verify_result(
        request,
        {
            "call_id": request["call_id"],
            "model_id": request["model_id"],
            "probabilities_by_code": probabilities,
            "candidate_token_ids": list(range(100, 100 + len(codes))),
            "candidate_token_strings": list(codes),
            "runtime_attestation": {"placeholder": True},
        },
    )
    assert verified["probabilities"][request["source_option_values"][-1]] == 1.0
    assert verified["display_option_values"] == list(
        reversed(verified["source_option_values"])
    )


def test_answer_order_authorizations_are_exact_and_scope_closed() -> None:
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
        model_id: f"{index + 1:064x}"
        for index, model_id in enumerate(MODEL_IDS)
    }
    execution = build_execution_authorization(
        freeze=freeze,
        plan=plan,
        modal_image_id="im-answer-order",
        cache_hashes=cache_hashes,
    )
    validate_execution_authorization(
        execution,
        freeze=freeze,
        plan=plan,
        modal_image_id="im-answer-order",
        cache_hashes=cache_hashes,
    )
    expanded = deepcopy(execution)
    expanded["next_stage_authorized"] = True
    with pytest.raises(PermissionError):
        validate_execution_authorization(
            expanded,
            freeze=freeze,
            plan=plan,
            modal_image_id="im-answer-order",
            cache_hashes=cache_hashes,
        )


def test_answer_order_remote_worker_has_no_generation_or_parser() -> None:
    source = (ROOT / "infra/modal/answer_order_app.py").read_text(encoding="utf-8")
    assert ".generate(" not in source
    assert "loaded_model(**inputs, use_cache=False)" in source
    assert "torch.softmax" in source
