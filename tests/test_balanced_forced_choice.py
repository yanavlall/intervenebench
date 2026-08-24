from __future__ import annotations

from pathlib import Path

import pytest

from intervenebench.balanced_forced_choice import (
    build_balanced_discovery_artifact,
    build_completed_full_action_artifact,
    build_execution_authorization,
    build_full_action_plan,
    build_materialization_authorization,
    prepare_new_requests,
    read_json_object,
    validate_execution_authorization,
    validate_materialization_authorization,
    verify_full_action_freeze,
    verify_new_result,
    weighted_distribution_average,
)
from intervenebench.modal_forced_choice import MODEL_IDS


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = ROOT / "artifacts/forced_choice_screen/discovery_screen_20260813_v1"
REVERSE_RUN = (
    ROOT / "artifacts/answer_order_canary/answer_order_canary_20260813_v1"
)
FULL_RUN = ROOT / "artifacts/balanced_full_action/balanced_full_action_20260813_v1"
FREEZE = ROOT / "configs/simulators/balanced_full_action_v1.json"
PLAN = ROOT / "data/manifests/simulators/balanced_full_action_plan_v1.json"


def test_balanced_average_is_symmetric_normalized_and_support_safe() -> None:
    first = {1: 0.8, 2: 0.2}
    second = {1: 0.1, 2: 0.9}
    expected = {1: 0.45, 2: 0.55}
    assert weighted_distribution_average(first, second) == pytest.approx(expected)
    assert weighted_distribution_average(second, first) == pytest.approx(expected)
    assert sum(weighted_distribution_average(first, second).values()) == 1.0
    with pytest.raises(ValueError):
        weighted_distribution_average(first, {1: 1.0})


def test_existing_paired_outputs_produce_forty_balanced_arm_predictions() -> None:
    result = build_balanced_discovery_artifact(
        ROOT, source_run_root=SOURCE_RUN, reverse_run_root=REVERSE_RUN
    )
    assert result["balanced_arm_prediction_count"] == 40
    assert result["screened_pair_recommendation_count"] == 20
    assert result["outcome_access"] == "not_accessed"
    for row in result["balanced_arm_predictions"]:
        assert sum(row["balanced_probabilities"].values()) == pytest.approx(1.0)
        assert row["source_order_weight"] == 0.5
        assert row["reverse_order_weight"] == 0.5
    assert all(
        row["full_action_set_recommendation"] is False
        for row in result["screened_pair_recommendations"]
    )


def test_full_action_plan_replays_and_reuses_all_existing_pairs() -> None:
    plan = read_json_object(PLAN)
    assert plan == build_full_action_plan(ROOT)
    assert plan["counts"] == {
        "experiment_count": 5,
        "model_count": 4,
        "unique_arm_count": 17,
        "balanced_arm_prediction_count": 68,
        "logical_ordered_call_count": 136,
        "reused_ordered_call_count": 80,
        "new_ordered_call_count": 56,
        "new_calls_per_model": 14,
        "full_action_recommendation_count": 20,
    }
    assert len(plan["logical_calls"]) == 136
    assert sum(call["acquisition"] == "reuse_verified_existing" for call in plan["logical_calls"]) == 80
    assert len(plan["new_calls"]) == 56
    assert all(call["model_exposure"] for call in plan["logical_calls"])
    assert all(
        call["display_option_values"]
        == (
            call["source_option_values"]
            if call["order_variant"] == "source"
            else list(reversed(call["source_option_values"]))
        )
        for call in plan["logical_calls"]
    )
    for model_id in plan["model_ids"]:
        assert sum(call["model_id"] == model_id for call in plan["new_calls"]) == 14
    for model_id in plan["model_ids"]:
        for experiment_id, expected in {
            "5vm8g": 4,
            "xc4yq": 6,
            "de5hx": 6,
            "turagaS11": 6,
            "wallaceS12": 12,
        }.items():
            assert sum(
                call["model_id"] == model_id
                and call["experiment_id"] == experiment_id
                for call in plan["logical_calls"]
            ) == expected


def test_full_action_freeze_is_zero_authority_and_exactly_bounded() -> None:
    summary = verify_full_action_freeze(ROOT, freeze_path=FREEZE, plan_path=PLAN)
    assert summary["logical_call_count"] == 136
    assert summary["reused_call_count"] == 80
    assert summary["new_call_count"] == 56
    assert summary["incremental_cost_cap_usd"] == 1.75
    freeze = read_json_object(FREEZE)
    assert not any(freeze["authority"].values())
    assert freeze["method"]["order_aggregation"] == (
        "equal_weight_source_and_full_reverse_after_inverse_mapping"
    )
    assert freeze["success_gate"]["next_stage_automatic"] is False


def test_new_requests_reconstruct_exactly_and_group_four_by_fourteen() -> None:
    requests = prepare_new_requests(ROOT, freeze_path=FREEZE, plan_path=PLAN)
    assert len(requests) == 56
    for model_id in MODEL_IDS:
        assert sum(request["model_id"] == model_id for request in requests) == 14
    assert all(request["prompt"] for request in requests)
    assert {request["order_variant"] for request in requests} == {
        "source",
        "reverse",
    }


def test_new_result_inverse_maps_display_codes() -> None:
    request = prepare_new_requests(ROOT, freeze_path=FREEZE, plan_path=PLAN)[0]
    probabilities = {code: 0.0 for code in request["answer_codes"]}
    probabilities[request["answer_codes"][0]] = 1.0
    verified = verify_new_result(
        request,
        {
            "call_id": request["call_id"],
            "model_id": request["model_id"],
            "probabilities_by_code": probabilities,
            "candidate_token_ids": list(range(200, 200 + len(probabilities))),
            "candidate_token_strings": request["answer_codes"],
            "runtime_attestation": {"placeholder": True},
        },
    )
    assert verified["probabilities"][request["display_option_values"][0]] == 1.0


def test_full_action_authorizations_are_exact_and_closed() -> None:
    freeze = read_json_object(FREEZE)
    plan = read_json_object(PLAN)
    materialization = build_materialization_authorization(freeze=freeze, plan=plan)
    validate_materialization_authorization(
        materialization, freeze=freeze, plan=plan
    )
    cache_hashes = {
        model_id: f"{index + 1:064x}"
        for index, model_id in enumerate(MODEL_IDS)
    }
    execution = build_execution_authorization(
        freeze=freeze,
        plan=plan,
        modal_image_id="im-balanced-full",
        cache_hashes=cache_hashes,
    )
    validate_execution_authorization(
        execution,
        freeze=freeze,
        plan=plan,
        modal_image_id="im-balanced-full",
        cache_hashes=cache_hashes,
    )
    execution["next_stage_authorized"] = True
    with pytest.raises(PermissionError):
        validate_execution_authorization(
            execution,
            freeze=freeze,
            plan=plan,
            modal_image_id="im-balanced-full",
            cache_hashes=cache_hashes,
        )


def test_full_action_worker_has_no_generation_or_parser() -> None:
    source = (ROOT / "infra/modal/balanced_full_action_app.py").read_text(
        encoding="utf-8"
    )
    assert ".generate(" not in source
    assert "loaded_model(**inputs, use_cache=False)" in source
    assert "len(requests) != 14" in source


def test_completed_full_action_artifact_is_exact_and_outcome_blind() -> None:
    result = build_completed_full_action_artifact(ROOT, new_run_root=FULL_RUN)
    assert result["ordered_component_call_count"] == 136
    assert result["balanced_arm_prediction_count"] == 68
    assert result["full_action_recommendation_count"] == 20
    assert result["outcome_access"] == "not_accessed"
    assert all(
        row["full_action_set_recommendation"] is True
        and row["human_outcome_accessed"] is False
        for row in result["full_action_recommendations"]
    )
    assert len(
        {
            (row["model_id"], row["experiment_id"])
            for row in result["full_action_recommendations"]
        }
    ) == 20
