from __future__ import annotations

from collections import Counter
from pathlib import Path

from intervenebench.confirmation_calls import (
    DEFAULT_CONFIRMATION_CALL_PLAN_PATH,
    build_confirmation_call_plan,
    prepare_confirmation_requests,
    verify_confirmation_call_plan,
)


ROOT = Path(__file__).resolve().parents[1]


def test_confirmation_call_plan_has_exact_stages_and_scope() -> None:
    plan = build_confirmation_call_plan(ROOT)
    assert plan["planned_call_count"] == 1464
    assert plan["maximum_attempt_count"] == 1700
    assert len(plan["calls"]) == 1700
    assert Counter(call["stage"] for call in plan["calls"]) == {
        "base": 1152,
        "primary_prompt_perturbation": 312,
        "outcome_free_adaptive_reserve": 236,
    }
    assert all(plan["authority"][key] is False for key in plan["authority"])


def test_paired_nuisance_paths_match_across_arms() -> None:
    plan = build_confirmation_call_plan(ROOT)
    shannon = [
        call
        for call in plan["calls"]
        if call["experiment_id"] == "ShannonS2"
        and call["stage"] == "base"
        and call["model_id"] == "qwen3_8b_generic"
        and call["answer_order"] == "source"
    ]
    by_nuisance = Counter(call["nuisance_id"] for call in shannon)
    assert len(by_nuisance) == 8
    assert set(by_nuisance.values()) == {6}
    assert all(call["sequence_seed"] is not None for call in shannon)

    pb2rr = [
        call
        for call in plan["calls"]
        if call["experiment_id"] == "pb2rr"
        and call["stage"] == "base"
        and call["model_id"] == "qwen3_vl_8b_primary"
        and call["answer_order"] == "source"
    ]
    assert len({call["nuisance_id"] for call in pb2rr}) == 16
    assert set(Counter(call["nuisance_id"] for call in pb2rr).values()) == {2}


def test_pb2rr_vision_and_text_assets_are_explicit() -> None:
    plan = build_confirmation_call_plan(ROOT)
    calls = [
        call
        for call in plan["calls"]
        if call["experiment_id"] == "pb2rr" and call["stage"] == "base"
    ]
    vision = [call for call in calls if "_vl_" in call["model_id"]]
    text = [call for call in calls if call["model_id"] == "qwen3_8b_text_ablation"]
    assert vision and text
    assert all(call["asset_path"].endswith(".png") for call in vision)
    assert all(len(call["asset_sha256"]) == 64 for call in vision)
    assert all(call["asset_path"] is None for call in text)
    assert all(call["asset_sha256"] is None for call in text)


def test_requests_reconstruct_every_prompt_without_outcomes() -> None:
    plan = build_confirmation_call_plan(ROOT)
    requests = prepare_confirmation_requests(ROOT, plan=plan, include_reserve=False)
    assert len(requests) == 1464
    assert {request["stage"] for request in requests} == {
        "base",
        "primary_prompt_perturbation",
    }
    assert all(request["prompt"] for request in requests)
    assert all("human_outcome" not in request for request in requests)
    assert all("response" not in request for request in requests)


def test_prompt_perturbation_changes_wrapper_not_cell_support() -> None:
    plan = build_confirmation_call_plan(ROOT)
    calls = plan["calls"]
    base = next(
        call
        for call in calls
        if call["experiment_id"] == "Blair1131"
        and call["stage"] == "base"
        and call["model_id"] == "qwen3_8b_generic"
        and call["arm_id"] == "stay_out_no_troops"
        and call["nuisance_id"] == "president_name_eric"
        and call["answer_order"] == "source"
    )
    perturbed = next(
        call
        for call in calls
        if call["experiment_id"] == base["experiment_id"]
        and call["stage"] == "primary_prompt_perturbation"
        and call["arm_id"] == base["arm_id"]
        and call["nuisance_id"] == base["nuisance_id"]
        and call["answer_order"] == base["answer_order"]
    )
    assert base["prompt_sha256"] != perturbed["prompt_sha256"]
    assert base["source_option_values"] == perturbed["source_option_values"]
    assert base["display_option_values"] == perturbed["display_option_values"]


def test_frozen_confirmation_call_plan_replays() -> None:
    payload = verify_confirmation_call_plan(
        ROOT, ROOT / DEFAULT_CONFIRMATION_CALL_PLAN_PATH
    )
    assert payload == build_confirmation_call_plan(ROOT)
