from __future__ import annotations

import json
from pathlib import Path

import pytest

from intervenebench.modal_runner import (
    build_cache_authorization_payload,
    build_execution_authorization_payload,
    build_materialization_authorization_payload,
    prepare_modal_calls,
    read_json_object,
    validate_cache_authorization,
    validate_execution_authorization,
    validate_materialization_authorization,
    validate_runtime_attestation,
    verify_remote_result,
)


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "configs/simulators/modal_discovery_preflight_v2.json"
PLAN = ROOT / "data/manifests/simulators/modal_preflight_call_plan_v1.json"


def test_prepare_calls_reconstructs_exact_40_blinded_prompts() -> None:
    calls = prepare_modal_calls(ROOT, freeze_path=FREEZE, call_plan_path=PLAN)
    assert len(calls) == 40
    assert len({call.call_id for call in calls}) == 40
    assert all(call.request["prompt"] == call.prompt for call in calls)
    assert all(call.request["option_values"] for call in calls)
    assert not any("human" in key for call in calls for key in call.request)


def test_remote_result_parses_strictly_without_repair() -> None:
    call = prepare_modal_calls(ROOT, freeze_path=FREEZE, call_plan_path=PLAN)[0]
    values = call.option_values
    probability = 1.0 / len(values)
    raw = json.dumps(
        {"probabilities": {str(value): probability for value in values}}
    )
    result = verify_remote_result(
        call,
        {
            "call_id": call.call_id,
            "model_id": call.model_id,
            "raw_text": raw,
            "runtime_attestation": {"gpu_name": "NVIDIA L40S"},
        },
    )
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="every response value"):
        verify_remote_result(
            call,
            {
                "call_id": call.call_id,
                "model_id": call.model_id,
                "raw_text": '{"probabilities":{"1":1.0}}',
                "runtime_attestation": {"gpu_name": "NVIDIA L40S"},
            },
        )


def test_materialization_and_cache_authorities_cannot_run_inference() -> None:
    freeze = read_json_object(FREEZE)
    plan = read_json_object(PLAN)
    materialization = build_materialization_authorization_payload(
        freeze=freeze,
        call_plan=plan,
        modal_profile="yanav",
        maximum_total_cost_usd=5.0,
    )
    assert materialization["image_materialization_authorized"] is True
    assert materialization["model_download_authorized"] is False
    assert materialization["paid_inference_authorized"] is False
    validate_materialization_authorization(
        materialization, freeze=freeze, call_plan=plan
    )

    cache = build_cache_authorization_payload(
        freeze=freeze,
        call_plan=plan,
        modal_profile="yanav",
        modal_image_id="im-test",
        maximum_total_cost_usd=5.0,
    )
    assert cache["model_download_authorized"] is True
    assert cache["paid_inference_authorized"] is False
    assert cache["model_ids"] == [model["model_id"] for model in freeze["models"]]
    validate_cache_authorization(
        cache,
        freeze=freeze,
        call_plan=plan,
        modal_image_id="im-test",
    )


def test_execution_authority_is_separate_and_denies_every_next_scope() -> None:
    freeze = read_json_object(FREEZE)
    payload = build_execution_authorization_payload(
        freeze=freeze,
        call_plan=read_json_object(PLAN),
        modal_profile="yanav",
        modal_image_id="im-test",
        cache_attestation_sha256_by_model={
            model["model_id"]: f"{index + 1:064x}"
            for index, model in enumerate(freeze["models"])
        },
        maximum_total_cost_usd=5.0,
    )
    assert payload["model_download_authorized"] is False
    assert payload["modal_execution_authorized"] is True
    assert payload["sealed_task_inference_authorized"] is False
    assert payload["outcome_access_authorized"] is False
    assert payload["fine_tuning_authorized"] is False
    assert payload["next_stage_authorized"] is False
    assert payload["maximum_planned_calls"] == 40
    validate_execution_authorization(
        payload,
        freeze=freeze,
        call_plan=read_json_object(PLAN),
        modal_image_id="im-test",
        cache_attestation_sha256_by_model={
            model["model_id"]: f"{index + 1:064x}"
            for index, model in enumerate(freeze["models"])
        },
    )


def test_authority_validation_rejects_image_and_cost_drift() -> None:
    freeze = read_json_object(FREEZE)
    plan = read_json_object(PLAN)
    materialization = build_materialization_authorization_payload(
        freeze=freeze,
        call_plan=plan,
        modal_profile="yanav",
        maximum_total_cost_usd=5.0,
    )
    changed = dict(materialization)
    changed["maximum_total_cost_usd"] = 6.0
    with pytest.raises(ValueError, match="cost cap"):
        validate_materialization_authorization(changed, freeze=freeze, call_plan=plan)

    cache = build_cache_authorization_payload(
        freeze=freeze,
        call_plan=plan,
        modal_profile="yanav",
        modal_image_id="im-test",
        maximum_total_cost_usd=5.0,
    )
    with pytest.raises(ValueError, match="different Modal image"):
        validate_cache_authorization(
            cache,
            freeze=freeze,
            call_plan=plan,
            modal_image_id="im-other",
        )


def test_runtime_attestation_must_match_image_model_cache_and_call() -> None:
    freeze = read_json_object(FREEZE)
    plan = read_json_object(PLAN)
    hashes = {
        model["model_id"]: f"{index + 1:064x}"
        for index, model in enumerate(freeze["models"])
    }
    authorization = build_execution_authorization_payload(
        freeze=freeze,
        call_plan=plan,
        modal_profile="yanav",
        modal_image_id="im-test",
        cache_attestation_sha256_by_model=hashes,
        maximum_total_cost_usd=5.0,
    )
    call = prepare_modal_calls(ROOT, freeze_path=FREEZE, call_plan_path=PLAN)[0]
    model = next(model for model in freeze["models"] if model["model_id"] == call.model_id)
    attestation = {
        "modal_sdk_version": "1.5.4",
        "modal_image_id": "im-test",
        "image_recipe_sha256": authorization["image_recipe_sha256"],
        "dependency_lock_sha256": authorization["dependency_lock_sha256"],
        "python_version": "3.11.13",
        "torch_version": "2.9.1+cu128",
        "transformers_version": "4.57.6",
        "cuda_runtime_version": "12.8",
        "gpu_name": "NVIDIA L40S",
        "checkpoint_commit": model["checkpoint_commit"],
        "weight_manifest_sha256": model["weight_file_manifest_sha256"],
        "tokenizer_manifest_sha256": model["tokenizer_manifest_sha256"],
        "cache_attestation_sha256": hashes[call.model_id],
        "call_id": call.call_id,
        "prompt_sha256": call.request["prompt_sha256"],
    }
    validate_runtime_attestation(
        call, attestation, freeze=freeze, authorization=authorization
    )
    changed = dict(attestation)
    changed["gpu_name"] = "NVIDIA A100"
    with pytest.raises(ValueError, match="gpu_name"):
        validate_runtime_attestation(
            call, changed, freeze=freeze, authorization=authorization
        )
