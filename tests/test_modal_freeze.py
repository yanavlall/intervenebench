from __future__ import annotations

import json
import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

from intervenebench.modal_freeze import (
    assert_modal_execution_ready,
    verify_modal_preflight_freeze,
)
from intervenebench.protocol import payload_hash


ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / "configs/simulators/modal_discovery_preflight_v2.json"
CALL_PLAN_PATH = (
    ROOT / "data/manifests/simulators/modal_preflight_call_plan_v1.json"
)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_freeze_is_exact_nonexecuting_and_budgeted() -> None:
    summary = verify_modal_preflight_freeze(
        ROOT, freeze_path=FREEZE_PATH, call_plan_path=CALL_PLAN_PATH
    )
    assert summary.call_count == 40
    assert summary.calls_per_model == 10
    assert summary.model_count == 4
    assert summary.maximum_gpu_container_seconds == 4000
    assert summary.maximum_gpu_cost_usd == pytest.approx(2.168)
    assert summary.hard_total_cost_cap_usd == 5.0
    assert summary.minimum_parse_successes == 40

    freeze = _read(FREEZE_PATH)
    assert freeze["runtime"]["provider"] == "modal"
    assert freeze["runtime"]["gpu_type"] == "L40S"
    assert freeze["runtime"]["gpu_fallback_allowed"] is False
    assert freeze["runtime"]["maximum_containers_per_parameterized_model"] == 1
    assert freeze["runtime"]["maximum_total_model_containers"] == 4
    assert freeze["runtime"]["python_version"] == "3.11"
    assert freeze["generation"]["malformed_output_retry_count"] == 0
    assert freeze["preflight"]["required_parse_successes_per_model"] == 10
    assert freeze["limits"]["abort_before_dispatch_when_next_call_exceeds_budget"]
    assert all(value is False for value in freeze["authority"].values())
    with pytest.raises(PermissionError, match="non-executing"):
        assert_modal_execution_ready(freeze)


def test_models_are_pinned_public_ungated_and_runtime_is_exact() -> None:
    freeze = _read(FREEZE_PATH)
    assert {model["checkpoint_commit"] for model in freeze["models"]}
    for model in freeze["models"]:
        assert len(model["checkpoint_commit"]) == 40
        int(model["checkpoint_commit"], 16)
        assert model["repository_private"] is False
        assert model["repository_gated"] is False
        assert model["license_id"] == "apache-2.0"
        assert model["trust_remote_code"] is False
        assert model["dtype"] == "bfloat16"
        for field in (
            "weight_file_manifest_sha256",
            "tokenizer_manifest_sha256",
            "chat_template_sha256",
            "config_sha256",
        ):
            assert len(model[field]) == 64
            int(model[field], 16)

    packages = freeze["runtime"]["packages"]
    assert packages
    assert all("==" in package for package in packages)
    assert all(not any(token in package for token in (">", "<", "~=", "*")) for package in packages)
    assert freeze["dependency_lock"]["target"] == (
        "CPython 3.11 x86_64 manylinux_2_28"
    )
    assert freeze["model_file_manifest"]["path"].endswith(
        "model_file_manifests_v1.json"
    )


def test_call_plan_covers_two_arms_per_task_per_model_and_is_hash_bound() -> None:
    verify_modal_preflight_freeze(
        ROOT, freeze_path=FREEZE_PATH, call_plan_path=CALL_PLAN_PATH
    )
    call_plan = _read(CALL_PLAN_PATH)
    calls = call_plan["calls"]
    assert len(calls) == len({call["call_id"] for call in calls}) == 40
    expected_tasks = {"5vm8g", "xc4yq", "de5hx", "turagaS11", "wallaceS12"}
    for model_id in {call["model_id"] for call in calls}:
        model_calls = [call for call in calls if call["model_id"] == model_id]
        assert len(model_calls) == 10
        assert {call["experiment_id"] for call in model_calls} == expected_tasks
        assert all(
            len(
                {
                    call["arm_id"]
                    for call in model_calls
                    if call["experiment_id"] == experiment_id
                }
            )
            == 2
            for experiment_id in expected_tasks
        )
        assert len({call["seed"] for call in model_calls}) == 10


def test_mutation_rejects_revision_authority_task_and_cost_drift() -> None:
    freeze = _read(FREEZE_PATH)
    plan = _read(CALL_PLAN_PATH)

    mutable_revision = deepcopy(freeze)
    mutable_revision["models"][0]["checkpoint_commit"] = "main"
    with pytest.raises(ValueError, match="immutable 40-character"):
        verify_modal_preflight_freeze(
            ROOT,
            freeze_path=FREEZE_PATH,
            call_plan_path=CALL_PLAN_PATH,
            freeze_override=mutable_revision,
        )

    authorized = deepcopy(freeze)
    authorized["authority"]["modal_execution_authorized"] = True
    with pytest.raises(ValueError, match="authority"):
        verify_modal_preflight_freeze(
            ROOT,
            freeze_path=FREEZE_PATH,
            call_plan_path=CALL_PLAN_PATH,
            freeze_override=authorized,
        )

    sealed_task = deepcopy(plan)
    sealed_task["calls"][0]["experiment_id"] = "tcg8p"
    with pytest.raises(ValueError, match="exact task allowlist"):
        verify_modal_preflight_freeze(
            ROOT,
            freeze_path=FREEZE_PATH,
            call_plan_path=CALL_PLAN_PATH,
            call_plan_override=sealed_task,
        )

    overrun = deepcopy(freeze)
    overrun["limits"]["maximum_gpu_container_seconds"] = 10000
    with pytest.raises(ValueError, match="cost cap|hard total"):
        verify_modal_preflight_freeze(
            ROOT,
            freeze_path=FREEZE_PATH,
            call_plan_path=CALL_PLAN_PATH,
            freeze_override=overrun,
        )


def test_call_plan_rejects_prompt_mutation_and_semantic_retry() -> None:
    plan = _read(CALL_PLAN_PATH)
    mutated = deepcopy(plan)
    mutated["calls"][0]["prompt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="prompt hash"):
        verify_modal_preflight_freeze(
            ROOT,
            freeze_path=FREEZE_PATH,
            call_plan_path=CALL_PLAN_PATH,
            call_plan_override=mutated,
        )

    freeze = _read(FREEZE_PATH)
    retry = deepcopy(freeze)
    retry["generation"]["malformed_output_retry_count"] = 1
    with pytest.raises(ValueError, match="malformed outputs"):
        verify_modal_preflight_freeze(
            ROOT,
            freeze_path=FREEZE_PATH,
            call_plan_path=CALL_PLAN_PATH,
            freeze_override=retry,
        )


def test_call_plan_builder_replays_exactly() -> None:
    script = ROOT / "scripts/build_modal_preflight_call_plan.py"
    spec = importlib.util.spec_from_file_location("modal_call_plan_builder", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert payload_hash(module.build(ROOT)) == payload_hash(_read(CALL_PLAN_PATH))
