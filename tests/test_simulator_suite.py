from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from intervenebench.pilot import PILOT_EXPERIMENTS
from intervenebench.simulator_suite import (
    PROSPECTIVE_EXPERIMENTS,
    assert_execution_ready,
    build_development_call_plan,
    validate_development_scope,
    verify_development_scope,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/simulators/development_v1.json"
SCOPE_PATH = ROOT / "data/manifests/benchmark/simulator_development_scope.json"


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_dry_run_scope_is_bound_response_free_and_non_executable() -> None:
    plan = verify_development_scope(
        ROOT, scope_path=SCOPE_PATH, config_path=CONFIG_PATH
    )
    scope = _read(SCOPE_PATH)
    assert tuple(scope["development_experiment_ids"]) == PILOT_EXPERIMENTS
    assert tuple(scope["excluded_prospective_experiment_ids"]) == PROSPECTIVE_EXPERIMENTS
    assert not set(scope["development_experiment_ids"]) & set(
        scope["excluded_prospective_experiment_ids"]
    )
    assert plan.maximum_calls == 3580
    assert dict(plan.stage_calls) == {
        "provider_parser_preflight": 40,
        "aggregate_model_screen": 240,
        "adaptive_sampling_reserve": 1020,
        "source_preserving_prompt_sensitivity": 360,
        "matched_base_specialist_microsimulation": 1920,
    }
    assert plan.maximum_gpu_seconds == 89500.0
    assert plan.estimated_gpu_cost_usd == pytest.approx(48.509)
    assert plan.estimated_total_cost_usd == pytest.approx(50.0)
    assert scope["dry_run_call_plan"]["recorded_spend_usd"] == 0.0


def test_dry_run_scope_keeps_every_execution_authority_false() -> None:
    scope = _read(SCOPE_PATH)
    config = _read(CONFIG_PATH)
    for field in (
        "local_model_execution_authorized",
        "model_download_authorized",
        "paid_inference_authorized",
        "modal_compute_authorized",
        "sealed_task_inference_authorized",
        "sealed_outcome_reveal_authorized",
        "fine_tuning_authorized",
    ):
        assert scope[field] is False
    with pytest.raises(PermissionError, match="dry-run only"):
        assert_execution_ready(scope, config)


def test_development_config_rejects_sealed_task_or_unknown_model() -> None:
    config = _read(CONFIG_PATH)
    with_sealed_task = deepcopy(config)
    with_sealed_task["experiments"]["tcg8p"] = 3
    with pytest.raises(ValueError, match="only the five"):
        build_development_call_plan(with_sealed_task)

    unknown_model = deepcopy(config)
    unknown_model["stages"][0]["model_ids"].append("undeclared-model")
    with pytest.raises(ValueError, match="unknown models"):
        build_development_call_plan(unknown_model)


def test_development_config_rejects_cost_overrun() -> None:
    config = _read(CONFIG_PATH)
    overrun = deepcopy(config)
    overrun["compute_ceiling"]["gpu_seconds_per_call_ceiling"] = 30.0
    with pytest.raises(ValueError, match="exceeds"):
        build_development_call_plan(overrun)


def test_scope_mutation_cannot_authorize_execution() -> None:
    scope = _read(SCOPE_PATH)
    config = _read(CONFIG_PATH)
    authorized = deepcopy(scope)
    authorized["modal_compute_authorized"] = True
    with pytest.raises(ValueError, match="must keep modal_compute_authorized false"):
        validate_development_scope(authorized, config=config)
