"""Response-free planning and validation for a multi-model simulator suite."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

from .pilot import PILOT_EXPERIMENTS
from .protocol import assert_blinded_payload, payload_hash


DEVELOPMENT_EXPERIMENTS = PILOT_EXPERIMENTS
PROSPECTIVE_EXPERIMENTS = (
    "tcg8p",
    "pb2rr",
    "z358z",
    "ShannonS2",
    "Blair1131",
    "KlarS44",
)
AUTHORITY_FIELDS = (
    "local_model_execution_authorized",
    "model_download_authorized",
    "paid_inference_authorized",
    "modal_compute_authorized",
    "sealed_task_inference_authorized",
    "sealed_outcome_reveal_authorized",
    "fine_tuning_authorized",
)


@dataclass(frozen=True, slots=True)
class DevelopmentCallPlan:
    stage_calls: tuple[tuple[str, int], ...]
    maximum_calls: int
    maximum_gpu_seconds: float
    estimated_gpu_cost_usd: float
    ancillary_cost_reserve_usd: float
    estimated_total_cost_usd: float
    hard_cost_cap_usd: float


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _model_ids(config: Mapping[str, Any]) -> set[str]:
    catalog = config.get("model_catalog")
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("model_catalog must be a non-empty list")
    identifiers: list[str] = []
    for model in catalog:
        if not isinstance(model, Mapping):
            raise ValueError("model catalog entries must be objects")
        identifier = model.get("model_id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("model_id must be a non-empty string")
        if model.get("execution_status") != "unresolved_dry_run_only":
            raise ValueError("dry-run models must remain unresolved and non-executable")
        if model.get("checkpoint_revision") is not None:
            raise ValueError("dry-run checkpoint revisions must remain unresolved")
        identifiers.append(identifier)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("model IDs must be unique")
    return set(identifiers)


def _stage_model_ids(stage: Mapping[str, Any], known_models: set[str]) -> tuple[str, ...]:
    raw = stage.get("model_ids")
    if not isinstance(raw, list) or not raw:
        raise ValueError("stage model_ids must be a non-empty list")
    model_ids = tuple(str(value) for value in raw)
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("stage model IDs must be unique")
    unknown = sorted(set(model_ids) - known_models)
    if unknown:
        raise ValueError(f"stage references unknown models: {unknown}")
    return model_ids


def build_development_call_plan(config: Mapping[str, Any]) -> DevelopmentCallPlan:
    """Validate a no-call development matrix and calculate its hard ceiling."""

    assert_blinded_payload(config)
    if config.get("schema_version") != "simulator_development_config.v1":
        raise ValueError("unsupported simulator-development config schema")
    if config.get("status") != "dry_run_only_no_execution_authority":
        raise ValueError("simulator-development config must remain dry-run only")
    experiments = config.get("experiments")
    if not isinstance(experiments, Mapping):
        raise ValueError("experiments must be an object")
    if tuple(experiments) != DEVELOPMENT_EXPERIMENTS:
        raise ValueError("development config must contain only the five frozen tasks")
    cell_counts = {
        experiment_id: _positive_integer(value, field=f"cells[{experiment_id}]")
        for experiment_id, value in experiments.items()
    }
    total_cells = sum(cell_counts.values())
    known_models = _model_ids(config)
    stages = config.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("stages must be a non-empty list")
    stage_calls: list[tuple[str, int]] = []
    seen_stage_ids: set[str] = set()
    for stage in stages:
        if not isinstance(stage, Mapping):
            raise ValueError("development stages must be objects")
        stage_id = stage.get("stage_id")
        stage_type = stage.get("type")
        if not isinstance(stage_id, str) or not stage_id.strip():
            raise ValueError("stage_id must be a non-empty string")
        if stage_id in seen_stage_ids:
            raise ValueError("stage IDs must be unique")
        seen_stage_ids.add(stage_id)
        if stage_type == "parser_preflight":
            models = _stage_model_ids(stage, known_models)
            calls = len(models) * _positive_integer(
                stage.get("calls_per_model"), field="calls_per_model"
            )
        elif stage_type == "aggregate_screen":
            models = _stage_model_ids(stage, known_models)
            calls = total_cells * len(models) * _positive_integer(
                stage.get("draws_per_cell"), field="draws_per_cell"
            )
        elif stage_type == "convergence_reserve":
            calls = (
                total_cells
                * _positive_integer(
                    stage.get("maximum_retained_models"),
                    field="maximum_retained_models",
                )
                * _positive_integer(
                    stage.get("maximum_added_draws_per_cell"),
                    field="maximum_added_draws_per_cell",
                )
            )
        elif stage_type == "prompt_sensitivity":
            calls = (
                total_cells
                * _positive_integer(
                    stage.get("maximum_retained_models"),
                    field="maximum_retained_models",
                )
                * _positive_integer(
                    stage.get("perturbation_count"), field="perturbation_count"
                )
                * _positive_integer(
                    stage.get("draws_per_cell"), field="draws_per_cell"
                )
            )
        elif stage_type == "matched_microsimulation":
            models = _stage_model_ids(stage, known_models)
            calls = (
                total_cells
                * len(models)
                * _positive_integer(stage.get("mode_count"), field="mode_count")
                * _positive_integer(
                    stage.get("samples_per_cell"), field="samples_per_cell"
                )
            )
        else:
            raise ValueError(f"unsupported development stage type: {stage_type}")
        stage_calls.append((stage_id, calls))

    maximum_calls = sum(calls for _, calls in stage_calls)
    compute = config.get("compute_ceiling")
    if not isinstance(compute, Mapping):
        raise ValueError("compute_ceiling must be an object")
    seconds_per_call = compute.get("gpu_seconds_per_call_ceiling")
    rate = compute.get("gpu_usd_per_second")
    reserve = compute.get("ancillary_cost_reserve_usd")
    cap = compute.get("hard_cost_cap_usd")
    for field, value in (
        ("gpu_seconds_per_call_ceiling", seconds_per_call),
        ("gpu_usd_per_second", rate),
        ("ancillary_cost_reserve_usd", reserve),
        ("hard_cost_cap_usd", cap),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{field} must be finite and non-negative")
    if float(seconds_per_call) <= 0.0 or float(rate) <= 0.0 or float(cap) <= 0.0:
        raise ValueError("GPU time, rate, and hard cap must be positive")
    maximum_gpu_seconds = maximum_calls * float(seconds_per_call)
    gpu_cost = maximum_gpu_seconds * float(rate)
    total_cost = gpu_cost + float(reserve)
    if total_cost > float(cap) + 1e-9:
        raise ValueError("planned development cost exceeds the hard cost cap")
    return DevelopmentCallPlan(
        stage_calls=tuple(stage_calls),
        maximum_calls=maximum_calls,
        maximum_gpu_seconds=maximum_gpu_seconds,
        estimated_gpu_cost_usd=gpu_cost,
        ancillary_cost_reserve_usd=float(reserve),
        estimated_total_cost_usd=total_cost,
        hard_cost_cap_usd=float(cap),
    )


def validate_development_scope(
    scope: Mapping[str, Any], *, config: Mapping[str, Any]
) -> DevelopmentCallPlan:
    """Verify that a scope authorizes planning only and excludes sealed tasks."""

    assert_blinded_payload(scope)
    if scope.get("schema_version") != "simulator_development_scope.v1":
        raise ValueError("unsupported simulator-development scope schema")
    if scope.get("status") != "response_free_dry_run_not_authorized_to_execute":
        raise ValueError("simulator-development scope must remain a dry run")
    if tuple(scope.get("development_experiment_ids", ())) != DEVELOPMENT_EXPERIMENTS:
        raise ValueError("scope must bind exactly the five development experiments")
    excluded = tuple(scope.get("excluded_prospective_experiment_ids", ()))
    if excluded != PROSPECTIVE_EXPERIMENTS:
        raise ValueError("scope must explicitly exclude all six prospective tasks")
    if set(DEVELOPMENT_EXPERIMENTS) & set(excluded):
        raise ValueError("development and prospective task scopes must be disjoint")
    for field in AUTHORITY_FIELDS:
        if scope.get(field) is not False:
            raise ValueError(f"dry-run scope must keep {field} false")
    if scope.get("config_payload_sha256") != payload_hash(config):
        raise ValueError("development scope is not bound to its config payload")
    plan = build_development_call_plan(config)
    recorded = scope.get("dry_run_call_plan")
    if not isinstance(recorded, Mapping):
        raise ValueError("scope must record its dry-run call plan")
    if recorded.get("maximum_calls") != plan.maximum_calls:
        raise ValueError("recorded maximum call count does not match the config")
    if abs(
        float(recorded.get("estimated_total_cost_usd", -1.0))
        - plan.estimated_total_cost_usd
    ) > 1e-9:
        raise ValueError("recorded dry-run cost does not match the config")
    return plan


def verify_development_scope(
    root: Path, *, scope_path: Path, config_path: Path
) -> DevelopmentCallPlan:
    scope = _read_object(scope_path)
    config = _read_object(config_path)
    expected_relative = str(config_path.relative_to(root))
    if scope.get("config_path") != expected_relative:
        raise ValueError("development scope references a different config path")
    return validate_development_scope(scope, config=config)


def assert_execution_ready(scope: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Fail closed until a separate authority and pinned revisions exist."""

    validate_development_scope(scope, config=config)
    blocked = [field for field in AUTHORITY_FIELDS if scope.get(field) is False]
    unresolved = [
        model.get("model_id")
        for model in config.get("model_catalog", ())
        if isinstance(model, Mapping) and model.get("checkpoint_revision") is None
    ]
    raise PermissionError(
        "simulator suite is dry-run only; separate authorization and immutable "
        f"checkpoint revisions are required (blocked={blocked}, unresolved={unresolved})"
    )
