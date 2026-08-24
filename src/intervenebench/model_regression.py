"""Experiment-paired regression gates for behavioral simulator versions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import fsum, isfinite
from pathlib import Path
from typing import Any, Mapping

from .experiment_statistics import paired_experiment_cluster_bootstrap


@dataclass(frozen=True, slots=True)
class ExperimentVersionResult:
    normalized_regret: float
    exact_choice: bool
    practically_reliable: bool


@dataclass(frozen=True, slots=True)
class ModelVersionEvaluation:
    model_version: str
    panel_sha256: str
    planned_output_count: int
    schema_valid_output_count: int
    experiments: Mapping[str, ExperimentVersionResult]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ModelVersionEvaluation":
        required = {
            "schema_version",
            "model_version",
            "panel_sha256",
            "planned_output_count",
            "schema_valid_output_count",
            "experiments",
        }
        if set(payload) != required:
            raise ValueError("model-version evaluation fields are invalid")
        if payload["schema_version"] != "intervenebench.model_version_evaluation.v1":
            raise ValueError("unsupported model-version evaluation schema")
        model_version = payload["model_version"]
        if not isinstance(model_version, str) or not model_version.strip():
            raise ValueError("model_version must be a non-empty string")
        panel_sha256 = payload["panel_sha256"]
        if (
            not isinstance(panel_sha256, str)
            or len(panel_sha256) != 64
            or any(character not in "0123456789abcdef" for character in panel_sha256)
        ):
            raise ValueError("panel_sha256 must be a lowercase SHA-256 digest")
        planned = payload["planned_output_count"]
        valid = payload["schema_valid_output_count"]
        if (
            isinstance(planned, bool)
            or not isinstance(planned, int)
            or planned <= 0
            or isinstance(valid, bool)
            or not isinstance(valid, int)
            or valid < 0
            or valid > planned
        ):
            raise ValueError("model output counts are invalid")
        raw_experiments = payload["experiments"]
        if not isinstance(raw_experiments, Mapping) or len(raw_experiments) < 2:
            raise ValueError("at least two experiment aggregates are required")
        experiments: dict[str, ExperimentVersionResult] = {}
        for experiment_id, raw in raw_experiments.items():
            if not isinstance(experiment_id, str) or not experiment_id.strip():
                raise ValueError("experiment IDs must be non-empty strings")
            if not isinstance(raw, Mapping) or set(raw) != {
                "normalized_regret",
                "exact_choice",
                "practically_reliable",
            }:
                raise ValueError("experiment result fields are invalid")
            regret = raw["normalized_regret"]
            if (
                isinstance(regret, bool)
                or not isinstance(regret, (int, float))
                or not isfinite(float(regret))
                or not 0.0 <= float(regret) <= 1.0
            ):
                raise ValueError("normalized regret must lie in [0, 1]")
            if not isinstance(raw["exact_choice"], bool) or not isinstance(
                raw["practically_reliable"], bool
            ):
                raise ValueError("choice and reliability fields must be boolean")
            experiments[experiment_id] = ExperimentVersionResult(
                normalized_regret=float(regret),
                exact_choice=raw["exact_choice"],
                practically_reliable=raw["practically_reliable"],
            )
        return cls(
            model_version=model_version,
            panel_sha256=panel_sha256,
            planned_output_count=planned,
            schema_valid_output_count=valid,
            experiments=experiments,
        )


def load_model_version_evaluation(path: Path) -> ModelVersionEvaluation:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError("model-version evaluation must be a JSON object")
    return ModelVersionEvaluation.from_mapping(payload)


@dataclass(frozen=True, slots=True)
class ModelVersionRegressionThresholds:
    paired_mean_regret_noninferiority_margin: float = 0.01
    worst_regret_increase_margin: float = 0.01
    maximum_exact_choice_rate_drop: float = 0.10
    maximum_practical_reliability_rate_drop: float = 0.10
    maximum_schema_validity_rate_drop: float = 0.01


def _mean(values: list[float]) -> float:
    return fsum(values) / len(values)


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values)


def compare_model_versions(
    candidate: ModelVersionEvaluation,
    reference: ModelVersionEvaluation,
    *,
    thresholds: ModelVersionRegressionThresholds,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Compare versions on identical experiments using paired uncertainty.

    This gate answers whether a candidate regressed relative to a reference. A
    pass never expands the operational release scope established separately by
    :mod:`intervenebench.release_decision`.
    """

    if set(candidate.experiments) != set(reference.experiments):
        raise ValueError("version comparison requires identical experiment IDs")
    if candidate.panel_sha256 != reference.panel_sha256:
        raise ValueError("versions must use the same frozen panel")

    ids = tuple(sorted(candidate.experiments))
    candidate_regret = {
        experiment_id: candidate.experiments[experiment_id].normalized_regret
        for experiment_id in ids
    }
    reference_regret = {
        experiment_id: reference.experiments[experiment_id].normalized_regret
        for experiment_id in ids
    }
    paired = paired_experiment_cluster_bootstrap(
        candidate_regret,
        reference_regret,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        confidence_level=0.95,
        lower_is_better=True,
    )
    candidate_exact = _rate(
        [candidate.experiments[experiment_id].exact_choice for experiment_id in ids]
    )
    reference_exact = _rate(
        [reference.experiments[experiment_id].exact_choice for experiment_id in ids]
    )
    candidate_reliable = _rate(
        [
            candidate.experiments[experiment_id].practically_reliable
            for experiment_id in ids
        ]
    )
    reference_reliable = _rate(
        [
            reference.experiments[experiment_id].practically_reliable
            for experiment_id in ids
        ]
    )
    candidate_schema = (
        candidate.schema_valid_output_count / candidate.planned_output_count
    )
    reference_schema = (
        reference.schema_valid_output_count / reference.planned_output_count
    )
    candidate_worst = max(candidate_regret.values())
    reference_worst = max(reference_regret.values())

    failures: list[str] = []
    if (
        paired.difference_confidence_interval[1]
        > thresholds.paired_mean_regret_noninferiority_margin
    ):
        failures.append("paired mean regret regressed")
    if candidate_worst - reference_worst > thresholds.worst_regret_increase_margin:
        failures.append("worst-case regret regressed")
    if reference_exact - candidate_exact > thresholds.maximum_exact_choice_rate_drop:
        failures.append("exact-choice rate regressed")
    if (
        reference_reliable - candidate_reliable
        > thresholds.maximum_practical_reliability_rate_drop
    ):
        failures.append("practical reliability regressed")
    if (
        reference_schema - candidate_schema
        > thresholds.maximum_schema_validity_rate_drop
    ):
        failures.append("schema validity regressed")

    paired_payload = asdict(paired)
    paired_payload["difference_confidence_interval"] = list(
        paired_payload["difference_confidence_interval"]
    )
    return {
        "schema_version": "intervenebench.model_version_regression.v1",
        "candidate_model_version": candidate.model_version,
        "reference_model_version": reference.model_version,
        "panel_sha256": candidate.panel_sha256,
        "experiment_count": len(ids),
        "promotion_decision": "pass_regression_gate" if not failures else "hold_regression",
        "failures": failures or ["all frozen regression gates passed"],
        "paired_regret": paired_payload,
        "worst_regret": {
            "candidate": candidate_worst,
            "reference": reference_worst,
            "difference": candidate_worst - reference_worst,
        },
        "exact_choice_rate": {
            "candidate": candidate_exact,
            "reference": reference_exact,
            "difference": candidate_exact - reference_exact,
        },
        "practical_reliability_rate": {
            "candidate": candidate_reliable,
            "reference": reference_reliable,
            "difference": candidate_reliable - reference_reliable,
        },
        "schema_validity_rate": {
            "candidate": candidate_schema,
            "reference": reference_schema,
            "difference": candidate_schema - reference_schema,
        },
        "thresholds": asdict(thresholds),
        "scope_boundary": (
            "regression pass is not an autonomous release authorization"
        ),
    }
