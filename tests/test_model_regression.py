from __future__ import annotations

import pytest

from intervenebench.model_regression import (
    ModelVersionEvaluation,
    ModelVersionRegressionThresholds,
    compare_model_versions,
)


def _evaluation(
    version: str,
    regrets: dict[str, float],
    *,
    exact: dict[str, bool] | None = None,
    valid: int = 300,
    planned: int = 300,
) -> ModelVersionEvaluation:
    exact = exact or {experiment_id: True for experiment_id in regrets}
    return ModelVersionEvaluation.from_mapping(
        {
            "schema_version": "intervenebench.model_version_evaluation.v1",
            "model_version": version,
            "panel_sha256": "a" * 64,
            "planned_output_count": planned,
            "schema_valid_output_count": valid,
            "experiments": {
                experiment_id: {
                    "normalized_regret": regret,
                    "exact_choice": exact[experiment_id],
                    "practically_reliable": regret <= 0.05,
                }
                for experiment_id, regret in regrets.items()
            },
        }
    )


def test_better_candidate_passes_paired_regression_gate() -> None:
    reference = _evaluation("reference", {"a": 0.03, "b": 0.04, "c": 0.02})
    candidate = _evaluation("candidate", {"a": 0.01, "b": 0.02, "c": 0.01})

    report = compare_model_versions(
        candidate,
        reference,
        thresholds=ModelVersionRegressionThresholds(),
        bootstrap_replicates=500,
        bootstrap_seed=17,
    )

    assert report["promotion_decision"] == "pass_regression_gate"
    assert report["paired_regret"]["mean_difference"] < 0
    assert report["paired_regret"]["experiment_count"] == 3
    assert report["scope_boundary"] == "regression pass is not an autonomous release authorization"


def test_candidate_with_material_regret_increase_is_held() -> None:
    reference = _evaluation("reference", {"a": 0.01, "b": 0.01, "c": 0.01})
    candidate = _evaluation("candidate", {"a": 0.05, "b": 0.05, "c": 0.05})

    report = compare_model_versions(
        candidate,
        reference,
        thresholds=ModelVersionRegressionThresholds(),
        bootstrap_replicates=200,
        bootstrap_seed=19,
    )

    assert report["promotion_decision"] == "hold_regression"
    assert "paired mean regret regressed" in report["failures"]
    assert "worst-case regret regressed" in report["failures"]


def test_schema_validity_regression_is_an_independent_failure() -> None:
    regrets = {"a": 0.01, "b": 0.02, "c": 0.01}
    reference = _evaluation("reference", regrets, valid=300, planned=300)
    candidate = _evaluation("candidate", regrets, valid=270, planned=300)

    report = compare_model_versions(
        candidate,
        reference,
        thresholds=ModelVersionRegressionThresholds(),
        bootstrap_replicates=100,
        bootstrap_seed=23,
    )

    assert report["promotion_decision"] == "hold_regression"
    assert "schema validity regressed" in report["failures"]


def test_comparison_rejects_panel_or_experiment_mismatch() -> None:
    reference = _evaluation("reference", {"a": 0.01, "b": 0.02})
    candidate = _evaluation("candidate", {"a": 0.01, "c": 0.02})
    with pytest.raises(ValueError, match="identical experiment IDs"):
        compare_model_versions(
            candidate,
            reference,
            thresholds=ModelVersionRegressionThresholds(),
            bootstrap_replicates=100,
            bootstrap_seed=1,
        )

    changed_panel = ModelVersionEvaluation.from_mapping(
        {
            "schema_version": "intervenebench.model_version_evaluation.v1",
            "model_version": "candidate",
            "panel_sha256": "b" * 64,
            "planned_output_count": 2,
            "schema_valid_output_count": 2,
            "experiments": {
                "a": {"normalized_regret": 0.01, "exact_choice": True, "practically_reliable": True},
                "b": {"normalized_regret": 0.02, "exact_choice": True, "practically_reliable": True},
            },
        }
    )
    with pytest.raises(ValueError, match="same frozen panel"):
        compare_model_versions(
            changed_panel,
            reference,
            thresholds=ModelVersionRegressionThresholds(),
            bootstrap_replicates=100,
            bootstrap_seed=1,
        )
