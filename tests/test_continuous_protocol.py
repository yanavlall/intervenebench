from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from intervenebench.phase2_continuous import (
    freeze_continuous_recommendation_from_outputs,
    replay_continuous_score,
    score_frozen_continuous_validation_recommendation,
)
from intervenebench.protocol import (
    assert_blinded_payload,
)


ROOT = Path(__file__).resolve().parents[1]


def test_tcg8p_candidate_is_source_verified_but_not_reveal_authorized() -> None:
    task = json.loads(
        (
            ROOT
            / "data/manifests/contracts/tcg8p_continuous_task_candidate.json"
        ).read_text()
    )
    bundle = json.loads(
        (
            ROOT
            / "data/manifests/contracts/tcg8p_continuous_blinded_bundle.json"
        ).read_text()
    )

    assert task["experiment_id"] == "tcg8p"
    assert task["canonical_split_status"] == "unassigned"
    assert task["reveal_authorized"] is False
    assert task["outcome_access"] == "sealed"
    assert task["estimator"]["location"] == "mean"
    assert task["estimator"]["robustness_locations"] == ["median"]
    assert task["estimator"]["regret_unit"] == "usd_per_month"
    assert task["estimator"]["normalized_for_pooled_regret"] is False
    assert task["valid_response"]["upper_bound"] is None
    assert task["valid_response"]["missing_codes"] == [77777, 99998, 99999]
    assert bundle["experiment_id"] == "tcg8p"
    assert_blinded_payload(bundle)


def _write_toy_continuous_artifacts(tmp_path: Path) -> tuple[Path, ...]:
    task = {
        "schema_version": "continuous_decision_task.v1",
        "experiment_id": "exp-continuous",
        "split": "validation",
        "socsci210_task_num": 0,
        "arms": [
            {"arm_id": "control", "condition_num": 0},
            {"arm_id": "notice", "condition_num": 1},
        ],
        "control_arm_id": "control",
        "outcome_family": "continuous",
        "direction": "lower_is_better",
        "outcome_unit": "usd_per_month",
        "valid_response": {
            "lower_bound": 0,
            "upper_bound": None,
            "integer_only": True,
            "missing_codes": [77777, 99998, 99999],
        },
        "estimator": {
            "location": "mean",
            "robustness_locations": ["median"],
            "practical_regret_tolerance": 0.0,
            "practical_regret_sensitivity": [0.0, 5.0, 10.0, 20.0],
            "regret_unit": "usd_per_month",
            "normalized_for_pooled_regret": False,
        },
    }
    split = {
        "experiment_to_split": {"exp-continuous": "validation"},
        "test_outcomes_sealed": True,
    }
    task_path = tmp_path / "task.json"
    split_path = tmp_path / "split.json"
    task_path.write_text(json.dumps(task))
    split_path.write_text(json.dumps(split))
    bundle = {
        "schema_version": "continuous_blinded_bundle.v1",
        "task_id": "exp-continuous:task-0",
        "experiment_id": "exp-continuous",
        "access_regime": "DESIGN_ONLY",
        "population": {
            "description": "Synthetic fixture population",
            "roster_id": "fixture-roster-v1",
        },
        "arms": [
            {"arm_id": "control", "message": "No notice"},
            {"arm_id": "notice", "message": "Advance notice"},
        ],
        "common_context": "Two planned outages.",
        "outcome_question": "Monthly willingness to pay to avoid the outages?",
        "response_contract": {
            "type": "integer",
            "unit": "usd_per_month",
            "minimum": 0,
            "maximum": None,
        },
        "source_material_sha256": "a" * 64,
        "outcome_access": "sealed",
        "reveal_authorized": False,
    }
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))
    raw_path = tmp_path / "raw.json"
    recommendation_path = tmp_path / "recommendation.json"
    freeze_continuous_recommendation_from_outputs(
        bundle_path=bundle_path,
        split_path=split_path,
        decision_task_path=task_path,
        outputs=(
            {"arm_id": "control", "draw_index": 0, "raw_response": '{"predicted_value": 20}'},
            {"arm_id": "notice", "draw_index": 0, "raw_response": '{"predicted_value": 10}'},
        ),
        raw_output_path=raw_path,
        recommendation_path=recommendation_path,
        simulator_id="mock",
        simulator_revision="1",
        draws=1,
        seed=17,
    )

    table = pa.table(
        {
            "study_id": ["exp-continuous"] * 7 + ["exp-test"],
            "sample_id": list(range(8)),
            "participant": list(range(10, 18)),
            "condition_num": [0, 0, 0, 1, 1, 1, 1, 0],
            "task_num": [0] * 8,
            "response": [10, 30, 99998, 0, 5, 15, 99999, 1],
        }
    )
    parquet_path = tmp_path / "data.parquet"
    pq.write_table(table, parquet_path)
    return task_path, split_path, raw_path, recommendation_path, parquet_path


def test_continuous_freeze_reveal_score_and_replay(tmp_path: Path) -> None:
    task, split, raw, recommendation, parquet = _write_toy_continuous_artifacts(
        tmp_path
    )
    score_path = tmp_path / "score.json"
    score_frozen_continuous_validation_recommendation(
        parquet_paths=(parquet,),
        decision_task_path=task,
        split_manifest_path=split,
        recommendation_path=recommendation,
        raw_output_path=raw,
        score_path=score_path,
        bootstrap_replicates=200,
        bootstrap_seed=19,
    )
    score = replay_continuous_score(
        score_path=score_path,
        recommendation_path=recommendation,
        raw_output_path=raw,
    )

    assert score["human_arm_locations"] == {"control": 20.0, "notice": 20 / 3}
    assert score["valid_observations_per_arm"] == {"control": 2, "notice": 3}
    assert score["missing_observations_per_arm"] == {"control": 1, "notice": 1}
    assert score["selected_arm_id"] == "notice"
    assert score["human_best_arm_id"] == "notice"
    assert score["raw_decision_regret"] == 0.0
    assert score["regret_unit"] == "usd_per_month"
    assert score["normalized_for_pooled_regret"] is False
    assert "normalized_decision_regret" not in score
    assert score["practical_reliability_by_tolerance"] == {
        "0.0": True,
        "5.0": True,
        "10.0": True,
        "20.0": True,
    }
    assert score["absolute_treatment_effect_error"] == pytest.approx(
        {"notice": 10 / 3}
    )
    assert score["treatment_effect_sign_correct"] == {"notice": True}
    assert score["no_effect_control_baseline"]["selected_arm_id"] == "control"
    assert score["robustness"]["median"]["human_arm_locations"] == {
        "control": 20.0,
        "notice": 5.0,
    }


def test_continuous_reveal_refuses_unassigned_task(tmp_path: Path) -> None:
    task, split, raw, recommendation, parquet = _write_toy_continuous_artifacts(
        tmp_path
    )
    task_payload = json.loads(task.read_text())
    task_payload["split"] = "unassigned"
    task.write_text(json.dumps(task_payload))

    with pytest.raises(ValueError, match="decision task"):
        score_frozen_continuous_validation_recommendation(
            parquet_paths=(parquet,),
            decision_task_path=task,
            split_manifest_path=split,
            recommendation_path=recommendation,
            raw_output_path=raw,
            score_path=tmp_path / "score.json",
            bootstrap_replicates=20,
            bootstrap_seed=1,
        )
