from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from intervenebench.phase1 import replay_score, score_frozen_validation_recommendation
from intervenebench.protocol import freeze_envelope, freeze_recommendation, payload_hash


def test_toy_freeze_reveal_score_and_replay(tmp_path) -> None:
    task = {
        "experiment_id": "exp-a",
        "split": "validation",
        "socsci210_task_num": 0,
        "arms": [
            {"arm_id": "arm_0", "condition_num": 0},
            {"arm_id": "arm_1", "condition_num": 1},
        ],
        "control_arm_id": "arm_0",
        "response_options": [
            {"raw_value": 1, "normalized_utility": 1.0},
            {"raw_value": 2, "normalized_utility": 0.0},
        ],
        "scale_lower": 1,
        "scale_upper": 2,
        "direction": "lower_is_better",
        "practical_regret_tolerance": 0.05,
    }
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task))
    split = {
        "experiment_to_split": {"exp-a": "validation"},
        "test_outcomes_sealed": True,
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    raw_payload = {
        "schema_version": "simulator_outputs.v1",
        "experiment_id": "exp-a",
        "outputs": [],
    }
    raw_path = tmp_path / "raw.json"
    raw_digest = freeze_envelope(raw_payload, raw_path, require_blinded=True)
    recommendation = {
        "schema_version": "recommendation.v1",
        "experiment_id": "exp-a",
        "split": "validation",
        "task_num": 0,
        "selected_arm_id": "arm_1",
        "synthetic_arm_means": {"arm_0": 0.4, "arm_1": 0.6},
        "synthetic_treatment_effects": {"arm_1": 0.2},
        "baselines": {
            "no_effect_control_policy": {
                "synthetic_arm_means": {"arm_0": 0.5, "arm_1": 0.5},
                "synthetic_treatment_effects": {"arm_1": 0.0},
                "selected_arm_id": "arm_0",
            }
        },
        "split_manifest_sha256": payload_hash(split),
        "decision_task_sha256": payload_hash(task),
        "blinded_bundle_sha256": "c" * 64,
        "simulator_outputs_sha256": raw_digest,
        "simulator": {"id": "mock", "revision": "1"},
        "provenance": {"created_at_utc": "2026-01-01T00:00:00Z"},
    }
    recommendation_path = tmp_path / "recommendation.json"
    freeze_recommendation(recommendation, recommendation_path)
    table = pa.table(
        {
            "study_id": ["exp-a"] * 4 + ["exp-test"],
            "sample_id": [0, 1, 2, 3, 4],
            "participant": [10, 11, 12, 13, 14],
            "condition_num": [0, 0, 1, 1, 0],
            "task_num": [0, 0, 0, 0, 0],
            "response": [2, 2, 1, 1, 1],
        }
    )
    parquet = tmp_path / "data.parquet"
    pq.write_table(table, parquet)
    score_path = tmp_path / "score.json"
    score_frozen_validation_recommendation(
        parquet_paths=(parquet,),
        decision_task_path=task_path,
        split_manifest_path=split_path,
        recommendation_path=recommendation_path,
        raw_output_path=raw_path,
        score_path=score_path,
        bootstrap_replicates=100,
        bootstrap_seed=7,
    )
    score = replay_score(
        score_path=score_path,
        recommendation_path=recommendation_path,
        raw_output_path=raw_path,
    )
    assert score["correct_choice"] is True
    assert score["normalized_decision_regret"] == 0.0
    assert score["human_arm_means"] == {"arm_0": 0.0, "arm_1": 1.0}
    assert score["no_effect_control_baseline"]["normalized_decision_regret"] == 1.0


def test_scoring_detects_raw_output_mismatch(tmp_path) -> None:
    raw_path = tmp_path / "raw.json"
    freeze_envelope({"safe": True}, raw_path, require_blinded=True)
    assert payload_hash({"safe": True}) != "0" * 64
