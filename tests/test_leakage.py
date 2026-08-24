from __future__ import annotations

import json

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from intervenebench.protocol import (
    assert_blinded_payload,
    freeze_recommendation,
    payload_hash,
    verify_frozen_recommendation,
)
from intervenebench.socsci210 import read_audit_view, read_revealed_outcomes


def reveal_files(tmp_path, *, experiment_id="exp-a", task_num=0):
    split = {
        "experiment_to_split": {experiment_id: "validation"},
        "test_outcomes_sealed": True,
    }
    task = {
        "experiment_id": experiment_id,
        "split": "validation",
        "socsci210_task_num": task_num,
    }
    split_path = tmp_path / "split.json"
    task_path = tmp_path / "task.json"
    split_path.write_text(json.dumps(split))
    task_path.write_text(json.dumps(task))
    return split, task, split_path, task_path


@pytest.mark.parametrize(
    "forbidden",
    ["response", "reasoning", "human_arm_means", "tau_h", "regret"],
)
def test_blinded_payload_rejects_forbidden_fields(forbidden: str) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        assert_blinded_payload({"safe": {"nested": [{forbidden: 1}]}})


def test_response_options_are_allowed() -> None:
    assert_blinded_payload({"response_options": [1, 2, 3, 4, 5]})


def test_audit_view_accepts_tuple_columns_without_opening_outcomes(tmp_path) -> None:
    path = tmp_path / "structural.parquet"
    pq.write_table(
        pa.table(
            {
                "study_id": ["exp-a"],
                "condition_num": [0],
                "response": ["hidden"],
            }
        ),
        path,
    )
    table = read_audit_view([path], ("study_id", "condition_num"))
    assert table.column_names == ["study_id", "condition_num"]
    with pytest.raises(ValueError, match="not allowed"):
        read_audit_view([path], ("study_id", "response"))


def test_frozen_artifact_detects_mutation(tmp_path) -> None:
    path = tmp_path / "recommendation.json"
    payload = {
        "experiment_id": "exp-a",
        "synthetic_arm_means": {"control": 0.4, "treatment": 0.6},
        "selected_arm_id": "treatment",
    }
    freeze_recommendation(payload, path)
    assert verify_frozen_recommendation(path) == payload

    envelope = json.loads(path.read_text())
    envelope["payload"]["selected_arm_id"] = "control"
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_frozen_recommendation(path)


def test_freeze_is_create_only(tmp_path) -> None:
    path = tmp_path / "recommendation.json"
    freeze_recommendation({"selected_arm_id": "a"}, path)
    with pytest.raises(FileExistsError):
        freeze_recommendation({"selected_arm_id": "b"}, path)


def test_reveal_refuses_missing_recommendation(tmp_path) -> None:
    _, _, split_path, task_path = reveal_files(tmp_path)
    with pytest.raises(FileNotFoundError):
        read_revealed_outcomes(
            [tmp_path / "data.parquet"],
            experiment_id="exp-a",
            recommendation_path=tmp_path / "missing.json",
            split_manifest_path=split_path,
            decision_task_path=task_path,
        )


def test_reveal_refuses_experiment_mismatch(tmp_path) -> None:
    split, task, split_path, task_path = reveal_files(tmp_path)
    path = tmp_path / "recommendation.json"
    freeze_recommendation(
        {
            "schema_version": "recommendation.v1",
            "experiment_id": "exp-a",
            "split": "validation",
            "task_num": 0,
            "selected_arm_id": "arm-1",
            "synthetic_arm_means": {"arm-0": 0.4, "arm-1": 0.6},
            "synthetic_treatment_effects": {"arm-1": 0.2},
            "split_manifest_sha256": payload_hash(split),
            "decision_task_sha256": payload_hash(task),
            "blinded_bundle_sha256": "c" * 64,
            "simulator": {"id": "mock", "revision": "1"},
            "provenance": {"created_at_utc": "2026-01-01T00:00:00Z"},
        },
        path,
    )
    with pytest.raises(ValueError, match="does not match"):
        read_revealed_outcomes(
            [tmp_path / "data.parquet"],
            experiment_id="exp-b",
            recommendation_path=path,
            split_manifest_path=split_path,
            decision_task_path=task_path,
        )


def test_reveal_refuses_malformed_recommendation(tmp_path) -> None:
    _, _, split_path, task_path = reveal_files(tmp_path)
    path = tmp_path / "recommendation.json"
    freeze_recommendation({"experiment_id": "exp-a"}, path)
    with pytest.raises(ValueError, match="missing required"):
        read_revealed_outcomes(
            [tmp_path / "data.parquet"],
            experiment_id="exp-a",
            recommendation_path=path,
            split_manifest_path=split_path,
            decision_task_path=task_path,
        )


def test_reveal_refuses_test_split(tmp_path) -> None:
    split, task, split_path, task_path = reveal_files(tmp_path)
    path = tmp_path / "recommendation.json"
    freeze_recommendation(
        {
            "schema_version": "recommendation.v1",
            "experiment_id": "exp-a",
            "split": "test",
            "task_num": 0,
            "selected_arm_id": "arm-0",
            "synthetic_arm_means": {"arm-0": 0.5, "arm-1": 0.5},
            "synthetic_treatment_effects": {"arm-1": 0.0},
            "split_manifest_sha256": payload_hash(split),
            "decision_task_sha256": payload_hash(task),
            "blinded_bundle_sha256": "c" * 64,
            "simulator": {"id": "mock", "revision": "1"},
            "provenance": {"created_at_utc": "2026-01-01T00:00:00Z"},
        },
        path,
    )
    with pytest.raises(ValueError, match="validation outcomes"):
        read_revealed_outcomes(
            [tmp_path / "data.parquet"],
            experiment_id="exp-a",
            recommendation_path=path,
            split_manifest_path=split_path,
            decision_task_path=task_path,
        )


def test_reveal_refuses_split_hash_mismatch(tmp_path) -> None:
    split, task, split_path, task_path = reveal_files(tmp_path)
    path = tmp_path / "recommendation.json"
    freeze_recommendation(
        {
            "schema_version": "recommendation.v1",
            "experiment_id": "exp-a",
            "split": "validation",
            "task_num": 0,
            "selected_arm_id": "arm-0",
            "synthetic_arm_means": {"arm-0": 0.5, "arm-1": 0.5},
            "synthetic_treatment_effects": {"arm-1": 0.0},
            "split_manifest_sha256": "a" * 64,
            "decision_task_sha256": payload_hash(task),
            "blinded_bundle_sha256": "c" * 64,
            "simulator": {"id": "mock", "revision": "1"},
            "provenance": {"created_at_utc": "2026-01-01T00:00:00Z"},
        },
        path,
    )
    assert payload_hash(split) != "a" * 64
    with pytest.raises(ValueError, match="split manifest"):
        read_revealed_outcomes(
            [tmp_path / "data.parquet"],
            experiment_id="exp-a",
            recommendation_path=path,
            split_manifest_path=split_path,
            decision_task_path=task_path,
        )
