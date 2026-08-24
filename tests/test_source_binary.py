from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from intervenebench.protocol import freeze_recommendation, payload_hash
from intervenebench.source_binary import read_revealed_source_binary_csv


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    csv_bytes = (
        b"CaseId,WEIGHT,P_COND70,T70_14,forbidden_extra\n"
        b"a,1.0,1,1,do-not-project\n"
        b"b,2.0,1,77,do-not-project\n"
        b"c,1.5,2,2,do-not-project\n"
    )
    archive_path = tmp_path / "source.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("source.csv", csv_bytes)
    import hashlib

    task = {
        "schema_version": "source_binary_sequence_decision_task_candidate.v1",
        "experiment_id": "fixture-source",
        "split": "validation",
        "source_question_id": "T70_14",
        "outcome_family": "binary",
        "arms": [{"arm_id": "control"}, {"arm_id": "treatment"}],
        "source_variable_mapping": {
            "participant_id_variable": "CaseId",
            "assignment_variable": "P_COND70",
            "outcome_variable": "T70_14",
            "weight_variable": "WEIGHT",
            "assignment_to_arm": {"1": "control", "2": "treatment"},
            "outcome_to_utility": {"1": 0.0, "2": 1.0},
            "valid_outcome_values": [1, 2],
            "missing_outcome_codes": [77, 98, 99],
        },
        "source_data_locator": {
            "container_path": archive_path.name,
            "container_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "csv_member": "source.csv",
            "csv_member_sha256": hashlib.sha256(csv_bytes).hexdigest(),
            "authorized_projection": ["CaseId", "WEIGHT", "P_COND70", "T70_14"],
        },
    }
    split = {
        "experiment_to_split": {"fixture-source": "validation"},
        "test_outcomes_sealed": True,
    }
    task_path = tmp_path / "task.json"
    split_path = tmp_path / "split.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    split_path.write_text(json.dumps(split), encoding="utf-8")
    recommendation = {
        "schema_version": "source_binary_recommendation.v1",
        "experiment_id": "fixture-source",
        "split": "validation",
        "source_question_id": "T70_14",
        "selected_arm_id": "treatment",
        "synthetic_arm_means": {"control": 0.3, "treatment": 0.6},
        "synthetic_treatment_effects": {"treatment": 0.3},
        "split_manifest_sha256": payload_hash(split),
        "decision_task_sha256": payload_hash(task),
        "blinded_bundle_sha256": "a" * 64,
        "simulator": {"id": "fixture", "revision": "1"},
        "provenance": {"created_at_utc": "2026-08-13T00:00:00Z"},
    }
    recommendation_path = tmp_path / "recommendation.json"
    freeze_recommendation(recommendation, recommendation_path)
    return archive_path, task_path, split_path, recommendation_path


def test_source_binary_fixture_reveal_and_fail_closed(tmp_path: Path) -> None:
    _, task, split, recommendation = _write_fixture(tmp_path)
    observations = read_revealed_source_binary_csv(
        root=tmp_path,
        experiment_id="fixture-source",
        recommendation_path=recommendation,
        split_manifest_path=split,
        decision_task_path=task,
    )
    assert [(row.participant_id, row.arm_id, row.value, row.weight) for row in observations] == [
        ("a", "control", 0.0, 1.0),
        ("c", "treatment", 1.0, 1.5),
    ]

    task_payload = json.loads(task.read_text(encoding="utf-8"))
    task_payload["split"] = "unassigned"
    task.write_text(json.dumps(task_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="bound|authorize"):
        read_revealed_source_binary_csv(
            root=tmp_path,
            experiment_id="fixture-source",
            recommendation_path=recommendation,
            split_manifest_path=split,
            decision_task_path=task,
        )
