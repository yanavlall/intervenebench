from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from intervenebench.protocol import freeze_envelope, freeze_recommendation, payload_hash
from intervenebench.source_ordinal import read_revealed_source_ordinal_sav


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    import hashlib

    fixture_csv = tmp_path / "fixture.csv"
    fixture_csv.write_text(
        "caseid,weight1,XTESS040,Q1,forbidden_extra\n"
        "1,1.0,1,1,do-not-project\n"
        "2,2.0,1,-1,do-not-project\n"
        "3,1.5,2,7,do-not-project\n",
        encoding="utf-8",
    )
    fixture_sav = tmp_path / "fixture.sav"
    import subprocess

    expression = (
        "args<-commandArgs(trailingOnly=TRUE);"
        "d<-read.csv(args[1],check.names=FALSE);"
        "haven::write_sav(d,args[2])"
    )
    subprocess.run(
        ["Rscript", "-e", expression, str(fixture_csv), str(fixture_sav)],
        check=True,
        capture_output=True,
        text=True,
    )
    sav_bytes = fixture_sav.read_bytes()
    archive_path = tmp_path / "source.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("source.sav", sav_bytes)

    task = {
        "schema_version": "source_ordinal_decision_task_candidate.v1",
        "experiment_id": "fixture-source",
        "canonical_split_status": "validation",
        "source_question_id": "Q1",
        "outcome_family": "ordinal",
        "arms": [
            {"arm_id": "picture_1", "source_assignment": "XTESS040=1"},
            {"arm_id": "picture_2", "source_assignment": "XTESS040=2"},
        ],
        "response_options": [
            {"raw_value": 1, "normalized_utility": 0.0},
            {"raw_value": 7, "normalized_utility": 1.0},
        ],
        "source_variable_mapping": {
            "participant_id_variable": "caseid",
            "weight_variable": "weight1",
            "assignment_variable": "XTESS040",
            "outcome_variable": "Q1",
            "assignment_to_arm": {"1": "picture_1", "2": "picture_2"},
            "missing_outcome_codes": [-1],
        },
        "source_data_locator": {
            "container_path": archive_path.name,
            "container_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "sav_member": "source.sav",
            "sav_member_sha256": hashlib.sha256(sav_bytes).hexdigest(),
            "authorized_projection": ["caseid", "weight1", "XTESS040", "Q1"],
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
        "schema_version": "source_ordinal_recommendation.v1",
        "experiment_id": "fixture-source",
        "split": "validation",
        "source_question_id": "Q1",
        "selected_arm_id": "picture_2",
        "synthetic_arm_means": {"picture_1": 0.2, "picture_2": 0.7},
        "synthetic_treatment_effects": {"picture_2": 0.5},
        "split_manifest_sha256": payload_hash(split),
        "decision_task_sha256": payload_hash(task),
        "blinded_bundle_sha256": "a" * 64,
        "simulator": {"id": "fixture", "revision": "1"},
        "provenance": {"created_at_utc": "2026-08-13T00:00:00Z"},
    }
    recommendation_path = tmp_path / "recommendation.json"
    freeze_recommendation(recommendation, recommendation_path)
    authorization = {
        "schema_version": "source_ordinal_reveal_authorization.v1",
        "status": "validation_source_outcome_reveal_authorized",
        "experiment_id": "fixture-source",
        "recommendation_sha256": payload_hash(recommendation),
        "split_manifest_sha256": payload_hash(split),
        "decision_task_sha256": payload_hash(task),
        "authorized_projection": ["caseid", "weight1", "XTESS040", "Q1"],
    }
    authorization_path = tmp_path / "authorization.json"
    freeze_envelope(authorization, authorization_path, require_blinded=True)
    return archive_path, task_path, split_path, recommendation_path, authorization_path


def test_source_ordinal_fixture_reads_only_frozen_projection(tmp_path: Path) -> None:
    _, task, split, recommendation, authorization = _write_fixture(tmp_path)
    observations = read_revealed_source_ordinal_sav(
        root=tmp_path,
        experiment_id="fixture-source",
        recommendation_path=recommendation,
        reveal_authorization_path=authorization,
        split_manifest_path=split,
        decision_task_path=task,
    )
    assert [
        (row.participant_id, row.arm_id, row.value, row.weight)
        for row in observations
    ] == [
        ("1", "picture_1", 0.0, 1.0),
        ("3", "picture_2", 1.0, 1.5),
    ]


def test_source_ordinal_fails_without_matching_separate_authorization(
    tmp_path: Path,
) -> None:
    _, task, split, recommendation, authorization = _write_fixture(tmp_path)
    authorization_payload = json.loads(
        authorization.read_text(encoding="utf-8")
    )
    authorization_payload["payload"]["authorized_projection"].append(
        "forbidden_extra"
    )
    authorization.write_text(json.dumps(authorization_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash|projection"):
        read_revealed_source_ordinal_sav(
            root=tmp_path,
            experiment_id="fixture-source",
            recommendation_path=recommendation,
            reveal_authorization_path=authorization,
            split_manifest_path=split,
            decision_task_path=task,
        )


def test_source_ordinal_fails_if_task_or_recommendation_changes(tmp_path: Path) -> None:
    _, task, split, recommendation, authorization = _write_fixture(tmp_path)
    task_payload = json.loads(task.read_text(encoding="utf-8"))
    task_payload["canonical_split_status"] = "unassigned"
    task.write_text(json.dumps(task_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="bound|authorize"):
        read_revealed_source_ordinal_sav(
            root=tmp_path,
            experiment_id="fixture-source",
            recommendation_path=recommendation,
            reveal_authorization_path=authorization,
            split_manifest_path=split,
            decision_task_path=task,
        )
