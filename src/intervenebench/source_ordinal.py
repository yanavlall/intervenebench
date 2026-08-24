"""Recommendation-bound, projection-limited source SAV ordinal ingestion."""

from __future__ import annotations

import csv
import hashlib
import json
from math import isfinite
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Any, Mapping
from zipfile import ZipFile

from .evaluation import WeightedObservation
from .protocol import payload_hash, verify_envelope, verify_frozen_recommendation


RECOMMENDATION_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "split",
        "source_question_id",
        "selected_arm_id",
        "synthetic_arm_means",
        "synthetic_treatment_effects",
        "split_manifest_sha256",
        "decision_task_sha256",
        "blinded_bundle_sha256",
        "simulator",
        "provenance",
    }
)
AUTHORIZATION_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "experiment_id",
        "recommendation_sha256",
        "split_manifest_sha256",
        "decision_task_sha256",
        "authorized_projection",
    }
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be a SHA-256 digest") from error
    return value


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _parse_integer(value: Any, *, field: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be integer-coded") from error
    if not isfinite(number) or not number.is_integer():
        raise ValueError(f"{field} must be integer-coded")
    return int(number)


def _verify_bound_authorization(
    recommendation_path: Path,
    reveal_authorization_path: Path,
    *,
    experiment_id: str,
    split_manifest_path: Path,
    decision_task_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    recommendation = verify_frozen_recommendation(recommendation_path)
    if set(recommendation) != RECOMMENDATION_REQUIRED_KEYS:
        raise ValueError("source ordinal recommendation has unexpected fields")
    if recommendation["schema_version"] != "source_ordinal_recommendation.v1":
        raise ValueError("unsupported source ordinal recommendation schema")
    if recommendation["experiment_id"] != experiment_id:
        raise ValueError("recommendation experiment changed")
    if recommendation["split"] != "validation":
        raise ValueError("only validation source outcomes may be authorized")
    means = recommendation["synthetic_arm_means"]
    if not isinstance(means, Mapping) or len(means) < 2 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
        for value in means.values()
    ):
        raise ValueError("synthetic arm means must be normalized finite utilities")
    if recommendation["selected_arm_id"] not in means:
        raise ValueError("selected arm is absent from synthetic means")
    if not isinstance(recommendation["synthetic_treatment_effects"], Mapping):
        raise ValueError("synthetic treatment effects must be a mapping")
    for field in (
        "split_manifest_sha256",
        "decision_task_sha256",
        "blinded_bundle_sha256",
    ):
        _digest(recommendation[field], field=field)
    if not isinstance(recommendation["simulator"], Mapping) or not all(
        str(recommendation["simulator"].get(field, "")).strip()
        for field in ("id", "revision")
    ):
        raise ValueError("simulator identity and revision are required")
    if not isinstance(recommendation["provenance"], Mapping) or not str(
        recommendation["provenance"].get("created_at_utc", "")
    ).strip():
        raise ValueError("recommendation creation time is required")

    authorization = verify_envelope(
        reveal_authorization_path, require_blinded=True
    )
    if set(authorization) != AUTHORIZATION_FIELDS:
        raise ValueError("source ordinal reveal authorization has unexpected fields")
    if (
        authorization["schema_version"]
        != "source_ordinal_reveal_authorization.v1"
        or authorization["status"]
        != "validation_source_outcome_reveal_authorized"
    ):
        raise ValueError("source ordinal outcome reveal is not authorized")
    if authorization["experiment_id"] != experiment_id:
        raise ValueError("reveal authorization experiment changed")
    if authorization["recommendation_sha256"] != payload_hash(recommendation):
        raise ValueError("reveal authorization is not bound to the recommendation")

    split = _load_object(split_manifest_path)
    task = _load_object(decision_task_path)
    if payload_hash(split) != recommendation["split_manifest_sha256"]:
        raise ValueError("recommendation is not bound to the split")
    if payload_hash(task) != recommendation["decision_task_sha256"]:
        raise ValueError("recommendation is not bound to the task")
    if authorization["split_manifest_sha256"] != payload_hash(split):
        raise ValueError("reveal authorization is not bound to the split")
    if authorization["decision_task_sha256"] != payload_hash(task):
        raise ValueError("reveal authorization is not bound to the task")
    if split.get("experiment_to_split", {}).get(experiment_id) != "validation":
        raise ValueError("split does not authorize a validation reveal")
    if split.get("test_outcomes_sealed") is not True:
        raise ValueError("test outcomes must remain sealed")
    task_split = task.get("split", task.get("canonical_split_status"))
    if task.get("experiment_id") != experiment_id or task_split != "validation":
        raise ValueError("decision task does not authorize this validation reveal")
    if task.get("outcome_family") != "ordinal":
        raise ValueError("source ordinal task must declare an ordinal outcome")
    if task.get("source_question_id") != recommendation["source_question_id"]:
        raise ValueError("recommendation question changed")
    if {arm["arm_id"] for arm in task.get("arms", ())} != set(means):
        raise ValueError("recommendation arms do not match the source task")

    mapping = task.get("source_variable_mapping")
    locator = task.get("source_data_locator")
    if not isinstance(mapping, Mapping) or not isinstance(locator, Mapping):
        raise ValueError("source ordinal task lacks frozen source metadata")
    projection = (
        str(mapping["participant_id_variable"]),
        str(mapping["weight_variable"]),
        str(mapping["assignment_variable"]),
        str(mapping["outcome_variable"]),
    )
    if locator.get("authorized_projection") != list(projection):
        raise ValueError("source locator projection changed")
    if authorization["authorized_projection"] != list(projection):
        raise ValueError("reveal authorization projection changed")
    return task, dict(mapping), projection


def _extract_projected_sav(
    sav_path: Path, csv_path: Path, projection: tuple[str, ...]
) -> None:
    expression = (
        "args<-commandArgs(trailingOnly=TRUE);"
        "cols<-strsplit(args[3],',',fixed=TRUE)[[1]];"
        "d<-haven::read_sav(args[1],col_select=tidyselect::all_of(cols));"
        "d<-haven::zap_labels(d);"
        "write.csv(data.frame(lapply(d,as.numeric),check.names=FALSE),"
        "args[2],row.names=FALSE,na='')"
    )
    subprocess.run(
        [
            "Rscript",
            "-e",
            expression,
            str(sav_path),
            str(csv_path),
            ",".join(projection),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )


def read_revealed_source_ordinal_sav(
    *,
    root: Path,
    experiment_id: str,
    recommendation_path: Path,
    reveal_authorization_path: Path,
    split_manifest_path: Path,
    decision_task_path: Path,
) -> tuple[WeightedObservation, ...]:
    """Project and read exactly four frozen columns after separate authorization."""

    task, mapping, projection = _verify_bound_authorization(
        recommendation_path,
        reveal_authorization_path,
        experiment_id=experiment_id,
        split_manifest_path=split_manifest_path,
        decision_task_path=decision_task_path,
    )
    locator = task["source_data_locator"]
    archive_path = root / str(locator["container_path"])
    if _sha256(archive_path) != locator["container_sha256"]:
        raise ValueError("source ordinal container hash mismatch")
    with ZipFile(archive_path) as archive:
        sav_member = str(locator["sav_member"])
        sav_bytes = archive.read(sav_member)
    if hashlib.sha256(sav_bytes).hexdigest() != locator["sav_member_sha256"]:
        raise ValueError("source ordinal SAV member hash mismatch")

    arm_map = {str(key): str(value) for key, value in mapping["assignment_to_arm"].items()}
    utility_map = {
        int(option["raw_value"]): float(option["normalized_utility"])
        for option in task["response_options"]
    }
    missing_codes = {int(value) for value in mapping["missing_outcome_codes"]}
    observations: list[WeightedObservation] = []
    participants: set[str] = set()
    with TemporaryDirectory(prefix=f"intervenebench-{experiment_id}-") as temporary:
        temporary_path = Path(temporary)
        sav_path = temporary_path / "source.sav"
        csv_path = temporary_path / "projection.csv"
        sav_path.write_bytes(sav_bytes)
        _extract_projected_sav(sav_path, csv_path, projection)
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != projection:
                raise ValueError("source ordinal projected columns changed")
            for row in reader:
                participant = str(
                    _parse_integer(
                        row[mapping["participant_id_variable"]],
                        field="source participant ID",
                    )
                )
                if participant in participants:
                    raise ValueError("source participant IDs must be unique")
                participants.add(participant)
                assignment = str(
                    _parse_integer(
                        row[mapping["assignment_variable"]],
                        field="source assignment",
                    )
                )
                if assignment not in arm_map:
                    raise ValueError("source assignment is outside the frozen action set")
                raw_text = row[mapping["outcome_variable"]]
                if raw_text == "":
                    continue
                raw_value = _parse_integer(raw_text, field="source ordinal outcome")
                if raw_value in missing_codes:
                    continue
                if raw_value not in utility_map:
                    raise ValueError("source outcome is outside the frozen ordinal scale")
                weight = float(row[mapping["weight_variable"]])
                if not isfinite(weight) or weight <= 0.0:
                    raise ValueError("source weights must be positive and finite")
                observations.append(
                    WeightedObservation(
                        participant_id=participant,
                        arm_id=arm_map[assignment],
                        value=utility_map[raw_value],
                        weight=weight,
                    )
                )
    if not observations or {row.arm_id for row in observations} != set(arm_map.values()):
        raise ValueError("every source ordinal arm must retain an observation")
    return tuple(observations)
