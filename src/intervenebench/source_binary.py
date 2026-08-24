"""Recommendation-bound ingestion for source-only binary decision outcomes."""

from __future__ import annotations

import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.csv as pacsv

from .evaluation import WeightedObservation
from .protocol import payload_hash, verify_frozen_recommendation


SOURCE_BINARY_REVEAL_REQUIRED_KEYS = frozenset(
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be a SHA-256 digest") from error
    return value


def verify_bound_source_binary_reveal_authorization(
    recommendation_path: Path,
    *,
    experiment_id: str,
    split_manifest_path: Path,
    decision_task_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authorize one source-only validation outcome after recommendation freeze."""

    recommendation = verify_frozen_recommendation(recommendation_path)
    missing = sorted(SOURCE_BINARY_REVEAL_REQUIRED_KEYS - set(recommendation))
    if missing:
        raise ValueError(f"recommendation is missing required fields: {missing}")
    if recommendation["schema_version"] != "source_binary_recommendation.v1":
        raise ValueError("unsupported source-binary recommendation schema")
    if recommendation["experiment_id"] != experiment_id:
        raise ValueError("recommendation experiment does not match requested reveal")
    if recommendation["split"] != "validation":
        raise ValueError("only validation source outcomes may be revealed")
    means = recommendation["synthetic_arm_means"]
    if not isinstance(means, Mapping) or len(means) < 2:
        raise ValueError("synthetic_arm_means must contain at least two arms")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0.0 <= value <= 1.0
        for value in means.values()
    ):
        raise ValueError("synthetic arm means must be normalized finite utilities")
    if recommendation["selected_arm_id"] not in means:
        raise ValueError("selected arm is absent from synthetic arm means")
    if not isinstance(recommendation["synthetic_treatment_effects"], Mapping):
        raise ValueError("synthetic_treatment_effects must be a mapping")
    for key in (
        "split_manifest_sha256",
        "decision_task_sha256",
        "blinded_bundle_sha256",
    ):
        _verify_digest(recommendation[key], label=key)
    if not isinstance(recommendation["simulator"], Mapping) or not all(
        str(recommendation["simulator"].get(key, "")).strip()
        for key in ("id", "revision")
    ):
        raise ValueError("simulator identity and revision are required")
    if not isinstance(recommendation["provenance"], Mapping) or not str(
        recommendation["provenance"].get("created_at_utc", "")
    ).strip():
        raise ValueError("recommendation creation time is required")

    split = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    task = json.loads(decision_task_path.read_text(encoding="utf-8"))
    if not isinstance(split, Mapping) or not isinstance(task, Mapping):
        raise ValueError("split manifest and decision task must be JSON objects")
    if payload_hash(split) != recommendation["split_manifest_sha256"]:
        raise ValueError("recommendation is not bound to the supplied split manifest")
    if payload_hash(task) != recommendation["decision_task_sha256"]:
        raise ValueError("recommendation is not bound to the supplied decision task")
    if split.get("experiment_to_split", {}).get(experiment_id) != "validation":
        raise ValueError("frozen split does not authorize a validation reveal")
    if split.get("test_outcomes_sealed") is not True:
        raise ValueError("split manifest must keep test outcomes sealed")
    task_split = task.get("split", task.get("canonical_split_status"))
    if task.get("experiment_id") != experiment_id or task_split != "validation":
        raise ValueError("frozen decision task does not authorize this validation reveal")
    if task.get("source_question_id") != recommendation["source_question_id"]:
        raise ValueError("recommendation question does not match decision task")
    if task.get("outcome_family") != "binary":
        raise ValueError("source decision task must declare a binary outcome")
    if {arm["arm_id"] for arm in task.get("arms", [])} != set(means):
        raise ValueError("recommendation arms do not match decision task")
    return dict(recommendation), dict(task)


def read_revealed_source_binary_csv(
    *,
    root: Path,
    experiment_id: str,
    recommendation_path: Path,
    split_manifest_path: Path,
    decision_task_path: Path,
) -> tuple[WeightedObservation, ...]:
    """Read the exact frozen source columns only after bound authorization."""

    _, task = verify_bound_source_binary_reveal_authorization(
        recommendation_path,
        experiment_id=experiment_id,
        split_manifest_path=split_manifest_path,
        decision_task_path=decision_task_path,
    )
    locator = task.get("source_data_locator")
    mapping = task.get("source_variable_mapping")
    if not isinstance(locator, Mapping) or not isinstance(mapping, Mapping):
        raise ValueError("decision task lacks source-data mapping metadata")
    expected_projection = [
        mapping["participant_id_variable"],
        mapping["weight_variable"],
        mapping["assignment_variable"],
        mapping["outcome_variable"],
    ]
    if locator.get("authorized_projection") != expected_projection:
        raise ValueError("source projection does not match the frozen task mapping")

    archive_path = root / str(locator["container_path"])
    if _sha256(archive_path) != locator["container_sha256"]:
        raise ValueError("source container hash mismatch")
    with ZipFile(archive_path) as archive:
        member_name = str(locator["csv_member"])
        member_bytes = archive.read(member_name)
    if hashlib.sha256(member_bytes).hexdigest() != locator["csv_member_sha256"]:
        raise ValueError("source CSV member hash mismatch")

    try:
        table = pacsv.read_csv(
            pa.BufferReader(member_bytes),
            convert_options=pacsv.ConvertOptions(include_columns=expected_projection),
        )
    except (pa.ArrowInvalid, KeyError) as error:
        raise ValueError("source CSV lacks a valid frozen projected column") from error
    if table.column_names != expected_projection:
        raise ValueError("source CSV projection order changed unexpectedly")
    arm_map = mapping["assignment_to_arm"]
    utility_map = mapping["outcome_to_utility"]
    missing_codes = {str(code) for code in mapping["missing_outcome_codes"]}
    observations: list[WeightedObservation] = []
    participant_ids: set[str] = set()
    for row in table.to_pylist():
        participant_id = str(row[mapping["participant_id_variable"]]).strip()
        if not participant_id or participant_id in participant_ids:
            raise ValueError("source participant IDs must be non-empty and unique")
        participant_ids.add(participant_id)
        assignment = str(row[mapping["assignment_variable"]]).strip()
        if assignment not in arm_map:
            raise ValueError("source assignment is absent from the frozen action set")
        raw_outcome = row[mapping["outcome_variable"]]
        outcome = "" if raw_outcome is None else str(raw_outcome).strip()
        if outcome in missing_codes or outcome == "":
            continue
        if outcome not in utility_map:
            raise ValueError("source outcome violates the frozen binary coding")
        weight = float(row[mapping["weight_variable"]])
        if not isfinite(weight) or weight <= 0.0:
            raise ValueError("source weights must be positive and finite")
        observations.append(
            WeightedObservation(
                participant_id=participant_id,
                arm_id=str(arm_map[assignment]),
                value=float(utility_map[outcome]),
                weight=weight,
            )
        )
    if not observations or set(observation.arm_id for observation in observations) != set(
        arm_map.values()
    ):
        raise ValueError("every frozen arm must retain a valid source observation")
    return tuple(observations)
