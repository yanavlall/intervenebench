"""Leakage checks and immutable recommendation envelopes."""

from __future__ import annotations

import json
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any, Mapping


FORBIDDEN_BLINDED_KEYS = frozenset(
    {
        "response",
        "reasoning",
        "human_outcomes",
        "human_arm_means",
        "human_treatment_effects",
        "tau_h",
        "human_winner",
        "regret",
        "p_value",
        "significance",
    }
)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def payload_hash(payload: Any) -> str:
    return sha256(canonical_json_bytes(payload)).hexdigest()


def _walk_keys(payload: Any, path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    violations: list[tuple[str, ...]] = []
    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = str(raw_key)
            current = (*path, key)
            if key.casefold() in FORBIDDEN_BLINDED_KEYS:
                violations.append(current)
            violations.extend(_walk_keys(value, current))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            violations.extend(_walk_keys(value, (*path, str(index))))
    return violations


def assert_blinded_payload(payload: Any) -> None:
    violations = _walk_keys(payload)
    if violations:
        rendered = [".".join(path) for path in violations]
        raise ValueError(f"forbidden outcome-derived fields in blinded payload: {rendered}")


def freeze_envelope(
    payload: Mapping[str, Any], path: Path, *, require_blinded: bool = False
) -> str:
    """Create a canonical, self-verifying JSON envelope without overwriting."""

    if require_blinded:
        assert_blinded_payload(payload)
    digest = payload_hash(payload)
    envelope = {"payload": payload, "sha256": digest}
    encoded = json.dumps(envelope, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    return digest


def verify_envelope(path: Path, *, require_blinded: bool = False) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        envelope = json.load(stream)
    if set(envelope) != {"payload", "sha256"}:
        raise ValueError("malformed artifact envelope")
    if not isinstance(envelope["payload"], Mapping):
        raise ValueError("artifact payload must be a JSON object")
    if require_blinded:
        assert_blinded_payload(envelope["payload"])
    actual = payload_hash(envelope["payload"])
    if actual != envelope["sha256"]:
        raise ValueError("artifact hash mismatch")
    return envelope["payload"]


def freeze_recommendation(payload: Mapping[str, Any], path: Path) -> str:
    """Write a self-verifying recommendation envelope without human outcomes."""

    return freeze_envelope(payload, path, require_blinded=True)


def verify_frozen_recommendation(path: Path) -> dict[str, Any]:
    return verify_envelope(path, require_blinded=True)


REVEAL_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "split",
        "task_num",
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

CONTINUOUS_REVEAL_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "split",
        "task_num",
        "selected_arm_id",
        "synthetic_arm_locations",
        "synthetic_treatment_effects",
        "outcome_family",
        "direction",
        "outcome_unit",
        "location_estimand",
        "normalized_for_pooled_regret",
        "split_manifest_sha256",
        "decision_task_sha256",
        "blinded_bundle_sha256",
        "simulator",
        "provenance",
    }
)


def verify_continuous_reveal_authorization(
    path: Path, *, experiment_id: str
) -> dict[str, Any]:
    """Verify a frozen validation recommendation for a continuous task."""

    payload = verify_frozen_recommendation(path)
    missing = sorted(CONTINUOUS_REVEAL_REQUIRED_KEYS - set(payload))
    if missing:
        raise ValueError(f"recommendation is missing required fields: {missing}")
    if payload["schema_version"] != "continuous_recommendation.v1":
        raise ValueError("unsupported continuous recommendation schema version")
    if payload["experiment_id"] != experiment_id:
        raise ValueError("recommendation experiment does not match requested reveal")
    if payload["split"] != "validation":
        raise ValueError("only validation outcomes may be revealed")
    if payload["outcome_family"] != "continuous":
        raise ValueError("continuous recommendation must declare continuous outcome family")
    if payload["location_estimand"] != "mean":
        raise ValueError("continuous recommendation must use the frozen mean estimand")
    if payload["normalized_for_pooled_regret"] is not False:
        raise ValueError("uncapped continuous outcomes cannot claim normalized pooled regret")
    robustness = payload.get("robustness")
    if (
        not isinstance(robustness, Mapping)
        or not isinstance(robustness.get("median"), Mapping)
        or not isinstance(
            robustness["median"].get("synthetic_arm_locations"), Mapping
        )
    ):
        raise ValueError("continuous recommendation requires synthetic median robustness")
    if not isinstance(payload["task_num"], int) or payload["task_num"] < 0:
        raise ValueError("recommendation task_num must be a non-negative integer")
    locations = payload["synthetic_arm_locations"]
    if not isinstance(locations, Mapping) or len(locations) < 2:
        raise ValueError("synthetic_arm_locations must contain at least two arms")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
        for value in locations.values()
    ):
        raise ValueError("synthetic arm locations must be finite numbers")
    if payload["selected_arm_id"] not in locations:
        raise ValueError("selected arm is absent from synthetic arm locations")
    if not isinstance(payload["synthetic_treatment_effects"], Mapping):
        raise ValueError("synthetic_treatment_effects must be a mapping")
    if not str(payload["outcome_unit"]).strip():
        raise ValueError("continuous outcome unit is required")
    if payload["direction"] not in {"higher_is_better", "lower_is_better"}:
        raise ValueError("continuous outcome direction is invalid")

    for key in ("split_manifest_sha256", "decision_task_sha256", "blinded_bundle_sha256"):
        value = payload[key]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{key} must be a SHA-256 digest")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError(f"{key} must be a SHA-256 digest") from error
    simulator = payload["simulator"]
    provenance = payload["provenance"]
    if not isinstance(simulator, Mapping) or not all(
        str(simulator.get(key, "")).strip() for key in ("id", "revision")
    ):
        raise ValueError("simulator identity and revision are required")
    if not isinstance(provenance, Mapping) or not str(
        provenance.get("created_at_utc", "")
    ).strip():
        raise ValueError("recommendation creation time is required")
    return payload


def verify_bound_continuous_reveal_authorization(
    path: Path,
    *,
    experiment_id: str,
    split_manifest_path: Path,
    decision_task_path: Path,
) -> dict[str, Any]:
    recommendation = verify_continuous_reveal_authorization(
        path, experiment_id=experiment_id
    )
    with split_manifest_path.open(encoding="utf-8") as stream:
        split = json.load(stream)
    with decision_task_path.open(encoding="utf-8") as stream:
        task = json.load(stream)
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
    if task.get("experiment_id") != experiment_id or task.get("split") != "validation":
        raise ValueError("frozen decision task does not authorize this validation reveal")
    if task.get("socsci210_task_num") != recommendation["task_num"]:
        raise ValueError("recommendation task does not match frozen decision task")
    if task.get("outcome_family") != "continuous":
        raise ValueError("decision task does not declare a continuous outcome")
    if task.get("estimator", {}).get("location") != recommendation["location_estimand"]:
        raise ValueError("recommendation estimator does not match frozen decision task")
    if task.get("estimator", {}).get("robustness_locations") != ["median"]:
        raise ValueError("decision task must freeze median robustness")
    return recommendation


def verify_reveal_authorization(path: Path, *, experiment_id: str) -> dict[str, Any]:
    """Verify that a frozen validation recommendation authorizes one outcome reveal."""

    payload = verify_frozen_recommendation(path)
    missing = sorted(REVEAL_REQUIRED_KEYS - set(payload))
    if missing:
        raise ValueError(f"recommendation is missing required fields: {missing}")
    if payload["schema_version"] != "recommendation.v1":
        raise ValueError("unsupported recommendation schema version")
    if payload["experiment_id"] != experiment_id:
        raise ValueError("recommendation experiment does not match requested reveal")
    if payload["split"] != "validation":
        raise ValueError("only validation outcomes may be revealed during Phase 1")
    if not isinstance(payload["task_num"], int) or payload["task_num"] < 0:
        raise ValueError("recommendation task_num must be a non-negative integer")

    means = payload["synthetic_arm_means"]
    if not isinstance(means, Mapping) or len(means) < 2:
        raise ValueError("synthetic_arm_means must contain at least two arms")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
        or not 0.0 <= value <= 1.0
        for value in means.values()
    ):
        raise ValueError("synthetic arm means must be finite normalized utilities")
    if payload["selected_arm_id"] not in means:
        raise ValueError("selected arm is absent from synthetic arm means")
    if not isinstance(payload["synthetic_treatment_effects"], Mapping):
        raise ValueError("synthetic_treatment_effects must be a mapping")

    for key in (
        "split_manifest_sha256",
        "decision_task_sha256",
        "blinded_bundle_sha256",
    ):
        value = payload[key]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{key} must be a SHA-256 digest")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError(f"{key} must be a SHA-256 digest") from error

    simulator = payload["simulator"]
    provenance = payload["provenance"]
    if not isinstance(simulator, Mapping) or not all(
        str(simulator.get(key, "")).strip() for key in ("id", "revision")
    ):
        raise ValueError("simulator identity and revision are required")
    if not isinstance(provenance, Mapping) or not str(
        provenance.get("created_at_utc", "")
    ).strip():
        raise ValueError("recommendation creation time is required")
    return payload


def verify_bound_reveal_authorization(
    path: Path,
    *,
    experiment_id: str,
    split_manifest_path: Path,
    decision_task_path: Path,
) -> dict[str, Any]:
    """Bind reveal authorization to the independently frozen split and task files."""

    recommendation = verify_reveal_authorization(path, experiment_id=experiment_id)
    with split_manifest_path.open(encoding="utf-8") as stream:
        split = json.load(stream)
    with decision_task_path.open(encoding="utf-8") as stream:
        task = json.load(stream)
    if not isinstance(split, Mapping) or not isinstance(task, Mapping):
        raise ValueError("split manifest and decision task must be JSON objects")
    if payload_hash(split) != recommendation["split_manifest_sha256"]:
        raise ValueError("recommendation is not bound to the supplied split manifest")
    if payload_hash(task) != recommendation["decision_task_sha256"]:
        raise ValueError("recommendation is not bound to the supplied decision task")
    if split.get("experiment_to_split", {}).get(experiment_id) != "validation":
        raise ValueError("frozen split does not authorize a validation reveal")
    if split.get("test_outcomes_sealed") is not True:
        raise ValueError("split manifest must keep test outcomes sealed during Phase 1")
    if task.get("experiment_id") != experiment_id or task.get("split") != "validation":
        raise ValueError("frozen decision task does not authorize this validation reveal")
    if task.get("socsci210_task_num") != recommendation["task_num"]:
        raise ValueError("recommendation task does not match frozen decision task")
    return recommendation
