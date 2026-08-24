"""Unified verification view over the prospective evaluation lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .confirmation_calls import (
    DEFAULT_CONFIRMATION_CALL_PLAN_PATH,
    verify_confirmation_call_plan,
)
from .confirmation_execution import (
    DEFAULT_CONFIRMATION_EXECUTION_FREEZE_PATH,
    verify_confirmation_execution_freeze,
)
from .confirmation_preparation import (
    DEFAULT_CONFIRMATION_PREPARATION_PATH,
    verify_confirmation_preparation,
)
from .protocol import payload_hash, verify_envelope
from .public_case_study import (
    DEFAULT_PUBLIC_CASE_STUDY_PATH,
    verify_public_case_study,
)


_ADJUDICATION_PATH = Path(
    "artifacts/confirmation/confirmation_20260814_v1/adjudicated_v1/final_manifest.json"
)
_AGGREGATION_PATH = Path(
    "artifacts/confirmation/confirmation_20260814_v1/aggregation_v1.json"
)
_SCORE_PATH = Path(
    "artifacts/confirmation/confirmation_20260814_v1/score_v1.json"
)
_VALUE_AUDIT_PATH = Path(
    "artifacts/confirmation/confirmation_20260814_v1/value_audit_v1.json"
)


def _stage(
    name: str,
    path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "stage": name,
        "verification": "passed",
        "artifact_path": path.as_posix(),
        "schema_version": payload["schema_version"],
        "artifact_status": payload["status"],
        "payload_sha256": payload_hash(payload),
    }


def evaluate_confirmation_lifecycle(root: Path) -> dict[str, Any]:
    """Verify the full prepare-to-release chain without making model calls."""

    preparation = verify_confirmation_preparation(
        root, root / DEFAULT_CONFIRMATION_PREPARATION_PATH
    )
    call_plan = verify_confirmation_call_plan(
        root, root / DEFAULT_CONFIRMATION_CALL_PLAN_PATH
    )
    execution = verify_confirmation_execution_freeze(
        root, root / DEFAULT_CONFIRMATION_EXECUTION_FREEZE_PATH
    )
    adjudication = verify_envelope(root / _ADJUDICATION_PATH)
    aggregation = verify_envelope(root / _AGGREGATION_PATH)
    score = verify_envelope(root / _SCORE_PATH)
    audit = verify_envelope(root / _VALUE_AUDIT_PATH)
    public_report = verify_public_case_study(root / DEFAULT_PUBLIC_CASE_STUDY_PATH)
    public_payload = public_report["payload"]

    preparation_hash = payload_hash(preparation)
    call_plan_hash = payload_hash(call_plan)
    adjudication_hash = payload_hash(adjudication)
    aggregation_hash = payload_hash(aggregation)
    score_hash = payload_hash(score)
    audit_hash = payload_hash(audit)
    public_sources = {
        source["path"]: source["payload_sha256"]
        for source in public_payload["provenance"]["source_artifacts"]
    }
    expected_public_sources = {
        _SCORE_PATH.as_posix(): score_hash,
        _VALUE_AUDIT_PATH.as_posix(): audit_hash,
        _AGGREGATION_PATH.as_posix(): aggregation_hash,
    }
    chain_checks = {
        "call_plan_to_preparation": (
            call_plan["preparation_payload_sha256"] == preparation_hash
        ),
        "execution_to_preparation": (
            execution["preparation"]["payload_sha256"] == preparation_hash
        ),
        "execution_to_call_plan": (
            execution["call_plan"]["payload_sha256"] == call_plan_hash
        ),
        "adjudication_to_call_plan": (
            adjudication["call_plan_payload_sha256"] == call_plan_hash
        ),
        "aggregation_to_adjudication": (
            aggregation["adjudication_manifest_payload_sha256"]
            == adjudication_hash
        ),
        "aggregation_to_preparation": (
            aggregation["preparation_payload_sha256"] == preparation_hash
        ),
        "aggregation_to_call_plan": (
            aggregation["call_plan_payload_sha256"] == call_plan_hash
        ),
        "score_to_aggregation": (
            score["aggregation_payload_sha256"] == aggregation_hash
        ),
        "audit_to_score": audit["score_payload_sha256"] == score_hash,
        "public_bundle_to_aggregate_sources": all(
            public_sources.get(path) == digest
            for path, digest in expected_public_sources.items()
        ),
    }
    failed = [name for name, passed in chain_checks.items() if not passed]
    if failed:
        raise ValueError(f"evaluation lifecycle hash chain failed: {failed}")

    participant_rows_serialized = max(
        int(score["participant_rows_serialized"]),
        int(audit["participant_rows_serialized"]),
        int(public_payload["run_integrity"]["participant_rows_serialized"]),
    )
    if participant_rows_serialized != 0:
        raise ValueError("evaluation lifecycle serialized participant rows")

    stages = [
        _stage("prepare", DEFAULT_CONFIRMATION_PREPARATION_PATH, preparation),
        _stage("freeze_call_plan", DEFAULT_CONFIRMATION_CALL_PLAN_PATH, call_plan),
        _stage(
            "freeze_execution", DEFAULT_CONFIRMATION_EXECUTION_FREEZE_PATH, execution
        ),
        _stage("adjudicate_outputs", _ADJUDICATION_PATH, adjudication),
        _stage("aggregate_recommendations", _AGGREGATION_PATH, aggregation),
        _stage("reveal_and_score", _SCORE_PATH, score),
        _stage("audit_value", _VALUE_AUDIT_PATH, audit),
        _stage("release_gate", DEFAULT_PUBLIC_CASE_STUDY_PATH, public_payload),
    ]
    return {
        "schema_version": "intervenebench.evaluation_lifecycle.v1",
        "overall_status": "complete_verified",
        "stages": stages,
        "integrity": {
            "hash_chain_verified": True,
            "chain_checks": chain_checks,
            "participant_rows_serialized": participant_rows_serialized,
            "model_calls_made_by_verification": 0,
            "outcome_access_performed_by_verification": False,
        },
        "release_decisions": public_report["release_decisions"],
        "authority": (
            "No execution, reveal, or model-call authority is granted by this status view."
        ),
    }


def render_confirmation_lifecycle(lifecycle: Mapping[str, Any]) -> str:
    lines = [
        "InterveneBench evaluation lifecycle",
        "===================================",
        f"Status: {str(lifecycle['overall_status']).replace('_', ' ').upper()}",
        f"{len(lifecycle['stages'])}/{len(lifecycle['stages'])} stages verified",
        "",
    ]
    for index, stage in enumerate(lifecycle["stages"], start=1):
        label = stage["stage"].replace("_", " ").upper()
        lines.append(
            f"{index}. {label}: {stage['verification'].upper()} "
            f"({stage['artifact_status']})"
        )
    lines.extend(["", "Release decisions", "-----------------"])
    labels = (
        ("Candidate screening", "candidate_screening"),
        ("Autonomous intervention selection", "autonomous_intervention_selection"),
        ("Confidence-based abstention", "confidence_based_abstention"),
        ("Small-sample human fallback", "small_sample_human_fallback"),
    )
    for label, key in labels:
        decision = lifecycle["release_decisions"][key]["decision"]
        lines.append(f"{label}: {decision.replace('_', ' ').upper()}")
    lines.extend(["", lifecycle["authority"]])
    return "\n".join(lines) + "\n"
