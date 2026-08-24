"""Outcome-blind readiness accounting for the finite replication queue."""

from __future__ import annotations

from math import isfinite
from typing import Any, Mapping

from .independent_replication import REQUIRED_STAGE_GATES


SCHEMA_VERSION = "intervenebench.replication_completion_status.v1"
COMPLETION_GATE_NAMES = REQUIRED_STAGE_GATES
GATE_STATES = frozenset({"passed", "blocked", "failed", "pending"})
FORBIDDEN_RESULT_KEYS = frozenset(
    {
        "result",
        "response",
        "human_outcomes",
        "human_arm_means",
        "human_treatment_effects",
        "human_winner",
        "treatment_effect",
        "decision_regret",
        "regret",
        "p_value",
        "significance",
        "reported_winner",
    }
)


def _assert_no_result_fields(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if key.casefold() in FORBIDDEN_RESULT_KEYS:
                location = ".".join((*path, key))
                raise ValueError(f"result-bearing field is forbidden: {location}")
            _assert_no_result_fields(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_result_fields(child, (*path, str(index)))


def _queue_rows(queue: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for wave_name in ("wave_1", "wave_2"):
        wave = queue.get(wave_name)
        if not isinstance(wave, list):
            raise ValueError("frozen queue waves must be lists")
        for row in wave:
            if not isinstance(row, Mapping):
                raise ValueError("frozen queue rows must be mappings")
            rows.append(row)
    expected_orders = list(range(1, len(rows) + 1))
    observed_orders = [row.get("queue_order") for row in rows]
    if observed_orders != expected_orders:
        raise ValueError("frozen queue orders must be contiguous")
    return rows


def _expected_status(gates: Mapping[str, str]) -> str:
    states = set(gates.values())
    if "failed" in states:
        return "failed"
    if "blocked" in states:
        return "blocked"
    if "pending" in states:
        return "pending"
    return "runnable"


def validate_replication_completion_status(
    payload: Mapping[str, Any], queue: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and count tasks without permitting optimistic readiness labels."""

    if not isinstance(payload, Mapping) or not isinstance(queue, Mapping):
        raise ValueError("completion status and frozen queue must be mappings")
    _assert_no_result_fields(payload)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported replication completion status schema")
    if payload.get("status") != "outcome_blind_completion_status":
        raise ValueError("completion status must remain outcome blind")
    queue_sha = payload.get("queue_sha256")
    if (
        not isinstance(queue_sha, str)
        or len(queue_sha) != 64
        or any(character not in "0123456789abcdef" for character in queue_sha)
    ):
        raise ValueError("queue_sha256 must be a lowercase SHA-256 digest")

    queue_rows = _queue_rows(queue)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(queue_rows):
        raise ValueError("completion status must cover the frozen queue exactly")

    counts = {state: 0 for state in ("runnable", "blocked", "failed", "pending")}
    runnable_socsci = 0
    for index, (candidate, queue_row) in enumerate(
        zip(candidates, queue_rows, strict=True)
    ):
        if not isinstance(candidate, Mapping):
            raise ValueError("completion candidate rows must be mappings")
        if (
            candidate.get("queue_order") != queue_row.get("queue_order")
            or candidate.get("candidate_id") != queue_row.get("candidate_id")
        ):
            raise ValueError("completion status must follow the frozen queue exactly")
        if (
            candidate.get("outcome_access") != "sealed"
            or candidate.get("result_text_exposed") is not False
        ):
            raise ValueError("every completion candidate must remain outcome sealed")
        gates = candidate.get("gates")
        if not isinstance(gates, Mapping) or tuple(gates) != COMPLETION_GATE_NAMES:
            raise ValueError("completion gate names and order must remain exact")
        if any(state not in GATE_STATES for state in gates.values()):
            raise ValueError("completion gate state is invalid")
        expected_status = _expected_status(gates)
        if candidate.get("overall_status") != expected_status:
            raise ValueError(
                f"candidate[{index}].overall_status does not follow its gate states"
            )
        counts[expected_status] += 1
        if expected_status == "runnable" and queue_row.get("source_stratum") == "socsci210":
            runnable_socsci += 1

    authority = payload.get("authority")
    if not isinstance(authority, Mapping):
        raise ValueError("completion status requires an authority boundary")
    spend = authority.get("authorized_spend_usd")
    zero_spend = (
        not isinstance(spend, bool)
        and isinstance(spend, (int, float))
        and isfinite(float(spend))
        and float(spend) == 0.0
    )
    if not zero_spend or any(
        authority.get(key) is not False
        for key in (
            "model_calls_authorized",
            "human_outcome_reveal_authorized",
            "participant_row_access_authorized",
        )
    ):
        raise ValueError("replication completion status must retain zero authority")

    panel_gate = queue.get("panel_gate")
    if not isinstance(panel_gate, Mapping):
        raise ValueError("frozen queue panel gate is required")
    minimum = panel_gate.get("minimum_runnable_tasks")
    strong = panel_gate.get("strong_target_tasks")
    if not isinstance(minimum, int) or not isinstance(strong, int):
        raise ValueError("frozen queue panel thresholds are invalid")
    return {
        "candidate_count": len(candidates),
        "runnable_count": counts["runnable"],
        "blocked_count": counts["blocked"],
        "failed_count": counts["failed"],
        "pending_count": counts["pending"],
        "runnable_socsci210_count": runnable_socsci,
        "minimum_panel_ready": counts["runnable"] >= minimum,
        "strong_panel_ready": counts["runnable"] >= strong,
    }
