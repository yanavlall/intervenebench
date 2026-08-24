"""Outcome-blind intake gates for an independent prospective replication panel."""

from __future__ import annotations

from math import isfinite
from typing import Any, Mapping


SCHEMA_VERSION = "intervenebench.independent_replication_intake.v1"
REQUIRED_STAGE_GATES = (
    "source_identity_and_dedup",
    "deployable_action_set",
    "stable_bounded_utility",
    "source_faithful_stimulus_and_sequence",
    "outcome_blind_human_mapping",
    "runnable_simulator_adapter",
)
FORBIDDEN_RESULT_KEYS = frozenset(
    {
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


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def validate_replication_intake(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a candidate intake without authorizing inference or reveal.

    The intake is deliberately stricter than the earlier design census: every
    candidate must still be outcome sealed, result-text clean, and provisionally
    independent by both fielding and conservative paradigm.  Scientific and
    engineering gates may subsequently remove rows; they may never add exposed
    rows to hit the target count.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("replication intake must be a mapping")
    _assert_no_result_fields(payload)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported independent-replication intake schema")
    if payload.get("status") != "outcome_blind_intake_frozen":
        raise ValueError("replication intake must be outcome-blind and frozen")

    panel_size = payload.get("panel_size")
    if not isinstance(panel_size, Mapping):
        raise ValueError("panel_size must be a mapping")
    minimum = panel_size.get("minimum_analyzable")
    target = panel_size.get("target")
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or minimum < 12
        or isinstance(target, bool)
        or not isinstance(target, int)
        or target < minimum
    ):
        raise ValueError("replication panel requires minimum 12 and target >= minimum")

    prior = payload.get("prior_revealed_experiment_ids")
    if not isinstance(prior, list) or any(
        not isinstance(value, str) or not value.strip() for value in prior
    ):
        raise ValueError("prior revealed experiment IDs must be a string list")
    if len(prior) != len(set(prior)):
        raise ValueError("prior revealed experiment IDs must be unique")
    prior_set = set(prior)

    candidates = payload.get("candidate_pool")
    if not isinstance(candidates, list) or len(candidates) < target:
        raise ValueError("candidate pool must contain at least the target panel size")
    identifiers: list[str] = []
    paradigms: list[str] = []
    fieldings: list[str] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError("candidate rows must be mappings")
        prefix = f"candidate_pool[{index}]"
        identifier = _nonempty_string(
            candidate.get("candidate_id"), name=f"{prefix}.candidate_id"
        )
        paradigm = _nonempty_string(
            candidate.get("paradigm_group"), name=f"{prefix}.paradigm_group"
        )
        fielding = _nonempty_string(
            candidate.get("fielding_cluster_id"),
            name=f"{prefix}.fielding_cluster_id",
        )
        _nonempty_string(
            candidate.get("audit_status"), name=f"{prefix}.audit_status"
        )
        if identifier in prior_set:
            raise ValueError("previously revealed experiments cannot enter replication")
        if (
            candidate.get("outcome_access") != "sealed"
            or candidate.get("result_text_exposed") is not False
        ):
            raise ValueError("every replication candidate must remain outcome sealed")
        identifiers.append(identifier)
        paradigms.append(paradigm)
        fieldings.append(fielding)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("replication candidate IDs must be unique")
    if len(paradigms) != len(set(paradigms)) or len(fieldings) != len(
        set(fieldings)
    ):
        raise ValueError("replication candidates must be independent units")

    stage_gates = payload.get("stage_gates")
    if tuple(stage_gates or ()) != REQUIRED_STAGE_GATES:
        raise ValueError("replication intake stage gates changed")

    compute = payload.get("compute_boundary")
    if not isinstance(compute, Mapping):
        raise ValueError("compute boundary is required")
    authorized_spend = compute.get("authorized_spend_usd")
    if (
        isinstance(authorized_spend, bool)
        or not isinstance(authorized_spend, (int, float))
        or not isfinite(float(authorized_spend))
        or float(authorized_spend) != 0.0
    ):
        raise ValueError("replication intake cannot authorize spending")
    if compute.get("paid_execution_requires_separate_authorization") is not True:
        raise ValueError("paid execution must require separate authorization")

    reveal = payload.get("reveal_boundary")
    if not isinstance(reveal, Mapping):
        raise ValueError("reveal boundary is required")
    if (
        reveal.get("human_outcome_reveal_authorized") is not False
        or reveal.get("participant_row_access_authorized") is not False
    ):
        raise ValueError("replication intake cannot authorize outcome reveal")

    return {
        "candidate_count": len(candidates),
        "minimum_analyzable": minimum,
        "target": target,
        "unique_paradigm_count": len(set(paradigms)),
        "unique_fielding_count": len(set(fieldings)),
        "outcomes_sealed": True,
        "authorized_spend_usd": 0,
        "human_outcome_reveal_authorized": False,
    }
