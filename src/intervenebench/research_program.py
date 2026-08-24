"""Validate the active depth-first research program without opening outcomes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


FORBIDDEN_RESULT_KEYS = {
    "human_mean",
    "human_effect",
    "human_winner",
    "regret",
    "treatment_effect",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_and_validate_research_program(path: Path) -> dict[str, Any]:
    """Return a validated scope manifest containing no target results."""

    program = json.loads(path.read_text(encoding="utf-8"))
    if program.get("schema_version") != "intervenebench.research_program.v1":
        raise ValueError("unsupported research-program schema")
    if program.get("status") != "active_scope_frozen":
        raise ValueError("research program is not active and frozen")

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key.lower() in FORBIDDEN_RESULT_KEYS:
                    raise ValueError(f"result-bearing key is forbidden: {key}")
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(program)
    discovery = program.get("discovery_tasks", [])
    prospective_development = program.get("prospective_development_tasks", [])
    prospective = program.get("prospective_evaluation_tasks", [])
    all_tasks = discovery + prospective_development + prospective
    identifiers = [task["candidate_id"] for task in all_tasks]
    paradigms = [task["paradigm_group"] for task in all_tasks]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate IDs must be unique")
    if len(paradigms) != len(set(paradigms)):
        raise ValueError("development and prospective paradigms must be disjoint")

    targets = program["task_count_target"]
    if not targets["minimum"] <= len(all_tasks) <= targets["maximum_without_scope_review"]:
        raise ValueError("task count falls outside frozen scope")
    if len(prospective) < 6:
        raise ValueError("at least six prospective experiments are required")
    if any(task["outcome_access"] == "sealed" for task in discovery):
        raise ValueError("sealed tasks cannot enter discovery")
    if any(
        task["outcome_access"] != "sealed" for task in prospective_development
    ):
        raise ValueError("prospective development outcomes must remain sealed")
    if any(task["outcome_access"] != "sealed" for task in prospective):
        raise ValueError("every prospective outcome must remain sealed")
    if program["compute_boundary"]["authorized_spend_usd"] != 0:
        raise ValueError("this scope manifest cannot authorize spending")
    if not program["compute_boundary"][
        "paid_or_modal_execution_requires_separate_authorization"
    ]:
        raise ValueError("paid execution must require separate authorization")
    return program


def verify_research_program(root: Path) -> dict[str, Any]:
    path = root / "data/manifests/research/depth_first_v1.json"
    program = load_and_validate_research_program(path)
    return {
        "program_id": program["program_id"],
        "manifest_sha256": _sha256(path),
        "discovery_experiments": len(program["discovery_tasks"]),
        "prospective_development_experiments": len(
            program["prospective_development_tasks"]
        ),
        "prospective_experiments": len(program["prospective_evaluation_tasks"]),
        "total_experiments": (
            len(program["discovery_tasks"])
            + len(program["prospective_development_tasks"])
            + len(program["prospective_evaluation_tasks"])
        ),
        "prospective_outcomes_sealed": True,
        "authorized_spend_usd": 0,
    }
