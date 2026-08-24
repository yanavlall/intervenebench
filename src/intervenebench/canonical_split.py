"""Fail-closed eligibility gate for the Benchmark v1 canonical split.

The canonical split is a scientific evaluation artifact, not an engineering
convenience.  This module deliberately refuses to produce one until enough
independent, source-faithful tasks exist for the declared claim tier.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


CANONICAL_SPLIT_FRACTIONS = {
    "train": 0.65,
    "validation": 0.15,
    "test": 0.20,
}
FULL_TRUST_MIN_INDEPENDENT_EXPERIMENTS = 100
FULL_TRUST_MIN_PARADIGM_GROUPS = 25
FULL_TRUST_MIN_TEST_EXPERIMENTS = 20
EXPLORATORY_MIN_INDEPENDENT_EXPERIMENTS = 60
EXPLORATORY_MIN_PARADIGM_GROUPS = 20


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _independent_count(rows: Iterable[Mapping[str, str]]) -> tuple[int, bool]:
    clusters = {row["fielding_cluster_id"] for row in rows}
    if "" in clusters:
        raise ValueError("runnable tasks must declare a non-empty fielding cluster")
    provisional = any(cluster.startswith("provisional:") for cluster in clusters)
    return len(clusters), provisional


def assess_canonical_split_readiness(rows: list[Mapping[str, str]]) -> dict[str, Any]:
    """Assess only outcome-blind contract and dependency metadata."""

    runnable = [
        row
        for row in rows
        if row["readiness_tier"] == "runnable_sealed_contract"
        and row["outcome_access"] == "sealed"
        and row["canonical_test_status"] == "eligible_after_contract"
    ]
    if any(row["canonical_split"] != "unassigned" for row in rows):
        raise ValueError("readiness input must remain canonical-split unassigned")
    fielding_counts = Counter(row["fielding_cluster_id"] for row in runnable)
    paradigm_count = len({row["paradigm_cluster_id"] for row in runnable})
    independent_count, provisional_clusters_remain = _independent_count(runnable)
    projected_test_count = int(
        independent_count * CANONICAL_SPLIT_FRACTIONS["test"]
    )

    full_gate = {
        "independent_experiments": independent_count
        >= FULL_TRUST_MIN_INDEPENDENT_EXPERIMENTS,
        "paradigm_groups": paradigm_count >= FULL_TRUST_MIN_PARADIGM_GROUPS,
        "projected_test_experiments": projected_test_count
        >= FULL_TRUST_MIN_TEST_EXPERIMENTS,
        "all_fielding_clusters_final": not provisional_clusters_remain,
    }
    exploratory_gate = {
        "independent_experiments": independent_count
        >= EXPLORATORY_MIN_INDEPENDENT_EXPERIMENTS,
        "paradigm_groups": paradigm_count >= EXPLORATORY_MIN_PARADIGM_GROUPS,
        "all_fielding_clusters_final": not provisional_clusters_remain,
    }
    full_ready = all(full_gate.values())
    exploratory_ready = all(exploratory_gate.values())
    return {
        "schema_version": "canonical_split_preflight.v1",
        "status": (
            "ready_full_trust_claim"
            if full_ready
            else "ready_exploratory_claim"
            if exploratory_ready
            else "blocked_insufficient_independent_tasks"
        ),
        "canonical_split_authorized": full_ready or exploratory_ready,
        "canonical_split_fractions": CANONICAL_SPLIT_FRACTIONS,
        "runnable_sealed_task_count": len(runnable),
        "independent_fielding_count": independent_count,
        "paradigm_group_count": paradigm_count,
        "projected_floor_test_experiment_count": projected_test_count,
        "provisional_fielding_clusters_remain": provisional_clusters_remain,
        "full_trust_claim_gate": {
            "minimum_independent_experiments": FULL_TRUST_MIN_INDEPENDENT_EXPERIMENTS,
            "minimum_paradigm_groups": FULL_TRUST_MIN_PARADIGM_GROUPS,
            "minimum_test_experiments": FULL_TRUST_MIN_TEST_EXPERIMENTS,
            "checks": full_gate,
            "passed": full_ready,
        },
        "exploratory_claim_gate": {
            "minimum_independent_experiments": EXPLORATORY_MIN_INDEPENDENT_EXPERIMENTS,
            "minimum_paradigm_groups": EXPLORATORY_MIN_PARADIGM_GROUPS,
            "checks": exploratory_gate,
            "passed": exploratory_ready,
        },
        "fielding_cluster_sizes": dict(sorted(fielding_counts.items())),
        "runnable_candidate_ids": [row["candidate_id"] for row in runnable],
        "forbidden_shortcut": (
            "Do not turn the current runnable set into a nominal canonical split. "
            "More arms, prompts, personas, simulators, seeds, outcomes, or tasks from "
            "one experiment do not increase the independent experiment count."
        ),
        "outcome_access": "sealed",
        "reveal_authorized": False,
    }


def build_canonical_split_preflight(root: Path) -> dict[str, Any]:
    rows = _read_csv(
        root / "data" / "manifests" / "benchmark" / "benchmark_v1_readiness.csv"
    )
    return assess_canonical_split_readiness(rows)


def write_canonical_split_preflight(root: Path) -> Path:
    payload = build_canonical_split_preflight(root)
    path = (
        root
        / "data"
        / "manifests"
        / "splits"
        / "benchmark_v1_canonical_split_preflight.json"
    )
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


if __name__ == "__main__":
    write_canonical_split_preflight(Path(__file__).resolve().parents[2])
