from __future__ import annotations

import csv
import json
from pathlib import Path

from intervenebench.canonical_split import (
    assess_canonical_split_readiness,
    build_canonical_split_preflight,
)


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_PATH = (
    ROOT
    / "data"
    / "manifests"
    / "splits"
    / "benchmark_v1_canonical_split_preflight.json"
)


def test_current_canonical_split_fails_closed_at_the_declared_scale_gates() -> None:
    payload = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    assert payload == build_canonical_split_preflight(ROOT)
    assert payload["status"] == "blocked_insufficient_independent_tasks"
    assert payload["canonical_split_authorized"] is False
    assert payload["runnable_sealed_task_count"] == 6
    assert payload["independent_fielding_count"] == 6
    assert payload["paradigm_group_count"] == 6
    assert payload["projected_floor_test_experiment_count"] == 1
    assert payload["provisional_fielding_clusters_remain"] is True
    assert payload["full_trust_claim_gate"]["passed"] is False
    assert payload["exploratory_claim_gate"]["passed"] is False
    assert payload["reveal_authorized"] is False


def test_preflight_deduplicates_shared_fieldings_and_requires_final_clusters() -> None:
    base = {
        "readiness_tier": "runnable_sealed_contract",
        "outcome_access": "sealed",
        "canonical_test_status": "eligible_after_contract",
        "canonical_split": "unassigned",
    }
    rows = []
    for index in range(100):
        rows.append(
            {
                **base,
                "candidate_id": f"candidate-{index}",
                "fielding_cluster_id": (
                    "known_shared:fielding-0"
                    if index in {0, 1}
                    else f"known_fielding:fielding-{index}"
                ),
                "paradigm_cluster_id": f"paradigm-{index % 25}",
            }
        )
    result = assess_canonical_split_readiness(rows)
    assert result["runnable_sealed_task_count"] == 100
    assert result["independent_fielding_count"] == 99
    assert result["full_trust_claim_gate"]["passed"] is False
    assert result["exploratory_claim_gate"]["passed"] is True
    assert result["canonical_split_authorized"] is True

    rows.append(
        {
            **base,
            "candidate_id": "candidate-100",
            "fielding_cluster_id": "known_fielding:fielding-100",
            "paradigm_cluster_id": "paradigm-0",
        }
    )
    result = assess_canonical_split_readiness(rows)
    assert result["independent_fielding_count"] == 100
    assert result["full_trust_claim_gate"]["passed"] is True
    assert result["canonical_split_authorized"] is True
