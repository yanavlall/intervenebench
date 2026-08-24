"""Build the outcome-blind Benchmark v1 dependency and readiness map.

The map orders engineering work; it does not inspect outcomes, assign splits,
or upgrade a source audit.  Paradigm consolidation is deliberately conservative
so related experiments cannot be separated merely because their earlier labels
were more granular.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .benchmark_v1 import build_candidate_rows


READINESS_COLUMNS = (
    "freeze_order",
    "candidate_id",
    "canonical_test_status",
    "outcome_access",
    "fielding_cluster_id",
    "paradigm_cluster_id",
    "dependency_status",
    "readiness_tier",
    "work_priority",
    "contract_status",
    "next_blocker",
    "canonical_split",
)

PARADIGM_CLUSTER_OVERRIDES = {
    "socsci210:d3agv": "demographic_change_status_threat",
    "socsci210:345ms": "demographic_change_status_threat",
    "socsci210:kryns": "demographic_change_status_threat",
    "socsci210:pb2rr": "demographic_change_status_threat",
    "socsci210:e2pyb": "racial_health_inequality_communication",
    "socsci210:hgmu6": "racial_health_inequality_communication",
    "socsci210:nj5dx": "inequality_and_power_framing",
    "socsci210:6wbd7": "inequality_and_power_framing",
}

# These identifiers are frozen from source/programming records, not inferred
# from participant overlap or outcome behavior.  Singleton IDs remain explicit
# until an exact co-fielding match is documented.
FIELDING_CLUSTER_OVERRIDES = {
    "socsci210:4w9pz": "known_fielding:tess_8041_teen2_065_070",
    "socsci210:z358z": "known_fielding:tess3_173_175_176",
    "external_archive_v1:KlarS44": "known_shared:tess_8041_040_043",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _readiness(row: dict[str, str]) -> tuple[str, str, str]:
    canonical_status = row["canonical_test_status"]
    contract_status = row["contract_status"]
    track = row["primary_track"]

    if row["outcome_access"] != "sealed":
        return (
            "development_only",
            "not_canonical_test_work",
            "development-only analysis may proceed without changing test claims",
        )
    if canonical_status.startswith("barred_"):
        return "terminally_barred", "not_scheduled", canonical_status
    if contract_status in {
        "sealed_contract_complete_split_pending",
        "sealed_scoring_contract_complete_split_pending",
        "implemented_continuous_contract_split_pending",
    }:
        blocker = (
            "pooled normalized-regret scale and estimator-tier freeze"
            if track == "extension_continuous"
            else "dependency review and canonical split"
        )
        return "runnable_sealed_contract", "1", blocker
    if contract_status == "sealed_simulator_contract_source_mapping_pending":
        return (
            "simulator_contract_scoring_mapping_pending",
            "2",
            "source variable, missing-code, weight, allocation, and support mapping",
        )
    if contract_status == "sealed_human_mapping_sequence_simulator_pending":
        return (
            "sequence_simulator_adapter_pending",
            "3",
            "faithful randomized survey-sequence adapter and blinded bundle",
        )
    if contract_status == "sealed_human_mapping_sequence_assets_pending":
        return (
            "sequence_source_assets_pending",
            "3",
            "exact co-fielded module, visual assets, order metadata, and sequence adapter",
        )
    if track in {
        "extension_factorial_action_subset",
        "extension_factorial_repeated_message",
        "extension_cofielded_message_policy",
        "extension_utility_sensitivity",
    }:
        return (
            "text_or_factorial_contract_next",
            "3",
            contract_status,
        )
    if track in {"extension_continuous", "extension_factorial"}:
        return "estimator_contract_pending", "4", contract_status
    if track == "extension_source_data":
        return "source_data_ingestion_pending", "5", contract_status
    if track == "extension_multimodal":
        return "multimodal_contract_later", "6", contract_status
    if track == "extension_interactive":
        return "interactive_or_longitudinal_later", "7", contract_status
    return "source_contract_pending", "4", contract_status


def build_readiness_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for candidate in build_candidate_rows(root):
        candidate_id = candidate["candidate_id"]
        readiness_tier, work_priority, next_blocker = _readiness(candidate)
        paradigm_cluster_id = PARADIGM_CLUSTER_OVERRIDES.get(
            candidate_id, candidate["paradigm_group"]
        )
        fielding_cluster_id = FIELDING_CLUSTER_OVERRIDES.get(
            candidate_id, candidate["fielding_cluster_id"]
        )
        dependency_status = (
            "known_shared_fielding"
            if fielding_cluster_id.startswith("known_shared:")
            else "known_cofielding_sequence_dependency"
            if fielding_cluster_id.startswith("known_fielding:")
            else "no_known_shared_fielding_match_in_audited_scope"
        )
        rows.append(
            {
                "freeze_order": candidate["freeze_order"],
                "candidate_id": candidate_id,
                "canonical_test_status": candidate["canonical_test_status"],
                "outcome_access": candidate["outcome_access"],
                "fielding_cluster_id": fielding_cluster_id,
                "paradigm_cluster_id": paradigm_cluster_id,
                "dependency_status": dependency_status,
                "readiness_tier": readiness_tier,
                "work_priority": work_priority,
                "contract_status": candidate["contract_status"],
                "next_blocker": next_blocker,
                "canonical_split": "unassigned",
            }
        )
    return rows


def write_readiness(root: Path) -> tuple[Path, Path]:
    benchmark_dir = root / "data" / "manifests" / "benchmark"
    rows = build_readiness_rows(root)
    path = benchmark_dir / "benchmark_v1_readiness.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=READINESS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    status_counts = Counter(row["readiness_tier"] for row in rows)
    cluster_sizes = Counter(row["paradigm_cluster_id"] for row in rows)
    freeze: dict[str, Any] = {
        "freeze_id": "benchmark-v1-readiness-20260813",
        "freeze_date": "2026-08-13",
        "candidate_count": len(rows),
        "canonical_split": "unassigned",
        "readiness_manifest": path.name,
        "readiness_manifest_sha256": _sha256(path),
        "readiness_tier_counts": dict(sorted(status_counts.items())),
        "paradigm_cluster_count": len(cluster_sizes),
        "multi_candidate_paradigm_clusters": {
            cluster_id: size
            for cluster_id, size in sorted(cluster_sizes.items())
            if size > 1
        },
        "new_participant_outcome_rows_opened": False,
        "aggregate_target_result_exposure_logged": True,
        "paid_simulator_calls": 0,
        "modal_compute_spend_usd": 0,
        "notes": (
            "Outcome-blind engineering readiness and conservative paradigm "
            "dependency map. It does not assign canonical splits or certify "
            "unresolved source-data mappings. The 4w9pz human mapping is complete "
            "but its exact co-fielded sequence assets are missing. Aggregate egmxd "
            "target frequencies were accidentally exposed during workbook inspection; "
            "no participant row was opened, and the candidate is development-only."
        ),
    }
    freeze_path = benchmark_dir / "benchmark_v1_readiness.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path, freeze_path


if __name__ == "__main__":
    write_readiness(Path(__file__).resolve().parents[2])
