"""Build and verify the outcome-blind Benchmark v1 scope freeze.

This module intentionally freezes only the candidate universe.  It does not
assign canonical splits, open outcomes, or claim that unresolved task contracts
are ready for scoring.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


FREEZE_ID = "benchmark-v1-scope-20260812"
FREEZE_DATE = "2026-08-12"
PORTFOLIO_DEVELOPMENT_IDS = frozenset(
    {"5vm8g", "xc4yq", "de5hx", "turagaS11", "wallaceS12"}
)
FORBIDDEN_COLUMNS = {
    "response",
    "reasoning",
    "arm_mean",
    "treatment_effect",
    "significance",
    "winner",
    "regret",
}

CANDIDATE_COLUMNS = (
    "freeze_order",
    "candidate_id",
    "dataset_stratum",
    "source_registry",
    "source_record_id",
    "paradigm_group",
    "primary_track",
    "contract_status",
    "outcome_access",
    "canonical_test_status",
    "canonical_split",
    "fielding_cluster_id",
    "fielding_cluster_status",
    "selection_hash",
    "notes",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_hash(candidate_id: str) -> str:
    payload = f"{FREEZE_ID}:{candidate_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def _phase2_contract_status(mapping_status: str) -> str:
    statuses = {
        "pending_primary_outcome_mapping": "primary_outcome_mapping_pending",
        "source_verified_continuous_contract": (
            "implemented_continuous_contract_split_pending"
        ),
        "source_verified_primary_outcome_missing_from_socsci": (
            "source_data_ingestion_blocked"
        ),
        "source_verified_primary_outcome_utility_ambiguous": (
            "utility_contract_pending"
        ),
        "source_verified_primary_composite": "composite_estimator_pending",
        "source_verified_pending_recode_provenance": (
            "response_recode_provenance_blocked"
        ),
        "source_verified_primary_outcome_action_subset_sequence_pending": (
            "sealed_human_mapping_sequence_simulator_pending"
        ),
        "source_verified_primary_outcome_action_subset_sequence_complete": (
            "sealed_scoring_contract_complete_split_pending"
        ),
        "source_verified_primary_outcome_action_subset_sequence_assets_pending": (
            "sealed_human_mapping_sequence_assets_pending"
        ),
        "source_verified_primary_outcome_source_data_multimodal_contract_complete": (
            "sealed_scoring_contract_complete_split_pending"
        ),
        "source_verified_primary_outcome_video_assets_missing": (
            "multimodal_source_assets_missing"
        ),
        "source_verified_primary_categorical_choice_multimodal_contract_complete": (
            "sealed_scoring_contract_complete_split_pending"
        ),
    }
    try:
        return statuses[mapping_status]
    except KeyError as error:
        raise ValueError(
            f"Unsupported Phase 2 mapping status: {mapping_status!r}"
        ) from error


def _external_contract_status(mapping_status: str) -> str:
    statuses = {
        "source_verified_scoring_contract_complete": (
            "sealed_scoring_contract_complete_split_pending"
        ),
        "source_verified": "scoring_contract_pending",
        "source_verified_pending_order_averaging_contract": (
            "order_averaging_contract_pending"
        ),
        "instrument_verified_source_data_mapping_pending": (
            "source_data_mapping_pending"
        ),
        "source_verified_weight_and_order_contract_pending": (
            "weight_and_order_contract_pending"
        ),
    }
    try:
        return statuses[mapping_status]
    except KeyError as error:
        raise ValueError(
            f"Unsupported external mapping status: {mapping_status!r}"
        ) from error


def _access_fields(outcome_access: str) -> tuple[str, str]:
    if outcome_access == "sealed":
        return "sealed", "eligible_after_contract"
    if outcome_access == "result_text_exposed_non_test":
        return "development_only_result_exposure", "barred_result_exposure"
    raise ValueError(f"Unsupported outcome access state: {outcome_access!r}")


def build_candidate_rows(root: Path) -> list[dict[str, str]]:
    manifest_dir = root / "data" / "manifests"
    audit_dir = manifest_dir / "audits"
    contract_dir = manifest_dir / "contracts"
    phase1 = _read_csv(audit_dir / "phase1_candidate_registry.csv")
    phase2 = _read_csv(audit_dir / "phase2_candidate_registry.csv")
    external = _read_csv(audit_dir / "external_source_audit_registry.csv")
    recode_adjudications = {
        row["experiment_id"]: row
        for row in _read_csv(audit_dir / "response_recode_adjudications.csv")
    }
    cross_audit_adjudications = {
        row["experiment_id"]: row
        for row in _read_csv(audit_dir / "cross_audit_adjudications.csv")
    }
    completed_contracts = {
        path.name.split("_", 1)[0]: json.loads(path.read_text(encoding="utf-8"))
        for path in contract_dir.glob("*_decision_task_candidate.json")
    }
    portfolio_reveal_path = (
        manifest_dir / "benchmark" / "portfolio_pilot_development_reveal.json"
    )
    portfolio_development_ids: frozenset[str] = frozenset()
    if portfolio_reveal_path.exists():
        portfolio_reveal = json.loads(portfolio_reveal_path.read_text(encoding="utf-8"))
        if (
            portfolio_reveal.get("status") != "development_reveal_authorized"
            or portfolio_reveal.get("permanent_role")
            != "development_only_portfolio_reveal"
            or portfolio_reveal.get("canonical_test_eligible") is not False
        ):
            raise ValueError("malformed portfolio development-reveal decision")
        portfolio_development_ids = frozenset(portfolio_reveal["experiment_ids"])
        if portfolio_development_ids != PORTFOLIO_DEVELOPMENT_IDS:
            raise ValueError("portfolio development-reveal set changed")

    candidates: list[dict[str, str]] = []

    for row in phase1:
        if row["phase1_eligible"] != "true":
            continue
        experiment_id = row["experiment_id"]
        if experiment_id == "jf46x":
            outcome_access = "development_only_validation_reveal"
            canonical_test_status = "barred_validation_reveal"
            contract_status = "completed_validation_smoke"
        elif experiment_id in portfolio_development_ids:
            outcome_access = "development_only_portfolio_reveal"
            canonical_test_status = "barred_portfolio_development_reveal"
            contract_status = "completed_development_portfolio"
        else:
            outcome_access = "sealed"
            canonical_test_status = "eligible_after_contract"
            contract_status = (
                "sealed_contract_complete_split_pending"
                if experiment_id in completed_contracts
                else "source_verified_task_pending_contract_artifact"
            )
        candidates.append(
            {
                "candidate_id": f"socsci210:{experiment_id}",
                "dataset_stratum": "socsci210_primary",
                "source_registry": "phase1_candidate_registry.csv",
                "source_record_id": experiment_id,
                "paradigm_group": row["paradigm_group"],
                "primary_track": "core_simple",
                "contract_status": contract_status,
                "outcome_access": outcome_access,
                "canonical_test_status": canonical_test_status,
                "canonical_split": "unassigned",
                "fielding_cluster_id": f"provisional:socsci210:{experiment_id}",
                "fielding_cluster_status": "provisional_singleton_pending_audit",
                "notes": (
                    "Previously source-audited Phase 1 candidate. The smoke split is "
                    "not the canonical benchmark split."
                ),
            }
        )

    for row in phase2:
        if row["scientific_status"] not in {
            "eligible_core",
            "eligible_extension",
        }:
            continue
        experiment_id = row["experiment_id"]
        outcome_access, canonical_test_status = _access_fields(
            row["outcome_access"]
        )
        contract_status = _phase2_contract_status(row["outcome_mapping_status"])
        contract = completed_contracts.get(experiment_id)
        if contract is not None:
            mapping_status = contract.get("source_data_mapping_status", "")
            if mapping_status == "complete_outcome_blind_schema_and_design_mapping":
                contract_status = "sealed_scoring_contract_complete_split_pending"
            elif mapping_status == (
                "complete_outcome_blind_human_mapping_sequence_simulator_pending"
            ):
                contract_status = "sealed_human_mapping_sequence_simulator_pending"
            elif mapping_status == (
                "complete_outcome_blind_human_mapping_sequence_assets_pending"
            ):
                contract_status = "sealed_human_mapping_sequence_assets_pending"
            elif mapping_status == (
                "complete_outcome_blind_schema_design_and_sequence_mapping"
            ):
                contract_status = "sealed_scoring_contract_complete_split_pending"
            elif mapping_status == (
                "complete_outcome_blind_schema_design_multimodal_and_nuisance_mapping"
            ):
                contract_status = "sealed_scoring_contract_complete_split_pending"
            elif mapping_status == (
                "complete_outcome_blind_schema_design_categorical_multimodal_mapping"
            ):
                contract_status = "sealed_scoring_contract_complete_split_pending"
        if experiment_id in portfolio_development_ids:
            outcome_access = "development_only_portfolio_reveal"
            canonical_test_status = "barred_portfolio_development_reveal"
            contract_status = "completed_development_portfolio"
        recode_adjudication = recode_adjudications.get(experiment_id)
        cross_audit_adjudication = cross_audit_adjudications.get(experiment_id)
        if recode_adjudication is not None and cross_audit_adjudication is not None:
            raise ValueError(
                f"Conflicting terminal adjudications for {experiment_id}"
            )
        if recode_adjudication is not None:
            if recode_adjudication["outcome_access"] != outcome_access:
                raise ValueError(
                    f"Outcome-access mismatch in recode adjudication: {experiment_id}"
                )
            contract_status = recode_adjudication["adjudication_status"]
            canonical_test_status = recode_adjudication["canonical_test_status"]
        if cross_audit_adjudication is not None:
            if cross_audit_adjudication["stable_source_id"] != experiment_id:
                raise ValueError(
                    f"Stable-source mismatch in cross audit: {experiment_id}"
                )
            if cross_audit_adjudication["outcome_access"] != outcome_access:
                raise ValueError(
                    f"Outcome-access mismatch in cross audit: {experiment_id}"
                )
            contract_status = cross_audit_adjudication["adjudication_status"]
            canonical_test_status = cross_audit_adjudication[
                "canonical_test_status"
            ]
        candidates.append(
            {
                "candidate_id": f"socsci210:{experiment_id}",
                "dataset_stratum": "socsci210_primary",
                "source_registry": "phase2_candidate_registry.csv",
                "source_record_id": experiment_id,
                "paradigm_group": row["paradigm_group"],
                "primary_track": row["primary_track"],
                "contract_status": contract_status,
                "outcome_access": outcome_access,
                "canonical_test_status": canonical_test_status,
                "canonical_split": "unassigned",
                "fielding_cluster_id": f"provisional:socsci210:{experiment_id}",
                "fielding_cluster_status": "provisional_singleton_pending_audit",
                "notes": row["notes"],
            }
        )
        if recode_adjudication is not None:
            candidates[-1]["notes"] = (
                row["notes"]
                + " "
                + recode_adjudication["terminal_reason"]
                + " The task is barred from Benchmark v1 scoring unless the documented "
                "reopen condition is satisfied before the canonical split."
            )
        if cross_audit_adjudication is not None:
            candidates[-1]["notes"] = (
                row["notes"]
                + " Cross-audit source reconciliation: "
                + cross_audit_adjudication["evidence_note"]
                + " The module remains in the frozen candidate census but is barred "
                "from Benchmark v1 scoring."
            )

    for row in external:
        if row["scientific_status"] != "eligible_extension":
            continue
        archive_id = row["archive_study_id"]
        outcome_access, canonical_test_status = _access_fields(
            row["outcome_access"]
        )
        if archive_id == "KlarS44":
            fielding_cluster_id = "known_shared:tess_8041_040_043"
            fielding_cluster_status = "known_shared_with_socsci210_xtvu5"
        else:
            fielding_cluster_id = f"provisional:external:{archive_id}"
            fielding_cluster_status = "provisional_singleton_pending_audit"
        contract_status = _external_contract_status(row["outcome_mapping_status"])
        contract = completed_contracts.get(archive_id)
        if contract is not None:
            mapping_status = contract.get("source_data_mapping_status", "")
            if mapping_status == "complete_outcome_blind_schema_and_design_mapping":
                contract_status = "sealed_scoring_contract_complete_split_pending"
            elif mapping_status == (
                "complete_outcome_blind_schema_design_and_sequence_mapping"
            ):
                contract_status = "sealed_scoring_contract_complete_split_pending"
            elif mapping_status == (
                "complete_outcome_blind_human_mapping_sequence_simulator_pending"
            ):
                contract_status = "sealed_human_mapping_sequence_simulator_pending"
            else:
                contract_status = "sealed_simulator_contract_source_mapping_pending"
        if archive_id in portfolio_development_ids:
            outcome_access = "development_only_portfolio_reveal"
            canonical_test_status = "barred_portfolio_development_reveal"
            contract_status = "completed_development_portfolio"
        candidates.append(
            {
                "candidate_id": f"external_archive_v1:{archive_id}",
                "dataset_stratum": "external_archive_v1",
                "source_registry": "external_source_audit_registry.csv",
                "source_record_id": archive_id,
                "paradigm_group": row["paradigm_group"],
                "primary_track": row["primary_track"],
                "contract_status": contract_status,
                "outcome_access": outcome_access,
                "canonical_test_status": canonical_test_status,
                "canonical_split": "unassigned",
                "fielding_cluster_id": fielding_cluster_id,
                "fielding_cluster_status": fielding_cluster_status,
                "notes": row["notes"],
            }
        )

    for freeze_order, row in enumerate(candidates, start=1):
        row["freeze_order"] = str(freeze_order)
        row["selection_hash"] = _candidate_hash(row["candidate_id"])

    validate_candidate_rows(candidates)
    return [{column: row[column] for column in CANDIDATE_COLUMNS} for row in candidates]


def validate_candidate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Benchmark v1 cannot be frozen with no candidates")
    if FORBIDDEN_COLUMNS.intersection(rows[0]):
        raise ValueError("Benchmark manifest contains forbidden result columns")
    ids = [row["candidate_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Benchmark candidate IDs must be unique")
    expected_orders = [str(index) for index in range(1, len(rows) + 1)]
    if [row["freeze_order"] for row in rows] != expected_orders:
        raise ValueError("Benchmark freeze order must be contiguous")
    if {row["canonical_split"] for row in rows} != {"unassigned"}:
        raise ValueError("Candidate-scope freeze must not assign canonical splits")
    for row in rows:
        if row["selection_hash"] != _candidate_hash(row["candidate_id"]):
            raise ValueError(f"Invalid selection hash for {row['candidate_id']}")
        if row["canonical_test_status"] == "eligible_after_contract":
            if row["outcome_access"] != "sealed":
                raise ValueError(
                    "Only sealed candidates may remain eligible for canonical test"
                )
        if row["canonical_test_status"] == "barred_unresolved_response_recode":
            if row["contract_status"] != "unscorable_recode_provenance_unresolved":
                raise ValueError("recode-barred tasks must carry the terminal blocker")
        if row["canonical_test_status"] == "barred_structural_ineligibility":
            if row["contract_status"] != "structurally_ineligible_cross_audit":
                raise ValueError(
                    "cross-audit barred tasks must carry the terminal blocker"
                )
        if row["canonical_test_status"] == "barred_portfolio_development_reveal":
            if (
                row["outcome_access"] != "development_only_portfolio_reveal"
                or row["contract_status"] != "completed_development_portfolio"
            ):
                raise ValueError(
                    "portfolio-revealed tasks must be permanently development-only"
                )


def write_freeze(root: Path) -> tuple[Path, Path]:
    manifest_dir = root / "data" / "manifests"
    audit_dir = manifest_dir / "audits"
    benchmark_dir = manifest_dir / "benchmark"
    rows = build_candidate_rows(root)
    candidate_path = benchmark_dir / "benchmark_v1_candidates.csv"
    with candidate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    source_names = (
        "phase1_candidate_registry.csv",
        "phase2_candidate_registry.csv",
        "external_source_audit_registry.csv",
        "response_recode_adjudications.csv",
        "cross_audit_adjudications.csv",
    )
    source_hashes = {
        source_name: _sha256(audit_dir / source_name)
        for source_name in source_names
    }
    access_counts = Counter(row["outcome_access"] for row in rows)
    stratum_counts = Counter(row["dataset_stratum"] for row in rows)
    freeze: dict[str, Any] = {
        "freeze_id": FREEZE_ID,
        "freeze_date": FREEZE_DATE,
        "scope_status": "candidate_scope_frozen_split_unassigned",
        "candidate_manifest": candidate_path.name,
        "candidate_manifest_sha256": _sha256(candidate_path),
        "source_registry_sha256": source_hashes,
        "candidate_count": len(rows),
        "dataset_stratum_counts": dict(sorted(stratum_counts.items())),
        "outcome_access_counts": dict(sorted(access_counts.items())),
        "canonical_test_eligible_after_contract_count": sum(
            row["canonical_test_status"] == "eligible_after_contract"
            for row in rows
        ),
        "canonical_test_barred_unresolved_recode_count": sum(
            row["canonical_test_status"] == "barred_unresolved_response_recode"
            for row in rows
        ),
        "canonical_test_barred_structural_ineligibility_count": sum(
            row["canonical_test_status"] == "barred_structural_ineligibility"
            for row in rows
        ),
        "canonical_split": "unassigned",
        "new_participant_outcome_rows_opened_during_freeze": False,
        "aggregate_target_result_exposure_logged_during_freeze": True,
        "paid_simulator_calls_during_freeze": 0,
        "modal_compute_spend_during_freeze_usd": 0,
        "notes": (
            "This freezes only the already audited Benchmark v1 candidate scope. "
            "It is not the canonical task registry or split. Contract, dependency, "
            "and fielding-cluster blockers must be resolved before splitting. "
            "A workbook sheet-selection failure exposed aggregate egmxd target "
            "frequencies; no participant row was opened, and egmxd is barred from "
            "canonical test evaluation."
        ),
    }
    freeze_path = benchmark_dir / "benchmark_v1_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return candidate_path, freeze_path


if __name__ == "__main__":
    write_freeze(Path(__file__).resolve().parents[2])
