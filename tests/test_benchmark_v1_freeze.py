import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from intervenebench.benchmark_v1 import (
    CANDIDATE_COLUMNS,
    FORBIDDEN_COLUMNS,
    build_candidate_rows,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "manifests"
AUDIT_DIR = MANIFEST_DIR / "audits"
BENCHMARK_DIR = MANIFEST_DIR / "benchmark"
MANIFEST_PATH = BENCHMARK_DIR / "benchmark_v1_candidates.csv"
FREEZE_PATH = BENCHMARK_DIR / "benchmark_v1_freeze.json"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_benchmark_v1_manifest_is_deterministic_and_matches_sources() -> None:
    frozen = _rows(MANIFEST_PATH)

    assert frozen == build_candidate_rows(ROOT)
    assert tuple(frozen[0]) == CANDIDATE_COLUMNS
    assert [row["freeze_order"] for row in frozen] == [
        str(index) for index in range(1, 39)
    ]
    assert len({row["candidate_id"] for row in frozen}) == 38


def test_benchmark_v1_freezes_only_audited_scientific_candidates() -> None:
    frozen = _rows(MANIFEST_PATH)

    assert Counter(row["dataset_stratum"] for row in frozen) == {
        "socsci210_primary": 31,
        "external_archive_v1": 7,
    }
    assert Counter(row["outcome_access"] for row in frozen) == {
        "sealed": 27,
        "development_only_portfolio_reveal": 5,
        "development_only_result_exposure": 5,
        "development_only_validation_reveal": 1,
    }
    assert {
        row["candidate_id"]
        for row in frozen
        if row["canonical_test_status"] != "eligible_after_contract"
    } == {
        "socsci210:jf46x",
        "socsci210:345ms",
        "socsci210:mzm26",
        "socsci210:egmxd",
        "socsci210:d3agv",
        "socsci210:ftwqy",
        "socsci210:gx6hp",
        "socsci210:hgmu6",
        "socsci210:yp736",
        "external_archive_v1:Harbridge-Yong1032",
        "external_archive_v1:AnsonBRIEF60",
        "socsci210:5vm8g",
        "socsci210:xc4yq",
        "socsci210:de5hx",
        "external_archive_v1:turagaS11",
        "external_archive_v1:wallaceS12",
    }
    assert {
        row["candidate_id"]
        for row in frozen
        if row["canonical_test_status"] == "barred_unresolved_response_recode"
    } == {"socsci210:gx6hp", "socsci210:hgmu6"}
    assert {
        row["candidate_id"]
        for row in frozen
        if row["canonical_test_status"] == "barred_structural_ineligibility"
    } == {"socsci210:d3agv", "socsci210:ftwqy", "socsci210:yp736"}
    assert {
        row["candidate_id"]
        for row in frozen
        if row["contract_status"] == "sealed_contract_complete_split_pending"
    } == set()
    assert {
        row["candidate_id"]
        for row in frozen
        if row["contract_status"] == "completed_development_portfolio"
    } == {
        "socsci210:5vm8g",
        "socsci210:xc4yq",
        "socsci210:de5hx",
        "external_archive_v1:turagaS11",
        "external_archive_v1:wallaceS12",
    }


def test_benchmark_v1_scope_freeze_does_not_prematurely_assign_splits() -> None:
    frozen = _rows(MANIFEST_PATH)

    assert {row["canonical_split"] for row in frozen} == {"unassigned"}
    assert all(row["contract_status"] for row in frozen)
    assert all(row["fielding_cluster_status"] for row in frozen)
    klar = next(
        row
        for row in frozen
        if row["candidate_id"] == "external_archive_v1:KlarS44"
    )
    assert klar["fielding_cluster_id"] == "known_shared:tess_8041_040_043"
    assert klar["fielding_cluster_status"] == "known_shared_with_socsci210_xtvu5"


def test_benchmark_v1_manifest_contains_no_result_columns() -> None:
    frozen = _rows(MANIFEST_PATH)

    assert FORBIDDEN_COLUMNS.isdisjoint(frozen[0])
    assert all("result_text" not in row["notes"] for row in frozen)


def test_benchmark_v1_freeze_hashes_and_compute_record_are_current() -> None:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    assert freeze["scope_status"] == "candidate_scope_frozen_split_unassigned"
    assert freeze["candidate_count"] == 38
    assert freeze["canonical_test_eligible_after_contract_count"] == 22
    assert freeze["canonical_test_barred_unresolved_recode_count"] == 2
    assert freeze["canonical_test_barred_structural_ineligibility_count"] == 3
    assert freeze["candidate_manifest_sha256"] == _sha256(MANIFEST_PATH)
    assert freeze["new_participant_outcome_rows_opened_during_freeze"] is False
    assert freeze["aggregate_target_result_exposure_logged_during_freeze"] is True
    assert freeze["paid_simulator_calls_during_freeze"] == 0
    assert freeze["modal_compute_spend_during_freeze_usd"] == 0
    for source_name, expected_hash in freeze["source_registry_sha256"].items():
        assert expected_hash == _sha256(AUDIT_DIR / source_name)
