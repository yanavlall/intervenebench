from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from intervenebench.readiness import READINESS_COLUMNS, build_readiness_rows


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "data" / "manifests" / "benchmark"
READINESS_PATH = BENCHMARK_DIR / "benchmark_v1_readiness.csv"
FREEZE_PATH = BENCHMARK_DIR / "benchmark_v1_readiness.json"


def _rows() -> list[dict[str, str]]:
    with READINESS_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_readiness_map_is_deterministic_outcome_blind_and_split_free() -> None:
    rows = _rows()
    assert rows == build_readiness_rows(ROOT)
    assert tuple(rows[0]) == READINESS_COLUMNS
    assert len(rows) == 38
    assert {row["canonical_split"] for row in rows} == {"unassigned"}
    assert all("winner" not in row and "regret" not in row for row in rows)


def test_readiness_map_consolidates_known_paradigm_dependencies() -> None:
    rows = _rows()
    groups = Counter(row["paradigm_cluster_id"] for row in rows)
    assert groups["demographic_change_status_threat"] == 4
    assert groups["racial_health_inequality_communication"] == 2
    assert groups["inequality_and_power_framing"] == 2
    klar = next(
        row for row in rows if row["candidate_id"] == "external_archive_v1:KlarS44"
    )
    assert klar["dependency_status"] == "known_shared_fielding"


def test_readiness_prioritizes_only_supported_sealed_contracts() -> None:
    rows = _rows()
    runnable = {
        row["candidate_id"]
        for row in rows
        if row["readiness_tier"] == "runnable_sealed_contract"
    }
    mapping_pending = {
        row["candidate_id"]
        for row in rows
        if row["readiness_tier"]
        == "simulator_contract_scoring_mapping_pending"
    }
    assert runnable == {
        "socsci210:tcg8p",
        "socsci210:pb2rr",
        "external_archive_v1:Blair1131",
        "external_archive_v1:ShannonS2",
        "external_archive_v1:KlarS44",
        "socsci210:z358z",
    }
    assert mapping_pending == set()


def test_4w9pz_human_mapping_is_frozen_but_sequence_assets_are_not_faked() -> None:
    rows = _rows()
    hecht = next(
        row for row in rows if row["candidate_id"] == "socsci210:4w9pz"
    )
    assert hecht["fielding_cluster_id"] == "known_fielding:tess_8041_teen2_065_070"
    assert hecht["dependency_status"] == "known_cofielding_sequence_dependency"
    assert hecht["readiness_tier"] == "sequence_source_assets_pending"
    assert hecht["contract_status"] == "sealed_human_mapping_sequence_assets_pending"


def test_sequence_deferred_tier_is_empty_after_source_programmed_adapters() -> None:
    rows = _rows()
    sequence_pending = {
        row["candidate_id"]
        for row in rows
        if row["readiness_tier"] == "sequence_simulator_adapter_pending"
    }
    assert sequence_pending == set()


def test_readiness_freeze_hash_and_compute_boundary_are_current() -> None:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    assert freeze["candidate_count"] == 38
    assert freeze["canonical_split"] == "unassigned"
    assert freeze["readiness_manifest_sha256"] == hashlib.sha256(
        READINESS_PATH.read_bytes()
    ).hexdigest()
    assert freeze["new_participant_outcome_rows_opened"] is False
    assert freeze["aggregate_target_result_exposure_logged"] is True
    assert freeze["paid_simulator_calls"] == 0
    assert freeze["modal_compute_spend_usd"] == 0
