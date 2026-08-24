from __future__ import annotations

import csv
import hashlib
import json
import re
import html
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "data"
    / "manifests"
    / "audits"
    / "tess_external_root_universe_v1.json"
)
SOURCE_REGISTRY = (
    ROOT
    / "data"
    / "manifests"
    / "audits"
    / "tess_external_root_audit_registry_v1.csv"
)
EXPOSURES = (
    ROOT
    / "data"
    / "manifests"
    / "audits"
    / "independent_replication_exposure_log_v1.json"
)
MANIFEST = (
    ROOT
    / "data"
    / "manifests"
    / "audits"
    / "tess_targeted_intervention_universe_v1.json"
)
CLOSURE = (
    ROOT
    / "data"
    / "manifests"
    / "audits"
    / "tess_external_root_random_lane_closure_v1.json"
)
REGISTRY = (
    ROOT
    / "data"
    / "manifests"
    / "audits"
    / "tess_targeted_intervention_audit_registry_v1.csv"
)
TARGETED_CLOSURE = (
    ROOT
    / "data"
    / "manifests"
    / "audits"
    / "tess_targeted_intervention_lane_closure_v1.json"
)


def _normalized(value: str) -> str:
    decoded = html.unescape(value)
    normalized = unicodedata.normalize("NFKC", decoded).casefold()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", normalized).split())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_random_lane_closure_binds_the_frozen_checkpoint() -> None:
    closure = json.loads(CLOSURE.read_text(encoding="utf-8"))
    assert closure["status"] == "closed_low_yield_at_frozen_checkpoint"
    assert closure["checkpoint_order"] == 20
    assert closure["clean_pass_count"] == 0
    assert closure["conditional_count"] == 3
    assert closure["terminal_or_blocked_count"] == 17
    assert closure["stop_rule_triggered"] is True
    assert closure["source_universe_sha256"] == _sha256(SOURCE)
    assert closure["audit_registry_sha256"] == _sha256(SOURCE_REGISTRY)


def test_targeted_universe_is_title_only_hash_ordered_and_deduplicated() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    source_by_id = {
        row["osf_node_id"]: row for row in source["candidate_universe"]
    }
    with SOURCE_REGISTRY.open(newline="", encoding="utf-8") as handle:
        audited = {row["osf_node_id"] for row in csv.DictReader(handle)}
    exposed: set[str] = set()
    for incident in json.loads(EXPOSURES.read_text(encoding="utf-8"))["incidents"]:
        exposed.add(incident["candidate_id"])
        if incident.get("source_lookup_alias"):
            exposed.add(incident["source_lookup_alias"])
    rows = manifest["candidate_universe"]
    assert rows
    assert [row["freeze_order"] for row in rows] == list(range(1, len(rows) + 1))
    assert {row["osf_node_id"] for row in rows}.isdisjoint(audited | exposed)
    expected = [
        source_by_id[row["osf_node_id"]]["selection_sha256"]
        for row in rows
    ]
    assert [row["selection_sha256"] for row in rows] == expected == sorted(expected)
    for row in rows:
        source_row = source_by_id[row["osf_node_id"]]
        assert row["title"] == source_row["title"]
        title = _normalized(row["title"])
        assert any(
            re.search(pattern, title)
            for pattern in manifest["selection_rule"]["include_patterns"]
        )
        assert not any(
            re.search(pattern, title)
            for pattern in manifest["selection_rule"]["exclude_patterns"]
        )


def test_targeted_audit_batch_and_authority_are_frozen() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = manifest["candidate_universe"]
    protocol = manifest["ordered_audit_protocol"]
    assert protocol["batch_candidate_ids"] == [
        row["osf_node_id"] for row in rows[:12]
    ]
    assert protocol["audit_contiguously_in_freeze_order"] is True
    assert protocol["no_selective_skipping"] is True
    assert protocol["block_size"] == 3
    assert protocol["checkpoint_after_order_6"]["minimum_scientific_survivors"] == 1
    assert protocol["checkpoint_after_order_9"]["minimum_scientific_survivors"] == 2
    assert protocol["hard_stop_after_order_12"] is True
    assert manifest["authority"] == {
        "authorized_spend_usd": 0,
        "model_calls_authorized": False,
        "human_outcome_reveal_authorized": False,
        "participant_row_access_authorized": False,
        "public_design_source_acquisition_authorized": True,
        "mixed_archive_member_listing_authorized": True,
        "participant_member_extraction_authorized": False,
    }


def test_targeted_manifest_binds_all_selection_inputs() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["inputs"] == {
        "source_universe_path": str(SOURCE.relative_to(ROOT)),
        "source_universe_sha256": _sha256(SOURCE),
        "source_registry_path": str(SOURCE_REGISTRY.relative_to(ROOT)),
        "source_registry_sha256": _sha256(SOURCE_REGISTRY),
        "exposure_log_path": str(EXPOSURES.relative_to(ROOT)),
        "exposure_log_sha256": _sha256(EXPOSURES),
    }


def test_targeted_registry_is_empty_or_a_contiguous_frozen_prefix() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    frozen = {
        row["freeze_order"]: row for row in manifest["candidate_universe"]
    }
    with REGISTRY.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["freeze_order"]) for row in rows] == list(
        range(1, len(rows) + 1)
    )
    for row in rows:
        source = frozen[int(row["freeze_order"])]
        assert row["osf_node_id"] == source["osf_node_id"]
        assert _normalized(row["study_title"]) == _normalized(source["title"])
        assert row["audit_status"] in {
            "eligible",
            "conditional",
            "ineligible",
            "source_blocked",
            "duplicate",
            "prospective_fail",
        }


def test_targeted_lane_closes_at_its_order_9_checkpoint() -> None:
    closure = json.loads(TARGETED_CLOSURE.read_text(encoding="utf-8"))
    assert closure["status"] == "closed_low_yield_at_order_9_checkpoint"
    assert closure["checkpoint_order"] == 9
    assert closure["scientific_survivor_count"] == 1
    assert closure["minimum_scientific_survivors"] == 2
    assert closure["stop_rule_triggered"] is True
    assert closure["scientific_survivor_ids"] == ["dvwu7"]
    assert closure["targeted_universe_sha256"] == _sha256(MANIFEST)
    assert closure["audit_registry_sha256"] == _sha256(REGISTRY)
    assert closure["remaining_authorized_batch_rows_not_opened"] == [
        "jk96b",
        "cwur6",
        "h6m8g",
    ]
