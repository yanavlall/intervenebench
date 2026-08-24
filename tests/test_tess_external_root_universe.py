from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "data"
    / "manifests"
    / "audits"
    / "tess_external_root_universe_v1.json"
)
REGISTRY = (
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
SOCSCI_NODES = (
    ROOT
    / "data"
    / "raw"
    / "socsci210"
    / "048481111a4425ed83dc0eacf15f8431f252b21a"
    / "metadata"
    / "osf_nodes"
)
PRIOR_EXTERNAL = (
    ROOT
    / "data"
    / "manifests"
    / "audits"
    / "external_candidate_universe_v1.csv"
)


def _normalized(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    return re.sub(r"[^a-z0-9]+", "", decomposed)


def _payload() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_external_tess_universe_is_hash_ordered_and_result_free() -> None:
    payload = _payload()
    label = payload["selection_label"]
    rows = payload["candidate_universe"]
    assert rows
    assert [row["freeze_order"] for row in rows] == list(range(1, len(rows) + 1))
    expected = [
        hashlib.sha256(f"{label}:{row['osf_node_id']}".encode()).hexdigest()
        for row in rows
    ]
    assert [row["selection_sha256"] for row in rows] == expected == sorted(expected)
    assert {row["outcome_access"] for row in rows} == {"sealed"}
    assert {row["result_text_exposed"] for row in rows} == {False}


def test_external_tess_universe_excludes_known_ids_and_normalized_titles() -> None:
    payload = _payload()
    rows = payload["candidate_universe"]
    socsci_payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in SOCSCI_NODES.glob("*.json")
    ]
    socsci_ids = {row["data"]["id"] for row in socsci_payloads}
    socsci_titles = {
        _normalized(row["data"]["attributes"]["title"])
        for row in socsci_payloads
    }
    with PRIOR_EXTERNAL.open(newline="", encoding="utf-8") as handle:
        external_titles = {
            _normalized(row["study_title"])
            for row in csv.DictReader(handle)
            if row.get("study_title")
        }
    assert {row["osf_node_id"] for row in rows}.isdisjoint(socsci_ids)
    assert {_normalized(row["title"]) for row in rows}.isdisjoint(socsci_titles)
    assert {_normalized(row["title"]) for row in rows}.isdisjoint(external_titles)


def test_external_tess_audit_batch_is_contiguous_and_zero_authority() -> None:
    payload = _payload()
    protocol = payload["ordered_audit_protocol"]
    expected = [
        row["osf_node_id"] for row in payload["candidate_universe"][:30]
    ]
    assert protocol["batch_candidate_ids"] == expected
    assert protocol["no_selective_skipping"] is True
    assert protocol["protocol_revision"] == "1.1_no_progress_stop"
    assert protocol["interim_stop_after_order_20_if_clean_passes_below"] == 2
    assert payload["authority"] == {
        "authorized_spend_usd": 0,
        "model_calls_authorized": False,
        "human_outcome_reveal_authorized": False,
        "participant_row_access_authorized": False,
        "public_design_source_acquisition_authorized": True,
        "mixed_archive_member_listing_authorized": True,
        "participant_member_extraction_authorized": False,
    }


def test_external_tess_audit_registry_is_contiguous_and_matches_frozen_rows() -> None:
    payload = _payload()
    frozen = {
        row["freeze_order"]: row for row in payload["candidate_universe"]
    }
    with REGISTRY.open(newline="", encoding="utf-8") as handle:
        registry = list(csv.DictReader(handle))
    orders = [int(row["freeze_order"]) for row in registry]
    assert orders == list(range(1, len(registry) + 1))
    assert {
        row["audit_status"] for row in registry
    } <= {
        "eligible",
        "conditional",
        "ineligible",
        "source_blocked",
        "duplicate",
        "prospective_fail",
    }
    for row in registry:
        source = frozen[int(row["freeze_order"])]
        assert row["osf_node_id"] == source["osf_node_id"]
        assert _normalized(row["study_title"]) == _normalized(source["title"])


def test_external_registry_result_exposure_matches_global_log() -> None:
    with REGISTRY.open(newline="", encoding="utf-8") as handle:
        registry = list(csv.DictReader(handle))
    exposed_registry_ids = {
        row["osf_node_id"]
        for row in registry
        if row["result_text_exposed"] == "true"
    }
    exposure_ids = {
        row["candidate_id"] for row in json.loads(EXPOSURES.read_text())["incidents"]
    }
    assert exposed_registry_ids <= exposure_ids
