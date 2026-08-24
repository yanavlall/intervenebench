from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESERVE = (
    ROOT
    / "data"
    / "manifests"
    / "audits"
    / "independent_replication_socsci_reserve_v1.json"
)
EXPOSURES = (
    ROOT
    / "data"
    / "manifests"
    / "audits"
    / "independent_replication_exposure_log_v1.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_replication_reserve_preserves_frozen_hash_order() -> None:
    payload = _load(RESERVE)
    label = payload["source_selection_label"]
    candidates = payload["candidates"]
    assert [row["order"] for row in candidates] == list(range(41, 58))
    assert len({row["candidate_id"] for row in candidates}) == 17
    for row in candidates:
        expected = hashlib.sha256(
            f"{label}:{row['candidate_id']}".encode("utf-8")
        ).hexdigest()
        assert row["selection_sha256"] == expected
    assert [row["selection_sha256"] for row in candidates] == sorted(
        row["selection_sha256"] for row in candidates
    )


def test_every_exposure_is_a_prospective_failure_in_reserve() -> None:
    reserve_rows = {
        row["candidate_id"]: row for row in _load(RESERVE)["candidates"]
    }
    exposure_ids = {
        row["candidate_id"] for row in _load(EXPOSURES)["incidents"]
    }
    assert exposure_ids
    reserve_exposure_ids = exposure_ids & reserve_rows.keys()
    assert reserve_exposure_ids
    assert all(
        reserve_rows[candidate_id]["audit_status"]
        == "prospective_fail_result_text_exposed"
        for candidate_id in reserve_exposure_ids
    )


def test_every_logged_exposure_is_excluded_from_pristine_replication() -> None:
    incidents = _load(EXPOSURES)["incidents"]
    assert len({row["candidate_id"] for row in incidents}) == len(incidents)
    assert {
        row["replication_disposition"] for row in incidents
    } == {"excluded_from_pristine_replication"}
    assert all(row["result_content_recorded"] is False for row in incidents)


def test_replication_reserve_authorizes_no_compute_or_reveal() -> None:
    boundary = _load(RESERVE)["compute_boundary"]
    assert boundary == {
        "authorized_spend_usd": 0,
        "paid_model_calls_authorized": False,
        "human_outcome_reveal_authorized": False,
        "participant_row_access_authorized": False,
    }
