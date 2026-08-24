from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE = (
    ROOT
    / "data"
    / "manifests"
    / "research"
    / "independent_replication_completion_queue_v1.json"
)
EXPOSURES = (
    ROOT
    / "data"
    / "manifests"
    / "audits"
    / "independent_replication_exposure_log_v1.json"
)
DEVELOPMENT_REVEAL = (
    ROOT
    / "data"
    / "manifests"
    / "benchmark"
    / "prospective_multimodal_development_reveal_v1.json"
)
CONFIRMATION_SCORING = (
    ROOT
    / "data"
    / "manifests"
    / "research"
    / "confirmation_scoring_protocol_v1.json"
)
PORTFOLIO_REVEAL = (
    ROOT
    / "data"
    / "manifests"
    / "benchmark"
    / "portfolio_pilot_development_reveal.json"
)


def _payload() -> dict:
    return json.loads(QUEUE.read_text(encoding="utf-8"))


def test_completion_queue_is_unique_contiguous_and_bounded() -> None:
    payload = _payload()
    rows = payload["wave_1"] + payload["wave_2"]
    assert [row["queue_order"] for row in rows] == list(range(1, 16))
    assert len({row["candidate_id"] for row in rows}) == 15
    assert payload["stop_rules"]["hard_stop_after_queue_order"] == 15
    assert payload["stop_rules"]["no_additional_corpus_search"] is True
    assert payload["stop_rules"]["no_gate_relaxation"] is True


def test_completion_queue_can_only_meet_the_panel_through_socsci_primary() -> None:
    payload = _payload()
    rows = payload["wave_1"] + payload["wave_2"]
    socsci = [row for row in rows if row["source_stratum"] == "socsci210"]
    assert len(socsci) == 11
    assert payload["panel_gate"]["minimum_socsci210_at_12"] == 8
    assert payload["panel_gate"]["minimum_runnable_tasks"] == 12
    assert payload["panel_gate"]["no_reveal_below_minimum"] is True


def test_completion_queue_excludes_every_exposure_id_and_alias() -> None:
    payload = _payload()
    queued = {
        row["candidate_id"] for row in payload["wave_1"] + payload["wave_2"]
    }
    exposed: set[str] = set()
    for incident in json.loads(EXPOSURES.read_text(encoding="utf-8"))["incidents"]:
        exposed.add(incident["candidate_id"])
        if incident.get("source_lookup_alias"):
            exposed.add(incident["source_lookup_alias"])
    assert queued.isdisjoint(exposed)


def test_completion_queue_excludes_all_previously_revealed_experiments() -> None:
    payload = _payload()
    queued = {
        row["candidate_id"] for row in payload["wave_1"] + payload["wave_2"]
    }
    revealed = set(
        json.loads(DEVELOPMENT_REVEAL.read_text(encoding="utf-8"))["experiment_ids"]
    )
    confirmation = json.loads(CONFIRMATION_SCORING.read_text(encoding="utf-8"))
    revealed.update(confirmation["payload"]["experiment_ids"])
    portfolio = json.loads(PORTFOLIO_REVEAL.read_text(encoding="utf-8"))
    revealed.update(portfolio["experiment_ids"])
    revealed.add("jf46x")
    assert queued.isdisjoint(revealed)


def test_completion_queue_has_zero_execution_and_outcome_authority() -> None:
    authority = _payload()["authority"]
    assert authority == {
        "authorized_spend_usd": 0,
        "model_calls_authorized": False,
        "human_outcome_reveal_authorized": False,
        "participant_row_access_authorized": False,
        "schema_only_participant_container_extraction_authorized": False,
        "public_design_source_acquisition_authorized": True,
    }
