from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "data" / "manifests" / "audits"
CONTRACT_DIR = ROOT / "data" / "manifests" / "contracts"
SOURCE = (
    ROOT
    / "data"
    / "raw"
    / "sources"
    / "h6txz"
    / "tess2_096_kelly_FINAL.doc"
)


def _load(name: str) -> dict:
    return json.loads((AUDIT_DIR / name).read_text(encoding="utf-8"))


def test_h6txz_retains_only_hash_bound_permitted_final_questionnaire() -> None:
    source = _load("h6txz_source_bundle_v1.json")
    adjudication = _load("h6txz_truthfulness_adjudication_v1.json")

    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == "93d1ddeaa210eb7d0312dfe176e5d242289960455eebda3585ab6e87ed060b9f"
    assert digest == source["retained_final_questionnaire"]["sha256"]
    assert digest == adjudication["source_questionnaire_sha256"]
    assert source["source_archive"]["retained"] is False
    assert source["participant_member_extracted"] is False
    assert source["participant_member_opened"] is False
    assert source["result_text_exposed"] is False

    retained = [path for path in SOURCE.parent.iterdir() if path.is_file()]
    assert retained == [SOURCE]


def test_h6txz_fails_the_fixed_world_truthfulness_gate_on_internal_chronology() -> None:
    adjudication = _load("h6txz_truthfulness_adjudication_v1.json")
    gate = adjudication["fixed_world_truthfulness_gate"]

    assert adjudication["evidence_scope"] == "final_questionnaire_only"
    assert gate["required"] is True
    assert gate["passed"] is False
    assert gate["decisive_arm_assignment_value"] == 1
    assert gate["broadcast_dateline_in_instrument"].startswith("December 6, 2010")
    assert "Dec. 10" in gate["postdated_evidence_statement_in_instrument"]
    assert "four calendar days after" in gate["calendar_conflict"]


def test_h6txz_preliminary_q11_design_is_not_promoted_to_a_task() -> None:
    adjudication = _load("h6txz_truthfulness_adjudication_v1.json")
    design = adjudication["preliminary_design_reconstruction"]
    disposition = adjudication["disposition"]

    assert [arm["assignment_value"] for arm in design["arms_as_labeled_in_instrument"]] == [
        1,
        2,
        3,
    ]
    assert design["candidate_control_assignment_value"] == 3
    assert design["candidate_outcome_question"] == "Q11"
    assert design["candidate_scale_bounds"] == [1, 10]
    assert design["candidate_scale_orientation"] == (
        "higher_means_more_perceived_program_credibility"
    )
    assert design["status"] == "observed_in_source_but_not_frozen_as_a_decision_task"
    assert disposition["scientific_survivor"] is False
    assert disposition["status"] == "excluded_fail_closed"
    assert disposition["eligible_for_independent_replication_panel"] is False


def test_h6txz_exclusion_has_no_runnable_contract_adapter_or_mapping_request() -> None:
    adjudication = _load("h6txz_truthfulness_adjudication_v1.json")
    disposition = adjudication["disposition"]

    assert disposition["runnable_decision_task_created"] is False
    assert disposition["blinded_bundle_created"] is False
    assert disposition["simulator_adapter_created"] is False
    assert disposition["human_mapping_pursued"] is False
    assert disposition["future_schema_only_request_created"] is False
    assert disposition["outcome_reveal_authorized"] is False
    assert not list(CONTRACT_DIR.glob("h6txz*.json"))
    assert not list((ROOT / "src" / "intervenebench").glob("h6txz*.py"))


def test_h6txz_exposure_attestation_remains_strictly_sealed() -> None:
    adjudication = _load("h6txz_truthfulness_adjudication_v1.json")
    attestation = adjudication["exposure_attestation"]

    false_keys = {
        "participant_file_extracted",
        "participant_file_opened",
        "participant_row_opened",
        "human_outcome_value_opened",
        "human_outcome_summary_opened",
        "winner_or_treatment_effect_opened",
        "report_opened",
        "methodology_file_opened",
        "proposal_opened",
        "manuscript_opened",
        "model_call_made",
    }
    assert all(attestation[key] is False for key in false_keys)
    assert attestation["paid_compute_spend_usd"] == 0
    assert attestation["outcome_access"] == "sealed"
