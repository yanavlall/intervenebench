from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from intervenebench.dvwu7_interactive import (
    load_dvwu7_interactive_bundle,
    materialize_dvwu7_pair,
    validate_dvwu7_interactive_bundle,
)
from intervenebench.protocol import assert_blinded_payload


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = ROOT / "data" / "manifests" / "contracts"
QUESTIONNAIRE = (
    ROOT
    / "data"
    / "raw"
    / "sources"
    / "dvwu7"
    / "TESS3_177_Ferrer_Questionnaire_final.doc"
)


def _load(name: str) -> dict:
    return json.loads((CONTRACT_DIR / name).read_text(encoding="utf-8"))


def _persona(*, sex: str = "female") -> dict:
    return {
        "persona_id": "synthetic-persona-017",
        "age": 52,
        "sex": sex,
        "bmi": 31.5,
        "most_important_value": "Relations with friends or family",
        "least_important_value": "Musical ability/appreciation",
        "profile_summary": "Works full time, lives with a partner, and has a regular doctor.",
    }


def test_dvwu7_contract_is_source_bound_sealed_and_mapping_blocked() -> None:
    task = _load("dvwu7_decision_task_candidate.json")
    bundle = load_dvwu7_interactive_bundle(ROOT)
    source = json.loads(
        (
            ROOT
            / "data"
            / "manifests"
            / "audits"
            / "dvwu7_source_bundle_v1.json"
        ).read_text(encoding="utf-8")
    )

    validate_dvwu7_interactive_bundle(bundle)
    assert_blinded_payload(task)
    assert_blinded_payload(bundle)
    assert task["task_id"] == bundle["task_id"] == "dvwu7:q9a:self_threat"
    assert task["experiment_id"] == bundle["experiment_id"] == "dvwu7"
    assert task["outcome_access"] == bundle["outcome_access"] == "sealed"
    assert task["reveal_authorized"] is bundle["reveal_authorized"] is False
    assert task["human_scoring_status"] == "blocked_pending_separate_schema_only_authorization"
    assert source["participant_member_extracted"] is False
    assert source["participant_member_opened"] is False

    digest = hashlib.sha256(QUESTIONNAIRE.read_bytes()).hexdigest()
    assert digest == task["source_hashes"]["final_questionnaire_sha256"]
    assert digest == bundle["source_material"]["sha256"]
    assert digest == source["retained_final_questionnaire"]["sha256"]


def test_dvwu7_freezes_cells_one_and_three_and_q9a_utility() -> None:
    task = _load("dvwu7_decision_task_candidate.json")
    bundle = load_dvwu7_interactive_bundle(ROOT)

    assert [arm["source_assignment_value"] for arm in task["arms"]] == [1, 3]
    assert [arm["source_assignment_value"] for arm in bundle["arms"]] == [1, 3]
    assert task["control_arm_id"] == bundle["control_arm_id"] == (
        "xtess177_3_active_control_self_threat"
    )
    assert task["source_question_id"] == "Q9a"
    assert task["scale_lower"] == 1 and task["scale_upper"] == 7
    assert task["direction"] == "higher_is_better"
    assert [option["value"] for option in bundle["response_options"]] == list(
        range(1, 8)
    )
    assert [option["normalized_utility"] for option in bundle["response_options"]] == [
        index / 6 for index in range(7)
    ]


def test_dvwu7_pair_is_deterministic_and_preserves_persona_across_arms() -> None:
    bundle = load_dvwu7_interactive_bundle(ROOT)
    first = materialize_dvwu7_pair(bundle, persona_state=_persona(), seed=177)
    again = materialize_dvwu7_pair(bundle, persona_state=_persona(), seed=177)
    different = materialize_dvwu7_pair(bundle, persona_state=_persona(), seed=178)

    assert first == again
    assert first.pair_id != different.pair_id
    assert len(first.arms) == 2
    assert {arm.pair_id for arm in first.arms} == {first.pair_id}
    assert {arm.persona_state_sha256 for arm in first.arms} == {
        first.persona_state_sha256
    }
    assert first.arms[0].treatment_prompt != first.arms[1].treatment_prompt
    assert first.arms[0].outcome_prompt == first.arms[1].outcome_prompt
    assert all("synthetic-persona-017" in arm.treatment_prompt for arm in first.arms)
    assert all("{{" not in arm.treatment_prompt for arm in first.arms)
    assert all("{{" not in arm.outcome_prompt for arm in first.arms)


def test_dvwu7_pair_uses_source_gender_specific_cancer_passage() -> None:
    bundle = load_dvwu7_interactive_bundle(ROOT)
    female = materialize_dvwu7_pair(bundle, persona_state=_persona(sex="female"), seed=1)
    male = materialize_dvwu7_pair(bundle, persona_state=_persona(sex="male"), seed=1)

    assert all("breast cancer" in arm.outcome_prompt.lower() for arm in female.arms)
    assert all("prostate cancer" in arm.outcome_prompt.lower() for arm in male.arms)
    assert all("prostate cancer" not in arm.outcome_prompt.lower() for arm in female.arms)


def test_dvwu7_strictly_rejects_outcome_leakage_and_unpaired_persona_fields() -> None:
    bundle = load_dvwu7_interactive_bundle(ROOT)
    leaked = deepcopy(bundle)
    leaked["arms"][0]["procedure"]["human_outcomes"] = [7]
    with pytest.raises(ValueError, match="forbidden|unexpected"):
        validate_dvwu7_interactive_bundle(leaked)

    persona = _persona()
    persona["reasoning"] = "copied from a target participant"
    with pytest.raises(ValueError, match="forbidden|unexpected"):
        materialize_dvwu7_pair(bundle, persona_state=persona, seed=1)

    persona = _persona()
    persona["q9a"] = 7
    with pytest.raises(ValueError, match="unexpected"):
        materialize_dvwu7_pair(bundle, persona_state=persona, seed=1)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("age", 39, "age"),
        ("bmi", 24.9, "BMI"),
        ("sex", "nonbinary", "sex"),
        ("least_important_value", "Relations with friends or family", "different"),
    ],
)
def test_dvwu7_persona_contract_fails_closed(
    field: str, value: object, message: str
) -> None:
    bundle = load_dvwu7_interactive_bundle(ROOT)
    persona = _persona()
    persona[field] = value
    with pytest.raises(ValueError, match=message):
        materialize_dvwu7_pair(bundle, persona_state=persona, seed=2)


def test_dvwu7_human_mapping_request_is_schema_only_and_minimal() -> None:
    task = _load("dvwu7_decision_task_candidate.json")
    request = task["future_schema_only_request"]

    assert request["authorization_required"] is True
    assert request["participant_rows_authorized"] is False
    assert request["frequencies_authorized"] is False
    assert request["outcome_values_authorized"] is False
    assert request["requested_known_variables"] == ["XTESS177", "Q9a"]
    assert request["requested_metadata"] == [
        "variable_names_and_labels_to_identify_unique_row_id_and_study_weight",
        "storage_types_for_XTESS177_Q9a_row_id_and_weight",
        "value_labels_for_XTESS177_and_Q9a",
        "user_missing_definitions_for_XTESS177_Q9a_and_weight",
    ]

