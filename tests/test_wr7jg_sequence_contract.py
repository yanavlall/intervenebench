from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from intervenebench.protocol import assert_blinded_payload
from intervenebench.wr7jg_sequence import (
    aggregate_wr7jg_predictions,
    load_wr7jg_sequence_bundle,
    materialize_wr7jg_pair,
    parse_wr7jg_sequence_response,
    parse_wr7jg_turnout_prediction,
    validate_wr7jg_sequence_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = ROOT / "data" / "manifests" / "contracts"
QUESTIONNAIRE = (
    ROOT
    / "data"
    / "raw"
    / "sources"
    / "wr7jg"
    / "tess2_017_bryan_FINAL.doc"
)


def _load_contract() -> dict:
    return json.loads(
        (CONTRACT_DIR / "wr7jg_decision_task_candidate.json").read_text(
            encoding="utf-8"
        )
    )


def _persona(*, already_voted: bool = False) -> dict:
    return {
        "persona_id": "synthetic-nj-voter-0042",
        "age": 44,
        "registered_to_vote_in_new_jersey": True,
        "already_voted": already_voted,
        "english_comfort": 1,
        "voter_identity": 4,
        "voter_identity_importance": 5,
        "election_care": 4,
        "election_excitement": 3,
        "party_identification": "Independent",
        "profile_summary": "Works full time and usually votes in statewide elections.",
    }


def _record(pair, arm_id: str, probability: float, *, persona_sha: str | None = None):
    return {
        "pair_id": pair.pair_id,
        "persona_state_sha256": persona_sha or pair.persona_state_sha256,
        "arm_id": arm_id,
        "turnout_probability": probability,
    }


def test_wr7jg_contract_is_source_bound_sealed_and_outcome_mapping_blocked() -> None:
    task = _load_contract()
    bundle = load_wr7jg_sequence_bundle(ROOT)
    source = json.loads(
        (
            ROOT / "data/manifests/audits/wr7jg_source_bundle_v1.json"
        ).read_text(encoding="utf-8")
    )

    assert_blinded_payload(task)
    assert_blinded_payload(bundle)
    assert task["task_id"] == bundle["task_id"] == (
        "wr7jg:matched-turnout:verb-vs-noun"
    )
    assert task["outcome_access"] == bundle["outcome_access"] == "sealed"
    assert task["reveal_authorized"] is bundle["reveal_authorized"] is False
    assert task["human_scoring_status"] == (
        "blocked_pending_separate_schema_only_authorization"
    )
    assert source["participant_member_extracted"] is False
    assert source["participant_member_opened"] is False
    assert source["result_text_exposed"] is False

    digest = hashlib.sha256(QUESTIONNAIRE.read_bytes()).hexdigest()
    assert digest == "957d2044bee7b941a4d8edadc12a2d7e96c98170a47c041f629a67bb2f625aad"
    assert digest == task["source_hashes"]["final_questionnaire_sha256"]
    assert digest == bundle["source_material"]["sha256"]
    assert digest == source["retained_final_questionnaire"]["sha256"]
    assert [path for path in QUESTIONNAIRE.parent.iterdir() if path.is_file()] == [
        QUESTIONNAIRE
    ]


def test_wr7jg_freezes_two_active_sequences_and_binary_matched_turnout() -> None:
    task = _load_contract()
    bundle = load_wr7jg_sequence_bundle(ROOT)

    assert [arm["source_assignment_value"] for arm in task["arms"]] == [1, 2]
    assert [arm["source_assignment_value"] for arm in bundle["arms"]] == [1, 2]
    assert task["control_arm_id"] == bundle["reference_arm_id"] == (
        "verb_phrase_action_frame"
    )
    assert task["control_arm_is_active_reference"] is True
    assert task["untreated_control_available"] is False
    assert bundle["untreated_control_available"] is False
    assert task["outcome_family"] == "binary"
    assert task["valid_outcome_values"] == [0, 1]
    assert task["utility_transform"] == "U(y)=y"
    assert bundle["terminal_outcome"]["not_a_survey_self_report"] is True


def test_wr7jg_bundle_preserves_exact_order_wording_and_date_routes() -> None:
    bundle = load_wr7jg_sequence_bundle(ROOT)
    verb, noun = bundle["arms"]

    assert [item["item_id"] for item in verb["items"]] == [
        f"V{index}" for index in range(1, 11)
    ]
    assert [item["item_id"] for item in noun["items"]] == [
        f"N{index}" for index in range(1, 11)
    ]
    assert verb["items"][0]["text"] == (
        "How important is it to you to vote in {{ELECTION_DAY_REFERENCE}} election?"
    )
    assert noun["items"][0]["text"] == (
        "How important is it to you to be a voter in {{ELECTION_DAY_REFERENCE}} election?"
    )
    assert [context["election_day_reference"] for context in bundle["fielding_contexts"]] == [
        "TOMORROW'S",
        "TODAY’S",
    ]


@pytest.mark.parametrize(
    ("context_id", "reference"),
    [
        ("november_2_before_midnight_tomorrows_election", "TOMORROW'S"),
        ("november_3_after_midnight_before_9am_todays_election", "TODAY’S"),
    ],
)
def test_wr7jg_pair_replays_deterministically_and_pairs_all_nuisance_state(
    context_id: str, reference: str
) -> None:
    bundle = load_wr7jg_sequence_bundle(ROOT)
    first = materialize_wr7jg_pair(
        bundle, persona_state=_persona(), fielding_context_id=context_id, seed=17
    )
    again = materialize_wr7jg_pair(
        bundle, persona_state=_persona(), fielding_context_id=context_id, seed=17
    )
    different = materialize_wr7jg_pair(
        bundle, persona_state=_persona(), fielding_context_id=context_id, seed=18
    )

    assert first == again
    assert first.pair_id != different.pair_id
    assert len(first.arms) == 2
    assert {arm.persona_state_sha256 for arm in first.arms} == {
        first.persona_state_sha256
    }
    assert {arm.fielding_context_id for arm in first.arms} == {context_id}
    assert first.arms[0].sequence_prompt != first.arms[1].sequence_prompt
    assert first.arms[0].outcome_prompt == first.arms[1].outcome_prompt
    assert all(reference in arm.sequence_prompt for arm in first.arms)
    assert all("{{ELECTION_DAY_REFERENCE}}" not in arm.sequence_prompt for arm in first.arms)
    assert all("synthetic-nj-voter-0042" in arm.sequence_prompt for arm in first.arms)


def test_wr7jg_common_pretreatment_responses_are_fixed_across_arms() -> None:
    bundle = load_wr7jg_sequence_bundle(ROOT)
    not_yet = materialize_wr7jg_pair(
        bundle,
        persona_state=_persona(already_voted=False),
        fielding_context_id="november_2_before_midnight_tomorrows_election",
        seed=1,
    )
    already = materialize_wr7jg_pair(
        bundle,
        persona_state=_persona(already_voted=True),
        fielding_context_id="november_2_before_midnight_tomorrows_election",
        seed=1,
    )

    assert all("ALL1." in arm.sequence_prompt for arm in not_yet.arms)
    assert all("ALL7." in arm.sequence_prompt for arm in not_yet.arms)
    assert all("Fixed pre-assignment response: 2=No" in arm.sequence_prompt for arm in not_yet.arms)
    assert all("Fixed pre-assignment response: 1=Yes" in arm.sequence_prompt for arm in already.arms)


def test_wr7jg_strictly_rejects_leakage_and_invalid_persona_state() -> None:
    bundle = load_wr7jg_sequence_bundle(ROOT)
    leaked = deepcopy(bundle)
    leaked["arms"][0]["human_outcomes"] = [1]
    with pytest.raises(ValueError, match="forbidden|unexpected"):
        validate_wr7jg_sequence_bundle(leaked)

    persona = _persona()
    persona["reasoning"] = "copied from a participant"
    with pytest.raises(ValueError, match="forbidden|unexpected"):
        materialize_wr7jg_pair(
            bundle,
            persona_state=persona,
            fielding_context_id="november_2_before_midnight_tomorrows_election",
            seed=2,
        )

    persona = _persona()
    persona["registered_to_vote_in_new_jersey"] = False
    with pytest.raises(ValueError, match="registered New Jersey"):
        materialize_wr7jg_pair(
            bundle,
            persona_state=persona,
            fielding_context_id="november_2_before_midnight_tomorrows_election",
            seed=2,
        )


def test_wr7jg_sequence_parser_is_exact_and_ordered() -> None:
    bundle = load_wr7jg_sequence_bundle(ROOT)
    pair = materialize_wr7jg_pair(
        bundle,
        persona_state=_persona(),
        fielding_context_id="november_3_after_midnight_before_9am_todays_election",
        seed=3,
    )
    item_ids = pair.arms[0].expected_sequence_item_ids
    payload = json.dumps(
        {"responses": {item_id: index % 5 + 1 for index, item_id in enumerate(item_ids)}}
    )
    parsed = parse_wr7jg_sequence_response(payload, expected_item_ids=item_ids)
    assert tuple(parsed) == item_ids

    with pytest.raises(ValueError, match="item support"):
        parse_wr7jg_sequence_response(
            '{"responses":{"V1":3}}', expected_item_ids=item_ids
        )
    bad = {"responses": {item_id: 3 for item_id in item_ids}}
    bad["responses"][item_ids[-1]] = True
    with pytest.raises(ValueError, match="integers"):
        parse_wr7jg_sequence_response(json.dumps(bad), expected_item_ids=item_ids)


@pytest.mark.parametrize("invalid", ["-0.1", "1.1", "true", '"0.5"', "null"])
def test_wr7jg_turnout_parser_is_strict(invalid: str) -> None:
    assert parse_wr7jg_turnout_prediction('{"turnout_probability":0.625}') == 0.625
    with pytest.raises(ValueError, match="finite|between"):
        parse_wr7jg_turnout_prediction(f'{{"turnout_probability":{invalid}}}')
    with pytest.raises(ValueError, match="unexpected fields"):
        parse_wr7jg_turnout_prediction(
            '{"turnout_probability":0.5,"explanation":"not allowed"}'
        )


def test_wr7jg_aggregation_requires_complete_pairs_and_uses_reference_on_tie() -> None:
    bundle = load_wr7jg_sequence_bundle(ROOT)
    first = materialize_wr7jg_pair(
        bundle,
        persona_state=_persona(),
        fielding_context_id="november_2_before_midnight_tomorrows_election",
        seed=4,
    )
    second = materialize_wr7jg_pair(
        bundle,
        persona_state={**_persona(), "persona_id": "synthetic-nj-voter-0043"},
        fielding_context_id="november_3_after_midnight_before_9am_todays_election",
        seed=4,
    )
    records = [
        _record(first, "verb_phrase_action_frame", 0.50),
        _record(first, "noun_phrase_voter_identity_frame", 0.70),
        _record(second, "verb_phrase_action_frame", 0.60),
        _record(second, "noun_phrase_voter_identity_frame", 0.80),
    ]
    aggregate = aggregate_wr7jg_predictions(records)
    assert aggregate.pair_count == 2
    assert dict(aggregate.arm_mean_probabilities) == {
        "verb_phrase_action_frame": 0.55,
        "noun_phrase_voter_identity_frame": 0.75,
    }
    assert aggregate.noun_minus_verb_effect == pytest.approx(0.20)
    assert aggregate.selected_arm_id == "noun_phrase_voter_identity_frame"

    tied = aggregate_wr7jg_predictions(
        [
            _record(first, "verb_phrase_action_frame", 0.5),
            _record(first, "noun_phrase_voter_identity_frame", 0.5),
        ]
    )
    assert tied.selected_arm_id == "verb_phrase_action_frame"


def test_wr7jg_aggregation_rejects_incomplete_duplicate_and_unpaired_personas() -> None:
    bundle = load_wr7jg_sequence_bundle(ROOT)
    pair = materialize_wr7jg_pair(
        bundle,
        persona_state=_persona(),
        fielding_context_id="november_2_before_midnight_tomorrows_election",
        seed=5,
    )
    verb = _record(pair, "verb_phrase_action_frame", 0.4)
    noun = _record(pair, "noun_phrase_voter_identity_frame", 0.6)

    with pytest.raises(ValueError, match="both arms"):
        aggregate_wr7jg_predictions([verb])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_wr7jg_predictions([verb, verb, noun])
    with pytest.raises(ValueError, match="persona-state"):
        aggregate_wr7jg_predictions(
            [verb, {**noun, "persona_state_sha256": "f" * 64}]
        )


def test_wr7jg_future_mapping_request_is_minimal_schema_only() -> None:
    task = _load_contract()
    request = task["future_schema_only_request"]

    assert request["authorization_required"] is True
    assert request["participant_rows_authorized"] is False
    assert request["frequencies_authorized"] is False
    assert request["outcome_values_authorized"] is False
    assert request["requested_known_variables"] == ["XTESS017", "ALL1", "ALL2"]
    assert request["requested_unknown_variable_roles"] == [
        "post_election_matched_new_jersey_voter_file_turnout",
        "unique_anonymous_row_identifier",
        "study_specific_weight_if_present",
    ]
    assert "direct_identifiers_or_linkage_keys" in request["forbidden_outputs"]
    assert "turnout_counts" in request["forbidden_outputs"]
