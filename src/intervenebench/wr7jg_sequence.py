"""Outcome-blind sequence contract for the wr7jg voter-identity task.

This module reconstructs the final questionnaire's two ten-item sequences and
the two programmed election-date routes.  It performs no inference and never
reads the participant SAV or matched voter-file outcome.  The adapter keeps
the persona, pretreatment responses, date route, and seed identical across
arms while isolating each arm's post-treatment sequence responses.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

from .protocol import assert_blinded_payload, canonical_json_bytes, payload_hash


_DATE_TOKEN = "{{ELECTION_DAY_REFERENCE}}"
_BUNDLE_FIELDS = frozenset(
    {
        "schema_version",
        "task_id",
        "experiment_id",
        "access_regime",
        "outcome_access",
        "reveal_authorized",
        "source_material",
        "population",
        "fielding_contexts",
        "response_scales",
        "common_pretreatment_items",
        "arms",
        "reference_arm_id",
        "untreated_control_available",
        "terminal_outcome",
        "interaction_contract",
        "reconstruction_policy",
    }
)
_PERSONA_FIELDS = frozenset(
    {
        "persona_id",
        "age",
        "registered_to_vote_in_new_jersey",
        "already_voted",
        "english_comfort",
        "voter_identity",
        "voter_identity_importance",
        "election_care",
        "election_excitement",
        "party_identification",
        "profile_summary",
    }
)
_ARM_IDS = (
    "verb_phrase_action_frame",
    "noun_phrase_voter_identity_frame",
)
_FIELDING_CONTEXTS = {
    "november_2_before_midnight_tomorrows_election": "TOMORROW'S",
    "november_3_after_midnight_before_9am_todays_election": "TODAY’S",
}
_COMMON_ITEMS = (
    ("ALL1", "Are you currently registered to vote in New Jersey?", "yes_no"),
    (
        "ALL2",
        "Have you already voted in the New Jersey gubernatorial election between Jon Corzine and Chris Christie?",
        "yes_no",
    ),
    (
        "ALL3",
        "Which of the following best describes your level of comfort with English?",
        "english_comfort",
    ),
    (
        "ALL4",
        "To what extent do you think of yourself as the kind of person who votes in elections?",
        "voter_identity",
    ),
    (
        "ALL5",
        "How important is it to you to be the kind of person who votes in elections?",
        "importance",
    ),
    (
        "ALL6",
        "How much do you care about the outcome of the New Jersey gubernatorial election between Jon Corzine and Chris Christie?",
        "care_all",
    ),
    (
        "ALL7",
        "How excited are you about the New Jersey gubernatorial election between Jon Corzine and Chris Christie?",
        "excitement",
    ),
)
_VERB_TEXTS = (
    "How important is it to you to vote in {{ELECTION_DAY_REFERENCE}} election?",
    "How much do you care about voting in {{ELECTION_DAY_REFERENCE}} election?",
    "How much do you want to vote in {{ELECTION_DAY_REFERENCE}} election?",
    "How personally relevant is it to you to vote in {{ELECTION_DAY_REFERENCE}} election?",
    "How easy do you think it is to vote in {{ELECTION_DAY_REFERENCE}} election?",
    "How convenient do you think it is to vote in {{ELECTION_DAY_REFERENCE}} election?",
    "How consistent are your thoughts and feelings about voting in {{ELECTION_DAY_REFERENCE}} election?",
    "How clear are your thoughts and feelings about voting in {{ELECTION_DAY_REFERENCE}} election?",
    "To what extent are your thoughts about voting in {{ELECTION_DAY_REFERENCE}} election similar to your feelings about voting?",
    "To what extent are your thoughts about voting in {{ELECTION_DAY_REFERENCE}} election different from your feelings about voting?",
)
_NOUN_TEXTS = (
    "How important is it to you to be a voter in {{ELECTION_DAY_REFERENCE}} election?",
    "How much do you care about being a voter in {{ELECTION_DAY_REFERENCE}} election?",
    "How much do you want to be a voter in {{ELECTION_DAY_REFERENCE}} election?",
    "How personally relevant is it to you to be a voter in {{ELECTION_DAY_REFERENCE}} election?",
    "How easy do you think it is to be a voter in {{ELECTION_DAY_REFERENCE}} election?",
    "How convenient do you think it is to be a voter in {{ELECTION_DAY_REFERENCE}} election?",
    "How consistent are your thoughts and feelings about being a voter in {{ELECTION_DAY_REFERENCE}} election?",
    "How clear are your thoughts and feelings about being a voter in {{ELECTION_DAY_REFERENCE}} election?",
    "To what extent are your thoughts about being a voter in {{ELECTION_DAY_REFERENCE}} election similar to your feelings about being a voter?",
    "To what extent are your thoughts about being a voter in {{ELECTION_DAY_REFERENCE}} election different from your feelings about being a voter?",
)
_TREATMENT_SCALE_IDS = (
    "importance",
    "care_treatment",
    "desire",
    "relevance",
    "ease",
    "convenience",
    "consistency",
    "clarity",
    "similarity",
    "difference",
)
_EXPECTED_SCALES = {
    "yes_no": ["Yes", "No"],
    "english_comfort": [
        "I am a native English speaker",
        "I am not a native English speaker but I am very comfortable in English",
        "I am moderately comfortable in English",
        "I am not very comfortable in English",
    ],
    "voter_identity": [
        "Not at all",
        "A little",
        "Moderately",
        "Quite a bit",
        "Very much",
    ],
    "importance": [
        "Not at all important",
        "Slightly important",
        "Moderately important",
        "Very important",
        "Extremely important",
    ],
    "care_all": [
        "Don’t care at all",
        "Care a little",
        "Care a moderate amount",
        "Care quite a bit",
        "Care a great deal",
    ],
    "excitement": [
        "Not at all excited",
        "Slightly excited",
        "Moderately excited",
        "Very excited",
        "Extremely excited",
    ],
    "care_treatment": [
        "Don’t care at all",
        "Care slightly",
        "Care somewhat",
        "Care quite a bit",
        "Care a great deal",
    ],
    "desire": [
        "Not at all",
        "A little",
        "A moderate amount",
        "Quite a bit",
        "A great deal",
    ],
    "relevance": [
        "Not at all relevant",
        "Slightly relevant",
        "Somewhat relevant",
        "Very relevant",
        "Extremely relevant",
    ],
    "ease": [
        "Not at all easy",
        "Slightly easy",
        "Somewhat easy",
        "Very easy",
        "Extremely easy",
    ],
    "convenience": [
        "Not at all convenient",
        "Slightly convenient",
        "Somewhat convenient",
        "Very convenient",
        "Extremely convenient",
    ],
    "consistency": [
        "Not at all consistent",
        "Slightly consistent",
        "Somewhat consistent",
        "Very consistent",
        "Extremely consistent",
    ],
    "clarity": [
        "Not at all clear",
        "Slightly clear",
        "Somewhat clear",
        "Very clear",
        "Extremely clear",
    ],
    "similarity": [
        "Not at all similar",
        "Slightly similar",
        "Somewhat similar",
        "Very similar",
        "Extremely similar",
    ],
    "difference": [
        "Not at all different",
        "Slightly different",
        "Somewhat different",
        "Very different",
        "Extremely different",
    ],
}


@dataclass(frozen=True, slots=True)
class Wr7jgArmEpisode:
    """One arm-local sequence and terminal prediction request."""

    arm_id: str
    pair_id: str
    persona_state_sha256: str
    seed: int
    fielding_context_id: str
    expected_sequence_item_ids: tuple[str, ...]
    sequence_prompt: str
    outcome_prompt: str


@dataclass(frozen=True, slots=True)
class Wr7jgSequencePair:
    """Paired counterfactual sequences sharing persona and date route."""

    pair_id: str
    persona_state_sha256: str
    seed: int
    fielding_context_id: str
    arms: tuple[Wr7jgArmEpisode, ...]


@dataclass(frozen=True, slots=True)
class Wr7jgSyntheticAggregate:
    """Aggregate synthetic prediction without any human outcome access."""

    pair_count: int
    arm_mean_probabilities: tuple[tuple[str, float], ...]
    noun_minus_verb_effect: float
    selected_arm_id: str


def _require_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_digest(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be a SHA-256 digest") from error
    return value


def validate_wr7jg_sequence_bundle(bundle: Mapping[str, Any]) -> None:
    """Validate the exact sealed, source-faithful wr7jg bundle."""

    assert_blinded_payload(bundle)
    if set(bundle) != _BUNDLE_FIELDS:
        raise ValueError("wr7jg bundle has unexpected or missing fields")
    if bundle["schema_version"] != "binary_sequence_blinded_bundle.v1":
        raise ValueError("unsupported wr7jg sequence-bundle schema")
    if bundle["task_id"] != "wr7jg:matched-turnout:verb-vs-noun":
        raise ValueError("wr7jg task ID is frozen")
    if bundle["experiment_id"] != "wr7jg":
        raise ValueError("wr7jg experiment ID is frozen")
    if bundle["access_regime"] != "DESIGN_ONLY":
        raise ValueError("wr7jg bundle must remain design-only")
    if bundle["outcome_access"] != "sealed" or bundle["reveal_authorized"] is not False:
        raise ValueError("wr7jg bundle must remain outcome sealed")

    source = bundle["source_material"]
    if not isinstance(source, Mapping) or set(source) != {"path", "sha256"}:
        raise ValueError("wr7jg source material has an unexpected shape")
    if source["path"] != "data/raw/sources/wr7jg/tess2_017_bryan_FINAL.doc":
        raise ValueError("wr7jg source path is frozen")
    _require_digest(source["sha256"], field="source_material.sha256")

    population = bundle["population"]
    if not isinstance(population, Mapping) or set(population) != {
        "description",
        "roster_id",
        "eligibility",
    }:
        raise ValueError("wr7jg population has an unexpected shape")
    for field, value in population.items():
        _require_text(value, field=f"population.{field}")

    contexts = bundle["fielding_contexts"]
    if not isinstance(contexts, list) or len(contexts) != 2:
        raise ValueError("wr7jg requires both programmed fielding contexts")
    recovered_contexts: dict[str, str] = {}
    for context in contexts:
        if not isinstance(context, Mapping) or set(context) != {
            "context_id",
            "source_window",
            "election_day_reference",
        }:
            raise ValueError("wr7jg fielding context has an unexpected shape")
        _require_text(context["source_window"], field="fielding_context.source_window")
        recovered_contexts[context["context_id"]] = context["election_day_reference"]
    if recovered_contexts != _FIELDING_CONTEXTS:
        raise ValueError("wr7jg fielding routes are not source faithful")

    scales = bundle["response_scales"]
    if scales != _EXPECTED_SCALES:
        raise ValueError("wr7jg response labels are not exact")

    common_items = bundle["common_pretreatment_items"]
    if not isinstance(common_items, list) or len(common_items) != len(_COMMON_ITEMS):
        raise ValueError("wr7jg must preserve all seven common items")
    for item, expected in zip(common_items, _COMMON_ITEMS, strict=True):
        if not isinstance(item, Mapping) or set(item) != {"item_id", "text", "scale_id"}:
            raise ValueError("wr7jg common item has an unexpected shape")
        if (item["item_id"], item["text"], item["scale_id"]) != expected:
            raise ValueError("wr7jg common item wording or order drifted")

    arms = bundle["arms"]
    if not isinstance(arms, list) or len(arms) != 2:
        raise ValueError("wr7jg requires exactly two arms")
    expected_by_arm = {
        _ARM_IDS[0]: (1, "V", _VERB_TEXTS),
        _ARM_IDS[1]: (2, "N", _NOUN_TEXTS),
    }
    for arm, arm_id in zip(arms, _ARM_IDS, strict=True):
        if not isinstance(arm, Mapping) or set(arm) != {
            "arm_id",
            "source_assignment_value",
            "source_item_prefix",
            "display_text",
            "items",
        }:
            raise ValueError("wr7jg arm has an unexpected shape")
        if arm["arm_id"] != arm_id:
            raise ValueError("wr7jg arm order and IDs are frozen")
        assignment, prefix, expected_texts = expected_by_arm[arm_id]
        if arm["source_assignment_value"] != assignment:
            raise ValueError("wr7jg source assignments are frozen")
        if arm["source_item_prefix"] != prefix:
            raise ValueError("wr7jg source item prefixes are frozen")
        if arm["display_text"] != (
            "In answering the questions that follow, please take your time and read "
            "each question carefully. Thank you, we appreciate it!"
        ):
            raise ValueError("wr7jg display text is not exact")
        items = arm["items"]
        if not isinstance(items, list) or len(items) != 10:
            raise ValueError("wr7jg arms require all ten ordered items")
        for index, (item, text, scale_id) in enumerate(
            zip(items, expected_texts, _TREATMENT_SCALE_IDS, strict=True), start=1
        ):
            if not isinstance(item, Mapping) or set(item) != {
                "item_id",
                "text",
                "scale_id",
            }:
                raise ValueError("wr7jg treatment item has an unexpected shape")
            if item != {
                "item_id": f"{prefix}{index}",
                "text": text,
                "scale_id": scale_id,
            }:
                raise ValueError("wr7jg treatment wording, order, or scale drifted")
            if item["text"].count(_DATE_TOKEN) != 1:
                raise ValueError("wr7jg item must contain exactly one date route token")

    if bundle["reference_arm_id"] != _ARM_IDS[0]:
        raise ValueError("wr7jg active reference arm is frozen")
    if bundle["untreated_control_available"] is not False:
        raise ValueError("wr7jg must not invent an untreated control")
    terminal = bundle["terminal_outcome"]
    if not isinstance(terminal, Mapping) or set(terminal) != {
        "human_endpoint",
        "synthetic_endpoint",
        "utility",
        "not_a_survey_self_report",
    }:
        raise ValueError("wr7jg terminal outcome has an unexpected shape")
    if terminal["not_a_survey_self_report"] is not True:
        raise ValueError("wr7jg turnout endpoint must remain an external record match")
    for field in ("human_endpoint", "synthetic_endpoint", "utility"):
        _require_text(terminal[field], field=f"terminal_outcome.{field}")

    if bundle["interaction_contract"] != {
        "sequence_unit": "synthetic_persona_by_fielding_context",
        "paired_across_arms": True,
        "same_persona_state_across_arms": True,
        "same_fielding_context_across_arms": True,
        "common_pretreatment_responses_fixed_across_arms": True,
        "separate_arm_conversations": True,
        "arm_generated_sequence_responses_reused": False,
        "target_position": "after_all_seven_common_items_and_all_ten_arm_specific_items",
    }:
        raise ValueError("wr7jg interaction contract is not exact")
    reconstruction = bundle["reconstruction_policy"]
    if not isinstance(reconstruction, Mapping) or set(reconstruction) != {
        "question_wording_order_and_response_labels",
        "normalization",
        "fielded_browser_typography_recovered",
        "claim_boundary",
    }:
        raise ValueError("wr7jg reconstruction policy has an unexpected shape")
    if reconstruction["fielded_browser_typography_recovered"] is not False:
        raise ValueError("wr7jg must not claim pixel-identical reconstruction")


def load_wr7jg_sequence_bundle(repository_root: Path) -> dict[str, Any]:
    """Load and validate the repository's frozen wr7jg bundle."""

    path = (
        repository_root
        / "data"
        / "manifests"
        / "contracts"
        / "wr7jg_sequence_blinded_bundle.json"
    )
    with path.open(encoding="utf-8") as stream:
        bundle = json.load(stream)
    validate_wr7jg_sequence_bundle(bundle)
    return bundle


def _validated_persona(persona_state: Mapping[str, Any]) -> dict[str, Any]:
    assert_blinded_payload(persona_state)
    if set(persona_state) != _PERSONA_FIELDS:
        raise ValueError("wr7jg persona state has unexpected or missing fields")
    persona = dict(persona_state)
    for field in ("persona_id", "party_identification", "profile_summary"):
        _require_text(persona[field], field=f"persona_state.{field}")
    age = persona["age"]
    if not isinstance(age, int) or isinstance(age, bool) or not 18 <= age <= 120:
        raise ValueError("wr7jg persona age must be an integer from 18 through 120")
    if persona["registered_to_vote_in_new_jersey"] is not True:
        raise ValueError("wr7jg personas must be registered New Jersey voters")
    if not isinstance(persona["already_voted"], bool):
        raise ValueError("wr7jg already_voted must be boolean")
    bounds = {
        "english_comfort": (1, 4),
        "voter_identity": (1, 5),
        "voter_identity_importance": (1, 5),
        "election_care": (1, 5),
        "election_excitement": (1, 5),
    }
    for field, (lower, upper) in bounds.items():
        value = persona[field]
        if not isinstance(value, int) or isinstance(value, bool) or not lower <= value <= upper:
            raise ValueError(f"wr7jg {field} must be an integer from {lower} through {upper}")
    return persona


def _common_response_values(persona: Mapping[str, Any]) -> dict[str, int]:
    return {
        "ALL1": 1,
        "ALL2": 1 if persona["already_voted"] else 2,
        "ALL3": persona["english_comfort"],
        "ALL4": persona["voter_identity"],
        "ALL5": persona["voter_identity_importance"],
        "ALL6": persona["election_care"],
        "ALL7": persona["election_excitement"],
    }


def _format_item(item: Mapping[str, Any], scales: Mapping[str, list[str]]) -> str:
    answers = " | ".join(
        f"{index}={label}" for index, label in enumerate(scales[item["scale_id"]], start=1)
    )
    return f"{item['item_id']}. {item['text']}\nAnswers: {answers}"


def materialize_wr7jg_pair(
    bundle: Mapping[str, Any],
    *,
    persona_state: Mapping[str, Any],
    fielding_context_id: str,
    seed: int,
) -> Wr7jgSequencePair:
    """Materialize paired arm-local sequences without executing a model."""

    validate_wr7jg_sequence_bundle(bundle)
    if fielding_context_id not in _FIELDING_CONTEXTS:
        raise ValueError("wr7jg fielding context is not one of the source routes")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("wr7jg paired seed must be a non-negative integer")
    persona = _validated_persona(persona_state)
    persona_sha = payload_hash(persona)
    pair_id = payload_hash(
        {
            "task_id": bundle["task_id"],
            "persona_state_sha256": persona_sha,
            "fielding_context_id": fielding_context_id,
            "seed": seed,
        }
    )
    day_reference = _FIELDING_CONTEXTS[fielding_context_id]
    persona_json = canonical_json_bytes(persona).decode("utf-8")
    common_values = _common_response_values(persona)
    common_lines: list[str] = []
    for item in bundle["common_pretreatment_items"]:
        scale = bundle["response_scales"][item["scale_id"]]
        value = common_values[item["item_id"]]
        common_lines.append(
            f"{item['item_id']}. {item['text']}\n"
            f"Fixed pre-assignment response: {value}={scale[value - 1]}"
        )
    stable_header = (
        "Simulate one registered New Jersey voter in an outcome-blind paired design. "
        "Do not use any target-human response, matched-turnout outcome, aggregate, "
        "winner, or reasoning. This arm runs in a fresh conversation; the persona, "
        "pretreatment responses, date route, and seed are byte-identical across arms.\n"
        f"Pair ID: {pair_id}\n"
        f"Persona-state SHA-256: {persona_sha}\n"
        f"Fielding context: {fielding_context_id}\n"
        f"Stable persona state: {persona_json}\n\n"
        "The following seven source responses occurred before random assignment and "
        "are fixed across both arms:\n\n"
        + "\n\n".join(common_lines)
    )
    outcome_prompt = (
        "Continue as the same synthetic persona after completing all ten questions in "
        "this arm. Do not copy, retrieve, or mention responses from the other arm. "
        "The source endpoint is not a survey self-report: after the November 3, 2009 "
        "election, the participant is matched to New Jersey voter turnout records. "
        "Predict the probability that this persona would be recorded as having voted. "
        "Return only one JSON object with exactly this shape: "
        '{"turnout_probability":NUMBER}. NUMBER must be finite and between 0 and 1.'
    )

    episodes: list[Wr7jgArmEpisode] = []
    for arm in bundle["arms"]:
        materialized_items: list[dict[str, str]] = []
        for item in arm["items"]:
            materialized = dict(item)
            materialized["text"] = item["text"].replace(_DATE_TOKEN, day_reference)
            materialized_items.append(materialized)
        item_ids = tuple(item["item_id"] for item in materialized_items)
        sequence_text = "\n\n".join(
            _format_item(item, bundle["response_scales"])
            for item in materialized_items
        )
        response_shape = ",".join(f'"{item_id}":INTEGER' for item_id in item_ids)
        sequence_prompt = (
            f"{stable_header}\n\n"
            f"Source display: {arm['display_text']}\n\n"
            "Privately answer every question below in source order as this persona. "
            "Each answer must use the displayed integer scale.\n\n"
            f"{sequence_text}\n\n"
            "Return only one JSON object with exactly this shape: "
            f'{{"responses":{{{response_shape}}}}}. Include each item exactly once.'
        )
        if _DATE_TOKEN in sequence_prompt:
            raise ValueError("wr7jg date route was not fully materialized")
        episodes.append(
            Wr7jgArmEpisode(
                arm_id=arm["arm_id"],
                pair_id=pair_id,
                persona_state_sha256=persona_sha,
                seed=seed,
                fielding_context_id=fielding_context_id,
                expected_sequence_item_ids=item_ids,
                sequence_prompt=sequence_prompt,
                outcome_prompt=outcome_prompt,
            )
        )
    return Wr7jgSequencePair(
        pair_id=pair_id,
        persona_state_sha256=persona_sha,
        seed=seed,
        fielding_context_id=fielding_context_id,
        arms=tuple(episodes),
    )


def parse_wr7jg_sequence_response(
    text: str, *, expected_item_ids: Sequence[str]
) -> dict[str, int]:
    """Strictly parse the arm-local ten-item response object."""

    _require_text(text, field="sequence response")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("wr7jg sequence response must be strict JSON") from error
    if not isinstance(payload, Mapping) or set(payload) != {"responses"}:
        raise ValueError("wr7jg sequence response must contain only responses")
    responses = payload["responses"]
    expected = tuple(expected_item_ids)
    if len(expected) != 10 or len(set(expected)) != 10:
        raise ValueError("wr7jg expected sequence must contain ten unique items")
    if not isinstance(responses, Mapping) or set(responses) != set(expected):
        raise ValueError("wr7jg response item support must match the frozen sequence")
    parsed: dict[str, int] = {}
    for item_id in expected:
        value = responses[item_id]
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
            raise ValueError("wr7jg sequence answers must be integers from 1 through 5")
        parsed[item_id] = value
    return parsed


def parse_wr7jg_turnout_prediction(text: str) -> float:
    """Strictly parse one outcome-blind synthetic turnout probability."""

    _require_text(text, field="turnout prediction")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("wr7jg turnout prediction must be strict JSON") from error
    if not isinstance(payload, Mapping) or set(payload) != {"turnout_probability"}:
        raise ValueError("wr7jg turnout prediction has unexpected fields")
    probability = payload["turnout_probability"]
    if (
        not isinstance(probability, (int, float))
        or isinstance(probability, bool)
        or not isfinite(float(probability))
        or not 0 <= float(probability) <= 1
    ):
        raise ValueError("wr7jg turnout probability must be finite and between 0 and 1")
    return float(probability)


def aggregate_wr7jg_predictions(
    records: Sequence[Mapping[str, Any]],
) -> Wr7jgSyntheticAggregate:
    """Aggregate paired synthetic probabilities without human outcome access."""

    if not records:
        raise ValueError("wr7jg aggregation requires at least one paired episode")
    expected_fields = {
        "pair_id",
        "persona_state_sha256",
        "arm_id",
        "turnout_probability",
    }
    by_pair: dict[str, dict[str, tuple[str, float]]] = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != expected_fields:
            raise ValueError("wr7jg aggregate record has unexpected or missing fields")
        pair_id = _require_digest(record["pair_id"], field="pair_id")
        persona_sha = _require_digest(
            record["persona_state_sha256"], field="persona_state_sha256"
        )
        arm_id = record["arm_id"]
        if arm_id not in _ARM_IDS:
            raise ValueError("wr7jg aggregate record has an unknown arm")
        probability = record["turnout_probability"]
        if (
            not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not isfinite(float(probability))
            or not 0 <= float(probability) <= 1
        ):
            raise ValueError("wr7jg aggregate probability must be finite and in [0, 1]")
        pair = by_pair.setdefault(pair_id, {})
        if arm_id in pair:
            raise ValueError("wr7jg aggregate contains a duplicate pair-arm record")
        pair[arm_id] = (persona_sha, float(probability))

    totals = {arm_id: 0.0 for arm_id in _ARM_IDS}
    for pair in by_pair.values():
        if set(pair) != set(_ARM_IDS):
            raise ValueError("wr7jg aggregate requires both arms for every pair")
        if len({persona_sha for persona_sha, _ in pair.values()}) != 1:
            raise ValueError("wr7jg paired arms must share one persona-state hash")
        for arm_id, (_, probability) in pair.items():
            totals[arm_id] += probability
    pair_count = len(by_pair)
    means = {arm_id: totals[arm_id] / pair_count for arm_id in _ARM_IDS}
    effect = means[_ARM_IDS[1]] - means[_ARM_IDS[0]]
    selected = _ARM_IDS[1] if effect > 0 else _ARM_IDS[0]
    return Wr7jgSyntheticAggregate(
        pair_count=pair_count,
        arm_mean_probabilities=tuple((arm_id, means[arm_id]) for arm_id in _ARM_IDS),
        noun_minus_verb_effect=effect,
        selected_arm_id=selected,
    )
