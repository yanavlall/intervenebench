"""Outcome-blind interactive contract for the dvwu7 self-affirmation task.

The adapter materializes two separate counterfactual conversations from one
immutable synthetic-persona state.  It never reads participant records and it
does not execute a model.  A future runner must keep each arm conversation
separate while using the same persona-state hash and paired seed in both arms.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from .protocol import assert_blinded_payload, canonical_json_bytes, payload_hash


BUNDLE_FIELDS = frozenset(
    {
        "schema_version",
        "task_id",
        "experiment_id",
        "access_regime",
        "outcome_access",
        "reveal_authorized",
        "source_material",
        "population",
        "arms",
        "control_arm_id",
        "common_context",
        "passage_routing",
        "outcome_question",
        "response_options",
        "interaction_contract",
    }
)
PERSONA_FIELDS = frozenset(
    {
        "persona_id",
        "age",
        "sex",
        "bmi",
        "most_important_value",
        "least_important_value",
        "profile_summary",
    }
)
ARM_FIELDS = frozenset({"arm_id", "source_assignment_value", "procedure"})
PROCEDURE_FIELDS = frozenset(
    {"value_selection", "self_reference_rule", "writing_prompt"}
)
EXPECTED_ARMS = (
    (
        "xtess177_1_self_affirmation_self_threat",
        1,
        "most_important_value",
        "self",
    ),
    (
        "xtess177_3_active_control_self_threat",
        3,
        "least_important_value",
        "another_person_only",
    ),
)


@dataclass(frozen=True, slots=True)
class Dvwu7ArmEpisode:
    """One arm-local conversation in a paired interactive episode."""

    arm_id: str
    pair_id: str
    persona_state_sha256: str
    seed: int
    treatment_prompt: str
    outcome_prompt: str


@dataclass(frozen=True, slots=True)
class Dvwu7InteractivePair:
    """Two counterfactual arm conversations sharing one persona state."""

    pair_id: str
    persona_state_sha256: str
    seed: int
    arms: tuple[Dvwu7ArmEpisode, ...]


def _require_digest(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be a SHA-256 digest") from error
    return value


def _require_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def validate_dvwu7_interactive_bundle(bundle: Mapping[str, Any]) -> None:
    """Validate the exact sealed dvwu7 interactive-bundle schema."""

    assert_blinded_payload(bundle)
    if set(bundle) != BUNDLE_FIELDS:
        raise ValueError("dvwu7 bundle has unexpected or missing fields")
    if bundle["schema_version"] != "interactive_ordinal_blinded_bundle.v1":
        raise ValueError("unsupported dvwu7 interactive-bundle schema")
    if bundle["task_id"] != "dvwu7:q9a:self_threat":
        raise ValueError("dvwu7 bundle has an unexpected task ID")
    if bundle["experiment_id"] != "dvwu7":
        raise ValueError("dvwu7 bundle has an unexpected experiment ID")
    if bundle["access_regime"] != "DESIGN_ONLY":
        raise ValueError("dvwu7 bundle must remain design-only")
    if bundle["outcome_access"] != "sealed" or bundle["reveal_authorized"] is not False:
        raise ValueError("dvwu7 bundle must remain outcome sealed")

    source = bundle["source_material"]
    if not isinstance(source, Mapping) or set(source) != {"path", "sha256"}:
        raise ValueError("dvwu7 source material has an unexpected shape")
    _require_text(source["path"], field="source_material.path")
    _require_digest(source["sha256"], field="source_material.sha256")

    population = bundle["population"]
    if not isinstance(population, Mapping) or set(population) != {
        "description",
        "roster_id",
        "eligibility",
    }:
        raise ValueError("dvwu7 population has an unexpected shape")
    for field in population:
        _require_text(population[field], field=f"population.{field}")

    arms = bundle["arms"]
    if not isinstance(arms, list) or len(arms) != 2:
        raise ValueError("dvwu7 bundle must contain exactly two frozen arms")
    for arm, expected in zip(arms, EXPECTED_ARMS, strict=True):
        if not isinstance(arm, Mapping) or set(arm) != ARM_FIELDS:
            raise ValueError("dvwu7 arm has an unexpected shape")
        arm_id, assignment, value_selection, reference_rule = expected
        if arm["arm_id"] != arm_id or arm["source_assignment_value"] != assignment:
            raise ValueError("dvwu7 arm IDs and source assignments are frozen")
        procedure = arm["procedure"]
        if not isinstance(procedure, Mapping) or set(procedure) != PROCEDURE_FIELDS:
            raise ValueError("dvwu7 treatment procedure has an unexpected shape")
        if procedure["value_selection"] != value_selection:
            raise ValueError("dvwu7 value-selection procedure is not source faithful")
        if procedure["self_reference_rule"] != reference_rule:
            raise ValueError("dvwu7 self-reference rule is not source faithful")
        _require_text(procedure["writing_prompt"], field="procedure.writing_prompt")
    if bundle["control_arm_id"] != EXPECTED_ARMS[1][0]:
        raise ValueError("dvwu7 active-control arm is frozen")

    _require_text(bundle["common_context"], field="common_context")
    passages = bundle["passage_routing"]
    if not isinstance(passages, Mapping) or set(passages) != {"female", "male"}:
        raise ValueError("dvwu7 passage routing must cover female and male personas")
    for sex, passage in passages.items():
        _require_text(passage, field=f"passage_routing.{sex}")
    if "breast cancer" not in passages["female"].casefold():
        raise ValueError("female route must preserve the breast-cancer passage")
    if "prostate cancer" not in passages["male"].casefold():
        raise ValueError("male route must preserve the prostate-cancer passage")

    _require_text(bundle["outcome_question"], field="outcome_question")
    if bundle["outcome_question"] != "I intend to lose weight in the next 6 months.":
        raise ValueError("dvwu7 target must remain source Q9a")
    options = bundle["response_options"]
    if not isinstance(options, list) or len(options) != 7:
        raise ValueError("dvwu7 requires the complete seven-point Q9a scale")
    if any(
        not isinstance(option, Mapping)
        or set(option) != {"value", "label", "normalized_utility"}
        for option in options
    ):
        raise ValueError("dvwu7 response option has an unexpected shape")
    values = [option["value"] for option in options]
    utilities = [option["normalized_utility"] for option in options]
    if values != list(range(1, 8)):
        raise ValueError("dvwu7 response values must be 1 through 7")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(float(value))
        for value in utilities
    ) or any(abs(float(value) - index / 6) > 1e-12 for index, value in enumerate(utilities)):
        raise ValueError("dvwu7 normalized utility must increase linearly from 0 to 1")
    if any(not isinstance(option["label"], str) or not option["label"].strip() for option in options):
        raise ValueError("dvwu7 response labels must be non-empty")

    interaction = bundle["interaction_contract"]
    expected_interaction = {
        "sequence_unit": "synthetic_persona",
        "paired_across_arms": True,
        "same_persona_state_across_arms": True,
        "separate_arm_conversations": True,
        "arm_generated_reflection_reused": False,
        "target_position": "after_arm_specific_writing_then_common_cancer_message",
    }
    if interaction != expected_interaction:
        raise ValueError("dvwu7 interactive pairing contract is not exact")


def _validated_persona(persona_state: Mapping[str, Any]) -> dict[str, Any]:
    assert_blinded_payload(persona_state)
    if set(persona_state) != PERSONA_FIELDS:
        raise ValueError("dvwu7 persona state has unexpected or missing fields")
    persona = dict(persona_state)
    for field in (
        "persona_id",
        "most_important_value",
        "least_important_value",
        "profile_summary",
    ):
        _require_text(persona[field], field=f"persona_state.{field}")
    age = persona["age"]
    if not isinstance(age, int) or isinstance(age, bool) or not 40 <= age <= 70:
        raise ValueError("dvwu7 persona age must be an integer from 40 through 70")
    bmi = persona["bmi"]
    if (
        not isinstance(bmi, (int, float))
        or isinstance(bmi, bool)
        or not isfinite(float(bmi))
        or not 25 <= float(bmi) <= 55
    ):
        raise ValueError("dvwu7 persona BMI must be finite and between 25 and 55")
    if persona["sex"] not in {"female", "male"}:
        raise ValueError("dvwu7 persona sex must follow the source female/male routing")
    if persona["most_important_value"].casefold() == persona[
        "least_important_value"
    ].casefold():
        raise ValueError("most- and least-important values must be different")
    return persona


def load_dvwu7_interactive_bundle(repository_root: Path) -> dict[str, Any]:
    """Load and validate the repository's frozen dvwu7 blinded bundle."""

    path = (
        repository_root
        / "data"
        / "manifests"
        / "contracts"
        / "dvwu7_interactive_blinded_bundle.json"
    )
    with path.open(encoding="utf-8") as stream:
        bundle = json.load(stream)
    validate_dvwu7_interactive_bundle(bundle)
    return bundle


def materialize_dvwu7_pair(
    bundle: Mapping[str, Any],
    *,
    persona_state: Mapping[str, Any],
    seed: int,
) -> Dvwu7InteractivePair:
    """Materialize two deterministic arm-local conversations for one persona.

    This function performs no inference.  ``seed`` identifies the paired
    synthetic replicate; it never selects a different persona or passage for
    one arm.  A future runner must begin each arm from a fresh conversation,
    issue ``treatment_prompt``, retain that arm's generated reflection only in
    that conversation, and then issue the common ``outcome_prompt``.
    """

    validate_dvwu7_interactive_bundle(bundle)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("dvwu7 paired seed must be a non-negative integer")
    persona = _validated_persona(persona_state)
    persona_sha = payload_hash(persona)
    pair_id = payload_hash(
        {
            "task_id": bundle["task_id"],
            "persona_state_sha256": persona_sha,
            "seed": seed,
        }
    )
    persona_json = canonical_json_bytes(persona).decode("utf-8")
    stable_header = (
        "Simulate exactly one synthetic persona in a design-only paired episode. "
        "Do not use any target-human response, outcome, aggregate, or reasoning. "
        "This arm runs in a fresh conversation, but the stable persona state below "
        "is byte-identical across both intervention arms.\n"
        f"Pair ID: {pair_id}\n"
        f"Persona-state SHA-256: {persona_sha}\n"
        f"Stable persona state: {persona_json}\n"
    )
    answers = ", ".join(
        f"{option['value']}={option['label']}"
        for option in bundle["response_options"]
    )
    passage = bundle["passage_routing"][persona["sex"]]
    outcome_prompt = (
        "Continue as the same synthetic persona after privately completing the "
        "preceding arm-specific writing exercise. Do not import or reuse a reflection "
        "from the other arm.\n\n"
        f"Context: {bundle['common_context']}\n\n"
        f"Cancer-risk information:\n{passage}\n\n"
        "Now predict this persona's response distribution for the target question. "
        "Return only one JSON object with exactly this shape: "
        '{"probabilities":{"1":NUMBER,"2":NUMBER,"3":NUMBER,"4":NUMBER,'
        '"5":NUMBER,"6":NUMBER,"7":NUMBER}}. Include every value exactly once; '
        "probabilities must be between 0 and 1 and sum to 1.\n"
        f"Target question: {bundle['outcome_question']}\n"
        f"Target answers: {answers}"
    )
    episodes: list[Dvwu7ArmEpisode] = []
    for arm in bundle["arms"]:
        procedure = arm["procedure"]
        selected_value = persona[procedure["value_selection"]]
        reference_instruction = (
            "Write only about why this value matters to you and how it has influenced "
            "your own past behavior."
            if procedure["self_reference_rule"] == "self"
            else "Write only about why this value could matter to another person; do "
            "not explain why it is important or unimportant to you."
        )
        treatment_prompt = (
            f"{stable_header}\n"
            "Privately complete the source-programmed values writing exercise as this "
            "persona before any cancer-risk information is shown. The generated "
            "reflection is arm-local treatment state and must never be copied to the "
            "other arm.\n"
            f"Selected value: {selected_value}\n"
            f"Instruction: {procedure['writing_prompt']}\n"
            f"Constraint: {reference_instruction}\n"
            'Return only one JSON object with exactly this shape: {"reflection":"STRING"}.'
        )
        episodes.append(
            Dvwu7ArmEpisode(
                arm_id=arm["arm_id"],
                pair_id=pair_id,
                persona_state_sha256=persona_sha,
                seed=seed,
                treatment_prompt=treatment_prompt,
                outcome_prompt=outcome_prompt,
            )
        )
    return Dvwu7InteractivePair(
        pair_id=pair_id,
        persona_state_sha256=persona_sha,
        seed=seed,
        arms=tuple(episodes),
    )

