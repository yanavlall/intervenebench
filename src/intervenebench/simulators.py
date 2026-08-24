"""Strict adapters and aggregation for outcome-blind simulators."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from math import fsum, isfinite
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

from .protocol import assert_blinded_payload


BUNDLE_FIELDS = frozenset(
    {
        "schema_version",
        "task_id",
        "experiment_id",
        "access_regime",
        "population",
        "arms",
        "common_context",
        "outcome_question",
        "response_options",
        "source_material_sha256",
    }
)
ORDINAL_BUNDLE_FIELDS = frozenset(
    {
        "schema_version",
        "task_id",
        "experiment_id",
        "access_regime",
        "population",
        "arms",
        "common_context",
        "outcome_question",
        "response_options",
        "source_material_sha256",
        "outcome_access",
        "reveal_authorized",
    }
)
ORDINAL_PNG_MULTIMODAL_BUNDLE_FIELDS = ORDINAL_BUNDLE_FIELDS
CONTINUOUS_BUNDLE_FIELDS = frozenset(
    {
        "schema_version",
        "task_id",
        "experiment_id",
        "access_regime",
        "population",
        "arms",
        "common_context",
        "outcome_question",
        "response_contract",
        "source_material_sha256",
        "outcome_access",
        "reveal_authorized",
    }
)
SEQUENCE_BUNDLE_FIELDS = frozenset(
    {
        "schema_version",
        "task_id",
        "experiment_id",
        "access_regime",
        "population",
        "arms",
        "common_context",
        "outcome_question",
        "response_options",
        "source_material_sha256",
        "outcome_access",
        "reveal_authorized",
        "sequence_contract",
    }
)
BOUNDED_MULTIMODAL_BUNDLE_FIELDS = frozenset(
    {
        "schema_version",
        "task_id",
        "experiment_id",
        "access_regime",
        "population",
        "arms",
        "common_context",
        "outcome_question",
        "response_contract",
        "nuisance_contract",
        "source_material_sha256",
        "outcome_access",
        "reveal_authorized",
    }
)
CATEGORICAL_MULTIMODAL_BUNDLE_FIELDS = frozenset(
    {
        "schema_version",
        "task_id",
        "experiment_id",
        "access_regime",
        "population",
        "arms",
        "common_context",
        "outcome_question",
        "response_options",
        "source_material_sha256",
        "outcome_access",
        "reveal_authorized",
    }
)


@dataclass(frozen=True, slots=True)
class BinaryProbability:
    yes_probability: float
    no_probability: float


@dataclass(frozen=True, slots=True)
class ContinuousPrediction:
    value: float


@dataclass(frozen=True, slots=True)
class OrdinalDistribution:
    probabilities: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class SequenceEpisode:
    """One reproducible nuisance-randomization path shared across all arms."""

    episode_id: str
    seed: int
    selections: tuple[tuple[str, str], ...]
    prior_exposure: str


@dataclass(frozen=True, slots=True)
class BoundedMultimodalPrompt:
    """One hash-verified prompt with exact local source assets."""

    text: str
    asset_paths: tuple[str, ...]
    asset_sha256: tuple[str, ...]


def validate_blinded_bundle(bundle: Mapping[str, Any]) -> None:
    assert_blinded_payload(bundle)
    extra = sorted(set(bundle) - BUNDLE_FIELDS)
    missing = sorted(BUNDLE_FIELDS - set(bundle))
    if extra:
        raise ValueError(f"unexpected bundle fields: {extra}")
    if missing:
        raise ValueError(f"missing bundle fields: {missing}")
    if bundle["schema_version"] != "blinded_bundle.v1":
        raise ValueError("unsupported blinded-bundle schema")
    if bundle["access_regime"] != "DESIGN_ONLY":
        raise ValueError("Phase 1 simulator bundle must use DESIGN_ONLY")
    arms = bundle["arms"]
    if not isinstance(arms, list) or not 2 <= len(arms) <= 4:
        raise ValueError("bundle must contain two to four arms")
    arm_ids = [arm.get("arm_id") for arm in arms if isinstance(arm, Mapping)]
    if len(arm_ids) != len(arms) or len(set(arm_ids)) != len(arms):
        raise ValueError("bundle arms must have unique IDs")
    if any(set(arm) != {"arm_id", "message"} for arm in arms):
        raise ValueError("bundle arms must contain only arm_id and message")
    options = bundle["response_options"]
    expected_options = [
        {"value": 1, "label": "Yes", "normalized_utility": 1.0},
        {"value": 2, "label": "No", "normalized_utility": 0.0},
    ]
    if options != expected_options:
        raise ValueError("Phase 1 binary bundle has unexpected response options")
    source_hash = bundle["source_material_sha256"]
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("source_material_sha256 must be a SHA-256 digest")


def parse_binary_probability(raw_text: str) -> BinaryProbability:
    """Parse an exact two-key probability object; never recover or impute."""

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ValueError("simulator output is not valid JSON") from error
    if not isinstance(parsed, Mapping) or set(parsed) != {
        "yes_probability",
        "no_probability",
    }:
        raise ValueError("simulator output must contain exactly two probability fields")
    yes = parsed["yes_probability"]
    no = parsed["no_probability"]
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
        or not 0.0 <= value <= 1.0
        for value in (yes, no)
    ):
        raise ValueError("probabilities must be finite numbers in [0, 1]")
    if abs(float(yes) + float(no) - 1.0) > 1e-6:
        raise ValueError("probabilities must sum to one")
    return BinaryProbability(float(yes), float(no))


def validate_continuous_blinded_bundle(bundle: Mapping[str, Any]) -> None:
    assert_blinded_payload(bundle)
    extra = sorted(set(bundle) - CONTINUOUS_BUNDLE_FIELDS)
    missing = sorted(CONTINUOUS_BUNDLE_FIELDS - set(bundle))
    if extra:
        raise ValueError(f"unexpected continuous bundle fields: {extra}")
    if missing:
        raise ValueError(f"missing continuous bundle fields: {missing}")
    if bundle["schema_version"] != "continuous_blinded_bundle.v1":
        raise ValueError("unsupported continuous blinded-bundle schema")
    if bundle["access_regime"] != "DESIGN_ONLY":
        raise ValueError("continuous simulator bundle must use DESIGN_ONLY")
    if bundle["outcome_access"] != "sealed" or bundle["reveal_authorized"] is not False:
        raise ValueError("continuous candidate bundle must remain outcome sealed")
    arms = bundle["arms"]
    if not isinstance(arms, list) or not 2 <= len(arms) <= 4:
        raise ValueError("continuous bundle must contain two to four arms")
    arm_ids = [arm.get("arm_id") for arm in arms if isinstance(arm, Mapping)]
    if len(arm_ids) != len(arms) or len(set(arm_ids)) != len(arms):
        raise ValueError("continuous bundle arms must have unique IDs")
    if any(set(arm) != {"arm_id", "message"} for arm in arms):
        raise ValueError("continuous bundle arms must contain only arm_id and message")
    contract = bundle["response_contract"]
    if not isinstance(contract, Mapping) or set(contract) != {
        "type",
        "unit",
        "minimum",
        "maximum",
    }:
        raise ValueError("continuous response contract has an unexpected shape")
    if contract["type"] != "integer" or contract["minimum"] != 0:
        raise ValueError("continuous bundle must declare a non-negative integer outcome")
    if contract["maximum"] is not None:
        raise ValueError("tcg8p source does not declare a substantive upper bound")
    if not str(contract["unit"]).strip():
        raise ValueError("continuous response unit is required")
    source_hash = bundle["source_material_sha256"]
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("source_material_sha256 must be a SHA-256 digest")


def validate_ordinal_blinded_bundle(bundle: Mapping[str, Any]) -> None:
    """Validate a sealed bounded ordinal simulator bundle.

    Unlike the Phase 1 binary-only adapter, this schema supports source-defined
    ordinal scales while preserving an exact allowlist and a closed reveal gate.
    """

    assert_blinded_payload(bundle)
    extra = sorted(set(bundle) - ORDINAL_BUNDLE_FIELDS)
    missing = sorted(ORDINAL_BUNDLE_FIELDS - set(bundle))
    if extra:
        raise ValueError(f"unexpected ordinal bundle fields: {extra}")
    if missing:
        raise ValueError(f"missing ordinal bundle fields: {missing}")
    if bundle["schema_version"] != "ordinal_blinded_bundle.v1":
        raise ValueError("unsupported ordinal blinded-bundle schema")
    if bundle["access_regime"] != "DESIGN_ONLY":
        raise ValueError("ordinal simulator bundle must use DESIGN_ONLY")
    if bundle["outcome_access"] != "sealed" or bundle["reveal_authorized"] is not False:
        raise ValueError("ordinal candidate bundle must remain outcome sealed")
    for field in ("task_id", "experiment_id", "common_context", "outcome_question"):
        if not isinstance(bundle[field], str) or not bundle[field].strip():
            raise ValueError(f"ordinal bundle requires non-empty {field}")
    population = bundle["population"]
    if not isinstance(population, Mapping) or set(population) != {
        "description",
        "roster_id",
    }:
        raise ValueError("ordinal population contract has an unexpected shape")
    if not all(
        isinstance(population[field], str) and population[field].strip()
        for field in ("description", "roster_id")
    ):
        raise ValueError("ordinal population fields must be non-empty strings")
    arms = bundle["arms"]
    if not isinstance(arms, list) or not 2 <= len(arms) <= 6:
        raise ValueError("ordinal bundle must contain two to six arms")
    arm_ids = [arm.get("arm_id") for arm in arms if isinstance(arm, Mapping)]
    if len(arm_ids) != len(arms) or len(set(arm_ids)) != len(arms):
        raise ValueError("ordinal bundle arms must have unique IDs")
    arm_shapes = {frozenset(arm) for arm in arms}
    direct_shape = frozenset({"arm_id", "message"})
    variant_shape = frozenset({"arm_id", "message_variants"})
    if arm_shapes not in ({direct_shape}, {variant_shape}):
        raise ValueError(
            "ordinal arms must uniformly contain arm_id plus message or "
            "message_variants"
        )
    if any(
        not isinstance(arm["arm_id"], str) or not arm["arm_id"].strip()
        for arm in arms
    ):
        raise ValueError("ordinal arm IDs must be non-empty strings")
    if arm_shapes == {direct_shape}:
        if any(
            not isinstance(arm["message"], str) or not arm["message"].strip()
            for arm in arms
        ):
            raise ValueError("ordinal arm messages must be non-empty strings")
    else:
        expected_variant_contract: tuple[tuple[str, float], ...] | None = None
        for arm in arms:
            variants = arm["message_variants"]
            if not isinstance(variants, list) or len(variants) < 2:
                raise ValueError(
                    "randomized-nuisance arms require at least two message variants"
                )
            required_variant_fields = {"variant_id", "weight", "message"}
            if any(
                not isinstance(variant, Mapping)
                or set(variant) != required_variant_fields
                for variant in variants
            ):
                raise ValueError("message variants have an unexpected shape")
            variant_ids = [variant["variant_id"] for variant in variants]
            weights = [variant["weight"] for variant in variants]
            if (
                any(
                    not isinstance(variant_id, str) or not variant_id.strip()
                    for variant_id in variant_ids
                )
                or len(set(variant_ids)) != len(variant_ids)
                or any(
                    not isinstance(variant["message"], str)
                    or not variant["message"].strip()
                    for variant in variants
                )
            ):
                raise ValueError("message variant IDs and text must be non-empty")
            if any(
                not isinstance(weight, (int, float))
                or isinstance(weight, bool)
                or not isfinite(weight)
                or weight <= 0.0
                for weight in weights
            ) or abs(fsum(float(weight) for weight in weights) - 1.0) > 1e-9:
                raise ValueError("message variant weights must be positive and sum to one")
            contract = tuple(
                (variant_id, float(weight))
                for variant_id, weight in zip(variant_ids, weights, strict=True)
            )
            if expected_variant_contract is None:
                expected_variant_contract = contract
            elif contract != expected_variant_contract:
                raise ValueError(
                    "randomized nuisance variants and weights must match across arms"
                )
    options = bundle["response_options"]
    if not isinstance(options, list) or len(options) < 2:
        raise ValueError("ordinal bundle requires at least two response options")
    required_option_fields = {"value", "label", "normalized_utility"}
    if any(
        not isinstance(option, Mapping) or set(option) != required_option_fields
        for option in options
    ):
        raise ValueError("ordinal response options have an unexpected shape")
    values = [option["value"] for option in options]
    utilities = [option["normalized_utility"] for option in options]
    if any(
        not isinstance(value, int) or isinstance(value, bool) for value in values
    ) or values != list(range(min(values), max(values) + 1)):
        raise ValueError("ordinal response values must be contiguous integers")
    if any(
        not isinstance(utility, (int, float))
        or isinstance(utility, bool)
        or not isfinite(utility)
        or not 0.0 <= utility <= 1.0
        for utility in utilities
    ):
        raise ValueError("ordinal utilities must be finite values in [0, 1]")
    if len(set(utilities)) != len(utilities):
        raise ValueError("ordinal utilities must define a strict ordering")
    if any(
        not isinstance(option["label"], str) or not option["label"].strip()
        for option in options
    ):
        raise ValueError("ordinal response labels must be non-empty strings")
    source_hash = bundle["source_material_sha256"]
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("source_material_sha256 must be a SHA-256 digest")


def _validate_sequence_randomization(randomization: Mapping[str, Any]) -> str:
    common = {"randomization_id", "token", "kind"}
    if not common.issubset(randomization):
        raise ValueError("sequence randomization is missing required fields")
    randomization_id = randomization["randomization_id"]
    token = randomization["token"]
    kind = randomization["kind"]
    if (
        not isinstance(randomization_id, str)
        or not randomization_id.strip()
        or not isinstance(token, str)
        or not token.startswith("{{")
        or not token.endswith("}}")
    ):
        raise ValueError("sequence randomization IDs and tokens must be explicit")
    if kind == "categorical":
        if set(randomization) != common | {"levels"}:
            raise ValueError("categorical sequence randomization has an unexpected shape")
        levels = randomization["levels"]
        if not isinstance(levels, list) or len(levels) < 2:
            raise ValueError("categorical sequence randomization needs at least two levels")
        if any(
            not isinstance(level, Mapping)
            or set(level) != {"level_id", "weight", "text"}
            for level in levels
        ):
            raise ValueError("categorical sequence levels have an unexpected shape")
        level_ids = [level["level_id"] for level in levels]
        weights = [level["weight"] for level in levels]
        if (
            any(not isinstance(value, str) or not value.strip() for value in level_ids)
            or len(set(level_ids)) != len(level_ids)
            or any(not isinstance(level["text"], str) for level in levels)
        ):
            raise ValueError("categorical level IDs must be unique and text must be strings")
        if any(
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not isfinite(weight)
            or weight <= 0.0
            for weight in weights
        ) or abs(fsum(float(weight) for weight in weights) - 1.0) > 1e-9:
            raise ValueError("categorical sequence weights must be positive and sum to one")
    elif kind == "permutation":
        if set(randomization) != common | {"items", "separator"}:
            raise ValueError("permutation sequence randomization has an unexpected shape")
        items = randomization["items"]
        separator = randomization["separator"]
        if not isinstance(items, list) or len(items) < 2:
            raise ValueError("permutation randomization needs at least two items")
        if any(
            not isinstance(item, Mapping) or set(item) != {"item_id", "text"}
            for item in items
        ):
            raise ValueError("permutation items have an unexpected shape")
        item_ids = [item["item_id"] for item in items]
        if (
            any(not isinstance(value, str) or not value.strip() for value in item_ids)
            or len(set(item_ids)) != len(item_ids)
            or any(
                not isinstance(item["text"], str) or not item["text"].strip()
                for item in items
            )
            or not isinstance(separator, str)
        ):
            raise ValueError("permutation items and separator are invalid")
    elif kind == "paired_profiles":
        expected = common | {
            "contexts",
            "pair_count",
            "candidate_labels",
            "traits",
            "trait_order_randomized",
            "questions",
        }
        if set(randomization) != expected:
            raise ValueError("paired-profile randomization has an unexpected shape")
        contexts = randomization["contexts"]
        if not isinstance(contexts, list) or len(contexts) < 2:
            raise ValueError("paired profiles need at least two contexts")
        if any(
            not isinstance(value, Mapping)
            or set(value) != {"level_id", "weight", "text"}
            for value in contexts
        ):
            raise ValueError("paired-profile contexts have an unexpected shape")
        context_ids = [value["level_id"] for value in contexts]
        context_weights = [value["weight"] for value in contexts]
        if (
            any(not isinstance(value, str) or not value.strip() for value in context_ids)
            or len(set(context_ids)) != len(context_ids)
            or any(
                not isinstance(value["text"], str) or not value["text"].strip()
                for value in contexts
            )
            or any(
                not isinstance(weight, (int, float))
                or isinstance(weight, bool)
                or not isfinite(weight)
                or weight <= 0.0
                for weight in context_weights
            )
            or abs(fsum(float(weight) for weight in context_weights) - 1.0) > 1e-9
        ):
            raise ValueError("paired-profile contexts are invalid")
        pair_count = randomization["pair_count"]
        candidate_labels = randomization["candidate_labels"]
        traits = randomization["traits"]
        if (
            not isinstance(pair_count, int)
            or isinstance(pair_count, bool)
            or pair_count < 1
            or not isinstance(candidate_labels, list)
            or len(candidate_labels) != 2
            or any(
                not isinstance(value, str) or not value.strip()
                for value in candidate_labels
            )
            or len(set(candidate_labels)) != 2
            or randomization["trait_order_randomized"] is not True
            or not isinstance(randomization["questions"], str)
            or not randomization["questions"].strip()
        ):
            raise ValueError("paired-profile layout is invalid")
        if not isinstance(traits, list) or len(traits) < 2:
            raise ValueError("paired profiles need at least two traits")
        if any(
            not isinstance(trait, Mapping)
            or set(trait) != {"trait_id", "label", "levels"}
            for trait in traits
        ):
            raise ValueError("paired-profile traits have an unexpected shape")
        trait_ids = [trait["trait_id"] for trait in traits]
        if (
            any(not isinstance(value, str) or not value.strip() for value in trait_ids)
            or len(set(trait_ids)) != len(trait_ids)
            or any(
                not isinstance(trait["label"], str)
                or not trait["label"].strip()
                or not isinstance(trait["levels"], list)
                or len(trait["levels"]) < 2
                for trait in traits
            )
        ):
            raise ValueError("paired-profile trait definitions are invalid")
        for trait in traits:
            levels = trait["levels"]
            if any(
                not isinstance(level, Mapping)
                or set(level) != {"level_id", "text"}
                for level in levels
            ):
                raise ValueError("paired-profile levels have an unexpected shape")
            level_ids = [level["level_id"] for level in levels]
            if (
                any(
                    not isinstance(value, str) or not value.strip()
                    for value in level_ids
                )
                or len(set(level_ids)) != len(level_ids)
                or any(
                    not isinstance(level["text"], str) or not level["text"].strip()
                    for level in levels
                )
            ):
                raise ValueError("paired-profile levels are invalid")
    else:
        raise ValueError("unsupported sequence randomization kind")
    return token


def validate_sequence_blinded_bundle(bundle: Mapping[str, Any]) -> None:
    """Validate a sealed bundle with explicit randomized prior survey exposure."""

    assert_blinded_payload(bundle)
    extra = sorted(set(bundle) - SEQUENCE_BUNDLE_FIELDS)
    missing = sorted(SEQUENCE_BUNDLE_FIELDS - set(bundle))
    if extra:
        raise ValueError(f"unexpected sequence bundle fields: {extra}")
    if missing:
        raise ValueError(f"missing sequence bundle fields: {missing}")
    if bundle["schema_version"] != "sequence_ordinal_blinded_bundle.v1":
        raise ValueError("unsupported sequence blinded-bundle schema")
    if bundle["access_regime"] != "DESIGN_ONLY":
        raise ValueError("sequence simulator bundle must use DESIGN_ONLY")
    if bundle["outcome_access"] != "sealed" or bundle["reveal_authorized"] is not False:
        raise ValueError("sequence candidate bundle must remain outcome sealed")
    for field in ("task_id", "experiment_id", "common_context", "outcome_question"):
        if not isinstance(bundle[field], str) or not bundle[field].strip():
            raise ValueError(f"sequence bundle requires non-empty {field}")
    population = bundle["population"]
    if not isinstance(population, Mapping) or set(population) != {
        "description",
        "roster_id",
    }:
        raise ValueError("sequence population contract has an unexpected shape")
    if any(not isinstance(value, str) or not value.strip() for value in population.values()):
        raise ValueError("sequence population fields must be non-empty strings")
    arms = bundle["arms"]
    if not isinstance(arms, list) or not 2 <= len(arms) <= 6:
        raise ValueError("sequence bundle must contain two to six arms")
    arm_shapes = {frozenset(arm) for arm in arms if isinstance(arm, Mapping)}
    direct_arm_shape = frozenset({"arm_id", "message"})
    templated_arm_shape = frozenset(
        {"arm_id", "message_template", "arm_substitutions"}
    )
    if len(arm_shapes) != 1 or arm_shapes not in (
        {direct_arm_shape},
        {templated_arm_shape},
    ):
        raise ValueError("sequence arms must use one supported shape uniformly")
    if any(
        not isinstance(arm["arm_id"], str) or not arm["arm_id"].strip()
        for arm in arms
    ):
        raise ValueError("sequence arm IDs must be non-empty strings")
    arm_substitution_tokens: set[str] = set()
    if arm_shapes == {direct_arm_shape}:
        if any(
            not isinstance(arm["message"], str) or not arm["message"].strip()
            for arm in arms
        ):
            raise ValueError("sequence arm messages must be non-empty strings")
    else:
        expected_substitutions: set[str] | None = None
        for arm in arms:
            if (
                not isinstance(arm["message_template"], str)
                or not arm["message_template"].strip()
                or not isinstance(arm["arm_substitutions"], Mapping)
                or not arm["arm_substitutions"]
            ):
                raise ValueError("templated sequence arms require template substitutions")
            keys = set(arm["arm_substitutions"])
            if any(
                not isinstance(key, str)
                or not key.startswith("{{")
                or not key.endswith("}}")
                or not isinstance(value, str)
                or not value.strip()
                for key, value in arm["arm_substitutions"].items()
            ):
                raise ValueError("arm substitutions must map explicit tokens to text")
            if expected_substitutions is None:
                expected_substitutions = keys
            elif keys != expected_substitutions:
                raise ValueError("arm substitution tokens must match across arms")
        arm_substitution_tokens = expected_substitutions or set()
    arm_ids = [arm["arm_id"] for arm in arms]
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError("sequence arm IDs must be unique")

    contract = bundle["sequence_contract"]
    required_contract_fields = {
        "sequence_unit",
        "paired_across_arms",
        "target_position",
        "prior_exposure_template",
        "randomizations",
    }
    if not isinstance(contract, Mapping) or set(contract) != required_contract_fields:
        raise ValueError("sequence contract has an unexpected shape")
    if contract["sequence_unit"] != "synthetic_persona":
        raise ValueError("sequence unit must be synthetic_persona")
    if contract["paired_across_arms"] is not True:
        raise ValueError("the same randomized sequence must be paired across arms")
    if contract["target_position"] != "stop_immediately_after_target_question":
        raise ValueError("sequence contract must stop immediately after the target question")
    template = contract["prior_exposure_template"]
    randomizations = contract["randomizations"]
    if not isinstance(template, str) or not isinstance(randomizations, list) or not randomizations:
        raise ValueError("sequence contract needs a template and randomizations")
    tokens = [_validate_sequence_randomization(value) for value in randomizations]
    randomization_ids = [value["randomization_id"] for value in randomizations]
    if len(set(tokens)) != len(tokens) or len(set(randomization_ids)) != len(randomization_ids):
        raise ValueError("sequence tokens and randomization IDs must be unique")
    declared_tokens = set(tokens)
    templates_to_scan = [template]
    if arm_shapes == {templated_arm_shape}:
        templates_to_scan.extend(arm["message_template"] for arm in arms)
    visible_tokens = {
        "{{" + part.split("}}", 1)[0] + "}}"
        for value in templates_to_scan
        for part in value.split("{{")[1:]
        if "}}" in part
    }
    for randomization in randomizations:
        if randomization["kind"] == "categorical":
            for level in randomization["levels"]:
                visible_tokens.update(
                    "{{" + part.split("}}", 1)[0] + "}}"
                    for part in level["text"].split("{{")[1:]
                    if "}}" in part
                )
    if visible_tokens != declared_tokens | arm_substitution_tokens:
        raise ValueError("sequence template tokens must exactly match declared randomizations")
    options = bundle["response_options"]
    if not isinstance(options, list) or len(options) < 2:
        raise ValueError("sequence bundle requires ordinal response options")
    if any(
        not isinstance(option, Mapping)
        or set(option) != {"value", "label", "normalized_utility"}
        for option in options
    ):
        raise ValueError("sequence response options have an unexpected shape")
    values = [option["value"] for option in options]
    if (
        any(not isinstance(value, int) or isinstance(value, bool) for value in values)
        or values != list(range(min(values), max(values) + 1))
        or any(
            not isinstance(option["normalized_utility"], (int, float))
            or isinstance(option["normalized_utility"], bool)
            or not 0.0 <= option["normalized_utility"] <= 1.0
            or not isinstance(option["label"], str)
            or not option["label"].strip()
            for option in options
        )
    ):
        raise ValueError("sequence response options are invalid")
    source_hash = bundle["source_material_sha256"]
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("source_material_sha256 must be a SHA-256 digest")


def validate_bounded_multimodal_bundle(bundle: Mapping[str, Any]) -> None:
    """Validate a sealed image/PDF bundle with a randomized nuisance factor."""

    assert_blinded_payload(bundle)
    extra = sorted(set(bundle) - BOUNDED_MULTIMODAL_BUNDLE_FIELDS)
    missing = sorted(BOUNDED_MULTIMODAL_BUNDLE_FIELDS - set(bundle))
    if extra:
        raise ValueError(f"unexpected multimodal bundle fields: {extra}")
    if missing:
        raise ValueError(f"missing multimodal bundle fields: {missing}")
    if bundle["schema_version"] != "bounded_numeric_multimodal_bundle.v1":
        raise ValueError("unsupported bounded multimodal bundle schema")
    if bundle["access_regime"] != "DESIGN_ONLY":
        raise ValueError("multimodal simulator bundle must use DESIGN_ONLY")
    if bundle["outcome_access"] != "sealed" or bundle["reveal_authorized"] is not False:
        raise ValueError("multimodal candidate bundle must remain outcome sealed")
    for field in ("task_id", "experiment_id", "common_context", "outcome_question"):
        if not isinstance(bundle[field], str) or not bundle[field].strip():
            raise ValueError(f"multimodal bundle requires non-empty {field}")
    population = bundle["population"]
    if not isinstance(population, Mapping) or set(population) != {
        "description",
        "roster_id",
    }:
        raise ValueError("multimodal population contract has an unexpected shape")

    arms = bundle["arms"]
    if not isinstance(arms, list) or not 2 <= len(arms) <= 6:
        raise ValueError("multimodal bundle must contain two to six arms")
    if any(
        not isinstance(arm, Mapping)
        or set(arm) != {"arm_id", "accessible_text", "asset"}
        for arm in arms
    ):
        raise ValueError("multimodal arms have an unexpected shape")
    arm_ids = [arm["arm_id"] for arm in arms]
    if (
        len(set(arm_ids)) != len(arm_ids)
        or any(not isinstance(value, str) or not value.strip() for value in arm_ids)
        or any(
            not isinstance(arm["accessible_text"], str)
            or not arm["accessible_text"].strip()
            for arm in arms
        )
    ):
        raise ValueError("multimodal arm IDs and accessible text must be non-empty")
    for arm in arms:
        asset = arm["asset"]
        if not isinstance(asset, Mapping) or set(asset) != {
            "path",
            "mime_type",
            "sha256",
            "page",
        }:
            raise ValueError("multimodal asset has an unexpected shape")
        path = asset["path"]
        if (
            not isinstance(path, str)
            or not path.startswith("data/raw/sources/")
            or ".." in Path(path).parts
            or asset["mime_type"] != "application/pdf"
            or asset["page"] != 1
            or not isinstance(asset["sha256"], str)
            or len(asset["sha256"]) != 64
        ):
            raise ValueError("multimodal source asset declaration is invalid")

    response = bundle["response_contract"]
    if not isinstance(response, Mapping) or set(response) != {
        "type",
        "unit",
        "minimum",
        "maximum",
    }:
        raise ValueError("bounded response contract has an unexpected shape")
    if (
        response["type"] != "integer"
        or not isinstance(response["minimum"], int)
        or isinstance(response["minimum"], bool)
        or not isinstance(response["maximum"], int)
        or isinstance(response["maximum"], bool)
        or response["minimum"] >= response["maximum"]
        or not isinstance(response["unit"], str)
        or not response["unit"].strip()
    ):
        raise ValueError("bounded response contract is invalid")

    nuisance = bundle["nuisance_contract"]
    if not isinstance(nuisance, Mapping) or set(nuisance) != {
        "paired_across_arms",
        "levels",
    } or nuisance["paired_across_arms"] is not True:
        raise ValueError("nuisance contract must be paired across arms")
    levels = nuisance["levels"]
    if not isinstance(levels, list) or len(levels) < 2 or any(
        not isinstance(level, Mapping)
        or set(level) != {"nuisance_id", "weight", "context"}
        for level in levels
    ):
        raise ValueError("nuisance levels have an unexpected shape")
    nuisance_ids = [level["nuisance_id"] for level in levels]
    nuisance_weights = [level["weight"] for level in levels]
    if (
        len(set(nuisance_ids)) != len(nuisance_ids)
        or any(not isinstance(value, str) or not value.strip() for value in nuisance_ids)
        or any(
            not isinstance(level["context"], str) or not level["context"].strip()
            for level in levels
        )
        or any(
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not isfinite(weight)
            or weight <= 0.0
            for weight in nuisance_weights
        )
        or abs(fsum(float(weight) for weight in nuisance_weights) - 1.0) > 1e-9
    ):
        raise ValueError("nuisance IDs and weights are invalid")
    source_hash = bundle["source_material_sha256"]
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("source_material_sha256 must be a SHA-256 digest")


def bounded_multimodal_prompt(
    bundle: Mapping[str, Any],
    *,
    arm_id: str,
    nuisance_id: str,
    repository_root: Path,
) -> BoundedMultimodalPrompt:
    """Materialize a prompt only after verifying the exact fielded asset bytes."""

    validate_bounded_multimodal_bundle(bundle)
    arms = {arm["arm_id"]: arm for arm in bundle["arms"]}
    nuisance = {
        level["nuisance_id"]: level for level in bundle["nuisance_contract"]["levels"]
    }
    if arm_id not in arms or nuisance_id not in nuisance:
        raise ValueError("requested arm or nuisance level is absent from bundle")
    arm = arms[arm_id]
    asset = arm["asset"]
    root = repository_root.resolve()
    path = (root / asset["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("multimodal asset path is missing or outside the repository")
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != asset["sha256"]:
        raise ValueError("multimodal asset hash does not match the frozen contract")
    response = bundle["response_contract"]
    keys = ",".join(
        f'"{value}":NUMBER'
        for value in range(response["minimum"], response["maximum"] + 1)
    )
    text = (
        "Estimate the response distribution for a random adult after viewing the "
        "attached exact fielded source asset. Do not answer as yourself. Return only "
        f'one JSON object with exactly this shape: {{"probabilities":{{{keys}}}}}. '
        "Include every value once; probabilities must be between 0 and 1 and sum to 1.\n\n"
        f"Population: {bundle['population']['description']}\n\n"
        f"Context: {bundle['common_context']}\n\n"
        f"Accessible text from the attached asset: {arm['accessible_text']}\n\n"
        f"Randomized recipient context: {nuisance[nuisance_id]['context']}\n\n"
        f"Question: {bundle['outcome_question']}\n"
        f"Valid answers: whole numbers {response['minimum']} through "
        f"{response['maximum']} {response['unit']}."
    )
    return BoundedMultimodalPrompt(
        text=text,
        asset_paths=(str(path),),
        asset_sha256=(digest,),
    )


def aggregate_bounded_multimodal_predictions(
    outputs: Iterable[Mapping[str, Any]],
    *,
    bundle: Mapping[str, Any],
    draws: int,
) -> dict[str, float]:
    """Aggregate expected normalized utility over paired nuisance paths and draws."""

    validate_bounded_multimodal_bundle(bundle)
    if draws <= 0:
        raise ValueError("draws must be positive")
    arm_ids = tuple(arm["arm_id"] for arm in bundle["arms"])
    levels = bundle["nuisance_contract"]["levels"]
    nuisance_weights = {
        level["nuisance_id"]: float(level["weight"]) for level in levels
    }
    response = bundle["response_contract"]
    option_values = tuple(range(response["minimum"], response["maximum"] + 1))
    expected = {
        (arm_id, nuisance_id, draw_index)
        for arm_id in arm_ids
        for nuisance_id in nuisance_weights
        for draw_index in range(draws)
    }
    predictions: dict[tuple[str, str, int], float] = {}
    denominator = response["maximum"] - response["minimum"]
    for output in outputs:
        key = (
            str(output.get("arm_id")),
            str(output.get("nuisance_id")),
            output.get("draw_index"),
        )
        if key in predictions:
            raise ValueError("duplicate multimodal arm/nuisance/draw prediction")
        probabilities = output.get("probabilities")
        if key not in expected or not isinstance(probabilities, Mapping):
            raise ValueError("invalid multimodal arm/nuisance/draw prediction")
        parsed = parse_ordinal_distribution(
            json.dumps({"probabilities": probabilities}),
            option_values=option_values,
        )
        predictions[key] = fsum(
            ((value - response["minimum"]) / denominator) * probability
            for value, probability in parsed.probabilities
        )
    if set(predictions) != expected:
        raise ValueError("multimodal predictions are not complete and paired")
    return {
        arm_id: fsum(
            nuisance_weights[nuisance_id]
            * fsum(
                predictions[(arm_id, nuisance_id, draw_index)]
                for draw_index in range(draws)
            )
            / draws
            for nuisance_id in nuisance_weights
        )
        for arm_id in arm_ids
    }


def validate_categorical_multimodal_bundle(bundle: Mapping[str, Any]) -> None:
    """Validate a result-free menu/image bundle with categorical choice utility."""

    assert_blinded_payload(bundle)
    extra = sorted(set(bundle) - CATEGORICAL_MULTIMODAL_BUNDLE_FIELDS)
    missing = sorted(CATEGORICAL_MULTIMODAL_BUNDLE_FIELDS - set(bundle))
    if extra or missing:
        raise ValueError(
            f"categorical multimodal fields mismatch; extra={extra}, missing={missing}"
        )
    if bundle["schema_version"] != "categorical_multimodal_bundle.v1":
        raise ValueError("unsupported categorical multimodal bundle schema")
    if bundle["access_regime"] != "DESIGN_ONLY":
        raise ValueError("categorical multimodal bundle must use DESIGN_ONLY")
    if bundle["outcome_access"] not in {
        "sealed",
        "result_text_exposed_non_test",
    } or bundle["reveal_authorized"] is not False:
        raise ValueError("categorical multimodal bundle has invalid access state")
    for field in ("task_id", "experiment_id", "common_context", "outcome_question"):
        if not isinstance(bundle[field], str) or not bundle[field].strip():
            raise ValueError(f"categorical multimodal bundle requires non-empty {field}")
    population = bundle["population"]
    if not isinstance(population, Mapping) or set(population) != {
        "description",
        "roster_id",
    }:
        raise ValueError("categorical multimodal population has an unexpected shape")
    arms = bundle["arms"]
    if not isinstance(arms, list) or not 2 <= len(arms) <= 6 or any(
        not isinstance(arm, Mapping)
        or set(arm) != {"arm_id", "accessible_text", "asset"}
        for arm in arms
    ):
        raise ValueError("categorical multimodal arms have an unexpected shape")
    arm_ids = [arm["arm_id"] for arm in arms]
    if len(set(arm_ids)) != len(arm_ids) or any(
        not isinstance(arm_id, str) or not arm_id.strip() for arm_id in arm_ids
    ):
        raise ValueError("categorical multimodal arm IDs must be unique")
    for arm in arms:
        if not isinstance(arm["accessible_text"], str) or not arm[
            "accessible_text"
        ].strip():
            raise ValueError("categorical multimodal accessible text is required")
        asset = arm["asset"]
        if not isinstance(asset, Mapping) or set(asset) != {
            "path",
            "mime_type",
            "sha256",
            "source_container_path",
            "source_member",
        }:
            raise ValueError("categorical multimodal asset has an unexpected shape")
        if (
            not isinstance(asset["path"], str)
            or not asset["path"].startswith("data/derived/stimuli/")
            or ".." in Path(asset["path"]).parts
            or asset["mime_type"] != "image/png"
            or not isinstance(asset["sha256"], str)
            or len(asset["sha256"]) != 64
            or not isinstance(asset["source_container_path"], str)
            or not asset["source_container_path"].startswith("data/raw/sources/")
            or not isinstance(asset["source_member"], str)
            or not asset["source_member"].startswith("word/media/")
        ):
            raise ValueError("categorical multimodal source asset declaration is invalid")
    options = bundle["response_options"]
    if not isinstance(options, list) or len(options) < 2 or any(
        not isinstance(option, Mapping)
        or set(option) != {"option_id", "label", "normalized_utility"}
        for option in options
    ):
        raise ValueError("categorical response options have an unexpected shape")
    option_ids = [option["option_id"] for option in options]
    if len(set(option_ids)) != len(option_ids) or any(
        not isinstance(option["option_id"], str)
        or not option["option_id"].strip()
        or not isinstance(option["label"], str)
        or not option["label"].strip()
        or not isinstance(option["normalized_utility"], (int, float))
        or isinstance(option["normalized_utility"], bool)
        or not 0.0 <= option["normalized_utility"] <= 1.0
        for option in options
    ):
        raise ValueError("categorical response options are invalid")
    source_hash = bundle["source_material_sha256"]
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("source_material_sha256 must be a SHA-256 digest")


def categorical_multimodal_prompt(
    bundle: Mapping[str, Any], *, arm_id: str, repository_root: Path
) -> BoundedMultimodalPrompt:
    """Materialize a categorical prompt after checking the embedded source image."""

    validate_categorical_multimodal_bundle(bundle)
    arms = {arm["arm_id"]: arm for arm in bundle["arms"]}
    if arm_id not in arms:
        raise ValueError("requested categorical arm is absent from bundle")
    arm = arms[arm_id]
    asset = arm["asset"]
    root = repository_root.resolve()
    path = (root / asset["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("categorical multimodal asset is missing or outside repository")
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != asset["sha256"]:
        raise ValueError("categorical multimodal asset hash does not match")
    option_keys = ",".join(
        f'"{option["option_id"]}":NUMBER' for option in bundle["response_options"]
    )
    option_list = "; ".join(
        f'{option["option_id"]}={option["label"]}'
        for option in bundle["response_options"]
    )
    text = (
        "Estimate which single menu item a random adult would choose after viewing "
        "the attached exact fielded menu. Do not answer as yourself. Return only one "
        f'JSON object with exactly this shape: {{"probabilities":{{{option_keys}}}}}. '
        "Include every option exactly once; probabilities must be between 0 and 1 "
        "and sum to 1.\n\n"
        f"Population: {bundle['population']['description']}\n\n"
        f"Context: {bundle['common_context']}\n\n"
        f"Accessible image description: {arm['accessible_text']}\n\n"
        f"Question: {bundle['outcome_question']}\n\nOptions: {option_list}"
    )
    return BoundedMultimodalPrompt(
        text=text, asset_paths=(str(path),), asset_sha256=(digest,)
    )


def aggregate_categorical_multimodal_predictions(
    outputs: Iterable[Mapping[str, Any]], *, bundle: Mapping[str, Any], draws: int
) -> dict[str, float]:
    """Aggregate expected normalized choice utility over repeated model draws."""

    validate_categorical_multimodal_bundle(bundle)
    if draws <= 0:
        raise ValueError("draws must be positive")
    arm_ids = tuple(arm["arm_id"] for arm in bundle["arms"])
    utilities = {
        option["option_id"]: float(option["normalized_utility"])
        for option in bundle["response_options"]
    }
    expected = {(arm_id, draw_index) for arm_id in arm_ids for draw_index in range(draws)}
    predictions: dict[tuple[str, int], float] = {}
    for output in outputs:
        key = (str(output.get("arm_id")), output.get("draw_index"))
        if key in predictions:
            raise ValueError("duplicate categorical multimodal arm/draw prediction")
        probabilities = output.get("probabilities")
        if key not in expected or not isinstance(probabilities, Mapping):
            raise ValueError("invalid categorical multimodal prediction")
        if set(probabilities) != set(utilities) or any(
            not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not isfinite(probability)
            or not 0.0 <= probability <= 1.0
            for probability in probabilities.values()
        ) or abs(fsum(float(value) for value in probabilities.values()) - 1.0) > 1e-6:
            raise ValueError("categorical probabilities must cover options and sum to one")
        predictions[key] = fsum(
            utilities[option_id] * float(probability)
            for option_id, probability in probabilities.items()
        )
    if set(predictions) != expected:
        raise ValueError("categorical predictions are not complete across arms and draws")
    return {
        arm_id: fsum(predictions[(arm_id, draw)] for draw in range(draws)) / draws
        for arm_id in arm_ids
    }


def validate_ordinal_png_multimodal_bundle(bundle: Mapping[str, Any]) -> None:
    """Validate a sealed ordinal task with one exact PNG asset per arm."""

    assert_blinded_payload(bundle)
    extra = sorted(set(bundle) - ORDINAL_PNG_MULTIMODAL_BUNDLE_FIELDS)
    missing = sorted(ORDINAL_PNG_MULTIMODAL_BUNDLE_FIELDS - set(bundle))
    if extra or missing:
        raise ValueError(
            f"ordinal PNG multimodal fields mismatch; extra={extra}, missing={missing}"
        )
    if bundle["schema_version"] != "ordinal_png_multimodal_bundle.v1":
        raise ValueError("unsupported ordinal PNG multimodal bundle schema")
    if bundle["access_regime"] != "DESIGN_ONLY":
        raise ValueError("ordinal PNG multimodal bundle must use DESIGN_ONLY")
    if bundle["outcome_access"] != "sealed" or bundle["reveal_authorized"] is not False:
        raise ValueError("ordinal PNG multimodal bundle must remain outcome sealed")

    arms = bundle["arms"]
    if not isinstance(arms, list) or not 2 <= len(arms) <= 6:
        raise ValueError("ordinal PNG multimodal bundle must contain two to six arms")
    if any(
        not isinstance(arm, Mapping)
        or set(arm) != {"arm_id", "accessible_text", "asset"}
        for arm in arms
    ):
        raise ValueError("ordinal PNG multimodal arms have an unexpected shape")
    for arm in arms:
        if (
            not isinstance(arm["arm_id"], str)
            or not arm["arm_id"].strip()
            or not isinstance(arm["accessible_text"], str)
            or not arm["accessible_text"].strip()
        ):
            raise ValueError("ordinal PNG arm IDs and accessible text are required")
        asset = arm["asset"]
        if not isinstance(asset, Mapping) or set(asset) != {
            "path",
            "mime_type",
            "sha256",
        }:
            raise ValueError("ordinal PNG asset has an unexpected shape")
        path = asset["path"]
        digest = asset["sha256"]
        if (
            not isinstance(path, str)
            or not path.startswith("data/derived/stimuli/")
            or ".." in Path(path).parts
            or Path(path).suffix != ".png"
            or asset["mime_type"] != "image/png"
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("ordinal PNG source asset declaration is invalid")

    # Reuse the ordinal validator for IDs, population, scale, utility ordering,
    # source provenance, and the closed outcome-access state.  Accessible image
    # text substitutes only for its required direct-message shape.
    ordinal_surrogate = dict(bundle)
    ordinal_surrogate["schema_version"] = "ordinal_blinded_bundle.v1"
    ordinal_surrogate["arms"] = [
        {"arm_id": arm["arm_id"], "message": arm["accessible_text"]}
        for arm in arms
    ]
    validate_ordinal_blinded_bundle(ordinal_surrogate)


def ordinal_png_multimodal_prompt(
    bundle: Mapping[str, Any],
    *,
    arm_id: str,
    repository_root: Path,
    option_order: str = "source",
) -> BoundedMultimodalPrompt:
    """Materialize one ordinal image prompt after exact path/hash validation."""

    validate_ordinal_png_multimodal_bundle(bundle)
    if option_order not in {"source", "reverse"}:
        raise ValueError("option_order must be source or reverse")
    arms = {arm["arm_id"]: arm for arm in bundle["arms"]}
    if arm_id not in arms:
        raise ValueError("requested ordinal PNG arm is absent from bundle")
    arm = arms[arm_id]
    asset = arm["asset"]
    root = repository_root.resolve()
    path = (root / asset["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("ordinal PNG asset is missing or outside the repository")
    contents = path.read_bytes()
    if not contents.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("ordinal PNG asset does not have a PNG signature")
    digest = sha256(contents).hexdigest()
    if digest != asset["sha256"]:
        raise ValueError("ordinal PNG asset hash does not match the frozen contract")

    options = list(bundle["response_options"])
    if option_order == "reverse":
        options.reverse()
    option_keys = ",".join(f'"{option["value"]}":NUMBER' for option in options)
    option_list = "; ".join(
        f'{option["value"]}={option["label"]}' for option in options
    )
    mapping_note = (
        " Output keys remain the original response values; do not renumber them."
        if option_order == "reverse"
        else ""
    )
    text = (
        "Estimate the response distribution for a random adult after viewing the "
        "attached exact fielded image. Do not answer as yourself. Return only one "
        f'JSON object with exactly this shape: {{"probabilities":{{{option_keys}}}}}. '
        "Include every response value exactly once; probabilities must be between "
        f"0 and 1 and sum to 1.{mapping_note}\n\n"
        f"Population: {bundle['population']['description']}\n\n"
        f"Context: {bundle['common_context']}\n\n"
        f"Accessible image description: {arm['accessible_text']}\n\n"
        f"Question: {bundle['outcome_question']}\n\nOptions: {option_list}"
    )
    return BoundedMultimodalPrompt(
        text=text, asset_paths=(str(path),), asset_sha256=(digest,)
    )


def aggregate_ordinal_png_multimodal_predictions(
    outputs: Iterable[Mapping[str, Any]],
    *,
    bundle: Mapping[str, Any],
    draws: int,
) -> dict[str, float]:
    """Aggregate a complete arm-by-draw grid using ordinal utility values."""

    validate_ordinal_png_multimodal_bundle(bundle)
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    rows = tuple(outputs)
    assert_blinded_payload(rows)
    arm_ids = tuple(arm["arm_id"] for arm in bundle["arms"])
    expected = {
        (arm_id, draw_index)
        for arm_id in arm_ids
        for draw_index in range(draws)
    }
    utilities = {
        int(option["value"]): float(option["normalized_utility"])
        for option in bundle["response_options"]
    }
    option_values = tuple(utilities)
    predictions: dict[tuple[str, int], float] = {}
    for output in rows:
        draw_index = output.get("draw_index")
        if isinstance(draw_index, bool) or not isinstance(draw_index, int):
            raise ValueError("invalid ordinal PNG arm/draw prediction")
        key = (str(output.get("arm_id")), draw_index)
        if key in predictions:
            raise ValueError("duplicate ordinal PNG arm/draw prediction")
        probabilities = output.get("probabilities")
        if key not in expected or not isinstance(probabilities, Mapping):
            raise ValueError("invalid ordinal PNG arm/draw prediction")
        parsed = parse_ordinal_distribution(
            json.dumps({"probabilities": probabilities}),
            option_values=option_values,
        )
        predictions[key] = fsum(
            utilities[value] * probability
            for value, probability in parsed.probabilities
        )
    if set(predictions) != expected:
        raise ValueError("ordinal PNG predictions are not complete and paired")
    return {
        arm_id: fsum(
            predictions[(arm_id, draw_index)] for draw_index in range(draws)
        )
        / draws
        for arm_id in arm_ids
    }


def _unit_interval(seed: int, randomization_id: str) -> float:
    digest = sha256(f"{seed}:{randomization_id}".encode("utf-8")).digest()
    return int.from_bytes(digest, "big") / (1 << (8 * len(digest)))


def _sequence_replacements(
    bundle: Mapping[str, Any], *, seed: int
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    contract = bundle["sequence_contract"]
    replacements: dict[str, str] = {}
    selections: list[tuple[str, str]] = []
    for randomization in contract["randomizations"]:
        randomization_id = randomization["randomization_id"]
        token = randomization["token"]
        if randomization["kind"] == "categorical":
            draw = _unit_interval(seed, randomization_id)
            cumulative = 0.0
            selected = randomization["levels"][-1]
            for level in randomization["levels"]:
                cumulative += float(level["weight"])
                if draw < cumulative:
                    selected = level
                    break
            replacements[token] = selected["text"]
            selections.append((randomization_id, selected["level_id"]))
        elif randomization["kind"] == "permutation":
            ordered = sorted(
                randomization["items"],
                key=lambda item: sha256(
                    f"{seed}:{randomization_id}:{item['item_id']}".encode("utf-8")
                ).digest(),
            )
            replacements[token] = randomization["separator"].join(
                item["text"] for item in ordered
            )
            selections.append(
                (randomization_id, ">".join(item["item_id"] for item in ordered))
            )
        else:
            contexts = randomization["contexts"]
            draw = _unit_interval(seed, f"{randomization_id}:context")
            cumulative = 0.0
            context = contexts[-1]
            for value in contexts:
                cumulative += float(value["weight"])
                if draw < cumulative:
                    context = value
                    break
            candidate_labels = randomization["candidate_labels"]
            traits = randomization["traits"]
            selection_parts = [f"context={context['level_id']}"]
            rendered_pairs: list[str] = []
            for pair_index in range(1, randomization["pair_count"] + 1):
                ordered_traits = sorted(
                    traits,
                    key=lambda trait: sha256(
                        f"{seed}:{randomization_id}:pair:{pair_index}:order:{trait['trait_id']}".encode(
                            "utf-8"
                        )
                    ).digest(),
                )
                lines = [
                    f"Pair {pair_index}. Office sought: {context['text']}.",
                    "Please review the two randomized candidate profiles:",
                ]
                selection_parts.append(
                    f"pair{pair_index}_order="
                    + ">".join(trait["trait_id"] for trait in ordered_traits)
                )
                for trait in ordered_traits:
                    candidate_values: list[str] = []
                    candidate_ids: list[str] = []
                    for candidate_index, candidate_label in enumerate(
                        candidate_labels, start=1
                    ):
                        levels = trait["levels"]
                        level_draw = _unit_interval(
                            seed,
                            f"{randomization_id}:pair:{pair_index}:candidate:{candidate_index}:trait:{trait['trait_id']}",
                        )
                        level = levels[min(int(level_draw * len(levels)), len(levels) - 1)]
                        candidate_values.append(f"{candidate_label}={level['text']}")
                        candidate_ids.append(
                            f"candidate{candidate_index}:{trait['trait_id']}={level['level_id']}"
                        )
                    lines.append(f"{trait['label']}: " + "; ".join(candidate_values))
                    selection_parts.extend(
                        f"pair{pair_index}_{value}" for value in candidate_ids
                    )
                lines.append(randomization["questions"])
                rendered_pairs.append("\n".join(lines))
            replacements[token] = (
                "A prior randomized candidate-profile module appeared.\n"
                + "\n\n".join(rendered_pairs)
            )
            selections.append((randomization_id, "|".join(selection_parts)))
    return replacements, selections


def _replace_tokens(template: str, replacements: Mapping[str, str]) -> str:
    rendered = template
    for _ in range(len(replacements) + 1):
        updated = rendered
        for token, value in replacements.items():
            updated = updated.replace(token, value)
        if updated == rendered:
            break
        rendered = updated
    return rendered


def materialize_sequence_episode(
    bundle: Mapping[str, Any], *, seed: int
) -> SequenceEpisode:
    """Materialize one deterministic nuisance path independent of intervention arm."""

    validate_sequence_blinded_bundle(bundle)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("sequence seed must be a non-negative integer")
    replacements, selections = _sequence_replacements(bundle, seed=seed)
    contract = bundle["sequence_contract"]
    prior_exposure = contract["prior_exposure_template"]
    prior_exposure = _replace_tokens(prior_exposure, replacements)
    if "{{" in prior_exposure or "}}" in prior_exposure:
        raise ValueError("sequence episode contains an unresolved randomization token")
    episode_payload = {
        "experiment_id": bundle["experiment_id"],
        "seed": seed,
        "selections": selections,
        "prior_exposure": prior_exposure,
    }
    episode_id = sha256(
        json.dumps(episode_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return SequenceEpisode(
        episode_id=episode_id,
        seed=seed,
        selections=tuple(selections),
        prior_exposure=prior_exposure.strip(),
    )


def sequence_probability_prompt(
    bundle: Mapping[str, Any], *, arm_id: str, episode: SequenceEpisode
) -> str:
    """Build one chronological target prompt from a pre-materialized paired episode."""

    validate_sequence_blinded_bundle(bundle)
    arms = {arm["arm_id"]: arm for arm in bundle["arms"]}
    if arm_id not in arms:
        raise ValueError("requested arm is absent from sequence bundle")
    expected_episode = materialize_sequence_episode(bundle, seed=episode.seed)
    if expected_episode != episode:
        raise ValueError("sequence episode does not belong to this bundle and seed")
    arm = arms[arm_id]
    if "message" in arm:
        arm_message = arm["message"]
    else:
        replacements, _ = _sequence_replacements(bundle, seed=episode.seed)
        replacements.update(arm["arm_substitutions"])
        arm_message = _replace_tokens(arm["message_template"], replacements)
        if "{{" in arm_message or "}}" in arm_message:
            raise ValueError("sequence arm message contains an unresolved token")
    answers = ", ".join(
        f"{option['value']}={option['label']}" for option in bundle["response_options"]
    )
    prior = (
        f"Prior randomized survey exposure:\n{episode.prior_exposure}\n\n"
        if episode.prior_exposure
        else "Prior randomized survey exposure: none; the target module came first.\n\n"
    )
    return (
        "Estimate the response distribution for a random adult completing the exact "
        "chronological survey path below. Do not answer as yourself. Privately account "
        "for how the adult would answer any preceding questions. Return only one JSON "
        'object with exactly this shape: {"probabilities":{"1":NUMBER,...}}. Include '
        "every listed answer value once; probabilities must be between 0 and 1 and sum "
        "to 1.\n\n"
        f"Population: {bundle['population']['description']}\n\n"
        f"Context: {bundle['common_context']}\n\n"
        f"{prior}"
        f"Target intervention and immediately preceding target-module item:\n{arm_message}\n\n"
        f"Target question: {bundle['outcome_question']}\n"
        f"Target answers: {answers}"
    )


def parse_ordinal_distribution(
    raw_text: str, *, option_values: tuple[int, ...]
) -> OrdinalDistribution:
    """Parse an exact ordinal probability object; never repair missing categories."""

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ValueError("simulator output is not valid JSON") from error
    if not isinstance(parsed, Mapping) or set(parsed) != {"probabilities"}:
        raise ValueError("ordinal output must contain exactly probabilities")
    probabilities = parsed["probabilities"]
    expected_keys = {str(value) for value in option_values}
    if not isinstance(probabilities, Mapping) or set(probabilities) != expected_keys:
        raise ValueError("ordinal probabilities must include every response value exactly")
    ordered: list[tuple[int, float]] = []
    for value in option_values:
        probability = probabilities[str(value)]
        if (
            not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not isfinite(probability)
            or not 0.0 <= probability <= 1.0
        ):
            raise ValueError("ordinal probabilities must be finite values in [0, 1]")
        ordered.append((value, float(probability)))
    if abs(fsum(probability for _, probability in ordered) - 1.0) > 1e-6:
        raise ValueError("ordinal probabilities must sum to one")
    return OrdinalDistribution(tuple(ordered))


def parse_ordinal_relative_weights(
    raw_text: str, *, option_values: tuple[int, ...]
) -> tuple[OrdinalDistribution, tuple[tuple[int, float], ...]]:
    """Parse declared relative weights and normalize them deterministically.

    The raw weights are returned alongside the normalized distribution so the
    transformation remains explicit and auditable. Missing options, negative or
    non-finite weights, and an all-zero response fail closed.
    """

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ValueError("simulator output is not valid JSON") from error
    if not isinstance(parsed, Mapping) or set(parsed) != {"relative_weights"}:
        raise ValueError("ordinal output must contain exactly relative_weights")
    weights = parsed["relative_weights"]
    expected_keys = {str(value) for value in option_values}
    if not isinstance(weights, Mapping) or set(weights) != expected_keys:
        raise ValueError("ordinal weights must include every response value exactly")
    ordered_weights: list[tuple[int, float]] = []
    for value in option_values:
        weight = weights[str(value)]
        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not isfinite(weight)
            or weight < 0.0
        ):
            raise ValueError("ordinal weights must be finite and non-negative")
        ordered_weights.append((value, float(weight)))
    total = fsum(weight for _, weight in ordered_weights)
    if total <= 0.0:
        raise ValueError("at least one ordinal weight must be positive")
    distribution = OrdinalDistribution(
        tuple((value, weight / total) for value, weight in ordered_weights)
    )
    return distribution, tuple(ordered_weights)


def ordinal_probability_prompt(
    bundle: Mapping[str, Any], *, arm_id: str, variant_id: str | None = None
) -> str:
    """Build one source-faithful prompt for a sealed bounded-ordinal task."""

    validate_ordinal_blinded_bundle(bundle)
    arms = {arm["arm_id"]: arm for arm in bundle["arms"]}
    if arm_id not in arms:
        raise ValueError("requested arm is absent from ordinal bundle")
    arm = arms[arm_id]
    if "message" in arm:
        if variant_id not in {None, "direct"}:
            raise ValueError("direct ordinal arm does not accept a message variant")
        message = arm["message"]
    else:
        variants = {
            variant["variant_id"]: variant for variant in arm["message_variants"]
        }
        if variant_id not in variants:
            raise ValueError("requested message variant is absent from ordinal arm")
        message = variants[str(variant_id)]["message"]
    answers = ", ".join(
        f"{option['value']}={option['label']}" for option in bundle["response_options"]
    )
    return (
        "Estimate the response distribution for a random adult in the population "
        "below. Do not answer as yourself. Return only one JSON object with exactly "
        'this shape: {"probabilities":{"1":NUMBER,...}}. Include every listed '
        "answer value once; probabilities must be between 0 and 1 and sum to 1.\n\n"
        f"Population: {bundle['population']['description']}\n\n"
        f"Context: {bundle['common_context']}\n\n"
        f"Intervention: {message}\n\n"
        f"Question: {bundle['outcome_question']}\n"
        f"Answers: {answers}"
    )


def ordinal_variant_contract(
    bundle: Mapping[str, Any], *, arm_id: str
) -> tuple[tuple[str, float], ...]:
    """Return the frozen nuisance-message mixture for one ordinal arm."""

    validate_ordinal_blinded_bundle(bundle)
    arms = {arm["arm_id"]: arm for arm in bundle["arms"]}
    if arm_id not in arms:
        raise ValueError("requested arm is absent from ordinal bundle")
    arm = arms[arm_id]
    if "message" in arm:
        return (("direct", 1.0),)
    return tuple(
        (str(variant["variant_id"]), float(variant["weight"]))
        for variant in arm["message_variants"]
    )


def aggregate_ordinal_predictions(
    outputs: Iterable[Mapping[str, Any]],
    *,
    bundle: Mapping[str, Any],
    draws: int,
) -> tuple[dict[str, float], dict[int, dict[str, float]]]:
    """Aggregate a complete arm-by-variant-by-draw ordinal prediction grid.

    Returns overall arm utilities and the draw-specific arm utilities used for
    outcome-free ranking-stability diagnostics.
    """

    validate_ordinal_blinded_bundle(bundle)
    if draws <= 0:
        raise ValueError("draws must be positive")
    arm_ids = tuple(arm["arm_id"] for arm in bundle["arms"])
    variants = {
        arm_id: ordinal_variant_contract(bundle, arm_id=arm_id)
        for arm_id in arm_ids
    }
    expected = {
        (arm_id, variant_id, draw_index)
        for arm_id in arm_ids
        for variant_id, _ in variants[arm_id]
        for draw_index in range(draws)
    }
    utilities = {
        int(option["value"]): float(option["normalized_utility"])
        for option in bundle["response_options"]
    }
    values: dict[tuple[str, str, int], float] = {}
    for output in outputs:
        key = (
            str(output.get("arm_id")),
            str(output.get("variant_id")),
            output.get("draw_index"),
        )
        if key in values:
            raise ValueError("duplicate ordinal arm/variant/draw prediction")
        probabilities = output.get("probabilities")
        if key not in expected or not isinstance(probabilities, Mapping):
            raise ValueError("invalid ordinal arm/variant/draw prediction")
        parsed = parse_ordinal_distribution(
            json.dumps({"probabilities": probabilities}),
            option_values=tuple(utilities),
        )
        values[key] = fsum(
            utilities[value] * probability
            for value, probability in parsed.probabilities
        )
    if set(values) != expected:
        raise ValueError("ordinal predictions are not complete across arms and draws")
    by_draw: dict[int, dict[str, float]] = {}
    for draw_index in range(draws):
        by_draw[draw_index] = {
            arm_id: fsum(
                weight * values[(arm_id, variant_id, draw_index)]
                for variant_id, weight in variants[arm_id]
            )
            for arm_id in arm_ids
        }
    overall = {
        arm_id: fsum(by_draw[draw_index][arm_id] for draw_index in range(draws))
        / draws
        for arm_id in arm_ids
    }
    return overall, by_draw


def aggregate_sequence_predictions(
    outputs: Iterable[Mapping[str, Any]],
    *,
    bundle: Mapping[str, Any],
    episode_ids: tuple[str, ...],
) -> dict[str, float]:
    """Aggregate normalized utility over complete paired arm-by-episode predictions."""

    validate_sequence_blinded_bundle(bundle)
    if not episode_ids or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("sequence episode IDs must be non-empty and unique")
    arm_ids = tuple(arm["arm_id"] for arm in bundle["arms"])
    expected = {
        (arm_id, episode_id) for arm_id in arm_ids for episode_id in episode_ids
    }
    utilities = {
        str(option["value"]): float(option["normalized_utility"])
        for option in bundle["response_options"]
    }
    values: dict[tuple[str, str], float] = {}
    for output in outputs:
        key = (str(output.get("arm_id")), str(output.get("episode_id")))
        if key in values:
            raise ValueError("duplicate sequence arm/episode prediction")
        probabilities = output.get("probabilities")
        if key not in expected or not isinstance(probabilities, Mapping):
            raise ValueError("invalid sequence arm/episode prediction")
        parsed = parse_ordinal_distribution(
            json.dumps({"probabilities": probabilities}),
            option_values=tuple(int(value) for value in utilities),
        )
        values[key] = fsum(
            utilities[str(value)] * probability
            for value, probability in parsed.probabilities
        )
    if set(values) != expected:
        raise ValueError("sequence predictions are not complete and paired across arms")
    return {
        arm_id: fsum(values[(arm_id, episode_id)] for episode_id in episode_ids)
        / len(episode_ids)
        for arm_id in arm_ids
    }


def parse_continuous_prediction(
    raw_text: str, *, integer_only: bool
) -> ContinuousPrediction:
    """Parse one exact continuous prediction; never recover or clamp output."""

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ValueError("simulator output is not valid JSON") from error
    if not isinstance(parsed, Mapping) or set(parsed) != {"predicted_value"}:
        raise ValueError("simulator output must contain exactly predicted_value")
    value = parsed["predicted_value"]
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
    ):
        raise ValueError("predicted value must be a finite number")
    numeric = float(value)
    if numeric < 0:
        raise ValueError("predicted value must be non-negative")
    if integer_only and not numeric.is_integer():
        raise ValueError("predicted value must satisfy the integer contract")
    return ContinuousPrediction(numeric)


def aggregate_binary_predictions(
    outputs: Iterable[Mapping[str, Any]], *, arm_ids: tuple[str, ...], draws: int
) -> dict[str, float]:
    if draws <= 0:
        raise ValueError("draws must be positive")
    expected = {(arm_id, draw_index) for arm_id in arm_ids for draw_index in range(draws)}
    values: dict[tuple[str, int], float] = {}
    for output in outputs:
        key = (str(output.get("arm_id")), output.get("draw_index"))
        if key in values:
            raise ValueError("duplicate arm/draw prediction")
        probability = output.get("yes_probability")
        if (
            key not in expected
            or not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not isfinite(probability)
            or not 0.0 <= probability <= 1.0
        ):
            raise ValueError("invalid arm/draw prediction")
        values[key] = float(probability)
    if set(values) != expected:
        raise ValueError("predictions are not complete for every arm and draw")
    return {
        arm_id: fsum(values[(arm_id, index)] for index in range(draws)) / draws
        for arm_id in arm_ids
    }


def aggregate_continuous_predictions(
    outputs: Iterable[Mapping[str, Any]],
    *,
    arm_ids: tuple[str, ...],
    draws: int,
    estimator: str,
) -> dict[str, float]:
    if draws <= 0:
        raise ValueError("draws must be positive")
    if estimator not in {"mean", "median"}:
        raise ValueError("unsupported continuous aggregation estimator")
    expected = {
        (arm_id, draw_index) for arm_id in arm_ids for draw_index in range(draws)
    }
    values: dict[tuple[str, int], float] = {}
    for output in outputs:
        key = (str(output.get("arm_id")), output.get("draw_index"))
        if key in values:
            raise ValueError("duplicate continuous arm/draw prediction")
        value = output.get("predicted_value")
        if (
            key not in expected
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(value)
            or value < 0
        ):
            raise ValueError("invalid continuous arm/draw prediction")
        values[key] = float(value)
    if set(values) != expected:
        raise ValueError("continuous predictions are not complete for every arm and draw")
    by_arm = {
        arm_id: tuple(values[(arm_id, index)] for index in range(draws))
        for arm_id in arm_ids
    }
    if estimator == "mean":
        return {
            arm_id: fsum(arm_values) / len(arm_values)
            for arm_id, arm_values in by_arm.items()
        }
    return {
        arm_id: float(median(arm_values)) for arm_id, arm_values in by_arm.items()
    }


def ollama_continuous_prompt(bundle: Mapping[str, Any], *, arm_id: str) -> str:
    validate_continuous_blinded_bundle(bundle)
    arms = {arm["arm_id"]: arm["message"] for arm in bundle["arms"]}
    if arm_id not in arms:
        raise ValueError("requested arm is absent from continuous bundle")
    contract = bundle["response_contract"]
    return (
        "Estimate the stated monthly willingness to pay for a random adult in the "
        "population below. Do not answer as yourself. Return only one JSON object "
        'with exactly this key: {"predicted_value": INTEGER}. The integer must be '
        f"at least {contract['minimum']} and represents {contract['unit']}.\n\n"
        f"Population: {bundle['population']['description']}\n\n"
        f"Scenario: {bundle['common_context']}\n\n"
        f"Notice policy: {arms[arm_id]}\n\n"
        f"Question: {bundle['outcome_question']}"
    )


def ollama_probability_prompt(bundle: Mapping[str, Any], *, arm_id: str) -> str:
    validate_blinded_bundle(bundle)
    arms = {arm["arm_id"]: arm["message"] for arm in bundle["arms"]}
    if arm_id not in arms:
        raise ValueError("requested arm is absent from bundle")
    return (
        "Estimate the response distribution for a random adult in the population below. "
        "Do not answer as yourself. Return only one JSON object with exactly these keys: "
        '{"yes_probability": NUMBER, "no_probability": NUMBER}. The two probabilities '
        "must be between 0 and 1 and sum to 1.\n\n"
        f"Population: {bundle['population']['description']}\n\n"
        f"Scenario: {bundle['common_context']}\n\n"
        f"Message: {arms[arm_id]}\n\n"
        f"Question: {bundle['outcome_question']}\n"
        "Answers: Yes or No."
    )
