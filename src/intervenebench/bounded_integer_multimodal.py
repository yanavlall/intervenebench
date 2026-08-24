"""Outcome-free planning and adaptive convergence for bounded-integer VLM tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from math import fsum, isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .protocol import assert_blinded_payload
from .sampling_convergence import SamplingDecision, choose_sampling_checkpoint


BOUNDED_INTEGER_PNG_MULTIMODAL_BUNDLE_FIELDS = frozenset(
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
        "sampling_contract",
        "representation_status",
        "source_material_sha256",
        "outcome_access",
        "reveal_authorized",
        "execution_status",
        "scoring_blocker",
    }
)


@dataclass(frozen=True, slots=True)
class BoundedIntegerPrediction:
    value: int


@dataclass(frozen=True, slots=True)
class BoundedIntegerMultimodalPrompt:
    text: str
    asset_paths: tuple[str, ...]
    asset_sha256: tuple[str, ...]


def validate_bounded_integer_png_multimodal_bundle(
    bundle: Mapping[str, Any],
) -> None:
    """Validate a sealed 0--100 task with one provenance-bound PNG per arm."""

    assert_blinded_payload(bundle)
    extra = sorted(set(bundle) - BOUNDED_INTEGER_PNG_MULTIMODAL_BUNDLE_FIELDS)
    missing = sorted(BOUNDED_INTEGER_PNG_MULTIMODAL_BUNDLE_FIELDS - set(bundle))
    if extra or missing:
        raise ValueError(
            "bounded integer PNG fields mismatch; "
            f"extra={extra}, missing={missing}"
        )
    if bundle["schema_version"] != "bounded_integer_png_multimodal_bundle.v1":
        raise ValueError("unsupported bounded integer PNG bundle schema")
    if bundle["access_regime"] != "DESIGN_ONLY":
        raise ValueError("bounded integer PNG bundle must use DESIGN_ONLY")
    if bundle["outcome_access"] != "sealed" or bundle["reveal_authorized"] is not False:
        raise ValueError("bounded integer PNG bundle must remain outcome sealed")
    if bundle["execution_status"] not in {
        "runnable",
        "design_only_adapter_ready_scoring_blocked",
    }:
        raise ValueError("bounded integer PNG execution status is invalid")
    blocker = bundle["scoring_blocker"]
    if not isinstance(blocker, str) or (
        bundle["execution_status"] == "runnable" and blocker
    ) or (
        bundle["execution_status"] != "runnable" and not blocker.strip()
    ):
        raise ValueError("execution status and scoring blocker are inconsistent")
    for field in ("task_id", "experiment_id", "common_context", "outcome_question"):
        if not isinstance(bundle[field], str) or not bundle[field].strip():
            raise ValueError(f"bounded integer PNG bundle requires non-empty {field}")
    population = bundle["population"]
    if (
        not isinstance(population, Mapping)
        or set(population) != {"description", "roster_id"}
        or any(
            not isinstance(value, str) or not value.strip()
            for value in population.values()
        )
    ):
        raise ValueError("bounded integer PNG population has an unexpected shape")

    arms = bundle["arms"]
    if not isinstance(arms, list) or not 2 <= len(arms) <= 6 or any(
        not isinstance(arm, Mapping)
        or set(arm) != {"arm_id", "accessible_text", "asset"}
        for arm in arms
    ):
        raise ValueError("bounded integer PNG arms have an unexpected shape")
    arm_ids = [arm["arm_id"] for arm in arms]
    if len(set(arm_ids)) != len(arm_ids) or any(
        not isinstance(arm_id, str) or not arm_id.strip() for arm_id in arm_ids
    ):
        raise ValueError("bounded integer PNG arm IDs must be unique")
    for arm in arms:
        if (
            not isinstance(arm["accessible_text"], str)
            or not arm["accessible_text"].strip()
        ):
            raise ValueError("bounded integer PNG accessible text is required")
        asset = arm["asset"]
        if not isinstance(asset, Mapping) or set(asset) != {
            "path",
            "mime_type",
            "sha256",
        }:
            raise ValueError("bounded integer PNG asset has an unexpected shape")
        path = asset["path"]
        digest = asset["sha256"]
        if (
            not isinstance(path, str)
            or not path.startswith("data/derived/stimuli/")
            or ".." in Path(path).parts
            or Path(path).suffix.lower() != ".png"
            or asset["mime_type"] != "image/png"
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("bounded integer PNG asset declaration is invalid")

    response = bundle["response_contract"]
    if (
        not isinstance(response, Mapping)
        or set(response)
        != {"type", "minimum", "maximum", "unit", "utility_direction"}
        or response["type"] != "integer"
        or response["minimum"] != 0
        or response["maximum"] != 100
        or not isinstance(response["unit"], str)
        or not response["unit"].strip()
        or response["utility_direction"]
        not in {"higher_is_better", "lower_is_better"}
    ):
        raise ValueError("bounded integer PNG response contract must be integer 0--100")

    sampling = bundle["sampling_contract"]
    if (
        not isinstance(sampling, Mapping)
        or set(sampling)
        != {
            "paired_across_arms",
            "checkpoints_per_arm",
            "arm_mean_tolerance",
            "margin_multiplier",
        }
        or sampling["paired_across_arms"] is not True
    ):
        raise ValueError("bounded integer PNG sampling contract must pair arms")
    checkpoints = sampling["checkpoints_per_arm"]
    if (
        not isinstance(checkpoints, list)
        or len(checkpoints) < 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in checkpoints
        )
        or checkpoints != sorted(set(checkpoints))
        or not isinstance(sampling["arm_mean_tolerance"], (int, float))
        or isinstance(sampling["arm_mean_tolerance"], bool)
        or not isfinite(sampling["arm_mean_tolerance"])
        or not 0.0 <= sampling["arm_mean_tolerance"] <= 1.0
        or not isinstance(sampling["margin_multiplier"], (int, float))
        or isinstance(sampling["margin_multiplier"], bool)
        or not isfinite(sampling["margin_multiplier"])
        or sampling["margin_multiplier"] < 0.0
    ):
        raise ValueError("bounded integer PNG sampling checkpoints are invalid")

    representation = bundle["representation_status"]
    if not isinstance(representation, Mapping) or set(representation) != {
        "status",
        "provenance_path",
        "provenance_sha256",
        "limitation",
    }:
        raise ValueError("bounded integer PNG representation status is invalid")
    provenance_path = representation["provenance_path"]
    provenance_digest = representation["provenance_sha256"]
    if (
        representation["status"] != "source_faithful_derived_composite"
        or not isinstance(provenance_path, str)
        or not provenance_path.startswith("data/manifests/stimuli/")
        or ".." in Path(provenance_path).parts
        or not isinstance(provenance_digest, str)
        or len(provenance_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in provenance_digest
        )
        or not isinstance(representation["limitation"], str)
        or not representation["limitation"].strip()
    ):
        raise ValueError("bounded integer PNG provenance declaration is invalid")
    source_hash = bundle["source_material_sha256"]
    if not isinstance(source_hash, str) or len(source_hash) != 64 or any(
        character not in "0123456789abcdef" for character in source_hash
    ):
        raise ValueError("source_material_sha256 must be a SHA-256 digest")


def parse_bounded_integer_prediction(
    raw_text: str, *, minimum: int = 0, maximum: int = 100
) -> BoundedIntegerPrediction:
    """Parse one exact bounded integer response; never round or clip."""

    if (
        not isinstance(minimum, int)
        or isinstance(minimum, bool)
        or not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or minimum >= maximum
    ):
        raise ValueError("bounded integer parser limits are invalid")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ValueError("simulator output is not valid JSON") from error
    if not isinstance(parsed, Mapping) or set(parsed) != {"predicted_value"}:
        raise ValueError("bounded integer output must contain exactly predicted_value")
    value = parsed["predicted_value"]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("predicted_value must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError("predicted_value is outside the declared bounds")
    return BoundedIntegerPrediction(value)


def bounded_integer_png_multimodal_prompt(
    bundle: Mapping[str, Any], *, arm_id: str, repository_root: Path
) -> BoundedIntegerMultimodalPrompt:
    """Materialize a prompt after checking the derived PNG and its provenance."""

    validate_bounded_integer_png_multimodal_bundle(bundle)
    if bundle["execution_status"] != "runnable":
        raise ValueError(f"bundle is not runnable: {bundle['scoring_blocker']}")
    arms = {arm["arm_id"]: arm for arm in bundle["arms"]}
    if arm_id not in arms:
        raise ValueError("requested bounded integer PNG arm is absent")
    root = repository_root.resolve()
    arm = arms[arm_id]
    asset = arm["asset"]
    asset_path = (root / asset["path"]).resolve()
    if not asset_path.is_relative_to(root) or not asset_path.is_file():
        raise ValueError("bounded integer PNG asset is missing or outside repository")
    contents = asset_path.read_bytes()
    if (
        not contents.startswith(b"\x89PNG\r\n\x1a\n")
        or sha256(contents).hexdigest() != asset["sha256"]
    ):
        raise ValueError("bounded integer PNG asset hash or signature does not match")
    representation = bundle["representation_status"]
    provenance_path = (root / representation["provenance_path"]).resolve()
    if not provenance_path.is_relative_to(root) or not provenance_path.is_file():
        raise ValueError("bounded integer PNG provenance is missing or outside repository")
    provenance_bytes = provenance_path.read_bytes()
    if sha256(provenance_bytes).hexdigest() != representation["provenance_sha256"]:
        raise ValueError("bounded integer PNG provenance hash does not match")
    provenance = json.loads(provenance_bytes)
    assert_blinded_payload(provenance)
    source_instrument = provenance.get("source_instrument")
    if (
        not isinstance(source_instrument, Mapping)
        or source_instrument.get("sha256") != bundle["source_material_sha256"]
    ):
        raise ValueError("bounded integer PNG source is not bound by provenance")
    entries = {
        entry.get("arm_id"): entry
        for entry in provenance.get("arms", [])
        if isinstance(entry, Mapping)
    }
    if (
        arm_id not in entries
        or entries[arm_id].get("output_path") != asset["path"]
        or entries[arm_id].get("output_sha256") != asset["sha256"]
    ):
        raise ValueError("bounded integer PNG asset is not bound by provenance")
    response = bundle["response_contract"]
    text = (
        "Predict the response of one randomly selected adult after viewing the "
        "attached source-faithful arm image. Do not answer as yourself. Return only "
        'one JSON object with exactly one integer field, for example '
        '{"predicted_value":50}. Do not explain, round, or add fields.\n\n'
        f"Population: {bundle['population']['description']}\n\n"
        f"Context: {bundle['common_context']}\n\n"
        f"Accessible arm text: {arm['accessible_text']}\n\n"
        f"Question: {bundle['outcome_question']}\n"
        f"Valid response: one whole number from {response['minimum']} through "
        f"{response['maximum']} inclusive ({response['unit']})."
    )
    return BoundedIntegerMultimodalPrompt(
        text=text,
        asset_paths=(str(asset_path),),
        asset_sha256=(asset["sha256"],),
    )


def aggregate_bounded_integer_png_predictions(
    outputs: Iterable[Mapping[str, Any]], *, bundle: Mapping[str, Any], draws: int
) -> dict[str, float]:
    """Aggregate a complete paired arm-by-draw grid to normalized utility."""

    validate_bounded_integer_png_multimodal_bundle(bundle)
    if not isinstance(draws, int) or isinstance(draws, bool) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    rows = tuple(outputs)
    assert_blinded_payload(rows)
    arm_ids = tuple(arm["arm_id"] for arm in bundle["arms"])
    expected = {
        (arm_id, draw_index)
        for arm_id in arm_ids
        for draw_index in range(draws)
    }
    predictions: dict[tuple[str, int], float] = {}
    response = bundle["response_contract"]
    denominator = response["maximum"] - response["minimum"]
    for output in rows:
        if set(output) != {"arm_id", "draw_index", "predicted_value"}:
            raise ValueError("bounded integer prediction row has an unexpected shape")
        draw_index = output["draw_index"]
        if not isinstance(draw_index, int) or isinstance(draw_index, bool):
            raise ValueError("bounded integer arm/draw prediction is invalid")
        key = (str(output["arm_id"]), draw_index)
        if key in predictions:
            raise ValueError("duplicate bounded integer arm/draw prediction")
        if key not in expected:
            raise ValueError("bounded integer arm/draw prediction is invalid")
        parsed = parse_bounded_integer_prediction(
            json.dumps({"predicted_value": output["predicted_value"]}),
            minimum=response["minimum"],
            maximum=response["maximum"],
        )
        scaled = (parsed.value - response["minimum"]) / denominator
        predictions[key] = (
            1.0 - scaled
            if response["utility_direction"] == "lower_is_better"
            else scaled
        )
    if set(predictions) != expected:
        raise ValueError("bounded integer predictions are not complete and paired")
    return {
        arm_id: fsum(predictions[(arm_id, draw)] for draw in range(draws)) / draws
        for arm_id in arm_ids
    }


def build_bounded_integer_png_call_plan(
    bundle: Mapping[str, Any], *, model_ids: Sequence[str]
) -> dict[str, Any]:
    """Build a zero-authority maximum call grid with draws paired across arms."""

    validate_bounded_integer_png_multimodal_bundle(bundle)
    if bundle["execution_status"] != "runnable":
        raise ValueError(f"bundle is not runnable: {bundle['scoring_blocker']}")
    models = tuple(str(model_id) for model_id in model_ids)
    if not models or any(not model_id.strip() for model_id in models) or len(set(models)) != len(models):
        raise ValueError("model_ids must be unique non-empty strings")
    checkpoints = tuple(bundle["sampling_contract"]["checkpoints_per_arm"])
    maximum_draws = checkpoints[-1]
    calls: list[dict[str, Any]] = []
    for model_id in models:
        for draw_index in range(maximum_draws):
            stage = "base" if draw_index < checkpoints[0] else "outcome_free_adaptive_reserve"
            paired_draw_id = f"draw_{draw_index:03d}"
            for arm in bundle["arms"]:
                payload = (
                    f"{bundle['experiment_id']}|{model_id}|{paired_draw_id}|"
                    f"{arm['arm_id']}|{arm['asset']['sha256']}"
                )
                calls.append(
                    {
                        "call_id": sha256(payload.encode("utf-8")).hexdigest(),
                        "experiment_id": bundle["experiment_id"],
                        "model_id": model_id,
                        "arm_id": arm["arm_id"],
                        "draw_index": draw_index,
                        "paired_draw_id": paired_draw_id,
                        "stage": stage,
                        "asset_path": arm["asset"]["path"],
                        "asset_sha256": arm["asset"]["sha256"],
                    }
                )
    plan = {
        "schema_version": "bounded_integer_png_call_plan.v1",
        "experiment_id": bundle["experiment_id"],
        "model_ids": list(models),
        "arm_count": len(bundle["arms"]),
        "checkpoints_per_arm": list(checkpoints),
        "maximum_draws_per_arm_model": maximum_draws,
        "planned_call_count": len(calls),
        "authority": {
            "model_calls_authorized": False,
            "paid_compute_authorized": False,
            "human_outcome_reveal_authorized": False,
        },
        "calls": calls,
    }
    assert_blinded_payload(plan)
    return plan


def choose_bounded_integer_png_checkpoint(
    outputs: Iterable[Mapping[str, Any]], *, bundle: Mapping[str, Any]
) -> SamplingDecision:
    """Apply the frozen convergence rule to complete cumulative paired draws."""

    validate_bounded_integer_png_multimodal_bundle(bundle)
    rows = tuple(outputs)
    assert_blinded_payload(rows)
    checkpoints = tuple(bundle["sampling_contract"]["checkpoints_per_arm"])
    means = []
    for count in checkpoints:
        selected = tuple(row for row in rows if row.get("draw_index", -1) < count)
        means.append(
            (
                count,
                aggregate_bounded_integer_png_predictions(
                    selected, bundle=bundle, draws=count
                ),
            )
        )
    sampling = bundle["sampling_contract"]
    return choose_sampling_checkpoint(
        means,
        minimum_samples=checkpoints[0],
        arm_mean_tolerance=float(sampling["arm_mean_tolerance"]),
        margin_multiplier=float(sampling["margin_multiplier"]),
    )
