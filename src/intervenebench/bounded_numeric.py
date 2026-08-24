"""Outcome-blind recommendation and reveal-bound scoring for bounded numeric tasks.

This module intentionally leaves the frozen uncapped-continuous adapter unchanged.
Bounded questionnaire responses have a declared utility scale, so their treatment
effects and regret are normalized before cross-task scoring.
"""

from __future__ import annotations

from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from .continuous import ContinuousObservation, continuous_arm_locations
from .evaluation import choose_best_arm, decision_regret, normalize_utility, treatment_effects
from .phase1 import read_json_object
from .protocol import (
    assert_blinded_payload,
    freeze_envelope,
    freeze_recommendation,
    payload_hash,
    verify_envelope,
    verify_frozen_recommendation,
)
from .schemas import OutcomeDirection
from .simulators import aggregate_continuous_predictions, parse_continuous_prediction
from .uncertainty import bootstrap_arm_location_optimality


BLINDED_BUNDLE_SCHEMA = "bounded_numeric_blinded_bundle.v1"
RECOMMENDATION_SCHEMA = "bounded_numeric_recommendation.v1"
REVEAL_AUTHORIZATION_SCHEMA = "bounded_numeric_reveal_authorization.v1"
SCORE_SCHEMA = "bounded_numeric_score.v1"
SUPPORTED_REVEAL_SPLITS = frozenset({"validation", "replication_test"})
REVEALED_OUTCOME_COLUMNS = (
    "study_id",
    "sample_id",
    "participant",
    "condition_num",
    "task_num",
    "response",
)


def _now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_digest(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be a SHA-256 digest") from error
    return value


def _bundle_bounds(bundle: Mapping[str, Any]) -> tuple[float, float, bool]:
    contract = bundle.get("response_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("bounded numeric bundle requires a response contract")
    lower = contract.get("minimum")
    upper = contract.get("maximum")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        for value in (lower, upper)
    ):
        raise ValueError("bounded numeric response bounds must be finite numbers")
    if float(lower) >= float(upper):
        raise ValueError("bounded numeric minimum must be smaller than maximum")
    response_type = contract.get("type")
    if response_type not in {"integer", "number"}:
        raise ValueError("bounded numeric response type must be integer or number")
    if not isinstance(contract.get("unit"), str) or not contract["unit"].strip():
        raise ValueError("bounded numeric response unit is required")
    return float(lower), float(upper), response_type == "integer"


def validate_bounded_numeric_blinded_bundle(bundle: Mapping[str, Any]) -> None:
    """Fail closed unless a design-only bounded numeric bundle is complete."""

    required = {
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
    if set(bundle) != required:
        raise ValueError("bounded numeric bundle fields do not match the frozen schema")
    if bundle["schema_version"] != BLINDED_BUNDLE_SCHEMA:
        raise ValueError("unsupported bounded numeric blinded bundle schema")
    if bundle["access_regime"] != "DESIGN_ONLY":
        raise ValueError("bounded numeric bundle must be design-only")
    if bundle["outcome_access"] != "sealed" or bundle["reveal_authorized"] is not False:
        raise ValueError("bounded numeric bundle must keep target outcomes sealed")
    assert_blinded_payload(bundle)
    for key in ("task_id", "experiment_id", "common_context", "outcome_question"):
        if not isinstance(bundle[key], str) or not bundle[key].strip():
            raise ValueError(f"bounded numeric bundle {key} must be non-empty")
    population = bundle["population"]
    if not isinstance(population, Mapping) or not all(
        isinstance(population.get(key), str) and population[key].strip()
        for key in ("description", "roster_id")
    ):
        raise ValueError("bounded numeric population description and roster are required")
    arms = bundle["arms"]
    if not isinstance(arms, list) or not 2 <= len(arms) <= 4:
        raise ValueError("bounded numeric bundle requires two to four arms")
    if any(
        not isinstance(arm, Mapping)
        or set(arm) != {"arm_id", "message"}
        or not all(isinstance(arm[key], str) and arm[key].strip() for key in arm)
        for arm in arms
    ):
        raise ValueError("bounded numeric arms require only non-empty arm_id and message")
    arm_ids = [str(arm["arm_id"]) for arm in arms]
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError("bounded numeric arm IDs must be unique")
    _bundle_bounds(bundle)
    _sha256_digest(bundle["source_material_sha256"], field="source_material_sha256")


def bounded_numeric_prompt(bundle: Mapping[str, Any], *, arm_id: str) -> str:
    """Construct the source-faithful prompt without target-outcome information."""

    validate_bounded_numeric_blinded_bundle(bundle)
    arms = {str(arm["arm_id"]): str(arm["message"]) for arm in bundle["arms"]}
    if arm_id not in arms:
        raise ValueError("requested arm is absent from bounded numeric bundle")
    lower, upper, integer_only = _bundle_bounds(bundle)
    numeric_type = "integer" if integer_only else "number"
    return (
        "Estimate the response of a random adult in the population below. Do not "
        "answer as yourself. Return only one JSON object with exactly this key: "
        f'{{"predicted_value": {numeric_type.upper()}}}. The value must be between '
        f"{lower:g} and {upper:g} inclusive.\n\n"
        f"Population: {bundle['population']['description']}\n\n"
        f"Context: {bundle['common_context']}\n\n"
        f"Intervention: {arms[arm_id]}\n\n"
        f"Question: {bundle['outcome_question']}"
    )


def _normalized_locations(
    locations: Mapping[str, float],
    *,
    lower: float,
    upper: float,
    direction: OutcomeDirection,
) -> dict[str, float]:
    return {
        arm_id: normalize_utility(
            float(location), lower=lower, upper=upper, direction=direction
        )
        for arm_id, location in locations.items()
    }


def _validate_task_bundle_binding(
    task: Mapping[str, Any], bundle: Mapping[str, Any]
) -> tuple[tuple[str, ...], float, float, bool, OutcomeDirection]:
    validate_bounded_numeric_blinded_bundle(bundle)
    if task.get("experiment_id") != bundle["experiment_id"]:
        raise ValueError("bounded numeric bundle disagrees with decision task")
    if task.get("task_id") != bundle["task_id"]:
        raise ValueError("bounded numeric bundle task ID disagrees with decision task")
    if task.get("outcome_family") != "continuous":
        raise ValueError("bounded numeric task must declare a continuous outcome")
    arms = task.get("arms")
    if not isinstance(arms, list):
        raise ValueError("bounded numeric task arms are required")
    arm_ids = tuple(str(arm.get("arm_id")) for arm in arms if isinstance(arm, Mapping))
    if arm_ids != tuple(str(arm["arm_id"]) for arm in bundle["arms"]):
        raise ValueError("bounded numeric bundle action order disagrees with decision task")
    if task.get("control_arm_id") not in arm_ids:
        raise ValueError("bounded numeric control is absent from the action set")
    lower, upper, integer_only = _bundle_bounds(bundle)
    valid = task.get("valid_response")
    if not isinstance(valid, Mapping) or (
        valid.get("lower_bound") != lower
        or valid.get("upper_bound") != upper
        or valid.get("integer_only") is not integer_only
    ):
        raise ValueError("bounded numeric response contract disagrees with decision task")
    if valid.get("missing_codes") != []:
        raise ValueError("bounded numeric adapter currently requires null-only missingness")
    estimator = task.get("estimator")
    if not isinstance(estimator, Mapping) or estimator.get("location") != "mean":
        raise ValueError("bounded numeric primary estimator must be the mean")
    if estimator.get("robustness_locations") != ["median"]:
        raise ValueError("bounded numeric task must freeze median robustness")
    if estimator.get("normalized_for_pooled_regret") is not True:
        raise ValueError("bounded numeric task must use normalized pooled regret")
    if task.get("utility_transform") != "U=y/100" or (lower, upper) != (0.0, 100.0):
        raise ValueError("this adapter revision is frozen to the declared U=y/100 transform")
    try:
        direction = OutcomeDirection(str(task["direction"]))
    except (KeyError, ValueError) as error:
        raise ValueError("bounded numeric task direction is invalid") from error
    return arm_ids, lower, upper, integer_only, direction


def freeze_bounded_numeric_recommendation_from_outputs(
    *,
    bundle_path: Path,
    split_path: Path,
    decision_task_path: Path,
    outputs: tuple[dict[str, Any], ...],
    raw_output_path: Path,
    recommendation_path: Path,
    simulator_id: str,
    simulator_revision: str,
    draws: int,
    seed: int,
) -> str:
    """Parse design-only outputs and freeze a normalized recommendation."""

    bundle = read_json_object(bundle_path)
    split = read_json_object(split_path)
    task = read_json_object(decision_task_path)
    arm_ids, lower, upper, integer_only, direction = _validate_task_bundle_binding(
        task, bundle
    )
    target_split = task.get("split")
    if target_split not in SUPPORTED_REVEAL_SPLITS:
        raise ValueError("bounded numeric target must have a frozen evaluation split")
    if split.get("experiment_to_split", {}).get(task["experiment_id"]) != target_split:
        raise ValueError("bounded numeric task disagrees with frozen split")
    if split.get("test_outcomes_sealed") is not True:
        raise ValueError("frozen split must keep test outcomes sealed")
    if draws <= 0:
        raise ValueError("draws must be positive")
    if not simulator_id.strip() or not simulator_revision.strip():
        raise ValueError("simulator identity and revision are required")

    parsed_outputs: list[dict[str, Any]] = []
    for output in outputs:
        raw_response = output.get("raw_response")
        if not isinstance(raw_response, str):
            raise ValueError("bounded numeric simulator output requires raw_response text")
        parsed = parse_continuous_prediction(raw_response, integer_only=integer_only)
        if not lower <= parsed.value <= upper:
            raise ValueError("predicted value lies outside the frozen response bounds")
        parsed_outputs.append(
            {
                "arm_id": output.get("arm_id"),
                "draw_index": output.get("draw_index"),
                "predicted_value": parsed.value,
                "raw_response": raw_response,
            }
        )
    raw_locations = aggregate_continuous_predictions(
        parsed_outputs, arm_ids=arm_ids, draws=draws, estimator="mean"
    )
    raw_medians = aggregate_continuous_predictions(
        parsed_outputs, arm_ids=arm_ids, draws=draws, estimator="median"
    )
    utilities = _normalized_locations(
        raw_locations, lower=lower, upper=upper, direction=direction
    )
    median_utilities = _normalized_locations(
        raw_medians, lower=lower, upper=upper, direction=direction
    )
    selected = choose_best_arm(utilities)
    control = str(task["control_arm_id"])
    now = _now_utc()
    raw_payload = {
        "schema_version": "bounded_numeric_simulator_outputs.v1",
        "experiment_id": task["experiment_id"],
        "draws_per_arm": draws,
        "seed": seed,
        "created_at_utc": now,
        "outputs": parsed_outputs,
    }
    raw_digest = freeze_envelope(raw_payload, raw_output_path, require_blinded=True)
    recommendation = {
        "schema_version": RECOMMENDATION_SCHEMA,
        "experiment_id": task["experiment_id"],
        "split": target_split,
        "task_num": task["socsci210_task_num"],
        "selected_arm_id": selected,
        "arm_ranking": sorted(arm_ids, key=lambda arm_id: (-utilities[arm_id], arm_id)),
        "synthetic_arm_locations_raw": raw_locations,
        "synthetic_arm_utilities": utilities,
        "synthetic_treatment_effects": treatment_effects(
            utilities, control_arm_id=control
        ),
        "outcome_family": "bounded_numeric",
        "direction": task["direction"],
        "outcome_unit": task["outcome_unit"],
        "location_estimand": "mean",
        "utility_transform": task["utility_transform"],
        "questionnaire_bounds": {"lower": lower, "upper": upper},
        "normalized_for_pooled_regret": True,
        "robustness": {
            "median": {
                "synthetic_arm_locations_raw": raw_medians,
                "synthetic_arm_utilities": median_utilities,
            }
        },
        "baselines": {
            "no_effect_control_policy": {
                "synthetic_arm_utilities": {arm_id: 0.0 for arm_id in arm_ids},
                "synthetic_treatment_effects": {
                    arm_id: 0.0 for arm_id in arm_ids if arm_id != control
                },
                "selected_arm_id": control,
            }
        },
        "split_manifest_sha256": payload_hash(split),
        "decision_task_sha256": payload_hash(task),
        "blinded_bundle_sha256": payload_hash(bundle),
        "simulator_outputs_sha256": raw_digest,
        "simulator": {"id": simulator_id, "revision": simulator_revision},
        "parser": {"id": "strict_bounded_numeric_integer.v1"},
        "persona_roster": bundle["population"]["roster_id"],
        "diagnostics": {
            "winner_margin_normalized_utility": max(utilities.values())
            - sorted(utilities.values(), reverse=True)[1],
            "parse_failures": 0,
        },
        "provenance": {
            "created_at_utc": now,
            "seed": seed,
            "draws_per_arm": draws,
            "raw_output_path": str(raw_output_path),
        },
    }
    return freeze_recommendation(recommendation, recommendation_path)


def _verify_bound_recommendation(
    *,
    recommendation_path: Path,
    raw_output_path: Path,
    split_manifest_path: Path,
    decision_task_path: Path,
    blinded_bundle_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    recommendation = verify_frozen_recommendation(recommendation_path)
    raw = verify_envelope(raw_output_path, require_blinded=True)
    split = read_json_object(split_manifest_path)
    task = read_json_object(decision_task_path)
    bundle = read_json_object(blinded_bundle_path)
    arm_ids, lower, upper, _, _ = _validate_task_bundle_binding(task, bundle)
    if recommendation.get("schema_version") != RECOMMENDATION_SCHEMA:
        raise ValueError("unsupported bounded numeric recommendation schema")
    if recommendation.get("experiment_id") != task["experiment_id"]:
        raise ValueError("recommendation experiment disagrees with decision task")
    if recommendation.get("task_num") != task["socsci210_task_num"]:
        raise ValueError("recommendation task number disagrees with decision task")
    target_split = task.get("split")
    if target_split not in SUPPORTED_REVEAL_SPLITS:
        raise ValueError("bounded numeric target lacks an evaluation split")
    if recommendation.get("split") != target_split:
        raise ValueError("recommendation split disagrees with decision task")
    if split.get("experiment_to_split", {}).get(task["experiment_id"]) != target_split:
        raise ValueError("split manifest does not authorize this evaluation target")
    if split.get("test_outcomes_sealed") is not True:
        raise ValueError("split manifest must keep test outcomes sealed")
    expected_hashes = {
        "split_manifest_sha256": payload_hash(split),
        "decision_task_sha256": payload_hash(task),
        "blinded_bundle_sha256": payload_hash(bundle),
        "simulator_outputs_sha256": payload_hash(raw),
    }
    for field, expected in expected_hashes.items():
        _sha256_digest(recommendation.get(field), field=field)
        if recommendation[field] != expected:
            raise ValueError(f"recommendation is not bound to the supplied {field}")
    if recommendation.get("normalized_for_pooled_regret") is not True:
        raise ValueError("bounded numeric recommendation must be normalized")
    if recommendation.get("questionnaire_bounds") != {
        "lower": lower,
        "upper": upper,
    }:
        raise ValueError("recommendation questionnaire bounds disagree with task")
    if set(recommendation.get("synthetic_arm_utilities", {})) != set(arm_ids):
        raise ValueError("recommendation utilities do not cover the action set")
    if recommendation.get("selected_arm_id") not in arm_ids:
        raise ValueError("recommendation selected arm is absent from action set")
    return recommendation, split, task, bundle


def freeze_bounded_numeric_reveal_authorization(
    *,
    recommendation_path: Path,
    raw_output_path: Path,
    split_manifest_path: Path,
    decision_task_path: Path,
    blinded_bundle_path: Path,
    authorization_path: Path,
) -> str:
    """Freeze an explicit authorization after the immutable recommendation exists."""

    recommendation, split, task, bundle = _verify_bound_recommendation(
        recommendation_path=recommendation_path,
        raw_output_path=raw_output_path,
        split_manifest_path=split_manifest_path,
        decision_task_path=decision_task_path,
        blinded_bundle_path=blinded_bundle_path,
    )
    authorization = {
        "schema_version": REVEAL_AUTHORIZATION_SCHEMA,
        "status": "target_outcome_reveal_authorized",
        "experiment_id": task["experiment_id"],
        "split": recommendation["split"],
        "task_num": task["socsci210_task_num"],
        "recommendation_sha256": payload_hash(recommendation),
        "simulator_outputs_sha256": recommendation["simulator_outputs_sha256"],
        "split_manifest_sha256": payload_hash(split),
        "decision_task_sha256": payload_hash(task),
        "blinded_bundle_sha256": payload_hash(bundle),
        "authorized_at_utc": _now_utc(),
    }
    return freeze_envelope(authorization, authorization_path, require_blinded=True)


def _verify_reveal_authorization(
    authorization_path: Path,
    *,
    recommendation: Mapping[str, Any],
    split: Mapping[str, Any],
    task: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    authorization = verify_envelope(authorization_path, require_blinded=True)
    expected = {
        "schema_version": REVEAL_AUTHORIZATION_SCHEMA,
        "status": "target_outcome_reveal_authorized",
        "experiment_id": task["experiment_id"],
        "split": recommendation["split"],
        "task_num": task["socsci210_task_num"],
        "recommendation_sha256": payload_hash(recommendation),
        "simulator_outputs_sha256": recommendation["simulator_outputs_sha256"],
        "split_manifest_sha256": payload_hash(split),
        "decision_task_sha256": payload_hash(task),
        "blinded_bundle_sha256": payload_hash(bundle),
    }
    for field, value in expected.items():
        if authorization.get(field) != value:
            raise ValueError(f"reveal authorization is not bound to {field}")
    if not isinstance(authorization.get("authorized_at_utc"), str) or not authorization[
        "authorized_at_utc"
    ].strip():
        raise ValueError("reveal authorization timestamp is required")
    return authorization


def _read_authorized_outcomes(
    parquet_paths: tuple[Path, ...], *, experiment_id: str, task_num: int
) -> pa.Table:
    if not parquet_paths:
        raise ValueError("at least one Parquet path is required")
    return pa.concat_tables(
        [
            pq.read_table(
                path,
                columns=list(REVEALED_OUTCOME_COLUMNS),
                filters=[("study_id", "=", experiment_id), ("task_num", "=", task_num)],
            )
            for path in parquet_paths
        ]
    )


def score_frozen_bounded_numeric_recommendation(
    *,
    parquet_paths: tuple[Path, ...],
    decision_task_path: Path,
    split_manifest_path: Path,
    blinded_bundle_path: Path,
    recommendation_path: Path,
    raw_output_path: Path,
    authorization_path: Path,
    score_path: Path,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: int = 2102026,
) -> str:
    """Verify every frozen binding, then reveal and score normalized utility."""

    recommendation, split, task, bundle = _verify_bound_recommendation(
        recommendation_path=recommendation_path,
        raw_output_path=raw_output_path,
        split_manifest_path=split_manifest_path,
        decision_task_path=decision_task_path,
        blinded_bundle_path=blinded_bundle_path,
    )
    authorization = _verify_reveal_authorization(
        authorization_path,
        recommendation=recommendation,
        split=split,
        task=task,
        bundle=bundle,
    )

    # This is the first line in the scoring path that may access target responses.
    table = _read_authorized_outcomes(
        parquet_paths,
        experiment_id=task["experiment_id"],
        task_num=task["socsci210_task_num"],
    )
    condition_to_arm = {
        int(arm["condition_num"]): str(arm["arm_id"]) for arm in task["arms"]
    }
    observations: list[ContinuousObservation] = []
    for row in table.to_pylist():
        arm_id = condition_to_arm.get(int(row["condition_num"]))
        if arm_id is None:
            raise ValueError("revealed condition is absent from decision task")
        observations.append(
            ContinuousObservation(
                participant_id=f"{row['study_id']}:{row['sample_id']}:{row['participant']}",
                arm_id=arm_id,
                value=row["response"],
            )
        )
    arm_ids, lower, upper, _, direction = _validate_task_bundle_binding(task, bundle)
    valid = task["valid_response"]
    primary = continuous_arm_locations(
        observations,
        arm_ids=arm_ids,
        missing_codes=(),
        valid_lower_bound=lower,
        valid_upper_bound=upper,
        integer_only=bool(valid["integer_only"]),
        estimator="mean",
    )
    human_utilities = _normalized_locations(
        primary.arm_locations, lower=lower, upper=upper, direction=direction
    )
    synthetic_utilities = {
        str(arm_id): float(value)
        for arm_id, value in recommendation["synthetic_arm_utilities"].items()
    }
    selected = str(recommendation["selected_arm_id"])
    if selected != choose_best_arm(synthetic_utilities):
        raise ValueError("frozen selected arm does not match synthetic utilities")
    human_best = choose_best_arm(human_utilities)
    regret = decision_regret(human_utilities, selected)
    control = str(task["control_arm_id"])
    human_effects = treatment_effects(human_utilities, control_arm_id=control)
    synthetic_effects = treatment_effects(synthetic_utilities, control_arm_id=control)
    median = continuous_arm_locations(
        observations,
        arm_ids=arm_ids,
        missing_codes=(),
        valid_lower_bound=lower,
        valid_upper_bound=upper,
        integer_only=bool(valid["integer_only"]),
        estimator="median",
    )
    median_utilities = _normalized_locations(
        median.arm_locations, lower=lower, upper=upper, direction=direction
    )
    synthetic_median_utilities = recommendation["robustness"]["median"][
        "synthetic_arm_utilities"
    ]
    median_selected = choose_best_arm(synthetic_median_utilities)
    median_human_best = choose_best_arm(median_utilities)
    median_regret = decision_regret(median_utilities, median_selected)
    bootstrap = bootstrap_arm_location_optimality(
        primary.arm_values,
        estimator="mean",
        direction=direction,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    tolerance = float(task["estimator"]["practical_regret_tolerance"])
    sensitivities = task["estimator"].get("practical_regret_sensitivity", [0.0])
    if not isinstance(sensitivities, list) or not sensitivities or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0
        for value in sensitivities
    ):
        raise ValueError("practical regret sensitivity must contain non-negative values")
    baseline_selected = recommendation["baselines"]["no_effect_control_policy"][
        "selected_arm_id"
    ]
    if baseline_selected != control:
        raise ValueError("no-effect baseline must select the declared control")
    score: dict[str, Any] = {
        "schema_version": SCORE_SCHEMA,
        "experiment_id": task["experiment_id"],
        "split": recommendation["split"],
        "recommendation_sha256": payload_hash(recommendation),
        "reveal_authorization_sha256": payload_hash(authorization),
        "location_estimand": "mean",
        "selected_arm_id": selected,
        "human_best_arm_id": human_best,
        "correct_choice": selected == human_best,
        "normalized_decision_regret": regret,
        "regret_unit": "normalized_utility",
        "normalized_for_pooled_regret": True,
        "practically_reliable": regret <= tolerance,
        "human_arm_locations_raw": primary.arm_locations,
        "human_arm_utilities": human_utilities,
        "synthetic_arm_locations_raw": recommendation["synthetic_arm_locations_raw"],
        "synthetic_arm_utilities": synthetic_utilities,
        "human_treatment_effects": human_effects,
        "synthetic_treatment_effects": synthetic_effects,
        "absolute_treatment_effect_error": {
            arm_id: abs(synthetic_effects[arm_id] - human_effects[arm_id])
            for arm_id in human_effects
        },
        "treatment_effect_sign_correct": {
            arm_id: (human_effects[arm_id] == 0 and synthetic_effects[arm_id] == 0)
            or human_effects[arm_id] * synthetic_effects[arm_id] > 0
            for arm_id in human_effects
        },
        "practical_reliability_by_tolerance": {
            str(float(value)): regret <= float(value) for value in sensitivities
        },
        "valid_observations_per_arm": primary.valid_counts,
        "missing_observations_per_arm": primary.missing_counts,
        "robustness": {
            "median": {
                "human_arm_locations_raw": median.arm_locations,
                "human_arm_utilities": median_utilities,
                "human_best_arm_id": median_human_best,
                "selected_arm_id": median_selected,
                "normalized_decision_regret": median_regret,
            }
        },
        "no_effect_control_baseline": {
            "selected_arm_id": baseline_selected,
            "correct_choice": baseline_selected == human_best,
            "normalized_decision_regret": decision_regret(
                human_utilities, baseline_selected
            ),
        },
        "bootstrap": {
            "replicates": bootstrap.replicates,
            "seed": bootstrap.seed,
            "optimal_probability": bootstrap.optimal_probability,
            "selected_arm_optimal_probability": bootstrap.optimal_probability[selected],
        },
        "created_at_utc": _now_utc(),
    }
    return freeze_envelope(score, score_path)


def replay_bounded_numeric_score(
    *,
    score_path: Path,
    recommendation_path: Path,
    raw_output_path: Path,
    authorization_path: Path,
) -> dict[str, Any]:
    score = verify_envelope(score_path)
    recommendation = verify_frozen_recommendation(recommendation_path)
    raw = verify_envelope(raw_output_path, require_blinded=True)
    authorization = verify_envelope(authorization_path, require_blinded=True)
    if score.get("schema_version") != SCORE_SCHEMA:
        raise ValueError("unsupported bounded numeric score schema")
    if score.get("recommendation_sha256") != payload_hash(recommendation):
        raise ValueError("score does not match recommendation")
    if recommendation.get("simulator_outputs_sha256") != payload_hash(raw):
        raise ValueError("recommendation does not match simulator outputs")
    if score.get("reveal_authorization_sha256") != payload_hash(authorization):
        raise ValueError("score does not match reveal authorization")
    if authorization.get("recommendation_sha256") != payload_hash(recommendation):
        raise ValueError("reveal authorization does not match recommendation")
    if score.get("normalized_for_pooled_regret") is not True:
        raise ValueError("bounded numeric score must use normalized pooled regret")
    return score
