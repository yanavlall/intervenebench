"""Leakage-safe scoring for source-verified continuous validation tasks."""

from __future__ import annotations

from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any

from .continuous import (
    ContinuousObservation,
    continuous_arm_locations,
    evaluate_continuous_decision,
    orient_continuous_locations,
)
from .evaluation import choose_best_arm, treatment_effects
from .phase1 import read_json_object
from .protocol import (
    freeze_envelope,
    payload_hash,
    verify_envelope,
    verify_frozen_recommendation,
)
from .schemas import OutcomeDirection
from .simulators import (
    aggregate_continuous_predictions,
    parse_continuous_prediction,
    validate_continuous_blinded_bundle,
)
from .socsci210 import read_revealed_continuous_outcomes
from .uncertainty import bootstrap_arm_location_optimality


def freeze_continuous_recommendation_from_outputs(
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
    """Parse outcome-blind simulator outputs and freeze a continuous recommendation."""

    bundle = read_json_object(bundle_path)
    split = read_json_object(split_path)
    task = read_json_object(decision_task_path)
    validate_continuous_blinded_bundle(bundle)
    if task.get("split") != "validation":
        raise ValueError("continuous simulation target must be in validation")
    experiment_id = task["experiment_id"]
    if split.get("experiment_to_split", {}).get(experiment_id) != "validation":
        raise ValueError("continuous decision task disagrees with frozen split")
    if bundle["experiment_id"] != experiment_id:
        raise ValueError("continuous blinded bundle disagrees with decision task")
    if draws <= 0:
        raise ValueError("draws must be positive")
    if not simulator_id.strip() or not simulator_revision.strip():
        raise ValueError("simulator identity and revision are required")

    arm_ids = tuple(str(arm["arm_id"]) for arm in bundle["arms"])
    parsed_outputs: list[dict[str, Any]] = []
    parse_failures = 0
    for output in outputs:
        raw_response = output.get("raw_response")
        if not isinstance(raw_response, str):
            raise ValueError("continuous simulator output requires raw_response text")
        try:
            parsed = parse_continuous_prediction(raw_response, integer_only=True)
        except ValueError:
            parse_failures += 1
            raise
        parsed_outputs.append(
            {
                "arm_id": output.get("arm_id"),
                "draw_index": output.get("draw_index"),
                "predicted_value": parsed.value,
                "raw_response": raw_response,
            }
        )
    synthetic_locations = aggregate_continuous_predictions(
        parsed_outputs,
        arm_ids=arm_ids,
        draws=draws,
        estimator=task["estimator"]["location"],
    )
    synthetic_median_locations = aggregate_continuous_predictions(
        parsed_outputs,
        arm_ids=arm_ids,
        draws=draws,
        estimator="median",
    )
    direction = OutcomeDirection(task["direction"])
    synthetic_utility = orient_continuous_locations(
        synthetic_locations, direction=direction
    )
    control = str(task["control_arm_id"])
    selected = choose_best_arm(
        synthetic_utility, tie_preferred_arm_id=control
    )
    effects = treatment_effects(synthetic_utility, control_arm_id=control)
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    raw_payload = {
        "schema_version": "continuous_simulator_outputs.v1",
        "experiment_id": experiment_id,
        "draws_per_arm": draws,
        "seed": seed,
        "parse_failures": parse_failures,
        "created_at_utc": now,
        "outputs": parsed_outputs,
    }
    raw_digest = freeze_envelope(
        raw_payload, raw_output_path, require_blinded=True
    )
    baseline_locations = {arm_id: 0.0 for arm_id in arm_ids}
    recommendation = {
        "schema_version": "continuous_recommendation.v1",
        "experiment_id": experiment_id,
        "split": "validation",
        "task_num": task["socsci210_task_num"],
        "selected_arm_id": selected,
        "arm_ranking": sorted(
            arm_ids, key=lambda arm_id: (-synthetic_utility[arm_id], arm_id)
        ),
        "synthetic_arm_locations": synthetic_locations,
        "synthetic_treatment_effects": effects,
        "outcome_family": "continuous",
        "direction": task["direction"],
        "outcome_unit": task["outcome_unit"],
        "location_estimand": task["estimator"]["location"],
        "normalized_for_pooled_regret": False,
        "robustness": {
            "median": {
                "synthetic_arm_locations": synthetic_median_locations,
            }
        },
        "baselines": {
            "no_effect_control_policy": {
                "synthetic_arm_locations": baseline_locations,
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
        "parser": {"id": "strict_continuous_integer.v1"},
        "persona_roster": bundle["population"]["roster_id"],
        "diagnostics": {
            "winner_margin_raw_utility": (
                max(synthetic_utility.values())
                - sorted(synthetic_utility.values(), reverse=True)[1]
            ),
            "parse_failures": parse_failures,
        },
        "provenance": {
            "created_at_utc": now,
            "seed": seed,
            "draws_per_arm": draws,
            "raw_output_path": str(raw_output_path),
        },
    }
    from .protocol import freeze_recommendation

    return freeze_recommendation(recommendation, recommendation_path)


def score_frozen_continuous_validation_recommendation(
    *,
    parquet_paths: tuple[Path, ...],
    decision_task_path: Path,
    split_manifest_path: Path,
    recommendation_path: Path,
    raw_output_path: Path,
    score_path: Path,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: int = 2102026,
) -> str:
    """Reveal one authorized validation outcome and score raw-unit mean regret."""

    task = read_json_object(decision_task_path)
    recommendation = verify_frozen_recommendation(recommendation_path)
    raw_payload = verify_envelope(raw_output_path, require_blinded=True)
    if payload_hash(raw_payload) != recommendation.get("simulator_outputs_sha256"):
        raise ValueError("recommendation does not match frozen simulator outputs")
    table = read_revealed_continuous_outcomes(
        parquet_paths,
        experiment_id=task["experiment_id"],
        recommendation_path=recommendation_path,
        split_manifest_path=split_manifest_path,
        decision_task_path=decision_task_path,
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
                participant_id=f"{row['study_id']}:{row['participant']}",
                arm_id=arm_id,
                value=row["response"],
            )
        )

    valid_response = task["valid_response"]
    locations = continuous_arm_locations(
        observations,
        arm_ids=tuple(arm["arm_id"] for arm in task["arms"]),
        missing_codes=tuple(float(code) for code in valid_response["missing_codes"]),
        valid_lower_bound=valid_response["lower_bound"],
        valid_upper_bound=valid_response["upper_bound"],
        integer_only=bool(valid_response["integer_only"]),
        estimator=task["estimator"]["location"],
    )
    evaluation = evaluate_continuous_decision(
        human_locations=locations.arm_locations,
        synthetic_locations=recommendation["synthetic_arm_locations"],
        control_arm_id=task["control_arm_id"],
        direction=OutcomeDirection(task["direction"]),
        practical_regret_tolerance=float(
            task["estimator"]["practical_regret_tolerance"]
        ),
        outcome_unit=task["outcome_unit"],
    )
    bootstrap = bootstrap_arm_location_optimality(
        locations.arm_values,
        estimator=task["estimator"]["location"],
        direction=OutcomeDirection(task["direction"]),
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    median_locations = continuous_arm_locations(
        observations,
        arm_ids=tuple(arm["arm_id"] for arm in task["arms"]),
        missing_codes=tuple(float(code) for code in valid_response["missing_codes"]),
        valid_lower_bound=valid_response["lower_bound"],
        valid_upper_bound=valid_response["upper_bound"],
        integer_only=bool(valid_response["integer_only"]),
        estimator="median",
    )
    median_evaluation = evaluate_continuous_decision(
        human_locations=median_locations.arm_locations,
        synthetic_locations=recommendation["robustness"]["median"][
            "synthetic_arm_locations"
        ],
        control_arm_id=task["control_arm_id"],
        direction=OutcomeDirection(task["direction"]),
        practical_regret_tolerance=float(
            task["estimator"]["practical_regret_tolerance"]
        ),
        outcome_unit=task["outcome_unit"],
    )
    baseline = recommendation["baselines"]["no_effect_control_policy"]
    if baseline.get("selected_arm_id") != task["control_arm_id"]:
        raise ValueError("no-effect baseline must select the declared control arm")
    baseline_evaluation = evaluate_continuous_decision(
        human_locations=locations.arm_locations,
        synthetic_locations=baseline["synthetic_arm_locations"],
        control_arm_id=task["control_arm_id"],
        direction=OutcomeDirection(task["direction"]),
        practical_regret_tolerance=float(
            task["estimator"]["practical_regret_tolerance"]
        ),
        outcome_unit=task["outcome_unit"],
    )
    effect_errors = {
        arm_id: abs(
            evaluation.synthetic_treatment_effects[arm_id]
            - evaluation.human_treatment_effects[arm_id]
        )
        for arm_id in evaluation.human_treatment_effects
    }
    sign_correct = {
        arm_id: (
            evaluation.human_treatment_effects[arm_id] == 0
            and evaluation.synthetic_treatment_effects[arm_id] == 0
        )
        or (
            evaluation.human_treatment_effects[arm_id]
            * evaluation.synthetic_treatment_effects[arm_id]
            > 0
        )
        for arm_id in evaluation.human_treatment_effects
    }
    tolerances = task["estimator"].get("practical_regret_sensitivity", [0.0])
    if (
        not isinstance(tolerances, list)
        or not tolerances
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(value)
            or value < 0
            for value in tolerances
        )
    ):
        raise ValueError("practical regret sensitivity must contain non-negative values")
    score: dict[str, Any] = {
        "schema_version": "continuous_score.v1",
        "experiment_id": task["experiment_id"],
        "split": "validation",
        "recommendation_sha256": payload_hash(recommendation),
        "location_estimand": task["estimator"]["location"],
        "selected_arm_id": evaluation.selected_arm_id,
        "human_best_arm_id": evaluation.human_best_arm_id,
        "correct_choice": evaluation.correct_choice,
        "raw_decision_regret": evaluation.regret,
        "regret_unit": evaluation.regret_unit,
        "normalized_for_pooled_regret": False,
        "practically_reliable": evaluation.practically_reliable,
        "human_arm_locations": evaluation.human_arm_locations,
        "synthetic_arm_locations": evaluation.synthetic_arm_locations,
        "human_treatment_effects": evaluation.human_treatment_effects,
        "synthetic_treatment_effects": evaluation.synthetic_treatment_effects,
        "absolute_treatment_effect_error": effect_errors,
        "treatment_effect_sign_correct": sign_correct,
        "practical_reliability_by_tolerance": {
            str(float(tolerance)): evaluation.regret <= float(tolerance)
            for tolerance in tolerances
        },
        "valid_observations_per_arm": locations.valid_counts,
        "missing_observations_per_arm": locations.missing_counts,
        "robustness": {
            "median": {
                "human_arm_locations": median_evaluation.human_arm_locations,
                "human_best_arm_id": median_evaluation.human_best_arm_id,
                "selected_arm_id": median_evaluation.selected_arm_id,
                "raw_decision_regret": median_evaluation.regret,
                "regret_unit": median_evaluation.regret_unit,
            }
        },
        "no_effect_control_baseline": {
            "selected_arm_id": baseline_evaluation.selected_arm_id,
            "correct_choice": baseline_evaluation.correct_choice,
            "raw_decision_regret": baseline_evaluation.regret,
            "regret_unit": baseline_evaluation.regret_unit,
        },
        "bootstrap": {
            "replicates": bootstrap.replicates,
            "seed": bootstrap.seed,
            "optimal_probability": bootstrap.optimal_probability,
            "selected_arm_optimal_probability": bootstrap.optimal_probability[
                evaluation.selected_arm_id
            ],
        },
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    return freeze_envelope(score, score_path)


def replay_continuous_score(
    *, score_path: Path, recommendation_path: Path, raw_output_path: Path
) -> dict[str, Any]:
    score = verify_envelope(score_path)
    recommendation = verify_frozen_recommendation(recommendation_path)
    raw = verify_envelope(raw_output_path, require_blinded=True)
    if score["recommendation_sha256"] != payload_hash(recommendation):
        raise ValueError("score does not match recommendation")
    if recommendation["simulator_outputs_sha256"] != payload_hash(raw):
        raise ValueError("recommendation does not match simulator outputs")
    if score.get("normalized_for_pooled_regret") is not False:
        raise ValueError("continuous score must not claim normalized pooled regret")
    return score
