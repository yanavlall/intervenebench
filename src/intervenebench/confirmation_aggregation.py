"""Outcome-blind aggregation for the frozen six-experiment confirmation panel."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import fsum, isfinite, log, sqrt
import random
from statistics import fmean
from typing import Any, Mapping, Sequence

from .protocol import assert_blinded_payload


TRUST_COMPONENT_DIRECTIONS = {
    "primary_normalized_top_two_margin": "larger",
    "primary_resampled_winner_stability": "larger",
    "primary_prompt_interface_sensitivity": "smaller",
    "cross_model_winner_agreement": "larger",
    "cross_model_arm_rank_dispersion": "smaller",
}


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def probability_call_summary(
    probabilities: Mapping[str, Any],
    *,
    value_to_utility: Mapping[int, float],
) -> dict[str, float]:
    """Map a strict source-value distribution to location, utility, and entropy."""

    if len(value_to_utility) < 2:
        raise ValueError("at least two response values are required")
    expected_keys = {str(value) for value in value_to_utility}
    if set(probabilities) != expected_keys:
        raise ValueError("probability support does not match response values")
    parsed = {
        int(key): _finite(value, field=f"probability[{key}]")
        for key, value in probabilities.items()
    }
    if any(value < 0.0 or value > 1.0 for value in parsed.values()):
        raise ValueError("probabilities must lie in [0, 1]")
    if abs(fsum(parsed.values()) - 1.0) > 1e-8:
        raise ValueError("probabilities must sum to one")
    source_location = fsum(value * probability for value, probability in parsed.items())
    decision_score = fsum(
        float(value_to_utility[value]) * probability
        for value, probability in parsed.items()
    )
    entropy = -fsum(probability * log(probability) for probability in parsed.values() if probability)
    return {
        "source_location": source_location,
        "decision_score": decision_score,
        "normalized_response_entropy": entropy / log(len(parsed)),
    }


def continuous_call_summary(
    predicted_value: Any, *, lower_is_better: bool
) -> dict[str, float | None]:
    """Represent an unbounded scalar draw without manufacturing normalization."""

    location = _finite(predicted_value, field="predicted_value")
    return {
        "source_location": location,
        "decision_score": -location if lower_is_better else location,
        "normalized_response_entropy": None,
    }


def _winner(means: Mapping[str, float]) -> str:
    best = max(means.values())
    return min(arm_id for arm_id, value in means.items() if value == best)


def _population_sd(values: Sequence[float]) -> float:
    center = fmean(values)
    return sqrt(fmean((value - center) ** 2 for value in values))


def _normalized_ranks(means: Mapping[str, float]) -> dict[str, float]:
    """Return 0=best and 1=worst midranks, preserving exact ties."""

    arm_ids = sorted(means)
    if len(arm_ids) < 2:
        raise ValueError("ranking requires at least two arms")
    ranks: dict[str, float] = {}
    for arm_id in arm_ids:
        better = sum(means[other] > means[arm_id] for other in arm_ids)
        tied_other = sum(means[other] == means[arm_id] for other in arm_ids) - 1
        ranks[arm_id] = (better + 0.5 * tied_other) / (len(arm_ids) - 1)
    return ranks


def _arm_means(rows: Sequence[Mapping[str, Any]], arm_ids: Sequence[str]) -> tuple[dict[str, float], dict[str, float]]:
    scores: dict[str, list[float]] = {arm_id: [] for arm_id in arm_ids}
    locations: dict[str, list[float]] = {arm_id: [] for arm_id in arm_ids}
    for row in rows:
        arm_id = str(row["arm_id"])
        if arm_id not in scores:
            raise ValueError("synthetic row contains an unexpected arm")
        scores[arm_id].append(_finite(row["decision_score"], field="decision_score"))
        locations[arm_id].append(_finite(row["source_location"], field="source_location"))
    if any(not values for values in scores.values()):
        raise ValueError("every arm requires synthetic outputs")
    return (
        {arm: fmean(scores[arm]) for arm in arm_ids},
        {arm: fmean(locations[arm]) for arm in arm_ids},
    )


def _validate_paired_grid(rows: Sequence[Mapping[str, Any]], arm_ids: Sequence[str]) -> None:
    cells_by_arm: dict[str, list[tuple[str, str]]] = {arm: [] for arm in arm_ids}
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        arm_id = str(row["arm_id"])
        key = (arm_id, str(row["nuisance_id"]), str(row["answer_order"]))
        if key in seen:
            raise ValueError("duplicate arm/nuisance/order cell")
        seen.add(key)
        cells_by_arm[arm_id].append((key[1], key[2]))
    reference = sorted(cells_by_arm[arm_ids[0]])
    if not reference or any(sorted(cells_by_arm[arm]) != reference for arm in arm_ids):
        raise ValueError("synthetic cells must be paired across every arm")


def _nuisance_arm_scores(
    rows: Sequence[Mapping[str, Any]], arm_ids: Sequence[str]
) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["nuisance_id"]), str(row["arm_id"]))].append(
            _finite(row["decision_score"], field="decision_score")
        )
    nuisance_ids = sorted({key[0] for key in grouped})
    return {
        nuisance_id: {
            arm_id: fmean(grouped[(nuisance_id, arm_id)]) for arm_id in arm_ids
        }
        for nuisance_id in nuisance_ids
    }


def _resampled_winner_stability(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm_ids: Sequence[str],
    selected_arm_id: str,
    resamples: int,
    rng: random.Random,
) -> float:
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    by_nuisance = _nuisance_arm_scores(rows, arm_ids)
    nuisance_ids = sorted(by_nuisance)
    winners: list[str] = []
    for _ in range(resamples):
        sampled = [rng.choice(nuisance_ids) for _ in nuisance_ids]
        means = {
            arm_id: fmean(by_nuisance[nuisance][arm_id] for nuisance in sampled)
            for arm_id in arm_ids
        }
        winners.append(_winner(means))
    return winners.count(selected_arm_id) / resamples


def _normalized_shift(
    base: float, perturbed: float, *, continuous_unbounded: bool
) -> float:
    difference = abs(base - perturbed)
    if not continuous_unbounded:
        return difference
    return difference / max(1.0, abs(base), abs(perturbed))


def aggregate_experiment(
    *,
    experiment_id: str,
    rows: Sequence[Mapping[str, Any]],
    arm_ids: Sequence[str],
    control_arm_id: str,
    primary_model_id: str,
    bootstrap_resamples: int,
    rng: random.Random,
    continuous_unbounded: bool,
) -> dict[str, Any]:
    """Aggregate base outputs and prespecified outcome-free diagnostics."""

    assert_blinded_payload(rows)
    if len(arm_ids) < 2 or len(set(arm_ids)) != len(arm_ids):
        raise ValueError("arm IDs must be unique and contain at least two arms")
    if control_arm_id not in arm_ids:
        raise ValueError("control arm is absent")
    base = [row for row in rows if row["stage"] == "base"]
    perturb = [
        row for row in rows if row["stage"] == "primary_prompt_perturbation"
    ]
    unsupported = {
        str(row["stage"])
        for row in rows
        if row["stage"] not in {"base", "primary_prompt_perturbation"}
    }
    if unsupported:
        raise ValueError(f"unsupported aggregation stages: {sorted(unsupported)}")
    models = sorted({str(row["model_id"]) for row in base})
    if primary_model_id not in models:
        raise ValueError("primary model is unavailable")
    recommendations: dict[str, dict[str, Any]] = {}
    base_by_model: dict[str, list[Mapping[str, Any]]] = {}
    for model_id in models:
        model_rows = [row for row in base if row["model_id"] == model_id]
        _validate_paired_grid(model_rows, arm_ids)
        base_by_model[model_id] = model_rows
        scores, locations = _arm_means(model_rows, arm_ids)
        selected = _winner(scores)
        control_score = scores[control_arm_id]
        recommendations[model_id] = {
            "selected_arm_id": selected,
            "arm_source_locations": locations,
            "arm_decision_scores": scores,
            "synthetic_treatment_effects": {
                arm_id: scores[arm_id] - control_score for arm_id in arm_ids
            },
            "base_call_count": len(model_rows),
            "tie_rule": "lexicographic_arm_id",
        }

    primary = recommendations[primary_model_id]
    primary_scores = primary["arm_decision_scores"]
    ordered_scores = sorted(primary_scores.values(), reverse=True)
    raw_margin = ordered_scores[0] - ordered_scores[1]
    if continuous_unbounded:
        margin = raw_margin / max(
            1.0, abs(ordered_scores[0]), abs(ordered_scores[1])
        )
    else:
        margin = raw_margin

    if perturb:
        if {str(row["model_id"]) for row in perturb} != {primary_model_id}:
            raise ValueError("prompt perturbations must belong only to the primary model")
        _validate_paired_grid(perturb, arm_ids)
        base_cells = {
            (str(row["arm_id"]), str(row["nuisance_id"]), str(row["answer_order"]))
            for row in base_by_model[primary_model_id]
        }
        perturb_cells = {
            (str(row["arm_id"]), str(row["nuisance_id"]), str(row["answer_order"]))
            for row in perturb
        }
        if perturb_cells != base_cells:
            raise ValueError("primary prompt cells do not match primary base cells")
        perturb_scores, _ = _arm_means(perturb, arm_ids)
        shifts = {
            arm_id: _normalized_shift(
                primary_scores[arm_id],
                perturb_scores[arm_id],
                continuous_unbounded=continuous_unbounded,
            )
            for arm_id in arm_ids
        }
        prompt_sensitivity: float | None = max(shifts.values())
        prompt_robustness: float | None = float(
            _winner(perturb_scores) == primary["selected_arm_id"]
        )
    else:
        perturb_scores = None
        shifts = None
        prompt_sensitivity = None
        prompt_robustness = None

    winners = {
        model_id: str(recommendations[model_id]["selected_arm_id"])
        for model_id in models
    }
    agreement = sum(
        winner == primary["selected_arm_id"] for winner in winners.values()
    ) / len(winners)
    normalized_ranks = {
        model_id: _normalized_ranks(
            recommendations[model_id]["arm_decision_scores"]
        )
        for model_id in models
    }
    rank_dispersion = fmean(
        _population_sd([normalized_ranks[model_id][arm_id] for model_id in models])
        for arm_id in arm_ids
    )
    chosen_entropy_values = [
        row["normalized_response_entropy"]
        for row in base_by_model[primary_model_id]
        if row["arm_id"] == primary["selected_arm_id"]
        and row["normalized_response_entropy"] is not None
    ]
    chosen_entropy = (
        fmean(_finite(value, field="normalized_response_entropy") for value in chosen_entropy_values)
        if chosen_entropy_values
        else None
    )
    stability = _resampled_winner_stability(
        base_by_model[primary_model_id],
        arm_ids=arm_ids,
        selected_arm_id=str(primary["selected_arm_id"]),
        resamples=bootstrap_resamples,
        rng=rng,
    )
    return {
        "experiment_id": experiment_id,
        "primary_model_id": primary_model_id,
        "control_arm_id": control_arm_id,
        "model_recommendations": recommendations,
        "primary_recommendation": dict(primary),
        "diagnostics": {
            "primary_normalized_top_two_margin": margin,
            "primary_resampled_winner_stability": stability,
            "primary_prompt_interface_sensitivity": prompt_sensitivity,
            "primary_prompt_winner_robustness": prompt_robustness,
            "primary_prompt_arm_mean_shifts": shifts,
            "primary_prompt_arm_decision_scores": perturb_scores,
            "cross_model_winner_agreement": agreement,
            "cross_model_arm_rank_dispersion": rank_dispersion,
            "primary_chosen_arm_normalized_response_entropy": chosen_entropy,
            "model_winners": winners,
            "normalized_arm_ranks_by_model": normalized_ranks,
        },
    }


def _midrank_scores(values: Mapping[str, float], *, larger_is_better: bool) -> dict[str, float]:
    directed = {
        key: value if larger_is_better else -value for key, value in values.items()
    }
    denominator = len(directed) - 1
    return {
        key: (
            sum(other < directed[key] for other in directed.values())
            + 0.5 * (sum(other == directed[key] for other in directed.values()) - 1)
        )
        / denominator
        for key in directed
    }


def build_trust_ranking(diagnostics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the frozen equal-directional-midrank score without target labels."""

    assert_blinded_payload(diagnostics)
    if len(diagnostics) < 2:
        raise ValueError("trust ranking requires at least two experiments")
    by_id = {str(row["experiment_id"]): row for row in diagnostics}
    if len(by_id) != len(diagnostics):
        raise ValueError("diagnostic experiment IDs must be unique")
    component_scores: dict[str, dict[str, float]] = {}
    raw_values: dict[str, dict[str, float]] = {}
    for component, direction in TRUST_COMPONENT_DIRECTIONS.items():
        values = {
            experiment_id: _finite(row[component], field=component)
            for experiment_id, row in by_id.items()
        }
        raw_values[component] = values
        component_scores[component] = _midrank_scores(
            values, larger_is_better=direction == "larger"
        )
    confidence = {
        experiment_id: fmean(
            component_scores[component][experiment_id]
            for component in TRUST_COMPONENT_DIRECTIONS
        )
        for experiment_id in by_id
    }
    ranking = [
        {
            "rank": index + 1,
            "experiment_id": experiment_id,
            "trust_score": confidence[experiment_id],
        }
        for index, experiment_id in enumerate(
            sorted(confidence, key=lambda item: (-confidence[item], item))
        )
    ]
    return {
        "method": "equal_mean_of_five_direction_aligned_midranks",
        "component_order": list(TRUST_COMPONENT_DIRECTIONS),
        "component_directions": dict(TRUST_COMPONENT_DIRECTIONS),
        "raw_outcome_free_values": raw_values,
        "component_midrank_scores": component_scores,
        "confidence_by_experiment": confidence,
        "ranking": ranking,
        "final_tie_rule": "lexicographic_experiment_id",
        "learned_threshold": None,
        "accept_abstain_policy": "not_validated_not_deployed",
    }


def validate_aggregation_authorization(
    authorization: Mapping[str, Any],
    *,
    run_id: str,
    adjudication_manifest_payload_sha256: str,
    call_plan_payload_sha256: str,
    preparation_payload_sha256: str,
    protocol_payload_sha256: str,
    strict_output_map_sha256: str,
) -> None:
    """Require explicit, hash-bound authority for aggregation and nothing else."""

    assert_blinded_payload(authorization)
    if authorization.get("schema_version") != "confirmation_aggregation_authorization.v1" or authorization.get("status") != "authorized_outcome_blind_aggregation_only":
        raise PermissionError("invalid confirmation aggregation authorization")
    if authorization.get("aggregation_authorized") is not True:
        raise PermissionError("confirmation aggregation is not authorized")
    forbidden = {
        "model_calls_authorized": False,
        "modal_compute_authorized": False,
        "model_download_authorized": False,
        "confirmation_outcome_reveal_authorized": False,
        "participant_row_access_authorized": False,
        "human_outcome_scoring_authorized": False,
        "automatic_next_stage_authorized": False,
    }
    if any(authorization.get(key) is not value for key, value in forbidden.items()):
        raise PermissionError("confirmation aggregation authority expanded")
    expected = {
        "run_id": run_id,
        "adjudication_manifest_payload_sha256": adjudication_manifest_payload_sha256,
        "call_plan_payload_sha256": call_plan_payload_sha256,
        "preparation_payload_sha256": preparation_payload_sha256,
        "protocol_payload_sha256": protocol_payload_sha256,
        "strict_output_map_sha256": strict_output_map_sha256,
        "expected_strict_output_count": 1404,
        "expected_unavailable_call_count": 60,
    }
    if any(authorization.get(key) != value for key, value in expected.items()):
        raise PermissionError("confirmation aggregation authorization binding drifted")

