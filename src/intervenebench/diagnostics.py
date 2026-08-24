"""Outcome-free, hash-bound diagnostics for prospective trust decisions."""

from __future__ import annotations

from collections import Counter
from math import fsum, isfinite, log, sqrt
from typing import Any, Mapping, Sequence

from .protocol import assert_blinded_payload, payload_hash


INPUT_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "primary_model_id",
        "control_arm_id",
        "utility_bounds",
        "decision_task_sha256",
        "blinded_bundle_sha256",
        "models",
    }
)
MODEL_FIELDS = frozenset(
    {
        "model_id",
        "model_revision",
        "recommendation_sha256",
        "outputs_sha256",
        "arm_means",
        "draw_arm_means",
        "prompt_variants",
    }
)
VARIANT_FIELDS = frozenset(
    {"variant_id", "inverse_mapping_applied", "arm_means"}
)


def _digest(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be a SHA-256 digest") from error
    return value


def _identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _means(
    value: Any, *, field: str, expected_arms: frozenset[str] | None = None
) -> dict[str, float]:
    if not isinstance(value, Mapping) or len(value) < 2:
        raise ValueError(f"{field} must contain at least two arms")
    means: dict[str, float] = {}
    for raw_arm, raw_mean in value.items():
        arm_id = _identifier(raw_arm, field=f"{field} arm ID")
        if (
            isinstance(raw_mean, bool)
            or not isinstance(raw_mean, (int, float))
            or not isfinite(float(raw_mean))
        ):
            raise ValueError(f"{field} values must be finite numbers")
        means[arm_id] = float(raw_mean)
    if expected_arms is not None and frozenset(means) != expected_arms:
        raise ValueError(f"{field} must cover the same arms")
    return means


def _winner(means: Mapping[str, float]) -> str:
    maximum = max(means.values())
    return min(arm_id for arm_id, value in means.items() if value == maximum)


def _winner_entropy(winners: Sequence[str], *, arm_count: int) -> float:
    if not winners or arm_count <= 1:
        return 0.0
    counts = Counter(winners)
    entropy = -fsum(
        (count / len(winners)) * log(count / len(winners))
        for count in counts.values()
    )
    return entropy / log(arm_count)


def _population_std(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("dispersion requires values")
    center = fsum(values) / len(values)
    return sqrt(fsum((value - center) ** 2 for value in values) / len(values))


def build_outcome_free_diagnostics(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Derive trust features using synthetic predictions and provenance only.

    The complete input object is recursively leakage-checked and hashed. The
    returned artifact deliberately contains no target-human labels or outcomes.
    """

    assert_blinded_payload(inputs)
    if set(inputs) != INPUT_FIELDS:
        raise ValueError("outcome-free diagnostic inputs have unexpected fields")
    if inputs.get("schema_version") != "outcome_free_diagnostic_inputs.v1":
        raise ValueError("unsupported diagnostic-input schema")
    experiment_id = _identifier(inputs.get("experiment_id"), field="experiment_id")
    primary_model_id = _identifier(
        inputs.get("primary_model_id"), field="primary_model_id"
    )
    control_arm_id = _identifier(
        inputs.get("control_arm_id"), field="control_arm_id"
    )
    bounds = inputs.get("utility_bounds")
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            for value in bounds
        )
        or float(bounds[0]) >= float(bounds[1])
    ):
        raise ValueError("utility_bounds must be two finite increasing numbers")
    utility_range = float(bounds[1]) - float(bounds[0])
    decision_task_sha = _digest(
        inputs.get("decision_task_sha256"), field="decision_task_sha256"
    )
    bundle_sha = _digest(
        inputs.get("blinded_bundle_sha256"), field="blinded_bundle_sha256"
    )

    raw_models = inputs.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("models must be a non-empty list")
    models: dict[str, dict[str, Any]] = {}
    expected_arms: frozenset[str] | None = None
    for raw_model in raw_models:
        if not isinstance(raw_model, Mapping) or set(raw_model) != MODEL_FIELDS:
            raise ValueError("diagnostic model entries have unexpected fields")
        model_id = _identifier(raw_model.get("model_id"), field="model_id")
        if model_id in models:
            raise ValueError("diagnostic model IDs must be unique")
        revision = _identifier(
            raw_model.get("model_revision"), field="model_revision"
        )
        recommendation_sha = _digest(
            raw_model.get("recommendation_sha256"),
            field="recommendation_sha256",
        )
        outputs_sha = _digest(
            raw_model.get("outputs_sha256"), field="outputs_sha256"
        )
        arm_means = _means(
            raw_model.get("arm_means"),
            field=f"models[{model_id}].arm_means",
            expected_arms=expected_arms,
        )
        if expected_arms is None:
            expected_arms = frozenset(arm_means)
        if control_arm_id not in arm_means:
            raise ValueError("control arm is absent from model arm means")
        raw_draws = raw_model.get("draw_arm_means")
        if not isinstance(raw_draws, list) or not raw_draws:
            raise ValueError("every model requires at least one synthetic draw")
        draws = [
            _means(
                draw,
                field=f"models[{model_id}].draw_arm_means",
                expected_arms=expected_arms,
            )
            for draw in raw_draws
        ]
        raw_variants = raw_model.get("prompt_variants")
        if not isinstance(raw_variants, list):
            raise ValueError("prompt_variants must be a list")
        variants: list[dict[str, Any]] = []
        variant_ids: set[str] = set()
        for raw_variant in raw_variants:
            if not isinstance(raw_variant, Mapping) or set(raw_variant) != VARIANT_FIELDS:
                raise ValueError("prompt variants have unexpected fields")
            variant_id = _identifier(
                raw_variant.get("variant_id"), field="variant_id"
            )
            if variant_id in variant_ids:
                raise ValueError("prompt variant IDs must be unique within a model")
            variant_ids.add(variant_id)
            if raw_variant.get("inverse_mapping_applied") is not True:
                raise ValueError("prompt perturbations must be exactly inverse mapped")
            variants.append(
                {
                    "variant_id": variant_id,
                    "arm_means": _means(
                        raw_variant.get("arm_means"),
                        field=f"models[{model_id}].prompt_variants[{variant_id}]",
                        expected_arms=expected_arms,
                    ),
                }
            )
        models[model_id] = {
            "revision": revision,
            "recommendation_sha256": recommendation_sha,
            "outputs_sha256": outputs_sha,
            "arm_means": arm_means,
            "draws": draws,
            "variants": variants,
        }

    if primary_model_id not in models:
        raise ValueError("primary model is absent from diagnostic inputs")
    assert expected_arms is not None
    primary = models[primary_model_id]
    primary_winner = _winner(primary["arm_means"])
    ordered_means = sorted(primary["arm_means"].values(), reverse=True)
    draw_winners = [_winner(draw) for draw in primary["draws"]]
    model_winners = {
        model_id: _winner(model["arm_means"])
        for model_id, model in models.items()
    }
    prompt_variants = primary["variants"]
    prompt_winners = [_winner(variant["arm_means"]) for variant in prompt_variants]
    prompt_shifts = [
        abs(variant["arm_means"][arm_id] - primary["arm_means"][arm_id])
        for variant in prompt_variants
        for arm_id in expected_arms
    ]
    effect_dispersions = [
        _population_std(
            [
                model["arm_means"][arm_id]
                - model["arm_means"][control_arm_id]
                for model in models.values()
            ]
        )
        for arm_id in sorted(expected_arms - {control_arm_id})
    ]
    features = {
        "normalized_winner_margin": (ordered_means[0] - ordered_means[1])
        / utility_range,
        "winner_stability": draw_winners.count(primary_winner) / len(draw_winners),
        "winner_rank_entropy": _winner_entropy(
            draw_winners, arm_count=len(expected_arms)
        ),
        "cross_model_winner_agreement": sum(
            winner == primary_winner for winner in model_winners.values()
        )
        / len(model_winners),
        "cross_model_effect_dispersion": fsum(effect_dispersions)
        / len(effect_dispersions),
        "prompt_winner_robustness": (
            sum(winner == primary_winner for winner in prompt_winners)
            / len(prompt_winners)
            if prompt_winners
            else None
        ),
        "prompt_max_arm_mean_shift": max(prompt_shifts) if prompt_shifts else None,
    }
    return {
        "schema_version": "outcome_free_diagnostics.v1",
        "experiment_id": experiment_id,
        "primary_model_id": primary_model_id,
        "primary_selected_arm_id": primary_winner,
        "control_arm_id": control_arm_id,
        "diagnostic_inputs_sha256": payload_hash(inputs),
        "decision_task_sha256": decision_task_sha,
        "blinded_bundle_sha256": bundle_sha,
        "model_recommendation_sha256s": {
            model_id: model["recommendation_sha256"]
            for model_id, model in models.items()
        },
        "model_outputs_sha256s": {
            model_id: model["outputs_sha256"]
            for model_id, model in models.items()
        },
        "model_winners": model_winners,
        "primary_draw_winners": draw_winners,
        "primary_prompt_variant_count": len(prompt_variants),
        "features": features,
        "target_human_outcomes_used": False,
    }
