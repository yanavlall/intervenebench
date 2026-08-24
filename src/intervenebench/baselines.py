"""Leakage-safe simple baselines for experiment-held-out evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from math import isfinite
from typing import Mapping, Sequence


FeatureValue = str | int | float | bool


@dataclass(frozen=True, slots=True)
class LabeledBaselineExample:
    experiment_id: str
    split: str
    features: Mapping[str, FeatureValue]
    normalized_utility: float


@dataclass(frozen=True, slots=True)
class HashedRidgeModel:
    coefficients: tuple[float, ...]
    feature_dimension: int
    l2_penalty: float
    training_experiment_ids: tuple[str, ...]

    def predict(self, features: Mapping[str, FeatureValue]) -> float:
        vector = _vectorize(features, self.feature_dimension)
        prediction = sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, vector, strict=True)
        )
        return min(1.0, max(0.0, prediction))


@dataclass(frozen=True, slots=True)
class LabeledEffectExample:
    """One aggregate arm-versus-reference effect from a revealed experiment."""

    experiment_id: str
    split: str
    features: Mapping[str, FeatureValue]
    normalized_effect: float
    sample_weight: float = 1.0


@dataclass(frozen=True, slots=True)
class HashedRidgeEffectModel:
    """Regularized classical model for normalized treatment effects."""

    coefficients: tuple[float, ...]
    feature_dimension: int
    l2_penalty: float
    training_experiment_ids: tuple[str, ...]
    training_experiment_weights: tuple[tuple[str, float], ...]

    def predict_effect(self, features: Mapping[str, FeatureValue]) -> float:
        vector = _vectorize(features, self.feature_dimension)
        prediction = sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, vector, strict=True)
        )
        return min(1.0, max(-1.0, prediction))


def _feature_slot(token: str, dimension: int) -> tuple[int, float]:
    digest = blake2b(token.encode("utf-8"), digest_size=16).digest()
    slot = 1 + int.from_bytes(digest[:8], "big") % (dimension - 1)
    sign = 1.0 if digest[8] % 2 == 0 else -1.0
    return slot, sign


def _vectorize(
    features: Mapping[str, FeatureValue], dimension: int
) -> tuple[float, ...]:
    if dimension < 4:
        raise ValueError("feature_dimension must be at least four")
    vector = [0.0] * dimension
    vector[0] = 1.0
    for raw_name, raw_value in sorted(features.items()):
        name = str(raw_name).strip()
        if not name:
            raise ValueError("feature names must be non-empty")
        if isinstance(raw_value, bool):
            slot, sign = _feature_slot(f"{name}={str(raw_value).lower()}", dimension)
            vector[slot] += sign
        elif isinstance(raw_value, (int, float)):
            value = float(raw_value)
            if not isfinite(value):
                raise ValueError("numeric baseline features must be finite")
            slot, sign = _feature_slot(f"numeric:{name}", dimension)
            vector[slot] += sign * value
        elif isinstance(raw_value, str) and raw_value.strip():
            slot, sign = _feature_slot(f"{name}={raw_value.strip()}", dimension)
            vector[slot] += sign
        else:
            raise ValueError("baseline features must be finite scalars or non-empty strings")
    return tuple(vector)


def _solve(matrix: list[list[float]], target: list[float]) -> tuple[float, ...]:
    """Solve a dense positive-definite system with pivoted elimination."""

    size = len(target)
    augmented = [row[:] + [target[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("regularized baseline system is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return tuple(augmented[row][-1] for row in range(size))


def fit_hashed_ridge(
    examples: Sequence[LabeledBaselineExample],
    *,
    experiment_to_split: Mapping[str, str],
    feature_dimension: int = 64,
    l2_penalty: float = 1.0,
) -> HashedRidgeModel:
    """Fit a bounded-utility ridge baseline using training experiments only.

    The caller supplies response-blind structured features. Experiment and arm IDs
    are retained only for provenance and are never automatically encoded.
    """

    if not examples:
        raise ValueError("at least one training example is required")
    if feature_dimension < 4:
        raise ValueError("feature_dimension must be at least four")
    if not isfinite(l2_penalty) or l2_penalty <= 0.0:
        raise ValueError("l2_penalty must be positive and finite")
    if any(example.split != "train" for example in examples):
        raise ValueError("classical baseline fitting may use train split labels only")
    if any(
        experiment_to_split.get(example.experiment_id) != "train"
        for example in examples
    ):
        raise ValueError(
            "classical baseline examples must be bound to train experiments in "
            "the supplied split manifest"
        )
    if any(
        not isfinite(example.normalized_utility)
        or not 0.0 <= example.normalized_utility <= 1.0
        for example in examples
    ):
        raise ValueError("training utilities must be finite and normalized to [0, 1]")
    if any(not example.experiment_id.strip() for example in examples):
        raise ValueError("training experiment IDs must be non-empty")

    vectors = [_vectorize(example.features, feature_dimension) for example in examples]
    matrix = [[0.0] * feature_dimension for _ in range(feature_dimension)]
    target = [0.0] * feature_dimension
    for vector, example in zip(vectors, examples, strict=True):
        for row, row_value in enumerate(vector):
            target[row] += row_value * example.normalized_utility
            for column, column_value in enumerate(vector):
                matrix[row][column] += row_value * column_value
    # Do not regularize the intercept; every other coefficient is ridge-penalized.
    for index in range(1, feature_dimension):
        matrix[index][index] += l2_penalty
    coefficients = _solve(matrix, target)
    return HashedRidgeModel(
        coefficients=coefficients,
        feature_dimension=feature_dimension,
        l2_penalty=l2_penalty,
        training_experiment_ids=tuple(
            sorted({example.experiment_id for example in examples})
        ),
    )


def fit_hashed_effect_ridge(
    examples: Sequence[LabeledEffectExample],
    *,
    experiment_to_split: Mapping[str, str],
    feature_dimension: int = 128,
    l2_penalty: float = 10.0,
) -> HashedRidgeEffectModel:
    """Fit signed normalized effects using training experiments only.

    Callers should give each experiment equal total sample weight so experiments
    with more arms do not dominate the fit. Target experiment labels are rejected
    through both the example split and the independently supplied split manifest.
    """

    if not examples:
        raise ValueError("at least one training example is required")
    if feature_dimension < 4:
        raise ValueError("feature_dimension must be at least four")
    if not isfinite(l2_penalty) or l2_penalty <= 0.0:
        raise ValueError("l2_penalty must be positive and finite")
    if any(example.split != "train" for example in examples):
        raise ValueError("classical baseline fitting may use train split labels only")
    if any(
        experiment_to_split.get(example.experiment_id) != "train"
        for example in examples
    ):
        raise ValueError(
            "classical baseline examples must be bound to train experiments in "
            "the supplied split manifest"
        )
    if any(
        not isfinite(example.normalized_effect)
        or not -1.0 <= example.normalized_effect <= 1.0
        for example in examples
    ):
        raise ValueError("training effects must be finite and normalized to [-1, 1]")
    if any(not example.experiment_id.strip() for example in examples):
        raise ValueError("training experiment IDs must be non-empty")
    if any(
        not isfinite(example.sample_weight) or example.sample_weight <= 0.0
        for example in examples
    ):
        raise ValueError("sample weights must be positive and finite")

    vectors = [_vectorize(example.features, feature_dimension) for example in examples]
    matrix = [[0.0] * feature_dimension for _ in range(feature_dimension)]
    target = [0.0] * feature_dimension
    experiment_weights: dict[str, float] = {}
    for vector, example in zip(vectors, examples, strict=True):
        weight = float(example.sample_weight)
        experiment_weights[example.experiment_id] = (
            experiment_weights.get(example.experiment_id, 0.0) + weight
        )
        for row, row_value in enumerate(vector):
            target[row] += weight * row_value * example.normalized_effect
            for column, column_value in enumerate(vector):
                matrix[row][column] += weight * row_value * column_value
    for index in range(1, feature_dimension):
        matrix[index][index] += l2_penalty
    coefficients = _solve(matrix, target)
    return HashedRidgeEffectModel(
        coefficients=coefficients,
        feature_dimension=feature_dimension,
        l2_penalty=l2_penalty,
        training_experiment_ids=tuple(sorted(experiment_weights)),
        training_experiment_weights=tuple(sorted(experiment_weights.items())),
    )
