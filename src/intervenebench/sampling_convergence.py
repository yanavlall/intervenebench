"""Outcome-free adaptive sampling convergence for simulator arm utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class SamplingDecision:
    sample_count: int
    converged: bool
    winner_arm_id: str
    winner_margin: float
    maximum_arm_shift: float | None
    reason: str


def _winner(means: Mapping[str, float]) -> str:
    maximum = max(means.values())
    return min(arm_id for arm_id, value in means.items() if value == maximum)


def _margin(means: Mapping[str, float]) -> float:
    ordered = sorted(means.values(), reverse=True)
    return ordered[0] - ordered[1]


def choose_sampling_checkpoint(
    checkpoints: Sequence[tuple[int, Mapping[str, float]]],
    *,
    minimum_samples: int = 5,
    arm_mean_tolerance: float = 0.01,
    margin_multiplier: float = 2.0,
) -> SamplingDecision:
    """Select the first stable cumulative checkpoint, or retain the cap.

    Convergence uses simulator outputs only.  A checkpoint is accepted when its
    winner is unchanged, every arm moves by at most the declared tolerance, and
    its winner margin exceeds the observed change by the declared multiplier.
    """

    if len(checkpoints) < 2:
        raise ValueError("at least two cumulative sampling checkpoints are required")
    if minimum_samples <= 0:
        raise ValueError("minimum_samples must be positive")
    if not isfinite(arm_mean_tolerance) or arm_mean_tolerance < 0.0:
        raise ValueError("arm_mean_tolerance must be finite and non-negative")
    if not isfinite(margin_multiplier) or margin_multiplier < 0.0:
        raise ValueError("margin_multiplier must be finite and non-negative")

    previous_count = 0
    expected_arms: set[str] | None = None
    normalized: list[tuple[int, dict[str, float]]] = []
    for sample_count, raw_means in checkpoints:
        if sample_count <= previous_count:
            raise ValueError("sampling checkpoints must be strictly increasing")
        if len(raw_means) < 2:
            raise ValueError("each checkpoint must contain at least two arms")
        means = {str(arm_id): float(value) for arm_id, value in raw_means.items()}
        if (
            any(not arm_id.strip() for arm_id in means)
            or any(not isfinite(value) for value in means.values())
        ):
            raise ValueError("checkpoint arms and means must be non-empty and finite")
        arm_ids = set(means)
        if expected_arms is None:
            expected_arms = arm_ids
        elif arm_ids != expected_arms:
            raise ValueError("every checkpoint must contain the same arm IDs")
        normalized.append((sample_count, means))
        previous_count = sample_count

    for index in range(1, len(normalized)):
        sample_count, current = normalized[index]
        if sample_count < minimum_samples:
            continue
        _, previous = normalized[index - 1]
        maximum_shift = max(
            abs(current[arm_id] - previous[arm_id]) for arm_id in current
        )
        winner = _winner(current)
        margin = _margin(current)
        if (
            winner == _winner(previous)
            and maximum_shift <= arm_mean_tolerance
            and margin > margin_multiplier * maximum_shift
        ):
            return SamplingDecision(
                sample_count=sample_count,
                converged=True,
                winner_arm_id=winner,
                winner_margin=margin,
                maximum_arm_shift=maximum_shift,
                reason="winner_and_arm_means_stable",
            )

    sample_count, capped = normalized[-1]
    previous = normalized[-2][1]
    maximum_shift = max(abs(capped[arm] - previous[arm]) for arm in capped)
    return SamplingDecision(
        sample_count=sample_count,
        converged=False,
        winner_arm_id=_winner(capped),
        winner_margin=_margin(capped),
        maximum_arm_shift=maximum_shift,
        reason="maximum_sampling_checkpoint_reached_without_convergence",
    )
