"""Checkpoint-training exposure checks for simulator comparisons.

These checks distinguish benchmark holdout from checkpoint holdout.  They use
only released split metadata and source-identity links; no response value is
loaded or summarized.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class CheckpointCompatibility:
    experiment_id: str
    checkpoint_experiment_id: str | None
    status: str
    primary_eligible: bool


def read_study_mapping(path: Path) -> dict[str, tuple[str, ...]]:
    """Read a released study-wise seen/unseen mapping and fail on ambiguity."""

    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or set(value) != {"seen", "unseen"}:
        raise ValueError("checkpoint mapping must contain exactly seen and unseen")
    normalized: dict[str, tuple[str, ...]] = {}
    for split_name in ("seen", "unseen"):
        identifiers = value[split_name]
        if (
            not isinstance(identifiers, list)
            or any(not isinstance(item, str) or not item.strip() for item in identifiers)
            or len(set(identifiers)) != len(identifiers)
        ):
            raise ValueError(f"checkpoint {split_name} IDs must be unique strings")
        normalized[split_name] = tuple(identifiers)
    overlap = set(normalized["seen"]) & set(normalized["unseen"])
    if overlap:
        raise ValueError(f"checkpoint seen/unseen mapping overlaps: {sorted(overlap)}")
    return normalized


def checkpoint_compatibility(
    *,
    experiment_id: str,
    source_stratum: str,
    mapping: Mapping[str, tuple[str, ...]],
    equivalent_socsci210_id: str | None = None,
) -> CheckpointCompatibility:
    """Classify whether a SocSci210-trained checkpoint is usable as held out.

    SocSci210 tasks must occur in exactly one released checkpoint split.  A
    genuinely external task is absent by construction, unless it shares a
    fielding with a SocSci210 study, in which case that equivalent ID controls
    the disposition.
    """

    if source_stratum not in {"socsci210", "external"}:
        raise ValueError("source_stratum must be socsci210 or external")
    if not experiment_id.strip():
        raise ValueError("experiment_id must be non-empty")
    seen = set(mapping.get("seen", ()))
    unseen = set(mapping.get("unseen", ()))
    if seen & unseen:
        raise ValueError("checkpoint mapping cannot overlap")

    checkpoint_id = equivalent_socsci210_id or (
        experiment_id if source_stratum == "socsci210" else None
    )
    if checkpoint_id is None:
        return CheckpointCompatibility(
            experiment_id=experiment_id,
            checkpoint_experiment_id=None,
            status="external_absent_from_socsci210_training_universe",
            primary_eligible=True,
        )
    in_seen = checkpoint_id in seen
    in_unseen = checkpoint_id in unseen
    if in_seen and in_unseen:
        raise ValueError("checkpoint experiment appears in both seen and unseen")
    if in_seen:
        return CheckpointCompatibility(
            experiment_id=experiment_id,
            checkpoint_experiment_id=checkpoint_id,
            status="training_exposed",
            primary_eligible=False,
        )
    if in_unseen:
        return CheckpointCompatibility(
            experiment_id=experiment_id,
            checkpoint_experiment_id=checkpoint_id,
            status="confirmed_checkpoint_unseen",
            primary_eligible=True,
        )
    if source_stratum == "socsci210" or equivalent_socsci210_id is not None:
        raise ValueError(
            f"checkpoint mapping has no disposition for SocSci210 ID {checkpoint_id}"
        )
    raise AssertionError("unreachable checkpoint-exposure state")
