"""Deterministic paradigm-group splitting for experiment-level evaluation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

from .schemas import ExperimentRecord, SplitName


@dataclass(frozen=True, slots=True)
class SplitManifest:
    seed: int
    fractions: dict[SplitName, float]
    experiment_to_paradigm: dict[str, str]
    experiment_to_split: dict[str, SplitName]

    @property
    def counts(self) -> dict[SplitName, int]:
        return {
            split: sum(value is split for value in self.experiment_to_split.values())
            for split in SplitName
        }

    def validate(self) -> None:
        if set(self.experiment_to_paradigm) != set(self.experiment_to_split):
            raise ValueError("split coverage must match experiment coverage")
        paradigm_to_splits: dict[str, set[SplitName]] = defaultdict(set)
        for experiment_id, paradigm in self.experiment_to_paradigm.items():
            paradigm_to_splits[paradigm].add(self.experiment_to_split[experiment_id])
        leaking = sorted(group for group, splits in paradigm_to_splits.items() if len(splits) != 1)
        if leaking:
            raise ValueError(f"paradigm groups cross splits: {leaking}")


def _stable_tiebreak(seed: int, value: str) -> str:
    return sha256(f"{seed}:{value}".encode()).hexdigest()


def build_grouped_split(
    experiments: Iterable[ExperimentRecord],
    *,
    seed: int,
    train_fraction: float = 0.65,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.20,
) -> SplitManifest:
    """Assign whole paradigm groups while approximately respecting target fractions.

    Assignment uses only experiment IDs, paradigm labels, the gold-audit indicator,
    and the declared seed. It cannot depend on outcomes or simulator performance.
    """

    records = tuple(experiments)
    if not records:
        raise ValueError("at least one experiment is required")
    experiment_ids = [record.experiment_id for record in records]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("experiment IDs must be unique")

    fractions = {
        SplitName.TRAIN: train_fraction,
        SplitName.VALIDATION: validation_fraction,
        SplitName.TEST: test_fraction,
    }
    if any(value <= 0 for value in fractions.values()):
        raise ValueError("split fractions must be positive")
    if abs(sum(fractions.values()) - 1.0) > 1e-12:
        raise ValueError("split fractions must sum to one")

    grouped: dict[str, list[ExperimentRecord]] = defaultdict(list)
    for record in records:
        grouped[record.paradigm_group].append(record)
    if len(grouped) < len(SplitName):
        raise ValueError(
            "at least three paradigm groups are required to populate train, "
            "validation, and test without splitting a paradigm"
        )

    total = len(records)
    total_gold = sum(record.gold_audit for record in records)
    targets = {split: total * fraction for split, fraction in fractions.items()}
    gold_targets = {split: total_gold * fraction for split, fraction in fractions.items()}
    counts = {split: 0 for split in SplitName}
    gold_counts = {split: 0 for split in SplitName}
    paradigm_to_split: dict[str, SplitName] = {}

    group_order = sorted(
        grouped,
        key=lambda group: (
            -len(grouped[group]),
            -sum(record.gold_audit for record in grouped[group]),
            _stable_tiebreak(seed, group),
        ),
    )

    # Seed each partition before greedy balancing. Without this coverage step, a
    # small registry can minimize the local ratio score by placing every group
    # in train, leaving no honest validation or test task.
    coverage_order = sorted(
        SplitName,
        key=lambda split: (-fractions[split], list(SplitName).index(split)),
    )
    for paradigm, split in zip(group_order[: len(SplitName)], coverage_order):
        group_size = len(grouped[paradigm])
        group_gold = sum(record.gold_audit for record in grouped[paradigm])
        paradigm_to_split[paradigm] = split
        counts[split] += group_size
        gold_counts[split] += group_gold

    for paradigm in group_order[len(SplitName) :]:
        group_size = len(grouped[paradigm])
        group_gold = sum(record.gold_audit for record in grouped[paradigm])

        def assignment_score(split: SplitName) -> tuple[float, float, int]:
            size_ratio = (counts[split] + group_size) / targets[split]
            gold_ratio = (
                (gold_counts[split] + group_gold) / gold_targets[split]
                if group_gold and gold_targets[split]
                else 0.0
            )
            split_order = list(SplitName).index(split)
            return (size_ratio + 0.20 * gold_ratio, size_ratio, split_order)

        chosen = min(SplitName, key=assignment_score)
        paradigm_to_split[paradigm] = chosen
        counts[chosen] += group_size
        gold_counts[chosen] += group_gold

    experiment_to_paradigm = {
        record.experiment_id: record.paradigm_group for record in records
    }
    experiment_to_split = {
        record.experiment_id: paradigm_to_split[record.paradigm_group]
        for record in records
    }
    manifest = SplitManifest(
        seed=seed,
        fractions=fractions,
        experiment_to_paradigm=experiment_to_paradigm,
        experiment_to_split=experiment_to_split,
    )
    manifest.validate()
    return manifest
