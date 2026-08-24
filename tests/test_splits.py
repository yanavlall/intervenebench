from __future__ import annotations

import pytest

from intervenebench.schemas import ExperimentRecord, SplitName
from intervenebench.splits import build_grouped_split


def records() -> list[ExperimentRecord]:
    return [
        ExperimentRecord(f"exp-{group}-{index}", f"group-{group}", index == 0)
        for group, size in enumerate((3, 2, 2, 1, 1, 1, 1, 1, 1, 1))
        for index in range(size)
    ]


def test_grouped_split_is_complete_disjoint_and_deterministic() -> None:
    first = build_grouped_split(records(), seed=17)
    second = build_grouped_split(reversed(records()), seed=17)
    assert first.experiment_to_split == second.experiment_to_split
    assert set(first.experiment_to_split) == {record.experiment_id for record in records()}
    for paradigm in {record.paradigm_group for record in records()}:
        assigned = {
            first.experiment_to_split[record.experiment_id]
            for record in records()
            if record.paradigm_group == paradigm
        }
        assert len(assigned) == 1
    assert set(first.experiment_to_split.values()) == set(SplitName)


def test_duplicate_experiment_ids_fail() -> None:
    duplicate = ExperimentRecord("same", "group-a")
    with pytest.raises(ValueError, match="unique"):
        build_grouped_split([duplicate, duplicate], seed=1)


def test_invalid_fractions_fail() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        build_grouped_split(
            records(),
            seed=1,
            train_fraction=0.5,
            validation_fraction=0.2,
            test_fraction=0.2,
        )


def test_three_paradigms_populate_every_split() -> None:
    small_registry = [
        ExperimentRecord("exp-a", "paradigm-a", True),
        ExperimentRecord("exp-b", "paradigm-b", True),
        ExperimentRecord("exp-c", "paradigm-c", True),
    ]
    manifest = build_grouped_split(small_registry, seed=2102026)
    assert set(manifest.experiment_to_split.values()) == set(SplitName)


def test_fewer_than_three_paradigms_fail_closed() -> None:
    with pytest.raises(ValueError, match="three paradigm groups"):
        build_grouped_split(
            [
                ExperimentRecord("exp-a", "paradigm-a"),
                ExperimentRecord("exp-b", "paradigm-b"),
            ],
            seed=1,
        )
