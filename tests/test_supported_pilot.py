from __future__ import annotations

import json
from pathlib import Path

from intervenebench.pilot import (
    PILOT_EXPERIMENTS,
    build_no_effect_baseline,
    build_supported_ordinal_pilot,
)
from intervenebench.protocol import assert_blinded_payload


ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / "data/manifests/splits/supported_ordinal_pilot.json"
BASELINE_PATH = ROOT / "data/manifests/benchmark/supported_ordinal_no_effect.json"


def test_supported_pilot_is_deterministic_sealed_and_not_canonical() -> None:
    split = json.loads(SPLIT_PATH.read_text())
    assert split == build_supported_ordinal_pilot(ROOT)
    assert set(split["experiment_to_split"]) == set(PILOT_EXPERIMENTS)
    assert split["counts"] == {"train": 3, "validation": 1, "test": 1}
    assert split["not_canonical_benchmark_split"] is True
    assert split["all_human_outcomes_sealed"] is True
    assert split["test_outcomes_sealed"] is True
    assert split["reveal_authorized"] is False
    assert set(split["sequence_contracts_excluded_from_strict_pilot"]) == {
        "ShannonS2",
        "KlarS44",
        "z358z",
    }
    assert_blinded_payload(split)


def test_no_effect_baseline_is_frozen_without_human_outcomes() -> None:
    split = json.loads(SPLIT_PATH.read_text())
    baseline = json.loads(BASELINE_PATH.read_text())
    assert baseline == build_no_effect_baseline(ROOT, split)
    assert baseline["outcome_access"] == "sealed"
    assert baseline["reveal_authorized"] is False
    assert baseline["human_outcomes_opened"] is False
    assert set(baseline["predictions"]) == set(PILOT_EXPERIMENTS)
    for experiment_id, prediction in baseline["predictions"].items():
        task = json.loads(
            (
                ROOT
                / "data/manifests/contracts"
                / f"{experiment_id}_decision_task_candidate.json"
            ).read_text()
        )
        assert prediction["selected_arm_id"] == task["control_arm_id"]
        assert set(prediction["synthetic_arm_means"]) == {
            arm["arm_id"] for arm in task["arms"]
        }
        assert set(prediction["synthetic_arm_means"].values()) == {0.5}
    assert_blinded_payload(baseline)
