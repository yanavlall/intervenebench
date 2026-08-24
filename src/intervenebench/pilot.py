"""Freeze the sealed ordinal pilot and its outcome-free baseline artifacts.

This is an engineering pilot over currently runnable bounded-ordinal contracts.
It is deliberately distinct from the later full Benchmark v1 canonical split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .protocol import assert_blinded_payload, payload_hash
from .schemas import ExperimentRecord
from .splits import build_grouped_split


PILOT_ID = "supported-ordinal-pilot-20260812"
PILOT_SEED = 2102026
PILOT_EXPERIMENTS = (
    "5vm8g",
    "xc4yq",
    "de5hx",
    "turagaS11",
    "wallaceS12",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def build_supported_ordinal_pilot(root: Path) -> dict[str, Any]:
    contract_dir = root / "data" / "manifests" / "contracts"
    tasks = {
        experiment_id: _read_json(
            contract_dir / f"{experiment_id}_decision_task_candidate.json"
        )
        for experiment_id in PILOT_EXPERIMENTS
    }
    bundles = {
        experiment_id: _read_json(
            contract_dir / f"{experiment_id}_blinded_bundle.json"
        )
        for experiment_id in PILOT_EXPERIMENTS
    }
    for experiment_id in PILOT_EXPERIMENTS:
        task = tasks[experiment_id]
        bundle = bundles[experiment_id]
        if task["outcome_access"] != "sealed" or task["reveal_authorized"] is not False:
            raise ValueError(f"pilot task is not sealed: {experiment_id}")
        if bundle["outcome_access"] != "sealed" or bundle["reveal_authorized"] is not False:
            raise ValueError(f"pilot bundle is not sealed: {experiment_id}")
        assert_blinded_payload(task)
        assert_blinded_payload(bundle)
        if task["outcome_family"] != "ordinal":
            raise ValueError("supported ordinal pilot cannot include another outcome family")

    split = build_grouped_split(
        [
            ExperimentRecord(
                experiment_id,
                tasks[experiment_id]["paradigm_group"],
                gold_audit=experiment_id in {"5vm8g", "xc4yq", "de5hx"},
            )
            for experiment_id in PILOT_EXPERIMENTS
        ],
        seed=PILOT_SEED,
    )
    return {
        "schema_version": "supported_ordinal_pilot_split.v1",
        "pilot_id": PILOT_ID,
        "freeze_date": "2026-08-12",
        "scope": "five_runnable_bounded_ordinal_contracts_only",
        "status": "engineering_split_frozen_all_human_outcomes_sealed",
        "not_canonical_benchmark_split": True,
        "seed": PILOT_SEED,
        "target_fractions": {key.value: value for key, value in split.fractions.items()},
        "experiment_to_paradigm": split.experiment_to_paradigm,
        "experiment_to_split": {
            experiment_id: split_name.value
            for experiment_id, split_name in split.experiment_to_split.items()
        },
        "counts": {key.value: value for key, value in split.counts.items()},
        "test_outcomes_sealed": True,
        "all_human_outcomes_sealed": True,
        "reveal_authorized": False,
        "excluded_runnable_contracts": {
            "tcg8p": "uncapped continuous outcome cannot share the bounded normalized-regret pilot",
            "Blair1131": "two retained arms fall below the strict 100-row pilot support floor",
        },
        "sequence_contracts_excluded_from_strict_pilot": {
            "ShannonS2": "adapter implemented; held outside the first cost-capped simple-prompt pilot because it requires randomized multi-module sequence simulation",
            "KlarS44": "adapter implemented; held outside the first cost-capped simple-prompt pilot because it requires randomized co-module sequence simulation and shares a fielding with xtvu5",
            "z358z": "adapter implemented; held outside the first cost-capped simple-prompt pilot because it requires randomized Kalla/Saperstein sequence simulation; released Socrates checkpoints are also training-exposed to this experiment",
        },
        "task_sha256": {
            experiment_id: payload_hash(tasks[experiment_id])
            for experiment_id in PILOT_EXPERIMENTS
        },
        "blinded_bundle_sha256": {
            experiment_id: payload_hash(bundles[experiment_id])
            for experiment_id in PILOT_EXPERIMENTS
        },
        "notes": (
            "This split validates multi-experiment orchestration without opening any "
            "human outcome. Its train/validation labels do not authorize outcome access "
            "or classical-model fitting. Full Benchmark v1 remains split-unassigned."
        ),
    }


def build_no_effect_baseline(root: Path, split: dict[str, Any]) -> dict[str, Any]:
    contract_dir = root / "data" / "manifests" / "contracts"
    tasks = {
        experiment_id: _read_json(
            contract_dir / f"{experiment_id}_decision_task_candidate.json"
        )
        for experiment_id in PILOT_EXPERIMENTS
    }
    predictions: dict[str, Any] = {}
    for experiment_id, task in tasks.items():
        arm_ids = [arm["arm_id"] for arm in task["arms"]]
        control = task["control_arm_id"]
        predictions[experiment_id] = {
            "selected_arm_id": control,
            "synthetic_arm_means": {arm_id: 0.5 for arm_id in arm_ids},
            "synthetic_treatment_effects": {
                arm_id: 0.0 for arm_id in arm_ids if arm_id != control
            },
            "selection_rule": "declared_control_arm_on_no_effect_tie",
        }
    payload = {
        "schema_version": "multi_experiment_no_effect_baseline.v1",
        "pilot_id": PILOT_ID,
        "split_sha256": payload_hash(split),
        "outcome_access": "sealed",
        "reveal_authorized": False,
        "human_outcomes_opened": False,
        "predictions": predictions,
    }
    assert_blinded_payload(payload)
    return payload


def write_supported_ordinal_pilot(root: Path) -> tuple[Path, Path]:
    split = build_supported_ordinal_pilot(root)
    split_path = root / "data" / "manifests" / "splits" / "supported_ordinal_pilot.json"
    split_path.write_text(
        json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    baseline = build_no_effect_baseline(root, split)
    baseline_path = (
        root / "data" / "manifests" / "benchmark" / "supported_ordinal_no_effect.json"
    )
    baseline_path.write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return split_path, baseline_path


if __name__ == "__main__":
    write_supported_ordinal_pilot(Path(__file__).resolve().parents[2])
