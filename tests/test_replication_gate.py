from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from intervenebench.replication_gate import (
    ReplicationTaskScore,
    evaluate_replication_gate,
    uniform_random_action_tail,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "data"
    / "manifests"
    / "research"
    / "independent_replication_protocol_v1.json"
)


def _tasks(count: int, *, improvement: float = 0.02) -> list[ReplicationTaskScore]:
    rows = []
    for index in range(count):
        source = "socsci210" if index < (10 if count >= 16 else 8) else "external_tess"
        primary = 0.0
        rows.append(
            ReplicationTaskScore(
                experiment_id=f"exp-{index:02d}",
                fielding_cluster_id=f"field-{index:02d}",
                paradigm_group=f"paradigm-{index:02d}",
                source_stratum=source,
                outcome_family="behavioral" if index < 4 else "attitudinal",
                primary_regret=primary,
                default_regret=0.005,
                classical_regret=0.006,
                arm_regrets={"best": primary, "other": 2.0 * improvement},
            )
        )
    return rows


def test_minimum_replication_gate_passes_only_all_conjunctive_requirements() -> None:
    result = evaluate_replication_gate(
        _tasks(12), bootstrap_replicates=2_000, bootstrap_seed=71
    )
    assert result["panel"]["experiment_count"] == 12
    assert result["primary_uniform_comparison"]["mean_difference"] == pytest.approx(
        -0.02
    )
    assert result["minimum_replication"]["passed"] is True
    assert result["strong_replication"]["passed"] is False
    assert result["completion_classification"] == "bounded_positive_replication"


def test_strong_replication_requires_sixteen_and_operational_superiority() -> None:
    result = evaluate_replication_gate(
        _tasks(16), bootstrap_replicates=2_000, bootstrap_seed=73
    )
    assert result["minimum_replication"]["passed"] is True
    assert result["strong_replication"]["passed"] is True
    assert result["completion_classification"] == "strong_positive_replication"


def test_exact_choice_cannot_rescue_insufficient_regret_value() -> None:
    rows = _tasks(12, improvement=0.005)
    result = evaluate_replication_gate(
        rows, bootstrap_replicates=1_000, bootstrap_seed=79
    )
    assert result["primary_uniform_comparison"]["mean_difference"] == pytest.approx(
        -0.005
    )
    assert result["minimum_replication"]["criteria"][
        "mean_improvement_at_least_0_01"
    ] is False
    assert result["minimum_replication"]["passed"] is False


def test_replication_gate_fails_closed_on_duplicate_units_or_missing_comparators() -> None:
    duplicate = _tasks(12)
    duplicate[1] = ReplicationTaskScore(
        experiment_id=duplicate[1].experiment_id,
        fielding_cluster_id=duplicate[0].fielding_cluster_id,
        paradigm_group=duplicate[1].paradigm_group,
        source_stratum=duplicate[1].source_stratum,
        outcome_family=duplicate[1].outcome_family,
        primary_regret=duplicate[1].primary_regret,
        default_regret=duplicate[1].default_regret,
        classical_regret=duplicate[1].classical_regret,
        arm_regrets=duplicate[1].arm_regrets,
    )
    with pytest.raises(ValueError, match="fielding"):
        evaluate_replication_gate(
            duplicate, bootstrap_replicates=100, bootstrap_seed=1
        )

    missing = _tasks(12)
    object.__setattr__(missing[0], "classical_regret", None)
    with pytest.raises(ValueError, match="classical_regret"):
        evaluate_replication_gate(
            missing, bootstrap_replicates=100, bootstrap_seed=1
        )


def test_replication_gate_enforces_socsci_primary_composition() -> None:
    rows = _tasks(12)
    for index in range(5):
        object.__setattr__(rows[index], "source_stratum", "external_tess")
    with pytest.raises(ValueError, match="SocSci210"):
        evaluate_replication_gate(rows, bootstrap_replicates=100, bootstrap_seed=1)


def test_uniform_random_action_tail_is_exact_when_action_space_is_small() -> None:
    tasks = {f"e{i}": {"a": 0.0, "b": 0.04} for i in range(4)}
    result = uniform_random_action_tail(
        tasks,
        observed_mean_regret=0.0,
        seed=13,
        monte_carlo_replicates=10_000,
    )
    assert result["method"] == "exact_enumeration"
    assert result["combination_count"] == 16
    assert result["tail_probability"] == pytest.approx(1 / 16)
    assert result["conservative_tail_probability"] == pytest.approx(1 / 16)


def test_uniform_random_action_tail_monte_carlo_is_deterministic_and_conservative() -> None:
    tasks = {f"e{i}": {f"a{j}": j / 10 for j in range(5)} for i in range(12)}
    first = uniform_random_action_tail(
        tasks,
        observed_mean_regret=0.12,
        seed=17,
        monte_carlo_replicates=20_000,
        exact_combination_limit=10,
    )
    second = uniform_random_action_tail(
        tasks,
        observed_mean_regret=0.12,
        seed=17,
        monte_carlo_replicates=20_000,
        exact_combination_limit=10,
    )
    assert first == second
    assert first["method"] == "deterministic_monte_carlo"
    assert first["conservative_tail_probability"] >= first["tail_probability"]


def test_harm_and_inconclusive_completion_are_distinguished() -> None:
    harmful = _tasks(12)
    for row in harmful:
        object.__setattr__(row, "primary_regret", 0.04)
        object.__setattr__(row, "arm_regrets", {"a": 0.0, "b": 0.04})
    harm = evaluate_replication_gate(
        harmful, bootstrap_replicates=1_000, bootstrap_seed=83
    )
    assert harm["completion_classification"] == "evidence_of_harm"

    mixed = _tasks(12)
    for index, row in enumerate(mixed):
        if index % 2:
            object.__setattr__(row, "primary_regret", 0.0)
            object.__setattr__(row, "arm_regrets", {"a": 0.0, "b": 0.04})
        else:
            object.__setattr__(row, "primary_regret", 0.03)
            object.__setattr__(row, "arm_regrets", {"a": 0.01, "b": 0.03})
    inconclusive = evaluate_replication_gate(
        mixed, bootstrap_replicates=1_000, bootstrap_seed=89
    )
    assert inconclusive["minimum_replication"]["passed"] is False
    assert inconclusive["completion_classification"] in {
        "non_replication",
        "inconclusive",
    }


def test_frozen_replication_protocol_binds_gate_code_and_zero_authority() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    module_path = ROOT / payload["implementation"]["gate_module"]
    assert sha256(module_path.read_bytes()).hexdigest() == payload["implementation"][
        "gate_module_sha256"
    ]
    assert payload["panel"]["target_experiment_count"] == 16
    assert payload["panel"]["minimum_analyzable_experiment_count"] == 12
    assert payload["primary_estimand"]["independent_unit"] == "experiment"
    assert payload["secondary_outcomes"][
        "exact_intervention_choice_is_secondary"
    ] is True
    assert payload["authority"] == {
        "authorized_spend_usd": 0,
        "model_calls_authorized": False,
        "human_outcome_reveal_authorized": False,
        "participant_row_access_authorized": False,
    }
