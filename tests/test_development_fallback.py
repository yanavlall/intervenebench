from __future__ import annotations

from pathlib import Path

from intervenebench.development_fallback import (
    DEFAULT_FALLBACK_PATH,
    FALLBACK_PROTOCOL_PATH,
    build_development_fallback_protocol,
    verify_development_fallback,
    verify_development_fallback_protocol,
)


ROOT = Path(__file__).resolve().parents[1]


def test_fallback_protocol_is_scope_bound_and_outcome_safe_for_confirmation() -> None:
    protocol = build_development_fallback_protocol(ROOT)
    assert protocol["development_experiment_count"] == 9
    assert protocol["confirmation_outcome_access_authorized"] is False
    assert protocol["target_prior_fit"] == "leave_one_experiment_out"
    assert protocol["budgets"] == [0, 10, 25, 50, 100, 250]
    assert protocol["partitions"] == 20
    assert protocol["fold_count"] == 10


def test_frozen_fallback_protocol_replays_exactly() -> None:
    protocol = verify_development_fallback_protocol(ROOT)
    assert protocol == build_development_fallback_protocol(ROOT)
    assert (ROOT / FALLBACK_PROTOCOL_PATH).exists()


def test_frozen_development_fallback_is_aggregate_only_and_target_excluded() -> None:
    result = verify_development_fallback(ROOT, ROOT / DEFAULT_FALLBACK_PATH)
    assert result["experiment_count"] == 9
    assert result["participant_rows_serialized"] == 0
    assert result["confirmation_outcomes_accessed"] == []
    for experiment_id, task in result["tasks"].items():
        assert experiment_id not in task["effect_prior"][
            "training_experiment_ids"
        ]

