from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from intervenebench.fallback_replication_audit import (
    DEFAULT_FALLBACK_REPLICATION_AUDIT_PATH,
    build_fallback_replication_audit,
    build_fallback_replication_audit_authorization,
    freeze_fallback_replication_audit,
    validate_fallback_replication_audit_authorization,
)
from intervenebench.protocol import payload_hash, verify_envelope


ROOT = Path(__file__).resolve().parents[1]


def test_fallback_replication_authority_is_aggregate_only_and_exact() -> None:
    authorization = build_fallback_replication_audit_authorization(ROOT)
    validate_fallback_replication_audit_authorization(authorization, root=ROOT)

    assert authorization["aggregate_human_outcome_access_authorized"] is True
    assert authorization["participant_row_access_authorized"] is False
    assert authorization["participant_row_serialization_authorized"] is False
    assert authorization["model_calls_authorized"] is False
    assert authorization["new_policy_authorized"] is False
    assert authorization["method_tuning_authorized"] is False
    assert authorization["automatic_next_stage_authorized"] is False

    expanded = deepcopy(authorization)
    expanded["method_tuning_authorized"] = True
    with pytest.raises(PermissionError, match="expanded"):
        validate_fallback_replication_audit_authorization(expanded, root=ROOT)


def test_fallback_negative_result_directionally_replicates_prospectively() -> None:
    audit = build_fallback_replication_audit(
        ROOT,
        authorization=build_fallback_replication_audit_authorization(ROOT),
    )

    assert audit["analysis_role"] == (
        "post_reveal_aggregate_only_replication_audit_of_a_pre_reveal_fallback_policy"
    )
    assert audit["development_panel"]["experiment_count"] == 9
    assert audit["confirmation_panel"]["experiment_count"] == 5
    assert audit["panels_are_experiment_disjoint"] is True
    assert audit["confirmation_panel"]["was_prospective_for_fallback_policy"] is True

    result = audit["balanced_eb_replication_result"]
    assert result["required_budgets"] == [25, 50, 100]
    assert result["development_stop_rule_triggered"] is True
    assert result["confirmation_directionally_replicated"] is True
    assert result["confirmation_statistically_resolved_at_every_required_budget"] is False
    assert result["decision"] == "stop_tuning_and_preserve_replicated_negative_result"
    for budget in (25, 50, 100):
        row = result["confirmation_by_budget"][str(budget)]
        assert row["candidate_minus_synthetic_mean_regret"] > 0.0
        assert row["candidate_improvement_over_synthetic"] < 0.0


def test_small_human_only_pilots_show_confirmatory_harm_but_claim_is_narrow() -> None:
    audit = build_fallback_replication_audit(
        ROOT,
        authorization=build_fallback_replication_audit_authorization(ROOT),
    )

    human = audit["human_only_confirmation_result"]
    assert human["budgets_with_95pct_paired_ci_entirely_worse_than_synthetic"] == [
        10,
        25,
        50,
        100,
    ]
    assert human["all_nonzero_budget_point_estimates_worse_than_synthetic"] is True
    assert audit["hedged_allocation_result"][
        "hedged_beats_matching_balanced_policy_at_any_confirmation_budget"
    ] is False
    assert audit["tcg8p_raw_unit_secondary"]["pooled_with_normalized_tasks"] is False
    assert audit["participant_rows_accessed"] == 0
    assert audit["participant_rows_serialized"] == 0
    assert audit["model_calls_made"] == 0
    assert audit["automatic_next_stage"] is False
    assert "small pilots" in audit["claim_boundary"]["supported_claim"]
    assert "humans are useless" in audit["claim_boundary"]["forbidden_claims"]


def test_secondary_pooled_summary_keeps_panels_visible_and_experiments_as_units() -> None:
    audit = build_fallback_replication_audit(
        ROOT,
        authorization=build_fallback_replication_audit_authorization(ROOT),
    )
    pooled = audit["secondary_pooled_descriptive_summary"]
    assert pooled["role"] == "secondary_descriptive_only_panels_reported_separately_first"
    assert pooled["experiment_is_resampling_unit"] is True
    assert pooled["development_experiment_count"] == 9
    assert pooled["confirmation_experiment_count"] == 5
    for budget in (25, 50, 100):
        assert pooled["balanced_eb_by_budget"][str(budget)][
            "candidate_minus_synthetic_mean_regret"
        ] > 0.0


def test_create_only_audit_replay(tmp_path: Path) -> None:
    authorization = build_fallback_replication_audit_authorization(ROOT)
    destination = tmp_path / "audit.json"
    digest = freeze_fallback_replication_audit(
        ROOT,
        authorization=authorization,
        destination=destination,
    )
    frozen = verify_envelope(destination)
    assert digest == payload_hash(frozen)
    assert frozen == build_fallback_replication_audit(
        ROOT, authorization=authorization
    )
    with pytest.raises(FileExistsError):
        freeze_fallback_replication_audit(
            ROOT,
            authorization=authorization,
            destination=destination,
        )


def test_repository_audit_replays_if_materialized() -> None:
    path = ROOT / DEFAULT_FALLBACK_REPLICATION_AUDIT_PATH
    if not path.exists():
        pytest.skip("fallback replication audit has not been materialized")
    authorization = build_fallback_replication_audit_authorization(ROOT)
    assert verify_envelope(path) == build_fallback_replication_audit(
        ROOT, authorization=authorization
    )
