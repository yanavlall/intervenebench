from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from intervenebench.fallback_failure_mechanism import (
    DEFAULT_MECHANISM_AUDIT_PATH,
    build_mechanism_audit,
    build_mechanism_audit_authorization,
    freeze_mechanism_audit,
    validate_mechanism_audit_authorization,
)
from intervenebench.protocol import payload_hash, verify_envelope


ROOT = Path(__file__).resolve().parents[1]


def test_mechanism_authority_is_aggregate_only_and_cannot_expand() -> None:
    authorization = build_mechanism_audit_authorization(ROOT)
    validate_mechanism_audit_authorization(authorization, root=ROOT)
    assert authorization["aggregate_human_outcome_access_authorized"] is True
    assert authorization["participant_row_access_authorized"] is False
    assert authorization["model_calls_authorized"] is False
    assert authorization["method_tuning_authorized"] is False
    assert authorization["causal_mechanism_claim_authorized"] is False
    assert authorization["automatic_next_stage_authorized"] is False

    expanded = deepcopy(authorization)
    expanded["causal_mechanism_claim_authorized"] = True
    with pytest.raises(PermissionError, match="expanded"):
        validate_mechanism_audit_authorization(expanded, root=ROOT)


def test_failure_pattern_is_harm_correction_magnitude_asymmetry() -> None:
    audit = build_mechanism_audit(
        ROOT, authorization=build_mechanism_audit_authorization(ROOT)
    )
    assert audit["analysis_role"] == "post_reveal_exploratory_failure_pattern_audit"
    assert audit["causal_mechanism_identified"] is False
    asymmetry = audit["eb_harm_correction_asymmetry"]
    assert asymmetry["task_budget_cell_count"] == 15
    assert asymmetry["worsened_cell_count"] == 6
    assert asymmetry["improved_cell_count"] == 3
    assert asymmetry["unchanged_cell_count"] == 6
    assert asymmetry["mean_harm_magnitude"] == pytest.approx(0.022626075140361246)
    assert asymmetry["mean_correction_magnitude"] == pytest.approx(
        0.001664893128460474
    )
    assert asymmetry["harm_to_correction_magnitude_ratio"] == pytest.approx(
        13.59010662821557
    )
    assert asymmetry["all_required_budget_leave_one_task_out_means_worse"] is True


def test_more_humans_and_regularization_attenuate_but_do_not_reverse_harm() -> None:
    audit = build_mechanism_audit(
        ROOT, authorization=build_mechanism_audit_authorization(ROOT)
    )
    budget = audit["budget_attenuation"]
    assert budget["human_only"]["regret_harm_reduction_10_to_100"] == pytest.approx(
        0.5865817381264878
    )
    assert budget["balanced_eb"]["regret_harm_reduction_10_to_100"] == pytest.approx(
        0.5303071454484563
    )
    assert budget["harm_remains_positive_at_100_for_both"] is True

    regularization = audit["regularization_result"]
    for value in regularization[
        "negative_value_rate_reduction_eb_vs_human_only_by_budget"
    ].values():
        assert value > 0.0
    assert regularization["regularization_reverses_mean_harm_at_any_required_budget"] is False


def test_task_heterogeneity_and_transition_limits_are_explicit() -> None:
    audit = build_mechanism_audit(
        ROOT, authorization=build_mechanism_audit_authorization(ROOT)
    )
    task_rows = {row["experiment_id"]: row for row in audit["task_patterns"]}
    assert task_rows["ShannonS2"]["balanced_eb_mean_delta_required_budgets"] > 0.02
    assert task_rows["KlarS44"]["balanced_eb_mean_delta_required_budgets"] > 0.01
    assert task_rows["z358z"]["balanced_eb_mean_delta_required_budgets"] < 0.0
    assert task_rows["Blair1131"]["balanced_eb_mean_delta_required_budgets"] == 0.0
    assert audit["transition_accounting"]["gross_harmful_flip_rate_recoverable"] is False
    assert audit["transition_accounting"]["gross_corrective_flip_rate_recoverable"] is False
    assert audit["transition_accounting"]["reason"] == (
        "replicate_level_transition_pairs_were_not_serialized"
    )


def test_create_only_mechanism_audit_replays(tmp_path: Path) -> None:
    authorization = build_mechanism_audit_authorization(ROOT)
    path = tmp_path / "mechanism.json"
    digest = freeze_mechanism_audit(
        ROOT, authorization=authorization, destination=path
    )
    frozen = verify_envelope(path)
    assert digest == payload_hash(frozen)
    assert frozen == build_mechanism_audit(ROOT, authorization=authorization)
    with pytest.raises(FileExistsError):
        freeze_mechanism_audit(ROOT, authorization=authorization, destination=path)


def test_repository_mechanism_audit_replays_if_present() -> None:
    path = ROOT / DEFAULT_MECHANISM_AUDIT_PATH
    if not path.exists():
        pytest.skip("mechanism audit has not been materialized")
    authorization = build_mechanism_audit_authorization(ROOT)
    assert verify_envelope(path) == build_mechanism_audit(
        ROOT, authorization=authorization
    )
