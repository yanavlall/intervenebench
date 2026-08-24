from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from intervenebench.cross_family_merge import (
    CANDIDATE_MODEL_ID,
    DEFAULT_MERGE_PATH,
    build_cross_family_merge,
    build_merge_authorization,
    freeze_cross_family_merge,
    validate_merge_authorization,
)
from intervenebench.protocol import payload_hash, verify_envelope


ROOT = Path(__file__).resolve().parents[1]


def test_merge_authorization_is_exact_and_outcome_blind() -> None:
    authorization = build_merge_authorization(ROOT)
    validate_merge_authorization(authorization, root=ROOT)
    assert authorization["aggregation_authorized"] is True
    assert authorization["model_calls_authorized"] is False
    assert authorization["model_downloads_authorized"] is False
    assert authorization["modal_compute_authorized"] is False
    assert authorization["human_outcome_access_authorized"] is False
    assert authorization["participant_row_access_authorized"] is False
    assert authorization["human_outcome_scoring_authorized"] is False
    assert authorization["automatic_next_stage_authorized"] is False

    widened = deepcopy(authorization)
    widened["human_outcome_access_authorized"] = True
    with pytest.raises(PermissionError, match="expanded"):
        validate_merge_authorization(widened, root=ROOT)


def test_real_merge_is_append_only_retrospective_architecture_robustness() -> None:
    authorization = build_merge_authorization(ROOT)
    merged = build_cross_family_merge(ROOT, authorization=authorization)

    assert merged["study_role"] == "retrospective_cross_family_robustness"
    assert merged["claim_boundary"]["prospective_confirmation"] is False
    assert merged["claim_boundary"]["adds_independent_experiments"] is False
    assert merged["panel_summary"]["experiment_count"] == 6
    assert merged["panel_summary"]["candidate_supported_experiment_count"] == 5
    assert merged["panel_summary"]["model_experiment_recommendation_count"] == 26
    assert merged["panel_summary"]["primary_candidate_winner_agreement_count"] == 3
    assert merged["panel_summary"]["primary_candidate_winner_agreement_rate"] == pytest.approx(0.6)
    assert merged["panel_summary"]["independent_experiment_n"] == 5
    assert merged["panel_summary"]["non_control_contrast_count_descriptive"] == 11
    assert merged["panel_summary"][
        "experiment_macro_mean_absolute_effect_disagreement"
    ] == pytest.approx(0.05795316906129887)
    assert merged["panel_summary"]["largest_arm_effect_disagreement"] == pytest.approx(
        0.1634176340004635
    )
    assert merged["panel_summary"][
        "experiment_macro_exact_effect_sign_agreement"
    ] == pytest.approx(0.72)
    assert merged["candidate_unavailable_experiments"] == [
        {
            "experiment_id": "tcg8p",
            "reason": "one_or_more_strict_parse_failures_no_rerun",
            "strict_parse_failure_count": 120,
        }
    ]

    by_id = {row["experiment_id"]: row for row in merged["experiment_results"]}
    assert CANDIDATE_MODEL_ID not in by_id["tcg8p"]["model_recommendations"]
    assert by_id["tcg8p"]["architecture_family_comparison"]["candidate_available"] is False
    assert by_id["pb2rr"]["architecture_family_comparison"]["winner_agreement"] is True
    assert by_id["z358z"]["architecture_family_comparison"]["winner_agreement"] is True
    assert by_id["Blair1131"]["architecture_family_comparison"]["winner_agreement"] is True
    assert by_id["ShannonS2"]["architecture_family_comparison"]["winner_agreement"] is False
    assert by_id["KlarS44"]["architecture_family_comparison"]["winner_agreement"] is False
    for experiment_id in ("pb2rr", "z358z", "ShannonS2", "Blair1131", "KlarS44"):
        comparison = by_id[experiment_id]["architecture_family_comparison"]
        assert comparison["non_control_contrast_count"] == len(
            by_id[experiment_id]["arm_order"]
        ) - 1
        assert len(comparison["non_control_effect_comparisons"]) == comparison[
            "non_control_contrast_count"
        ]
        assert 0.0 <= comparison["exact_effect_sign_agreement_rate"] <= 1.0
    assert by_id["tcg8p"]["architecture_family_comparison"][
        "non_control_effect_comparisons"
    ] == []

    expected = {
        "tcg8p": (1.0, 0.07856742013183861),
        "pb2rr": (1.0, 0.0),
        "z358z": (1.0, 0.0),
        "ShannonS2": (0.4, 0.12316727010356458),
        "Blair1131": (0.8, 0.26666666666666666),
        "KlarS44": (0.6, 0.3640541899260919),
    }
    for experiment_id, (agreement, dispersion) in expected.items():
        diagnostics = by_id[experiment_id]["retrospective_augmented_diagnostics"]
        assert diagnostics["cross_model_winner_agreement"] == pytest.approx(agreement)
        assert diagnostics["cross_model_arm_rank_dispersion"] == pytest.approx(dispersion)

    original_order = [
        row["experiment_id"] for row in merged["original_pre_reveal_trust_ranking"]["ranking"]
    ]
    augmented_order = [
        row["experiment_id"]
        for row in merged["retrospective_augmented_legacy_trust_ranking"]["ranking"]
    ]
    assert original_order == ["pb2rr", "z358z", "Blair1131", "tcg8p", "ShannonS2", "KlarS44"]
    assert augmented_order == original_order
    assert merged["retrospective_augmented_legacy_trust_ranking"]["learned_threshold"] is None
    assert (
        merged["retrospective_augmented_legacy_trust_ranking"]["accept_abstain_policy"]
        == "not_validated_not_deployed"
    )
    assert merged["additional_model_calls_made_during_merge"] == 0
    assert merged["human_outcomes_accessed_during_merge"] is False
    assert merged["participant_rows_accessed_during_merge"] == 0
    assert merged["human_outcome_scoring_performed_during_merge"] is False
    assert merged["automatic_next_stage"] is False
    assert merged["source_file_sha256"] == {
        "confirmation_aggregation": "458305866b887524beaafe2c2dad7d8d0eb2014a3e95c3ababbccaadd88d503e",
        "candidate_recommendations": "50da5ba663d77ad41d6323a041e1c1b06347bf59fd87abe0fe853e29093bdaf2",
        "candidate_final_manifest": "8c5cf9ca954caa5d41bae4993eedaf6d4a3d89e12cb1916c9ae2afabec7b2e81",
    }
    assert set(merged["implementation_file_sha256"]) == {
        "cross_family_merge_module",
        "authorization_builder",
        "merge_builder",
    }


def test_merge_rejects_contract_or_primary_mutation() -> None:
    authorization = build_merge_authorization(ROOT)
    merged = build_cross_family_merge(ROOT, authorization=authorization)
    assert payload_hash(merged)

    changed = deepcopy(authorization)
    changed["candidate_recommendations_payload_sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="binding drifted"):
        validate_merge_authorization(changed, root=ROOT)


def test_freeze_is_create_only_and_self_verifying(tmp_path: Path) -> None:
    authorization = build_merge_authorization(ROOT)
    destination = tmp_path / "merge.json"
    digest = freeze_cross_family_merge(
        ROOT,
        authorization=authorization,
        destination=destination,
    )
    stored = verify_envelope(destination, require_blinded=True)
    assert digest == payload_hash(stored)
    assert stored["panel_summary"]["primary_candidate_winner_agreement_rate"] == pytest.approx(0.6)
    with pytest.raises(FileExistsError):
        freeze_cross_family_merge(
            ROOT,
            authorization=authorization,
            destination=destination,
        )


def test_repository_merge_artifact_if_present_replays_exactly() -> None:
    path = ROOT / DEFAULT_MERGE_PATH
    if not path.exists():
        pytest.skip("create-only merge artifact has not been materialized yet")
    stored = verify_envelope(path, require_blinded=True)
    rebuilt = build_cross_family_merge(
        ROOT,
        authorization=build_merge_authorization(ROOT),
    )
    assert stored == rebuilt
