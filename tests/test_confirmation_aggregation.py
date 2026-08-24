from __future__ import annotations

from copy import deepcopy
import random

import pytest

from intervenebench.confirmation_aggregation import (
    aggregate_experiment,
    build_trust_ranking,
    continuous_call_summary,
    probability_call_summary,
    validate_aggregation_authorization,
)


def _row(
    *,
    model: str,
    arm: str,
    nuisance: str,
    score: float,
    stage: str = "base",
    order: str = "source",
    entropy: float | None = 0.5,
) -> dict:
    return {
        "call_id": f"{stage}-{model}-{arm}-{nuisance}-{order}",
        "model_id": model,
        "arm_id": arm,
        "nuisance_id": nuisance,
        "answer_order": order,
        "stage": stage,
        "source_location": score,
        "decision_score": score,
        "normalized_response_entropy": entropy,
    }


def test_probability_and_continuous_call_summaries_are_exact() -> None:
    bounded = probability_call_summary(
        {"1": 0.25, "2": 0.75},
        value_to_utility={1: 0.0, 2: 1.0},
    )
    assert bounded["source_location"] == pytest.approx(1.75)
    assert bounded["decision_score"] == pytest.approx(0.75)
    assert bounded["normalized_response_entropy"] == pytest.approx(
        0.8112781244591328
    )

    continuous = continuous_call_summary(25, lower_is_better=True)
    assert continuous == {
        "source_location": 25.0,
        "decision_score": -25.0,
        "normalized_response_entropy": None,
    }


def test_aggregate_experiment_preserves_pairing_and_prompt_separation() -> None:
    rows = []
    for model, values in {
        "primary": {"a": (0.2, 0.4), "b": (0.8, 0.6)},
        "other": {"a": (0.7, 0.7), "b": (0.3, 0.3)},
    }.items():
        for arm, nuisance_values in values.items():
            for index, value in enumerate(nuisance_values):
                for order in ("source", "reverse"):
                    rows.append(
                        _row(
                            model=model,
                            arm=arm,
                            nuisance=f"n{index}",
                            score=value,
                            order=order,
                        )
                    )
    for arm, values in {"a": (0.3, 0.5), "b": (0.7, 0.5)}.items():
        for index, value in enumerate(values):
            for order in ("source", "reverse"):
                rows.append(
                    _row(
                        model="primary",
                        arm=arm,
                        nuisance=f"n{index}",
                        score=value,
                        stage="primary_prompt_perturbation",
                        order=order,
                    )
                )

    result = aggregate_experiment(
        experiment_id="toy",
        rows=rows,
        arm_ids=["a", "b"],
        control_arm_id="a",
        primary_model_id="primary",
        bootstrap_resamples=100,
        rng=random.Random(7),
        continuous_unbounded=False,
    )
    assert result["model_recommendations"]["primary"]["selected_arm_id"] == "b"
    assert result["model_recommendations"]["primary"]["arm_decision_scores"] == {
        "a": pytest.approx(0.3),
        "b": pytest.approx(0.7),
    }
    assert result["model_recommendations"]["primary"][
        "synthetic_treatment_effects"
    ] == {"a": 0.0, "b": pytest.approx(0.4)}
    diagnostics = result["diagnostics"]
    assert diagnostics["primary_normalized_top_two_margin"] == pytest.approx(0.4)
    assert diagnostics["primary_prompt_interface_sensitivity"] == pytest.approx(
        0.1
    )
    assert diagnostics["primary_prompt_winner_robustness"] == 1.0
    assert diagnostics["cross_model_winner_agreement"] == 0.5
    assert diagnostics["cross_model_arm_rank_dispersion"] == 0.5
    assert 0.0 <= diagnostics["primary_resampled_winner_stability"] <= 1.0


def test_continuous_margin_is_scale_normalized() -> None:
    rows = [
        _row(model="primary", arm="a", nuisance="n0", score=-100, entropy=None),
        _row(model="primary", arm="b", nuisance="n0", score=-80, entropy=None),
    ]
    result = aggregate_experiment(
        experiment_id="continuous",
        rows=rows,
        arm_ids=["a", "b"],
        control_arm_id="a",
        primary_model_id="primary",
        bootstrap_resamples=5,
        rng=random.Random(1),
        continuous_unbounded=True,
    )
    assert result["diagnostics"]["primary_normalized_top_two_margin"] == 0.2
    assert result["diagnostics"]["primary_chosen_arm_normalized_response_entropy"] is None


def test_trust_ranking_uses_direction_aligned_midranks_and_lexicographic_ties() -> None:
    rows = [
        {
            "experiment_id": "b",
            "primary_normalized_top_two_margin": 0.8,
            "primary_resampled_winner_stability": 0.9,
            "primary_prompt_interface_sensitivity": 0.1,
            "cross_model_winner_agreement": 1.0,
            "cross_model_arm_rank_dispersion": 0.1,
        },
        {
            "experiment_id": "a",
            "primary_normalized_top_two_margin": 0.8,
            "primary_resampled_winner_stability": 0.9,
            "primary_prompt_interface_sensitivity": 0.1,
            "cross_model_winner_agreement": 1.0,
            "cross_model_arm_rank_dispersion": 0.1,
        },
        {
            "experiment_id": "c",
            "primary_normalized_top_two_margin": 0.1,
            "primary_resampled_winner_stability": 0.2,
            "primary_prompt_interface_sensitivity": 0.9,
            "cross_model_winner_agreement": 0.5,
            "cross_model_arm_rank_dispersion": 0.8,
        },
    ]
    result = build_trust_ranking(rows)
    assert [row["experiment_id"] for row in result["ranking"]] == ["a", "b", "c"]
    assert result["accept_abstain_policy"] == "not_validated_not_deployed"


def _authorization() -> dict:
    return {
        "schema_version": "confirmation_aggregation_authorization.v1",
        "status": "authorized_outcome_blind_aggregation_only",
        "run_id": "confirmation_20260814_v1",
        "adjudication_manifest_payload_sha256": "a" * 64,
        "call_plan_payload_sha256": "b" * 64,
        "preparation_payload_sha256": "c" * 64,
        "protocol_payload_sha256": "d" * 64,
        "strict_output_map_sha256": "e" * 64,
        "expected_strict_output_count": 1404,
        "expected_unavailable_call_count": 60,
        "aggregation_authorized": True,
        "model_calls_authorized": False,
        "modal_compute_authorized": False,
        "model_download_authorized": False,
        "confirmation_outcome_reveal_authorized": False,
        "participant_row_access_authorized": False,
        "human_outcome_scoring_authorized": False,
        "automatic_next_stage_authorized": False,
    }


def test_aggregation_authority_is_narrow_and_hash_bound() -> None:
    validate_aggregation_authorization(
        _authorization(),
        run_id="confirmation_20260814_v1",
        adjudication_manifest_payload_sha256="a" * 64,
        call_plan_payload_sha256="b" * 64,
        preparation_payload_sha256="c" * 64,
        protocol_payload_sha256="d" * 64,
        strict_output_map_sha256="e" * 64,
    )
    expanded = deepcopy(_authorization())
    expanded["human_outcome_scoring_authorized"] = True
    with pytest.raises(PermissionError, match="expanded"):
        validate_aggregation_authorization(
            expanded,
            run_id="confirmation_20260814_v1",
            adjudication_manifest_payload_sha256="a" * 64,
            call_plan_payload_sha256="b" * 64,
            preparation_payload_sha256="c" * 64,
            protocol_payload_sha256="d" * 64,
            strict_output_map_sha256="e" * 64,
        )
