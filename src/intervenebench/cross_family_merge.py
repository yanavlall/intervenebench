"""Zero-call retrospective merge of Mistral and frozen confirmation recommendations.

The original confirmation recommendations remain the immutable primary policy.
This module appends one independent architecture-family comparator where it is
available and recomputes outcome-free disagreement diagnostics.  The six human
outcomes had already been revealed before the Mistral run, so every output is
explicitly retrospective development evidence, never prospective confirmation.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from math import fsum, isclose, isfinite, sqrt
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from .confirmation_aggregation import build_trust_ranking
from .cross_family_regression import (
    CANDIDATE_MODEL_ID,
    EXPERIMENT_IDS,
    validate_cross_family_protocol,
)
from .protocol import assert_blinded_payload, freeze_envelope, payload_hash, verify_envelope


DEFAULT_PROTOCOL_PATH = Path("data/manifests/research/cross_family_regression_protocol_v1.json")
DEFAULT_CONFIRMATION_AGGREGATION_PATH = Path(
    "artifacts/confirmation/confirmation_20260814_v1/aggregation_v1.json"
)
DEFAULT_CANDIDATE_RUN_ROOT = Path(
    "artifacts/cross_family_target/target_run_20260815_v1_continuation_seedfix_v2"
)
DEFAULT_CANDIDATE_RECOMMENDATIONS_PATH = DEFAULT_CANDIDATE_RUN_ROOT / "recommendations_v1.json"
DEFAULT_CANDIDATE_FINAL_MANIFEST_PATH = DEFAULT_CANDIDATE_RUN_ROOT / "final_manifest.json"
DEFAULT_AUTHORIZATION_PATH = Path(
    "artifacts/cross_family_target/authorizations/outcome_blind_merge_20260815_v1.json"
)
DEFAULT_MERGE_PATH = DEFAULT_CANDIDATE_RUN_ROOT / "retrospective_cross_family_merge_v1.json"

EXPECTED_PROTOCOL_PAYLOAD_SHA256 = "a74daf32a0c42d13909e01c2b7c54766cf058e429aa3b69d3ff499614cd4904e"
EXPECTED_CONFIRMATION_AGGREGATION_PAYLOAD_SHA256 = (
    "03d824f757593e4d51940567086c7cd3a526ca51bb590886c086cbc5860237e4"
)
EXPECTED_CANDIDATE_RECOMMENDATIONS_PAYLOAD_SHA256 = (
    "718c2af1ea453c00d52ca80ab73e5baf145668ccc5a69a80888824db79ee0ea7"
)
EXPECTED_CANDIDATE_FINAL_MANIFEST_PAYLOAD_SHA256 = (
    "4db1ae4459f798acfe04fa3086889aeafa87490a360d3b5d752186689221697a"
)
EXPECTED_SOURCE_FILE_SHA256 = {
    "confirmation_aggregation": "458305866b887524beaafe2c2dad7d8d0eb2014a3e95c3ababbccaadd88d503e",
    "candidate_recommendations": "50da5ba663d77ad41d6323a041e1c1b06347bf59fd87abe0fe853e29093bdaf2",
    "candidate_final_manifest": "8c5cf9ca954caa5d41bae4993eedaf6d4a3d89e12cb1916c9ae2afabec7b2e81",
}
EXPECTED_SUPPORTED_EXPERIMENTS = (
    "pb2rr",
    "z358z",
    "ShannonS2",
    "Blair1131",
    "KlarS44",
)
EXPECTED_UNAVAILABLE = (
    {
        "experiment_id": "tcg8p",
        "reason": "one_or_more_strict_parse_failures_no_rerun",
        "strict_parse_failure_count": 120,
    },
)

_CONTRACT_FIELDS = (
    "candidate_payload_sha256",
    "blinded_bundle_payload_sha256",
    "arm_order",
    "control_arm_id",
    "direction",
    "outcome_family",
    "outcome_unit",
)
_RECOMMENDATION_FIELDS = frozenset(
    {
        "selected_arm_id",
        "arm_source_locations",
        "arm_decision_scores",
        "synthetic_treatment_effects",
        "base_call_count",
        "tie_rule",
    }
)
_AUTHORITY = {
    "aggregation_authorized": True,
    "model_calls_authorized": False,
    "model_downloads_authorized": False,
    "modal_compute_authorized": False,
    "human_outcome_access_authorized": False,
    "participant_row_access_authorized": False,
    "participant_row_serialization_authorized": False,
    "human_outcome_scoring_authorized": False,
    "trust_threshold_selection_authorized": False,
    "automatic_next_stage_authorized": False,
}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_file_sha256(root: Path) -> dict[str, str]:
    paths = {
        "cross_family_merge_module": Path("src/intervenebench/cross_family_merge.py"),
        "authorization_builder": Path(
            "scripts/build_cross_family_merge_authorization.py"
        ),
        "merge_builder": Path("scripts/build_cross_family_merge.py"),
    }
    return {label: _file_sha256(root / path) for label, path in paths.items()}


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _winner(scores: Mapping[str, Any]) -> str:
    parsed = {str(key): _finite(value, field=f"score[{key}]") for key, value in scores.items()}
    if len(parsed) < 2:
        raise ValueError("at least two arms are required")
    best = max(parsed.values())
    return min(arm_id for arm_id, value in parsed.items() if value == best)


def _normalized_ranks(scores: Mapping[str, Any]) -> dict[str, float]:
    parsed = {str(key): _finite(value, field=f"score[{key}]") for key, value in scores.items()}
    if len(parsed) < 2:
        raise ValueError("at least two arms are required")
    denominator = len(parsed) - 1
    return {
        arm_id: (
            sum(other > parsed[arm_id] for other in parsed.values())
            + 0.5 * (sum(other == parsed[arm_id] for other in parsed.values()) - 1)
        )
        / denominator
        for arm_id in sorted(parsed)
    }


def _population_sd(values: Sequence[float]) -> float:
    center = fmean(values)
    return sqrt(fmean((value - center) ** 2 for value in values))


def _pearson(left: Mapping[str, float], right: Mapping[str, float]) -> float | None:
    if set(left) != set(right):
        raise ValueError("rank maps must cover identical arms")
    keys = sorted(left)
    left_values = [left[key] for key in keys]
    right_values = [right[key] for key in keys]
    left_center = fmean(left_values)
    right_center = fmean(right_values)
    left_ss = fsum((value - left_center) ** 2 for value in left_values)
    right_ss = fsum((value - right_center) ** 2 for value in right_values)
    if left_ss == 0.0 or right_ss == 0.0:
        return None
    covariance = fsum(
        (left - left_center) * (right - right_center)
        for left, right in zip(left_values, right_values, strict=True)
    )
    return covariance / sqrt(left_ss * right_ss)


def _mean_absolute_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    if set(left) != set(right):
        raise ValueError("arm maps must cover identical arms")
    return fmean(
        abs(_finite(left[key], field=f"left[{key}]") - _finite(right[key], field=f"right[{key}]"))
        for key in sorted(left)
    )


def _effect_sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _non_control_effect_comparison(
    primary: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    arm_ids: Sequence[str],
    control_arm_id: str,
) -> dict[str, Any]:
    primary_effects = primary["synthetic_treatment_effects"]
    candidate_effects = candidate["synthetic_treatment_effects"]
    comparisons = []
    for arm_id in arm_ids:
        if arm_id == control_arm_id:
            continue
        primary_effect = _finite(primary_effects[arm_id], field="primary effect")
        candidate_effect = _finite(candidate_effects[arm_id], field="candidate effect")
        signed_difference = candidate_effect - primary_effect
        comparisons.append(
            {
                "arm_id": arm_id,
                "primary_effect": primary_effect,
                "candidate_effect": candidate_effect,
                "signed_candidate_minus_primary": signed_difference,
                "absolute_effect_disagreement": abs(signed_difference),
                "exact_effect_sign_agreement": (
                    _effect_sign(primary_effect) == _effect_sign(candidate_effect)
                ),
            }
        )
    if not comparisons:
        raise ValueError("at least one non-control contrast is required")
    return {
        "non_control_effect_comparisons": comparisons,
        "non_control_contrast_count": len(comparisons),
        "mean_absolute_effect_disagreement": fmean(
            row["absolute_effect_disagreement"] for row in comparisons
        ),
        "maximum_absolute_effect_disagreement": max(
            row["absolute_effect_disagreement"] for row in comparisons
        ),
        "exact_effect_sign_agreement_rate": fmean(
            row["exact_effect_sign_agreement"] for row in comparisons
        ),
    }


def _validate_recommendation(
    recommendation: Mapping[str, Any], *, arm_ids: Sequence[str], control_arm_id: str
) -> None:
    if set(recommendation) != _RECOMMENDATION_FIELDS:
        raise ValueError("model recommendation schema drifted")
    if recommendation.get("tie_rule") != "lexicographic_arm_id":
        raise ValueError("model recommendation tie rule drifted")
    if isinstance(recommendation.get("base_call_count"), bool) or not isinstance(
        recommendation.get("base_call_count"), int
    ) or recommendation["base_call_count"] <= 0:
        raise ValueError("model recommendation base-call count is invalid")
    expected = set(arm_ids)
    for field in ("arm_source_locations", "arm_decision_scores", "synthetic_treatment_effects"):
        values = recommendation.get(field)
        if not isinstance(values, Mapping) or set(values) != expected:
            raise ValueError(f"{field} does not cover the declared arms")
        for arm_id, value in values.items():
            _finite(value, field=f"{field}[{arm_id}]")
    selected = recommendation.get("selected_arm_id")
    if selected != _winner(recommendation["arm_decision_scores"]):
        raise ValueError("model recommendation winner does not match arm scores")
    if control_arm_id not in expected:
        raise ValueError("control arm is absent")
    control_score = _finite(
        recommendation["arm_decision_scores"][control_arm_id], field="control score"
    )
    for arm_id in arm_ids:
        expected_effect = (
            _finite(recommendation["arm_decision_scores"][arm_id], field="arm score")
            - control_score
        )
        observed_effect = _finite(
            recommendation["synthetic_treatment_effects"][arm_id], field="effect"
        )
        if not isclose(expected_effect, observed_effect, abs_tol=1e-12):
            raise ValueError("model recommendation treatment effects are inconsistent")


def _load_inputs(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_paths = {
        "confirmation_aggregation": root / DEFAULT_CONFIRMATION_AGGREGATION_PATH,
        "candidate_recommendations": root / DEFAULT_CANDIDATE_RECOMMENDATIONS_PATH,
        "candidate_final_manifest": root / DEFAULT_CANDIDATE_FINAL_MANIFEST_PATH,
    }
    for label, path in source_paths.items():
        if _file_sha256(path) != EXPECTED_SOURCE_FILE_SHA256[label]:
            raise ValueError(f"{label} file hash drifted")
    protocol = verify_envelope(root / DEFAULT_PROTOCOL_PATH, require_blinded=True)
    confirmation = verify_envelope(
        root / DEFAULT_CONFIRMATION_AGGREGATION_PATH, require_blinded=True
    )
    candidate = verify_envelope(
        root / DEFAULT_CANDIDATE_RECOMMENDATIONS_PATH, require_blinded=True
    )
    final = verify_envelope(
        root / DEFAULT_CANDIDATE_FINAL_MANIFEST_PATH, require_blinded=True
    )
    expected_hashes = (
        (protocol, EXPECTED_PROTOCOL_PAYLOAD_SHA256, "protocol"),
        (
            confirmation,
            EXPECTED_CONFIRMATION_AGGREGATION_PAYLOAD_SHA256,
            "confirmation aggregation",
        ),
        (
            candidate,
            EXPECTED_CANDIDATE_RECOMMENDATIONS_PAYLOAD_SHA256,
            "candidate recommendations",
        ),
        (
            final,
            EXPECTED_CANDIDATE_FINAL_MANIFEST_PAYLOAD_SHA256,
            "candidate final manifest",
        ),
    )
    for payload, expected, label in expected_hashes:
        if payload_hash(payload) != expected:
            raise ValueError(f"{label} payload hash drifted")
    validate_cross_family_protocol(protocol)
    return protocol, confirmation, candidate, final


def _validate_source_safety(
    confirmation: Mapping[str, Any], candidate: Mapping[str, Any], final: Mapping[str, Any]
) -> None:
    if confirmation.get("schema_version") != "confirmation_outcome_blind_aggregation.v1":
        raise ValueError("confirmation aggregation schema drifted")
    if confirmation.get("status") != "complete_frozen_outcome_blind_confirmation_aggregation_stop":
        raise ValueError("confirmation aggregation status drifted")
    if tuple(confirmation.get("experiment_order", ())) != EXPERIMENT_IDS:
        raise ValueError("confirmation experiment order drifted")
    if (
        confirmation.get("confirmation_outcomes_accessed") is not False
        or confirmation.get("participant_rows_accessed") != 0
        or confirmation.get("human_outcome_scoring_performed") is not False
    ):
        raise PermissionError("confirmation aggregation is not outcome blind")

    if candidate.get("schema_version") != "intervenebench.cross_family_recommendations.v1":
        raise ValueError("candidate recommendation schema drifted")
    if candidate.get("status") != "frozen_outcome_blind_recommendations_stop":
        raise ValueError("candidate recommendation status drifted")
    if tuple(candidate.get("experiment_order", ())) != EXPERIMENT_IDS:
        raise ValueError("candidate experiment order drifted")
    if (
        candidate.get("human_outcomes_accessed") is not False
        or candidate.get("participant_rows_accessed") != 0
        or candidate.get("human_outcome_scoring_performed") is not False
        or candidate.get("automatic_next_stage") is not False
    ):
        raise PermissionError("candidate recommendations are not outcome blind")
    if candidate.get("recommendation_count") != 5 or candidate.get(
        "unavailable_experiment_count"
    ) != 1:
        raise ValueError("candidate recommendation availability drifted")
    if tuple(candidate.get("unavailable_experiments", ())) != EXPECTED_UNAVAILABLE:
        raise ValueError("candidate unavailable experiment drifted")

    if final.get("schema_version") != "intervenebench.cross_family_no_rerun_continuation.v1":
        raise ValueError("candidate final manifest schema drifted")
    if final.get("recommendations_payload_sha256") != payload_hash(candidate):
        raise ValueError("candidate final manifest does not bind recommendations")
    if (
        final.get("continuation_attempt_count") != 504
        or final.get("strict_output_count") != 504
        or final.get("new_strict_parse_failure_count") != 0
        or final.get("tcg8p_reruns") != 0
        or final.get("duplicated_target_inference_count") != 0
        or final.get("semantic_repairs") != 0
        or final.get("model_downloads") != 0
        or final.get("human_outcomes_accessed") is not False
        or final.get("participant_rows_accessed") != 0
        or final.get("human_outcome_scoring_performed") is not False
        or final.get("automatic_next_stage") is not False
    ):
        raise PermissionError("candidate final manifest safety boundary drifted")


def build_merge_authorization(root: Path) -> dict[str, Any]:
    protocol, confirmation, candidate, final = _load_inputs(root)
    _validate_source_safety(confirmation, candidate, final)
    payload = {
        "schema_version": "intervenebench.cross_family_merge_authorization.v1",
        "status": "authorized_outcome_blind_retrospective_merge_only",
        "study_role": "retrospective_cross_family_robustness",
        "protocol_payload_sha256": payload_hash(protocol),
        "confirmation_aggregation_payload_sha256": payload_hash(confirmation),
        "candidate_recommendations_payload_sha256": payload_hash(candidate),
        "candidate_final_manifest_payload_sha256": payload_hash(final),
        "source_file_sha256": dict(EXPECTED_SOURCE_FILE_SHA256),
        "implementation_file_sha256": _implementation_file_sha256(root),
        "expected_experiment_order": list(EXPERIMENT_IDS),
        "expected_candidate_supported_experiments": list(EXPECTED_SUPPORTED_EXPERIMENTS),
        "expected_candidate_unavailable_experiments": [dict(row) for row in EXPECTED_UNAVAILABLE],
        **_AUTHORITY,
    }
    assert_blinded_payload(payload)
    return payload


def validate_merge_authorization(authorization: Mapping[str, Any], *, root: Path) -> None:
    assert_blinded_payload(authorization)
    expected = build_merge_authorization(root)
    authority_fields = set(_AUTHORITY)
    if any(authorization.get(key) is not value for key, value in _AUTHORITY.items()):
        raise PermissionError("cross-family merge authority expanded")
    if set(authorization) != set(expected):
        raise PermissionError("cross-family merge authorization schema expanded")
    if any(authorization.get(key) != value for key, value in expected.items() if key not in authority_fields):
        raise PermissionError("cross-family merge authorization binding drifted")


def _merge_experiment(
    original: Mapping[str, Any], candidate: Mapping[str, Any] | None
) -> dict[str, Any]:
    experiment_id = str(original["experiment_id"])
    arm_ids = tuple(str(value) for value in original["arm_order"])
    control_arm_id = str(original["control_arm_id"])
    primary_model_id = str(original["primary_model_id"])
    original_recommendations = deepcopy(dict(original["model_recommendations"]))
    if primary_model_id not in original_recommendations:
        raise ValueError("original primary recommendation is absent")
    for recommendation in original_recommendations.values():
        _validate_recommendation(
            recommendation, arm_ids=arm_ids, control_arm_id=control_arm_id
        )

    comparison: dict[str, Any]
    candidate_strict_output_count = 0
    if candidate is None:
        comparison = {
            "candidate_available": False,
            "candidate_unavailable_reason": "one_or_more_strict_parse_failures_no_rerun",
            "primary_model_id": primary_model_id,
            "primary_model_family": "qwen3",
            "candidate_model_id": CANDIDATE_MODEL_ID,
            "candidate_model_family": "mistral",
            "winner_agreement": None,
            "arm_rank_correlation": None,
            "arm_decision_score_mae": None,
            "non_control_effect_comparisons": [],
            "non_control_contrast_count": 0,
            "mean_absolute_effect_disagreement": None,
            "maximum_absolute_effect_disagreement": None,
            "exact_effect_sign_agreement_rate": None,
        }
    else:
        for field in _CONTRACT_FIELDS:
            left = original.get(field)
            right = candidate.get(field)
            if left != right:
                raise ValueError(f"{experiment_id} contract field {field} drifted")
        candidate_models = candidate.get("model_recommendations")
        if not isinstance(candidate_models, Mapping) or set(candidate_models) != {
            CANDIDATE_MODEL_ID
        }:
            raise ValueError("candidate model recommendation set drifted")
        if CANDIDATE_MODEL_ID in original_recommendations:
            raise ValueError("candidate model is already present in original aggregation")
        candidate_recommendation = deepcopy(candidate_models[CANDIDATE_MODEL_ID])
        _validate_recommendation(
            candidate_recommendation,
            arm_ids=arm_ids,
            control_arm_id=control_arm_id,
        )
        original_recommendations[CANDIDATE_MODEL_ID] = candidate_recommendation
        candidate_strict_output_count = int(candidate["strict_output_count"])
        primary = original_recommendations[primary_model_id]
        primary_ranks = _normalized_ranks(primary["arm_decision_scores"])
        candidate_ranks = _normalized_ranks(candidate_recommendation["arm_decision_scores"])
        effect_comparison = _non_control_effect_comparison(
            primary,
            candidate_recommendation,
            arm_ids=arm_ids,
            control_arm_id=control_arm_id,
        )
        comparison = {
            "candidate_available": True,
            "candidate_unavailable_reason": None,
            "primary_model_id": primary_model_id,
            "primary_model_family": "qwen3",
            "candidate_model_id": CANDIDATE_MODEL_ID,
            "candidate_model_family": "mistral",
            "primary_selected_arm_id": primary["selected_arm_id"],
            "candidate_selected_arm_id": candidate_recommendation["selected_arm_id"],
            "winner_agreement": (
                primary["selected_arm_id"] == candidate_recommendation["selected_arm_id"]
            ),
            "arm_rank_correlation": _pearson(primary_ranks, candidate_ranks),
            "arm_decision_score_mae": _mean_absolute_difference(
                primary["arm_decision_scores"],
                candidate_recommendation["arm_decision_scores"],
            ),
            **effect_comparison,
        }

    model_ids = sorted(original_recommendations)
    winners = {
        model_id: str(original_recommendations[model_id]["selected_arm_id"])
        for model_id in model_ids
    }
    primary_selected = str(original_recommendations[primary_model_id]["selected_arm_id"])
    agreement = sum(winner == primary_selected for winner in winners.values()) / len(winners)
    normalized_ranks = {
        model_id: _normalized_ranks(
            original_recommendations[model_id]["arm_decision_scores"]
        )
        for model_id in model_ids
    }
    rank_dispersion = fmean(
        _population_sd([normalized_ranks[model_id][arm_id] for model_id in model_ids])
        for arm_id in arm_ids
    )
    augmented_diagnostics = deepcopy(dict(original["diagnostics"]))
    augmented_diagnostics.update(
        {
            "cross_model_winner_agreement": agreement,
            "cross_model_arm_rank_dispersion": rank_dispersion,
            "model_winners": winners,
            "normalized_arm_ranks_by_model": normalized_ranks,
        }
    )
    return {
        "experiment_id": experiment_id,
        "candidate_payload_sha256": original["candidate_payload_sha256"],
        "blinded_bundle_payload_sha256": original["blinded_bundle_payload_sha256"],
        "arm_order": list(arm_ids),
        "control_arm_id": control_arm_id,
        "direction": original["direction"],
        "outcome_family": original["outcome_family"],
        "outcome_unit": original["outcome_unit"],
        "normalized_for_pooled_regret": original["normalized_for_pooled_regret"],
        "primary_model_id": primary_model_id,
        "primary_recommendation": deepcopy(original["primary_recommendation"]),
        "model_recommendations": original_recommendations,
        "available_model_ids": model_ids,
        "available_model_count": len(model_ids),
        "original_strict_output_count": original["strict_output_count"],
        "candidate_strict_output_count": candidate_strict_output_count,
        "original_pre_reveal_diagnostics": deepcopy(original["diagnostics"]),
        "retrospective_augmented_diagnostics": augmented_diagnostics,
        "architecture_family_comparison": comparison,
    }


def build_cross_family_merge(
    root: Path, *, authorization: Mapping[str, Any]
) -> dict[str, Any]:
    validate_merge_authorization(authorization, root=root)
    protocol, confirmation, candidate, final = _load_inputs(root)
    _validate_source_safety(confirmation, candidate, final)

    original_rows = {
        str(row["experiment_id"]): row for row in confirmation["experiment_results"]
    }
    candidate_rows = {
        str(row["experiment_id"]): row for row in candidate["experiment_results"]
    }
    if tuple(original_rows) != EXPERIMENT_IDS:
        raise ValueError("original experiment result order drifted")
    if tuple(candidate_rows) != EXPECTED_SUPPORTED_EXPERIMENTS:
        raise ValueError("candidate supported experiment order drifted")

    merged_rows = [
        _merge_experiment(original_rows[experiment_id], candidate_rows.get(experiment_id))
        for experiment_id in EXPERIMENT_IDS
    ]
    supported = [
        row
        for row in merged_rows
        if row["architecture_family_comparison"]["candidate_available"] is True
    ]
    agreement_count = sum(
        row["architecture_family_comparison"]["winner_agreement"] is True
        for row in supported
    )
    total_non_control_contrasts = sum(
        row["architecture_family_comparison"]["non_control_contrast_count"]
        for row in supported
    )
    experiment_macro_effect_disagreement = fmean(
        row["architecture_family_comparison"]["mean_absolute_effect_disagreement"]
        for row in supported
    )
    largest_arm_effect_disagreement = max(
        row["architecture_family_comparison"]["maximum_absolute_effect_disagreement"]
        for row in supported
    )
    experiment_macro_sign_agreement = fmean(
        row["architecture_family_comparison"]["exact_effect_sign_agreement_rate"]
        for row in supported
    )
    augmented_ranking = build_trust_ranking(
        [
            {
                "experiment_id": row["experiment_id"],
                **{
                    key: row["retrospective_augmented_diagnostics"][key]
                    for key in (
                        "primary_normalized_top_two_margin",
                        "primary_resampled_winner_stability",
                        "primary_prompt_interface_sensitivity",
                        "cross_model_winner_agreement",
                        "cross_model_arm_rank_dispersion",
                    )
                },
            }
            for row in merged_rows
        ]
    )
    payload = {
        "schema_version": "intervenebench.retrospective_cross_family_merge.v1",
        "status": "complete_frozen_outcome_blind_retrospective_merge_stop",
        "study_role": "retrospective_cross_family_robustness",
        "claim_boundary": deepcopy(protocol["claim_boundary"]),
        "panel_outcome_state_at_candidate_model_freeze": "previously_revealed",
        "recommendation_timing": {
            "original_confirmation_models": "frozen_before_confirmation_human_reveal",
            CANDIDATE_MODEL_ID: "frozen_after_panel_reveal_but_without_outcome_access",
        },
        "foundation_pretraining_exposure": {
            "original_confirmation_models": "unknown",
            CANDIDATE_MODEL_ID: "unknown",
        },
        "socrates_checkpoint_exposure_handling": "inherited_from_original_confirmation_preparation",
        "protocol_payload_sha256": payload_hash(protocol),
        "authorization_payload_sha256": payload_hash(authorization),
        "confirmation_aggregation_payload_sha256": payload_hash(confirmation),
        "candidate_recommendations_payload_sha256": payload_hash(candidate),
        "candidate_final_manifest_payload_sha256": payload_hash(final),
        "source_file_sha256": dict(EXPECTED_SOURCE_FILE_SHA256),
        "implementation_file_sha256": _implementation_file_sha256(root),
        "experiment_order": list(EXPERIMENT_IDS),
        "candidate_model_id": CANDIDATE_MODEL_ID,
        "candidate_checkpoint_commit": final["checkpoint_commit"],
        "candidate_unavailable_experiments": [dict(row) for row in EXPECTED_UNAVAILABLE],
        "original_primary_policy_immutable": True,
        "original_pre_reveal_trust_ranking": deepcopy(confirmation["trust_ranking"]),
        "retrospective_augmented_legacy_trust_ranking": augmented_ranking,
        "experiment_results": merged_rows,
        "panel_summary": {
            "experiment_count": len(merged_rows),
            "candidate_supported_experiment_count": len(supported),
            "candidate_unavailable_experiment_count": 1,
            "model_experiment_recommendation_count": sum(
                row["available_model_count"] for row in merged_rows
            ),
            "available_model_count_by_experiment": {
                row["experiment_id"]: row["available_model_count"] for row in merged_rows
            },
            "primary_candidate_winner_agreement_count": agreement_count,
            "primary_candidate_winner_agreement_rate": agreement_count / len(supported),
            "primary_candidate_winner_disagreement_experiments": [
                row["experiment_id"]
                for row in supported
                if row["architecture_family_comparison"]["winner_agreement"] is False
            ],
            "independent_experiment_n": len(supported),
            "non_control_contrast_count_descriptive": total_non_control_contrasts,
            "experiment_macro_mean_absolute_effect_disagreement": (
                experiment_macro_effect_disagreement
            ),
            "largest_arm_effect_disagreement": largest_arm_effect_disagreement,
            "experiment_macro_exact_effect_sign_agreement": (
                experiment_macro_sign_agreement
            ),
            "aggregation_unit_note": (
                "Effect-disagreement panel means weight each experiment equally; "
                "the contrast count is descriptive and does not increase N."
            ),
        },
        "additional_model_calls_made_during_merge": 0,
        "model_downloads_made_during_merge": 0,
        "modal_compute_used_during_merge": False,
        "human_outcomes_accessed_during_merge": False,
        "participant_rows_accessed_during_merge": 0,
        "human_outcome_scoring_performed_during_merge": False,
        "trust_threshold_selected_during_merge": False,
        "automatic_next_stage": False,
    }
    assert_blinded_payload(payload)
    return payload


def freeze_cross_family_merge(
    root: Path,
    *,
    authorization: Mapping[str, Any],
    destination: Path | None = None,
) -> str:
    path = destination or (root / DEFAULT_MERGE_PATH)
    return freeze_envelope(
        build_cross_family_merge(root, authorization=authorization),
        path,
        require_blinded=True,
    )
