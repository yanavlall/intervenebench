"""Aggregate-only retrospective scoring for the frozen cross-family comparator.

The candidate recommendations were generated after the confirmation panel had
already been revealed, but the inference process itself accessed no outcomes.
Accordingly, these scores are development-only evidence. They may motivate a
frozen diagnostic for future untouched experiments, but cannot revise the
original prospective recommendation, trust ranking, or confirmation claims.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from math import fsum
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

from .confirmation_scoring import HumanArmSummary, score_synthetic_recommendation
from .cross_family_merge import CANDIDATE_MODEL_ID
from .experiment_statistics import paired_experiment_cluster_bootstrap
from .protocol import freeze_envelope, payload_hash, verify_envelope


DEFAULT_CONFIRMATION_SCORE_PATH = Path(
    "artifacts/confirmation/confirmation_20260814_v1/score_v1.json"
)
DEFAULT_CROSS_FAMILY_MERGE_PATH = Path(
    "artifacts/cross_family_target/"
    "target_run_20260815_v1_continuation_seedfix_v2/"
    "retrospective_cross_family_merge_v1.json"
)
DEFAULT_RETROSPECTIVE_SCORE_AUTHORIZATION_PATH = Path(
    "artifacts/cross_family_target/authorizations/"
    "retrospective_score_20260815_v2.json"
)
DEFAULT_RETROSPECTIVE_SCORE_PATH = Path(
    "artifacts/cross_family_target/"
    "target_run_20260815_v1_continuation_seedfix_v2/"
    "retrospective_cross_family_score_v2.json"
)
DEFAULT_DIAGNOSTIC_SPEC_PATH = Path(
    "data/manifests/research/cross_family_disagreement_diagnostic_v2.json"
)

EXPECTED_CONFIRMATION_SCORE_PAYLOAD_SHA256 = (
    "fa2acc4661f8397658178a1b4d53e7806b2a35acf032520e625ffdcb79aaf1a7"
)
EXPECTED_CONFIRMATION_SCORE_FILE_SHA256 = (
    "8562d148ce04bc44af1481858b94f5a43f62edf94120199b2156c8920a51c2ec"
)
EXPECTED_CROSS_FAMILY_MERGE_PAYLOAD_SHA256 = (
    "2fde47af33a2928f8d63358b2404855332fe1fb22331a52c7066283f318de865"
)
EXPECTED_CROSS_FAMILY_MERGE_FILE_SHA256 = (
    "364b3d42faa1a99c1894993db13d3946369c96c7ede40eea5e92889ef1bca298"
)
ALL_EXPERIMENT_IDS = (
    "tcg8p",
    "pb2rr",
    "z358z",
    "ShannonS2",
    "Blair1131",
    "KlarS44",
)
SUPPORTED_EXPERIMENT_IDS = (
    "pb2rr",
    "z358z",
    "ShannonS2",
    "Blair1131",
    "KlarS44",
)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2026081501

_AUTHORITY = {
    "retrospective_scoring_authorized": True,
    "aggregate_human_outcome_access_authorized": True,
    "future_diagnostic_spec_freeze_authorized": True,
    "participant_row_access_authorized": False,
    "participant_row_serialization_authorized": False,
    "model_calls_authorized": False,
    "model_downloads_authorized": False,
    "modal_compute_authorized": False,
    "recommendation_changes_authorized": False,
    "primary_policy_changes_authorized": False,
    "trust_threshold_tuning_authorized": False,
    "model_switching_rule_authorized": False,
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
        "retrospective_score_module": Path(
            "src/intervenebench/cross_family_retrospective_score.py"
        ),
        "authorization_builder": Path(
            "scripts/build_cross_family_retrospective_score_authorization.py"
        ),
        "score_builder": Path("scripts/build_cross_family_retrospective_score.py"),
        "diagnostic_spec_builder": Path(
            "scripts/build_future_cross_family_diagnostic_spec.py"
        ),
    }
    return {label: _file_sha256(root / path) for label, path in paths.items()}


def _load_sources(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    score_path = root / DEFAULT_CONFIRMATION_SCORE_PATH
    merge_path = root / DEFAULT_CROSS_FAMILY_MERGE_PATH
    if _file_sha256(score_path) != EXPECTED_CONFIRMATION_SCORE_FILE_SHA256:
        raise ValueError("confirmation score file hash drifted")
    if _file_sha256(merge_path) != EXPECTED_CROSS_FAMILY_MERGE_FILE_SHA256:
        raise ValueError("cross-family merge file hash drifted")
    score = verify_envelope(score_path)
    merge = verify_envelope(merge_path)
    if payload_hash(score) != EXPECTED_CONFIRMATION_SCORE_PAYLOAD_SHA256:
        raise ValueError("confirmation score payload hash drifted")
    if payload_hash(merge) != EXPECTED_CROSS_FAMILY_MERGE_PAYLOAD_SHA256:
        raise ValueError("cross-family merge payload hash drifted")
    _validate_sources(score, merge)
    return score, merge


def _validate_sources(score: Mapping[str, Any], merge: Mapping[str, Any]) -> None:
    if (
        score.get("schema_version") != "confirmation_score.v1"
        or score.get("status") != "complete_prospective_confirmation_scoring_stop"
        or tuple(score.get("experiment_ids", ())) != ALL_EXPERIMENT_IDS
        or tuple(score.get("confirmation_outcomes_accessed", ()))
        != ALL_EXPERIMENT_IDS
    ):
        raise ValueError("confirmation score identity drifted")
    if (
        score.get("participant_rows_serialized") != 0
        or score.get("model_calls_made") != 0
        or score.get("modal_compute_used") is not False
        or score.get("recommendations_changed_after_reveal") is not False
        or score.get("diagnostics_changed_after_reveal") is not False
        or score.get("threshold_tuned_after_reveal") is not False
        or score.get("automatic_followup_authorized") is not False
    ):
        raise PermissionError("confirmation score safety boundary drifted")
    if set(score.get("experiment_scores", {})) != set(ALL_EXPERIMENT_IDS):
        raise ValueError("confirmation aggregate experiment support drifted")

    if (
        merge.get("schema_version")
        != "intervenebench.retrospective_cross_family_merge.v1"
        or merge.get("status")
        != "complete_frozen_outcome_blind_retrospective_merge_stop"
        or tuple(merge.get("experiment_order", ())) != ALL_EXPERIMENT_IDS
        or merge.get("candidate_model_id") != CANDIDATE_MODEL_ID
    ):
        raise ValueError("cross-family merge identity drifted")
    if (
        merge.get("additional_model_calls_made_during_merge") != 0
        or merge.get("human_outcomes_accessed_during_merge") is not False
        or merge.get("participant_rows_accessed_during_merge") != 0
        or merge.get("human_outcome_scoring_performed_during_merge") is not False
        or merge.get("trust_threshold_selected_during_merge") is not False
        or merge.get("automatic_next_stage") is not False
        or merge.get("original_primary_policy_immutable") is not True
    ):
        raise PermissionError("cross-family merge safety boundary drifted")
    unavailable = merge.get("candidate_unavailable_experiments")
    if not isinstance(unavailable, list) or [row.get("experiment_id") for row in unavailable] != [
        "tcg8p"
    ]:
        raise ValueError("candidate unavailability drifted")


def build_retrospective_score_authorization(root: Path) -> dict[str, Any]:
    score, merge = _load_sources(root)
    return {
        "schema_version": "intervenebench.cross_family_retrospective_score_authorization.v2",
        "status": "authorized_aggregate_only_retrospective_development_scoring",
        "confirmation_score_payload_sha256": payload_hash(score),
        "cross_family_merge_payload_sha256": payload_hash(merge),
        "confirmation_score_file_sha256": EXPECTED_CONFIRMATION_SCORE_FILE_SHA256,
        "cross_family_merge_file_sha256": EXPECTED_CROSS_FAMILY_MERGE_FILE_SHA256,
        "authorized_experiment_ids": list(SUPPORTED_EXPERIMENT_IDS),
        "candidate_unavailable_experiment_ids": ["tcg8p"],
        "implementation_file_sha256": _implementation_file_sha256(root),
        **_AUTHORITY,
    }


def validate_retrospective_score_authorization(
    authorization: Mapping[str, Any], *, root: Path
) -> None:
    expected = build_retrospective_score_authorization(root)
    if set(authorization) != set(expected):
        raise PermissionError("retrospective score authority expanded")
    if any(authorization.get(key) is not value for key, value in _AUTHORITY.items()):
        raise PermissionError("retrospective score authority expanded")
    if dict(authorization) != expected:
        raise PermissionError("retrospective score authorization binding drifted")


def _recompute_scores(
    confirmation_score: Mapping[str, Any], merge: Mapping[str, Any]
) -> list[dict[str, Any]]:
    merge_rows = {
        row["experiment_id"]: row for row in merge["experiment_results"]
    }
    rows: list[dict[str, Any]] = []
    for experiment_id in SUPPORTED_EXPERIMENT_IDS:
        merged = merge_rows[experiment_id]
        if merged.get("normalized_for_pooled_regret") is not True:
            raise ValueError("retrospective pooled task is not bounded-normalized")
        comparison = merged["architecture_family_comparison"]
        if comparison.get("candidate_available") is not True:
            raise ValueError("supported candidate recommendation is unavailable")
        source_score = confirmation_score["experiment_scores"][experiment_id]
        frozen_primary = source_score["primary_score"]
        human = HumanArmSummary(
            arm_means=frozen_primary["human_arm_means"],
            complete_case_count_by_arm=frozen_primary["complete_case_count_by_arm"],
            outcome_unit=frozen_primary["outcome_unit"],
        )
        primary_recommendation = merged["model_recommendations"][
            merged["primary_model_id"]
        ]
        candidate_recommendation = merged["model_recommendations"][CANDIDATE_MODEL_ID]
        primary = score_synthetic_recommendation(
            arm_ids=merged["arm_order"],
            control_arm_id=merged["control_arm_id"],
            human=human,
            synthetic_arm_scores=primary_recommendation["arm_decision_scores"],
            selected_arm_id=primary_recommendation["selected_arm_id"],
            practical_tolerance=frozen_primary["practical_tolerance"],
        )
        candidate = score_synthetic_recommendation(
            arm_ids=merged["arm_order"],
            control_arm_id=merged["control_arm_id"],
            human=human,
            synthetic_arm_scores=candidate_recommendation["arm_decision_scores"],
            selected_arm_id=candidate_recommendation["selected_arm_id"],
            practical_tolerance=frozen_primary["practical_tolerance"],
        )
        for field in (
            "human_selected_arm_id",
            "synthetic_selected_arm_id",
            "exact_choice",
            "decision_regret",
            "mean_absolute_treatment_effect_error",
            "treatment_effect_sign_accuracy",
        ):
            if primary[field] != frozen_primary[field]:
                raise ValueError(f"frozen primary score mismatch for {experiment_id}: {field}")
        rows.append(
            {
                "experiment_id": experiment_id,
                "paradigm_group": source_score["paradigm_group"],
                "winner_disagreement": comparison["winner_agreement"] is False,
                "primary_model_id": merged["primary_model_id"],
                "candidate_model_id": CANDIDATE_MODEL_ID,
                "human_selected_arm_id": candidate["human_selected_arm_id"],
                "primary_selected_arm_id": primary["synthetic_selected_arm_id"],
                "candidate_selected_arm_id": candidate["synthetic_selected_arm_id"],
                "primary_exact_choice": primary["exact_choice"],
                "candidate_exact_choice": candidate["exact_choice"],
                "primary_decision_regret": primary["decision_regret"],
                "candidate_decision_regret": candidate["decision_regret"],
                "candidate_minus_primary_regret": (
                    candidate["decision_regret"] - primary["decision_regret"]
                ),
                "primary_treatment_effect_mae": primary[
                    "mean_absolute_treatment_effect_error"
                ],
                "candidate_treatment_effect_mae": candidate[
                    "mean_absolute_treatment_effect_error"
                ],
                "primary_effect_sign_accuracy": primary[
                    "treatment_effect_sign_accuracy"
                ],
                "candidate_effect_sign_accuracy": candidate[
                    "treatment_effect_sign_accuracy"
                ],
                "primary_practically_reliable": primary["practically_reliable"],
                "candidate_practically_reliable": candidate["practically_reliable"],
                "practical_tolerance": primary["practical_tolerance"],
                "candidate_score": candidate,
            }
        )
    return rows


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return fsum(float(row[field]) for row in rows) / len(rows)


def _diagnostic_group(rows: list[dict[str, Any]], *, disagreement: bool) -> dict[str, Any]:
    selected = [row for row in rows if row["winner_disagreement"] is disagreement]
    if not selected:
        raise ValueError("winner-disagreement evaluation requires both groups")
    return {
        "experiment_count": len(selected),
        "experiment_ids": [row["experiment_id"] for row in selected],
        "primary_error_rate": _mean(selected, "primary_exact_choice") * -1.0 + 1.0,
        "primary_mean_regret": _mean(selected, "primary_decision_regret"),
        "candidate_error_rate": _mean(selected, "candidate_exact_choice") * -1.0 + 1.0,
        "candidate_mean_regret": _mean(selected, "candidate_decision_regret"),
    }


def build_retrospective_cross_family_score(
    root: Path, *, authorization: Mapping[str, Any]
) -> dict[str, Any]:
    validate_retrospective_score_authorization(authorization, root=root)
    confirmation_score, merge = _load_sources(root)
    rows = _recompute_scores(confirmation_score, merge)
    primary_regret = {row["experiment_id"]: row["primary_decision_regret"] for row in rows}
    candidate_regret = {
        row["experiment_id"]: row["candidate_decision_regret"] for row in rows
    }
    regret_bootstrap = paired_experiment_cluster_bootstrap(
        candidate_regret,
        primary_regret,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        confidence_level=0.95,
        lower_is_better=True,
    )
    regret_bootstrap_payload = asdict(regret_bootstrap)
    regret_bootstrap_payload["difference_confidence_interval"] = list(
        regret_bootstrap.difference_confidence_interval
    )
    disagree = _diagnostic_group(rows, disagreement=True)
    agree = _diagnostic_group(rows, disagreement=False)
    positive_signal = (
        disagree["primary_error_rate"] > agree["primary_error_rate"]
        and disagree["primary_mean_regret"] > agree["primary_mean_regret"]
    )
    transitions = {
        "fixed_by_candidate": [
            row["experiment_id"]
            for row in rows
            if not row["primary_exact_choice"] and row["candidate_exact_choice"]
        ],
        "harmed_by_candidate": [
            row["experiment_id"]
            for row in rows
            if row["primary_exact_choice"] and not row["candidate_exact_choice"]
        ],
        "unchanged_incorrect": [
            row["experiment_id"]
            for row in rows
            if not row["primary_exact_choice"] and not row["candidate_exact_choice"]
        ],
        "unchanged_correct": [
            row["experiment_id"]
            for row in rows
            if row["primary_exact_choice"] and row["candidate_exact_choice"]
        ],
    }
    payload = {
        "schema_version": "intervenebench.cross_family_retrospective_score.v2",
        "status": "complete_aggregate_only_retrospective_development_score_stop",
        "study_role": "retrospective_development_only_cross_family_score",
        "evidence_tier": "already_revealed_panel_not_prospective_validation",
        "claim_boundary": {
            "permitted": (
                "development-only architecture sensitivity and scoring on five "
                "previously revealed experiments"
            ),
            "forbidden": (
                "new prospective confirmation, calibrated trust, a deployable "
                "model-switching rule, or additional independent experiment N"
            ),
        },
        "confirmation_score_payload_sha256": payload_hash(confirmation_score),
        "cross_family_merge_payload_sha256": payload_hash(merge),
        "authorization_payload_sha256": payload_hash(authorization),
        "source_file_sha256": {
            "confirmation_score": EXPECTED_CONFIRMATION_SCORE_FILE_SHA256,
            "cross_family_merge": EXPECTED_CROSS_FAMILY_MERGE_FILE_SHA256,
        },
        "implementation_file_sha256": _implementation_file_sha256(root),
        "superseded_serialization_only_artifacts": {
            "authorization_v1_payload_sha256": (
                "62be4ff7caeaa5bf80bab29888628f3d104f7193085b23336a5471ba39825a20"
            ),
            "score_v1_payload_sha256": (
                "0bcbb4bd8c2a2a30198718177addb8357245f08afda697e6509308706bbbb27b"
            ),
            "diagnostic_v1_payload_sha256": (
                "55c4f76fd0f98a90275d34027fe1a4a0776932a4720ec76acb5cc4bcc0257231"
            ),
            "reason": "tuple_to_json_list_roundtrip_mismatch_no_scientific_value_change",
        },
        "candidate_model_id": CANDIDATE_MODEL_ID,
        "candidate_checkpoint_commit": merge["candidate_checkpoint_commit"],
        "candidate_recommendation_timing": (
            "post_panel_reveal_but_generated_and_frozen_without_outcome_access"
        ),
        "experiment_order": list(SUPPORTED_EXPERIMENT_IDS),
        "candidate_unavailable_experiments": ["tcg8p"],
        "independent_experiment_n": len(rows),
        "primary_policy_changed": False,
        "primary_exact_choice_rate": _mean(rows, "primary_exact_choice"),
        "candidate_exact_choice_rate": _mean(rows, "candidate_exact_choice"),
        "primary_mean_decision_regret": _mean(rows, "primary_decision_regret"),
        "candidate_mean_decision_regret": _mean(rows, "candidate_decision_regret"),
        "paired_mean_regret_delta_candidate_minus_primary": _mean(
            rows, "candidate_minus_primary_regret"
        ),
        "paired_regret_experiment_cluster_bootstrap": regret_bootstrap_payload,
        "primary_mean_treatment_effect_mae": _mean(
            rows, "primary_treatment_effect_mae"
        ),
        "candidate_mean_treatment_effect_mae": _mean(
            rows, "candidate_treatment_effect_mae"
        ),
        "primary_mean_effect_sign_accuracy": _mean(
            rows, "primary_effect_sign_accuracy"
        ),
        "candidate_mean_effect_sign_accuracy": _mean(
            rows, "candidate_effect_sign_accuracy"
        ),
        "decision_transitions": transitions,
        "diagnostic_evaluation": {
            "winner_disagreement": {
                "disagreement_group": disagree,
                "agreement_group": agree,
                "primary_error_rate_when_disagree": disagree["primary_error_rate"],
                "primary_error_rate_when_agree": agree["primary_error_rate"],
                "primary_mean_regret_when_disagree": disagree["primary_mean_regret"],
                "primary_mean_regret_when_agree": agree["primary_mean_regret"],
                "positive_signal_under_frozen_direction": positive_signal,
            },
            "conclusion": (
                "positive_retrospective_signal_still_unvalidated"
                if positive_signal
                else "no_positive_retrospective_signal_do_not_deploy_or_tune_threshold"
            ),
        },
        "experiment_scores": rows,
        "human_outcome_access_scope": "existing_aggregate_confirmation_score_only",
        "participant_rows_accessed": 0,
        "participant_rows_serialized": 0,
        "model_calls_made": 0,
        "model_downloads_made": 0,
        "modal_compute_used": False,
        "recommendations_changed": False,
        "trust_threshold_tuned": False,
        "model_switching_rule_deployed": False,
        "automatic_next_stage": False,
    }
    return payload


def build_future_cross_family_diagnostic_spec(
    retrospective_score: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        retrospective_score.get("schema_version")
        != "intervenebench.cross_family_retrospective_score.v2"
        or retrospective_score.get("status")
        != "complete_aggregate_only_retrospective_development_score_stop"
        or retrospective_score.get("participant_rows_accessed") != 0
        or retrospective_score.get("model_calls_made") != 0
        or retrospective_score.get("trust_threshold_tuned") is not False
    ):
        raise ValueError("retrospective development score is not eligible for spec freeze")
    return {
        "schema_version": "intervenebench.cross_family_disagreement_diagnostic.v2",
        "status": "frozen_for_future_untouched_replication_only",
        "diagnostic_role": "secondary_unvalidated_failure_diagnostic",
        "development_evidence_payload_sha256": payload_hash(retrospective_score),
        "development_result": (
            "no_positive_retrospective_signal_do_not_deploy_or_tune_threshold"
        ),
        "hypothesized_direction": "more_disagreement_means_lower_trust",
        "validated_on_development_panel": False,
        "diagnostics": {
            "winner_disagreement": "primary_selected_arm_id != candidate_selected_arm_id",
            "effect_sign_disagreement_rate": (
                "one minus exact sign agreement over non-control contrasts; sign(0)=0"
            ),
            "mean_absolute_effect_disagreement": (
                "mean absolute primary-vs-candidate normalized effect difference over "
                "non-control contrasts"
            ),
            "arm_rank_disagreement": "one minus arm-rank correlation when defined",
        },
        "candidate_model_id": retrospective_score["candidate_model_id"],
        "candidate_checkpoint_commit": retrospective_score[
            "candidate_checkpoint_commit"
        ],
        "task_eligibility": (
            "diagnostic is available only when the frozen primary and exact candidate "
            "model both complete the same outcome-blind arm grid"
        ),
        "missing_candidate_rule": "mark_diagnostic_unavailable_never_impute_disagreement",
        "trust_threshold": None,
        "accept_abstain_policy": "not_validated_not_deployed",
        "model_switching_rule": "forbidden",
        "experiment_is_independent_unit": True,
        "models_calls_arms_and_seeds_do_not_increase_n": True,
        "target_human_outcomes_must_be_hidden_until_diagnostic_freeze": True,
        "required_target_order": (
            "freeze both recommendations and all disagreement diagnostics before reveal"
        ),
        "future_evaluation": {
            "primary_target": "frozen_primary_normalized_decision_regret",
            "secondary_target": "frozen_primary_exact_choice_error",
            "analysis": (
                "report fixed-direction risk-coverage descriptively; do not fit or tune "
                "a threshold unless a separate prospective protocol authorizes it"
            ),
            "minimum_label_gate": (
                "classification metrics not_estimable unless both classes have at least "
                "three independent experiments"
            ),
        },
        "claim_boundary": (
            "A future result may test whether cross-family disagreement predicts risk; "
            "this manifest alone establishes no validated trust policy."
        ),
        "automatic_next_stage": False,
    }


def freeze_retrospective_cross_family_score(
    root: Path,
    *,
    authorization: Mapping[str, Any],
    destination: Path | None = None,
) -> str:
    return freeze_envelope(
        build_retrospective_cross_family_score(root, authorization=authorization),
        destination or (root / DEFAULT_RETROSPECTIVE_SCORE_PATH),
        require_blinded=False,
    )


def freeze_future_cross_family_diagnostic_spec(
    retrospective_score: Mapping[str, Any], *, destination: Path
) -> str:
    return freeze_envelope(
        build_future_cross_family_diagnostic_spec(retrospective_score),
        destination,
        require_blinded=False,
    )
