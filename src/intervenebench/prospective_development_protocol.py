"""Pre-reveal protocol and authorization for three image experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .balanced_forced_choice import read_json_object
from .multimodal_freeze import sha256_file, verify_prospective_multimodal_freeze
from .multimodal_prospective import EXPERIMENT_IDS
from .protocol import assert_blinded_payload, payload_hash, verify_envelope


PROTOCOL_PATH = Path(
    "data/manifests/benchmark/prospective_multimodal_development_protocol_v1.json"
)
AUTHORIZATION_PATH = Path(
    "data/manifests/benchmark/prospective_multimodal_development_reveal_v1.json"
)
FREEZE_PATH = Path("configs/simulators/prospective_multimodal_v4.json")
PLAN_PATH = Path("data/manifests/simulators/prospective_multimodal_plan_v1.json")
RUN_ROOT = Path(
    "artifacts/prospective_multimodal/prospective_multimodal_20260813_v4"
)
RECOMMENDATIONS_PATH = RUN_ROOT / "prospective_recommendations.json"
REPORT_PATH = Path("docs/reports/prospective_multimodal_development_results.md")
SOCSCI_ROOT = Path(
    "data/raw/socsci210/048481111a4425ed83dc0eacf15f8431f252b21a/data"
)


def _midrank_scores(
    values: Mapping[str, float], *, larger_is_better: bool
) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError("rank confidence requires at least two experiments")
    transformed = {
        key: float(value) if larger_is_better else -float(value)
        for key, value in values.items()
    }
    denominator = len(values) - 1
    return {
        key: (
            sum(other < transformed[key] for other in transformed.values())
            + 0.5
            * (sum(other == transformed[key] for other in transformed.values()) - 1)
        )
        / denominator
        for key in transformed
    }


def build_equal_rank_confidence(
    diagnostics: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create the frozen equal-rank reliability score without human outcomes."""

    by_id = {str(row["experiment_id"]): row for row in diagnostics}
    if set(by_id) != set(EXPERIMENT_IDS):
        raise ValueError("diagnostics do not cover the prospective experiments")
    definitions = {
        "winner_margin": (
            "primary_model_balanced_winner_margin",
            True,
        ),
        "source_reverse_choice_stability": (
            "primary_model_source_reverse_choice_stability",
            True,
        ),
        "low_order_total_variation": (
            "primary_model_mean_arm_source_reverse_total_variation",
            False,
        ),
        "two_vlm_choice_agreement": (
            "two_vlm_complete_action_choice_agreement",
            True,
        ),
        "vision_text_choice_agreement": (
            "vision_vs_accessible_text_choice_agreement",
            True,
        ),
    }
    component_scores: dict[str, dict[str, float]] = {}
    raw_values: dict[str, dict[str, float]] = {}
    for feature_id, (field, larger) in definitions.items():
        values = {
            experiment_id: float(by_id[experiment_id][field])
            for experiment_id in EXPERIMENT_IDS
        }
        raw_values[feature_id] = values
        component_scores[feature_id] = _midrank_scores(
            values, larger_is_better=larger
        )
    confidence = {
        experiment_id: sum(
            component_scores[feature_id][experiment_id]
            for feature_id in definitions
        )
        / len(definitions)
        for experiment_id in EXPERIMENT_IDS
    }
    return {
        "method": "equal_mean_of_five_directional_midrank_scores",
        "feature_order": list(definitions),
        "raw_outcome_free_values": raw_values,
        "component_rank_scores": component_scores,
        "confidence_by_experiment": confidence,
        "tie_break": "experiment_id_ascending_for_fixed_coverage_only",
    }


def _source_file_manifest(root: Path) -> dict[str, Any]:
    parquet_paths = tuple(sorted((root / SOCSCI_ROOT).glob("*.parquet")))
    if len(parquet_paths) != 17:
        raise ValueError("SocSci210 revision must contain exactly 17 Parquet shards")
    es4xw_zip = root / "data/raw/sources/es4xw/Bauman024.zip"
    return {
        "socsci210": {
            "dataset_revision": "048481111a4425ed83dc0eacf15f8431f252b21a",
            "shards": [
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in parquet_paths
            ],
        },
        "es4xw": {
            "path": es4xw_zip.relative_to(root).as_posix(),
            "size_bytes": es4xw_zip.stat().st_size,
            "sha256": sha256_file(es4xw_zip),
            "sav_member": "TESS2_040_Bauman_Client.sav",
            "sav_member_sha256": (
                "4688a240f4a7746249fc3164538d7f8a5b1cccba82b95dd3de3fea352ab3473c"
            ),
        },
    }


def build_pre_reveal_protocol(root: Path) -> dict[str, Any]:
    """Build the exact zero-outcome-access prospective-development protocol."""

    freeze = read_json_object(root / FREEZE_PATH)
    verify_prospective_multimodal_freeze(root, freeze)
    plan = read_json_object(root / PLAN_PATH)
    final = verify_envelope(root / RUN_ROOT / "final_manifest.json", require_blinded=True)
    recommendations = verify_envelope(
        root / RECOMMENDATIONS_PATH, require_blinded=True
    )
    if (
        recommendations["status"]
        != "complete_outcome_blind_multimodal_recommendations_stop"
        or recommendations["outcome_access"] != "not_accessed"
        or recommendations["run_manifest_payload_sha256"] != payload_hash(final)
    ):
        raise ValueError("prospective recommendations are not reveal-ready")
    tasks: list[dict[str, Any]] = []
    for experiment_id in EXPERIMENT_IDS:
        task_path = Path(
            f"data/manifests/contracts/{experiment_id}_decision_task_candidate.json"
        )
        bundle_path = Path(
            f"data/manifests/contracts/{experiment_id}_blinded_bundle.json"
        )
        task = read_json_object(root / task_path)
        bundle = read_json_object(root / bundle_path)
        if (
            task["outcome_access"] != "sealed"
            or task["reveal_authorized"] is not False
            or bundle["outcome_access"] != "sealed"
            or bundle["reveal_authorized"] is not False
        ):
            raise PermissionError("prospective task is not sealed before reveal")
        tasks.append(
            {
                "experiment_id": experiment_id,
                "permanent_role": "prospective_development_noncanonical",
                "canonical_test_eligible": False,
                "task_path": task_path.as_posix(),
                "task_payload_sha256": payload_hash(task),
                "bundle_path": bundle_path.as_posix(),
                "bundle_payload_sha256": payload_hash(bundle),
                "control_arm_id": task["control_arm_id"],
                "practical_regret_tolerance": task[
                    "practical_regret_tolerance"
                ],
                "primary_estimand": task["estimand"],
            }
        )
    implementation_paths = (
        "src/intervenebench/prospective_development_protocol.py",
        "src/intervenebench/prospective_development_score.py",
        "src/intervenebench/human_fallback.py",
        "src/intervenebench/multimodal_recommendations.py",
        "scripts/build_prospective_development_protocol.py",
        "scripts/authorize_prospective_development_reveal.py",
        "scripts/score_prospective_multimodal_development.py",
    )
    protocol = {
        "schema_version": "prospective_multimodal_development_protocol.v1",
        "protocol_id": "intervenebench-prospective-mm-development-20260813-v1",
        "status": "frozen_pre_reveal_zero_outcome_access",
        "evidence_tier": "prospective_development_noncanonical",
        "experiment_ids": list(EXPERIMENT_IDS),
        "tasks": tasks,
        "synthetic_inputs": {
            "simulator_freeze_path": FREEZE_PATH.as_posix(),
            "simulator_freeze_payload_sha256": payload_hash(freeze),
            "call_plan_path": PLAN_PATH.as_posix(),
            "call_plan_payload_sha256": payload_hash(plan),
            "run_manifest_path": (RUN_ROOT / "final_manifest.json").as_posix(),
            "run_manifest_payload_sha256": payload_hash(final),
            "recommendations_path": RECOMMENDATIONS_PATH.as_posix(),
            "recommendations_payload_sha256": payload_hash(recommendations),
        },
        "outcome_sources": _source_file_manifest(root),
        "outcome_column_allowlist": {
            "nj5dx": {
                "source": "socsci210",
                "columns": [
                    "study_id",
                    "sample_id",
                    "participant",
                    "condition_num",
                    "task_num",
                    "response",
                ],
                "task_num": 0,
            },
            "e2pyb": {
                "source": "socsci210",
                "columns": [
                    "study_id",
                    "sample_id",
                    "participant",
                    "condition_num",
                    "task_num",
                    "response",
                ],
                "task_num": 0,
            },
            "es4xw": {
                "source": "official_source_sav",
                "columns": ["caseid", "weight1", "XTESS040", "Q1"],
                "assignment_column": "XTESS040",
                "outcome_column": "Q1",
                "weight_column": "weight1",
            },
        },
        "primary_scoring": {
            "human_arm_estimators": "exactly_as_frozen_in_each_task_contract",
            "tie_rule": "lexicographic_arm_id",
            "treatment_effect_reference": "task_control_arm_id",
            "headline_metrics": [
                "treatment_effect_mae",
                "effect_sign_accuracy",
                "correct_intervention_choice",
                "decision_regret",
                "practically_reliable_at_task_tolerance",
            ],
            "descriptive_metrics": [
                "total_variation",
                "ordinal_wasserstein_normalized",
                "jensen_shannon_divergence_bits",
            ],
            "within_experiment_bootstrap": {
                "unit": "participant_within_randomized_arm",
                "replicates": 5000,
                "seed": 2026081301,
                "confidence_level": 0.95,
                "weights_travel_with_participant": True,
            },
            "experiment_cluster_bootstrap": {
                "unit": "experiment",
                "replicates": 10000,
                "seed": 2026081302,
                "confidence_level": 0.95,
                "claim_boundary": "descriptive_only_at_three_experiments",
            },
        },
        "selective_decision": {
            "primary_model_id": "qwen3_vl_8b_primary",
            "confidence": build_equal_rank_confidence(
                recommendations["outcome_free_experiment_diagnostics"]
            ),
            "fixed_coverage_counts": [1, 2, 3],
            "minimum_class_count_for_binary_metrics": 3,
            "classifier_or_calibration_authorized": False,
            "threshold_selection_authorized": False,
            "evaluation": "risk_coverage_and_continuous_regret_ranking_only",
        },
        "human_fallback": {
            "budgets_total_outcome_observations": [0, 10, 25, 50, 100, 250],
            "policies": [
                "synthetic_only",
                "human_only_balanced",
                "synthetic_plus_balanced_fixed10",
                "synthetic_plus_hedged_fixed10",
            ],
            "partitions": 20,
            "fold_count": 10,
            "seed": 2026081303,
            "pilot_evaluation_people_disjoint": True,
            "sampling_without_replacement": True,
            "nested_arm_prefixes_within_policy": True,
            "same_folds_and_arm_orders_across_policies": True,
            "balanced_allocation_tie_rule": "source_arm_order",
            "hedged_allocation": {
                "uniform_exploration_fraction": 0.25,
                "minimum_one_per_arm_when_budget_positive": True,
                "remaining_weight": (
                    "Laplace-smoothed winner votes from three models crossed with "
                    "source/reverse order; no human target outcomes"
                ),
                "integerization": "largest_remainder_then_source_arm_order",
            },
            "fusion": {
                "synthetic_prior_pseudocount_per_arm": 10,
                "formula": "(10*synthetic_mean+n_arm*pilot_hajek_mean)/(10+n_arm)",
                "status": "transparent_fixed_baseline_not_empirical_bayes",
            },
            "outputs": [
                "mean_regret",
                "exact_choice_rate",
                "practical_reliability_rate",
                "paired_mean_regret_change_vs_synthetic",
                "negative_value_rate_vs_synthetic",
                "marginal_regret_reduction_per_human",
            ],
        },
        "report": {
            "skeleton_path": REPORT_PATH.as_posix(),
            "skeleton_sha256": sha256_file(root / REPORT_PATH),
            "preserve_null_and_negative_results": True,
        },
        "implementation_hashes": [
            {"path": path, "sha256": sha256_file(root / path)}
            for path in implementation_paths
        ],
        "other_experiments_must_remain_sealed": [
            "tcg8p",
            "pb2rr",
            "z358z",
            "ShannonS2",
            "Blair1131",
            "KlarS44",
        ],
        "authority": {
            "human_outcome_reveal_authorized": False,
            "modal_compute_authorized": False,
            "paid_inference_authorized": False,
            "canonical_test_claim_authorized": False,
            "trust_model_claim_authorized": False,
            "automatic_next_stage_authorized": False,
        },
        "claim_boundary": (
            "Three prospective-development experiments can test frozen methods "
            "descriptively but cannot establish a canonical trust model, calibrated "
            "threshold, or general human-simulation reliability."
        ),
    }
    assert_blinded_payload(protocol)
    return protocol


def verify_pre_reveal_protocol(root: Path) -> dict[str, Any]:
    protocol = read_json_object(root / PROTOCOL_PATH)
    expected = build_pre_reveal_protocol(root)
    if protocol != expected:
        raise ValueError("prospective-development protocol does not replay exactly")
    if any(protocol["authority"].values()):
        raise PermissionError("pre-reveal protocol cannot grant authority")
    return protocol


def build_reveal_authorization(root: Path) -> dict[str, Any]:
    protocol = verify_pre_reveal_protocol(root)
    return {
        "schema_version": "prospective_multimodal_development_reveal.v1",
        "authorization_id": "intervenebench-prospective-mm-reveal-20260813-v1",
        "status": "prospective_development_reveal_authorized",
        "experiment_ids": list(EXPERIMENT_IDS),
        "permanent_role": "prospective_development_noncanonical",
        "canonical_test_eligible": False,
        "protocol_path": PROTOCOL_PATH.as_posix(),
        "protocol_payload_sha256": payload_hash(protocol),
        "recommendations_payload_sha256": protocol["synthetic_inputs"][
            "recommendations_payload_sha256"
        ],
        "outcome_column_allowlist": protocol["outcome_column_allowlist"],
        "other_experiments_must_remain_sealed": protocol[
            "other_experiments_must_remain_sealed"
        ],
        "human_outcome_reveal_authorized": True,
        "participant_row_serialization_authorized": False,
        "modal_compute_authorized": False,
        "paid_inference_authorized": False,
        "canonical_test_claim_authorized": False,
        "trust_model_claim_authorized": False,
        "automatic_next_stage_authorized": False,
        "claim_boundary": protocol["claim_boundary"],
    }


def verify_reveal_authorization(root: Path) -> dict[str, Any]:
    authorization = read_json_object(root / AUTHORIZATION_PATH)
    expected = build_reveal_authorization(root)
    if authorization != expected:
        raise PermissionError("prospective-development reveal authorization mismatch")
    return authorization
