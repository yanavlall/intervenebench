"""Aggregate-only evidence registry for the nine revealed development tasks."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from .experiment_statistics import experiment_cluster_bootstrap
from .phase1 import replay_score
from .portfolio_development import verify_development_score
from .prospective_development_score import (
    DEFAULT_SCORE_PATH as PROSPECTIVE_SCORE_PATH,
    verify_prospective_development_score,
)
from .protocol import freeze_envelope, payload_hash, verify_envelope
from .selective_decision import (
    SelectiveDecisionRecord,
    selective_decision_summary,
)


DEFAULT_DEVELOPMENT_EVIDENCE_PATH = Path(
    "artifacts/development/development_evidence_v1.json"
)
PHASE1_SCORE_PATH = Path("artifacts/phase1/jf46x_score.json")
PHASE1_RECOMMENDATION_PATH = Path("artifacts/phase1/jf46x_recommendation.json")
PHASE1_RAW_PATH = Path("artifacts/phase1/jf46x_ollama_outputs.json")
PORTFOLIO_SCORE_PATH = Path("artifacts/portfolio_pilot/development_score_v2.json")
DISCOVERY_SCORE_PATH = Path(
    "artifacts/balanced_full_action/balanced_full_action_20260813_v1/"
    "retrospective_discovery_score.json"
)
DISCOVERY_DIAGNOSTICS_PATH = Path(
    "artifacts/balanced_full_action/balanced_full_action_20260813_v1/"
    "outcome_free_diagnostics_v2.json"
)
PROSPECTIVE_RECOMMENDATIONS_PATH = Path(
    "artifacts/prospective_multimodal/prospective_multimodal_20260813_v4/"
    "prospective_recommendations.json"
)

DISCOVERY_MODEL_ID = "qwen3_8b_generic"
PROSPECTIVE_MODEL_ID = "qwen3_vl_8b_primary"
DEVELOPMENT_IDS = (
    "jf46x",
    "5vm8g",
    "xc4yq",
    "de5hx",
    "turagaS11",
    "wallaceS12",
    "nj5dx",
    "es4xw",
    "e2pyb",
)
RICH_DIAGNOSTIC_IDS = (
    "5vm8g",
    "xc4yq",
    "de5hx",
    "turagaS11",
    "wallaceS12",
    "nj5dx",
    "es4xw",
    "e2pyb",
)
SEALED_CONFIRMATION_IDS = (
    "tcg8p",
    "pb2rr",
    "z358z",
    "ShannonS2",
    "Blair1131",
    "KlarS44",
)
PARADIGMS = {
    "jf46x": "vaccine_risk_trust",
    "5vm8g": "racial_discrimination_beliefs",
    "xc4yq": "terrorism_threat_emotion",
    "de5hx": "electoral_risk_framing",
    "turagaS11": "environmental_risk_information_and_personal_responsibility",
    "wallaceS12": "international_law_commitment_compliance_and_reputation",
    "nj5dx": "inequality_and_power_framing",
    "es4xw": "workplace_diversity_composition",
    "e2pyb": "racial_disparity_dimension_messaging",
}
FEATURE_DIRECTIONS = {
    "winner_margin": "larger",
    "order_choice_stability": "larger",
    "order_total_variation": "smaller",
    "cross_model_choice_agreement": "larger",
    "cross_model_utility_dispersion": "smaller",
    "response_entropy": "smaller",
}


def _finite(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _midranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    if len(values) == 1:
        return [1.0]
    return [rank / (len(values) - 1) for rank in ranks]


def equal_rank_confidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_directions: Mapping[str, str],
) -> dict[str, float]:
    """Average direction-aware midranks; larger output means more trustworthy."""

    if len(rows) < 2:
        raise ValueError("equal-rank confidence requires at least two experiments")
    identifiers = [str(row["experiment_id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("experiment IDs must be unique")
    scores = {experiment_id: [] for experiment_id in identifiers}
    for feature, direction in feature_directions.items():
        if direction not in {"larger", "smaller"}:
            raise ValueError(f"unsupported direction for {feature}")
        values = [_finite(row[feature], name=feature) for row in rows]
        directed = values if direction == "larger" else [-value for value in values]
        for experiment_id, rank in zip(identifiers, _midranks(directed), strict=True):
            scores[experiment_id].append(rank)
    return {
        experiment_id: fmean(feature_scores)
        for experiment_id, feature_scores in scores.items()
    }


def _effect_mae(human: Mapping[str, Any], synthetic: Mapping[str, Any]) -> float:
    if set(human) != set(synthetic) or not human:
        raise ValueError("treatment-effect keys must match and be nonempty")
    return fmean(abs(float(synthetic[key]) - float(human[key])) for key in human)


def _phase1_row(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    score = replay_score(
        score_path=root / PHASE1_SCORE_PATH,
        recommendation_path=root / PHASE1_RECOMMENDATION_PATH,
        raw_output_path=root / PHASE1_RAW_PATH,
    )
    recommendation = verify_envelope(
        root / PHASE1_RECOMMENDATION_PATH, require_blinded=True
    )
    row = {
        "experiment_id": "jf46x",
        "paradigm_group": PARADIGMS["jf46x"],
        "development_role": "phase1_validation_reveal",
        "evidence_tier": "development_only_nonprospective",
        "primary_model_id": "llama3_2_3b_phase1",
        "selected_arm_id": score["selected_arm_id"],
        "human_best_arm_id": score["human_best_arm_id"],
        "correct_intervention_choice": bool(score["correct_choice"]),
        "decision_regret": float(score["normalized_decision_regret"]),
        "practically_reliable_at_0_05": bool(score["practically_reliable"]),
        "human_treatment_effects": score["human_treatment_effects"],
        "synthetic_treatment_effects": score["synthetic_treatment_effects"],
        "treatment_effect_mae": _effect_mae(
            score["human_treatment_effects"], score["synthetic_treatment_effects"]
        ),
        "rich_diagnostics_available": False,
    }
    diagnostics = {
        "experiment_id": "jf46x",
        "winner_margin": float(recommendation["diagnostics"]["winner_margin"]),
        "availability": "winner_margin_only_not_in_rich_trust_screen",
        "target_human_outcomes_used": False,
    }
    return row, diagnostics


def _discovery_rows(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    portfolio = verify_development_score(root, root / PORTFOLIO_SCORE_PATH)
    score = verify_envelope(root / DISCOVERY_SCORE_PATH, require_blinded=False)
    diagnostics = verify_envelope(
        root / DISCOVERY_DIAGNOSTICS_PATH, require_blinded=True
    )
    if (
        score.get("schema_version") != "modal_full_action_discovery_score.v1"
        or score.get("revealed_development_score_payload_sha256")
        != payload_hash(portfolio)
        or score.get("selected_primary_model_id_for_future_freeze")
        != DISCOVERY_MODEL_ID
    ):
        raise ValueError("retrospective discovery score contract is invalid")
    task_scores = {
        row["experiment_id"]: row
        for row in score["task_scores"]
        if row["model_id"] == DISCOVERY_MODEL_ID
    }
    decisions = {
        row["experiment_id"]: row
        for row in diagnostics["decision_diagnostics"]
        if row["model_id"] == DISCOVERY_MODEL_ID
    }
    experiments = {
        row["experiment_id"]: row for row in diagnostics["experiment_diagnostics"]
    }
    expected = set(DEVELOPMENT_IDS[1:6])
    if set(task_scores) != expected or set(decisions) != expected or set(experiments) != expected:
        raise ValueError("discovery task or diagnostic support drifted")
    rows: list[dict[str, Any]] = []
    mapped: list[dict[str, Any]] = []
    for experiment_id in DEVELOPMENT_IDS[1:6]:
        task = task_scores[experiment_id]
        decision = decisions[experiment_id]
        experiment = experiments[experiment_id]
        rows.append(
            {
                "experiment_id": experiment_id,
                "paradigm_group": PARADIGMS[experiment_id],
                "development_role": "retrospective_discovery",
                "evidence_tier": "development_only_model_selected_on_same_five",
                "primary_model_id": DISCOVERY_MODEL_ID,
                "selected_arm_id": task["selected_arm_id"],
                "human_best_arm_id": task["human_best_arm_id"],
                "correct_intervention_choice": bool(
                    task["correct_intervention_choice"]
                ),
                "decision_regret": float(task["decision_regret"]),
                "practically_reliable_at_0_05": bool(
                    task["practically_reliable_at_0_05"]
                ),
                "human_treatment_effects": task["human_treatment_effects"],
                "synthetic_treatment_effects": task["synthetic_treatment_effects"],
                "treatment_effect_mae": float(task["treatment_effect_mae"]),
                "rich_diagnostics_available": True,
            }
        )
        mapped.append(
            {
                "experiment_id": experiment_id,
                "winner_margin": float(decision["balanced_winner_margin"]),
                "order_choice_stability": float(
                    decision["source_reverse_full_action_choice_stable"]
                ),
                "order_total_variation": float(
                    decision["mean_arm_order_total_variation"]
                ),
                "cross_model_choice_agreement": float(
                    experiment["pairwise_model_choice_agreement"]
                ),
                "cross_model_utility_dispersion": float(
                    experiment["mean_cross_model_arm_utility_population_sd"]
                ),
                "response_entropy": float(decision["chosen_arm_response_entropy"]),
                "mapping_note": "four-model text discovery suite",
                "target_human_outcomes_used": False,
            }
        )
    hashes = {
        "portfolio_score_payload_sha256": payload_hash(portfolio),
        "discovery_score_payload_sha256": payload_hash(score),
        "discovery_diagnostics_payload_sha256": payload_hash(diagnostics),
    }
    return rows, mapped, hashes


def _prospective_rows(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    score = verify_prospective_development_score(
        root, root / PROSPECTIVE_SCORE_PATH
    )
    recommendations = verify_envelope(
        root / PROSPECTIVE_RECOMMENDATIONS_PATH, require_blinded=True
    )
    experiment_diagnostics = {
        row["experiment_id"]: row
        for row in recommendations["outcome_free_experiment_diagnostics"]
    }
    expected = set(DEVELOPMENT_IDS[6:])
    if set(score["tasks"]) != expected or set(experiment_diagnostics) != expected:
        raise ValueError("prospective-development support drifted")
    rows: list[dict[str, Any]] = []
    mapped: list[dict[str, Any]] = []
    for experiment_id in DEVELOPMENT_IDS[6:]:
        task = score["tasks"][experiment_id]
        model = task["models"][PROSPECTIVE_MODEL_ID]
        diagnostic = experiment_diagnostics[experiment_id]
        rows.append(
            {
                "experiment_id": experiment_id,
                "paradigm_group": PARADIGMS[experiment_id],
                "development_role": "prospective_development",
                "evidence_tier": "prospective_development_noncanonical",
                "primary_model_id": PROSPECTIVE_MODEL_ID,
                "selected_arm_id": model["selected_arm_id"],
                "human_best_arm_id": model["human_best_arm_id"],
                "correct_intervention_choice": bool(
                    model["correct_intervention_choice"]
                ),
                "decision_regret": float(model["decision_regret"]),
                "practically_reliable_at_0_05": bool(
                    model["practically_reliable_at_task_tolerance"]
                ),
                "human_treatment_effects": task["human_treatment_effects"],
                "synthetic_treatment_effects": model[
                    "synthetic_treatment_effects"
                ],
                "treatment_effect_mae": float(model["treatment_effect_mae"]),
                "rich_diagnostics_available": True,
            }
        )
        mapped.append(
            {
                "experiment_id": experiment_id,
                "winner_margin": float(
                    diagnostic["primary_model_balanced_winner_margin"]
                ),
                "order_choice_stability": float(
                    diagnostic["primary_model_source_reverse_choice_stability"]
                ),
                "order_total_variation": float(
                    diagnostic[
                        "primary_model_mean_arm_source_reverse_total_variation"
                    ]
                ),
                "cross_model_choice_agreement": float(
                    diagnostic["two_vlm_complete_action_choice_agreement"]
                ),
                "cross_model_utility_dispersion": float(
                    diagnostic["mean_two_vlm_expected_utility_population_sd"]
                ),
                "response_entropy": float(
                    diagnostic[
                        "primary_model_chosen_arm_normalized_response_entropy"
                    ]
                ),
                "mapping_note": "two-model vision-language suite",
                "target_human_outcomes_used": False,
            }
        )
    return rows, mapped, {
        "prospective_score_payload_sha256": payload_hash(score),
        "prospective_recommendations_payload_sha256": payload_hash(
            recommendations
        ),
    }


def _screen(
    task_rows: Sequence[Mapping[str, Any]],
    diagnostic_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    tasks = {row["experiment_id"]: row for row in task_rows}
    if tuple(row["experiment_id"] for row in diagnostic_rows) != RICH_DIAGNOSTIC_IDS:
        raise ValueError("rich diagnostic order or support drifted")
    analyses: dict[str, Any] = {}
    for feature, direction in FEATURE_DIRECTIONS.items():
        confidence = equal_rank_confidence(
            diagnostic_rows,
            feature_directions={feature: direction},
        )
        analyses[feature] = asdict(
            selective_decision_summary(
                [
                    SelectiveDecisionRecord(
                        experiment_id=experiment_id,
                        confidence=confidence[experiment_id],
                        regret=float(tasks[experiment_id]["decision_regret"]),
                        exact_correct=bool(
                            tasks[experiment_id]["correct_intervention_choice"]
                        ),
                        practically_reliable=bool(
                            tasks[experiment_id]["practically_reliable_at_0_05"]
                        ),
                    )
                    for experiment_id in RICH_DIAGNOSTIC_IDS
                ],
                minimum_class_count=3,
            )
        )
    composite = equal_rank_confidence(
        diagnostic_rows, feature_directions=FEATURE_DIRECTIONS
    )
    composite_analysis = asdict(
        selective_decision_summary(
            [
                SelectiveDecisionRecord(
                    experiment_id=experiment_id,
                    confidence=composite[experiment_id],
                    regret=float(tasks[experiment_id]["decision_regret"]),
                    exact_correct=bool(
                        tasks[experiment_id]["correct_intervention_choice"]
                    ),
                    practically_reliable=bool(
                        tasks[experiment_id]["practically_reliable_at_0_05"]
                    ),
                )
                for experiment_id in RICH_DIAGNOSTIC_IDS
            ],
            minimum_class_count=3,
        )
    )
    successes = sum(
        bool(tasks[experiment_id]["correct_intervention_choice"])
        for experiment_id in RICH_DIAGNOSTIC_IDS
    )
    practical = sum(
        bool(tasks[experiment_id]["practically_reliable_at_0_05"])
        for experiment_id in RICH_DIAGNOSTIC_IDS
    )
    return {
        "experiment_ids": list(RICH_DIAGNOSTIC_IDS),
        "experiment_count": len(RICH_DIAGNOSTIC_IDS),
        "feature_directions": FEATURE_DIRECTIONS,
        "individual_diagnostics": analyses,
        "equal_rank_composite_confidence": composite,
        "equal_rank_composite_analysis": composite_analysis,
        "exact_success_count": successes,
        "exact_failure_count": len(RICH_DIAGNOSTIC_IDS) - successes,
        "practical_success_count": practical,
        "practical_failure_count": len(RICH_DIAGNOSTIC_IDS) - practical,
        "classifier_status": "not_estimable",
        "classifier_reason": (
            "Only one exact-choice failure and zero practical-regret failures are "
            "available; the frozen minimum is three experiments in each class."
        ),
        "threshold_status": "no_validated_abstention_threshold",
        "allowed_interpretation": (
            "continuous-regret diagnostic ranking is exploratory development "
            "evidence; no calibrated probability or deployment threshold"
        ),
    }


def build_development_evidence(root: Path) -> dict[str, Any]:
    phase1, phase1_diagnostic = _phase1_row(root)
    discovery_rows, discovery_diagnostics, discovery_hashes = _discovery_rows(root)
    prospective_rows, prospective_diagnostics, prospective_hashes = (
        _prospective_rows(root)
    )
    tasks = [phase1, *discovery_rows, *prospective_rows]
    diagnostics = [*discovery_diagnostics, *prospective_diagnostics]
    if tuple(row["experiment_id"] for row in tasks) != DEVELOPMENT_IDS:
        raise ValueError("development task order or support drifted")
    regrets = {
        row["experiment_id"]: float(row["decision_regret"]) for row in tasks
    }
    regret_bootstrap = asdict(
        experiment_cluster_bootstrap(
            regrets,
            replicates=10000,
            seed=2026081305,
            confidence_level=0.95,
        )
    )
    regret_bootstrap["confidence_interval"] = list(
        regret_bootstrap["confidence_interval"]
    )
    primary_summary = {
        "experiment_count": len(tasks),
        "paradigm_count": len({row["paradigm_group"] for row in tasks}),
        "correct_intervention_count": sum(
            bool(row["correct_intervention_choice"]) for row in tasks
        ),
        "practically_reliable_count": sum(
            bool(row["practically_reliable_at_0_05"]) for row in tasks
        ),
        "mean_decision_regret": fmean(regrets.values()),
        "worst_case_decision_regret": max(regrets.values()),
        "mean_treatment_effect_mae": fmean(
            float(row["treatment_effect_mae"]) for row in tasks
        ),
        "decision_regret_experiment_cluster_bootstrap": regret_bootstrap,
    }
    phase1_score = verify_envelope(root / PHASE1_SCORE_PATH, require_blinded=False)
    payload = {
        "schema_version": "development_evidence.v1",
        "status": "complete_aggregate_only_nine_experiment_registry",
        "development_only": True,
        "canonical_test_claim": False,
        "experiment_ids": list(DEVELOPMENT_IDS),
        "experiment_count": len(DEVELOPMENT_IDS),
        "rich_diagnostic_experiment_ids": list(RICH_DIAGNOSTIC_IDS),
        "rich_diagnostic_experiment_count": len(RICH_DIAGNOSTIC_IDS),
        "sealed_confirmation_experiment_ids": list(SEALED_CONFIRMATION_IDS),
        "participant_rows_read": 0,
        "participant_rows_serialized": 0,
        "source_artifact_payload_hashes": {
            "phase1_score_payload_sha256": payload_hash(phase1_score),
            **discovery_hashes,
            **prospective_hashes,
        },
        "tasks": tasks,
        "diagnostics": {
            "jf46x_partial": phase1_diagnostic,
            "rich_rows": diagnostics,
        },
        "primary_summary": primary_summary,
        "trust_screening": _screen(tasks, diagnostics),
        "fallback_compatibility": {
            "status": "requires_common_reanalysis_before_pooling",
            "reason": (
                "The Phase 1 task has no fallback run, the five portfolio tasks use "
                "a fixed evaluation-third protocol, and the three multimodal tasks "
                "use repeated ten-fold evaluation. Existing curves are not pooled."
            ),
            "next_method": (
                "one common repeated disjoint-fold evaluator with nested budgets and "
                "target-experiment-excluded empirical-Bayes effect fusion"
            ),
        },
        "claim_boundary": (
            "Nine development experiments support method development and failure "
            "analysis. The five retrospective discovery tasks include model-selection "
            "optimism; the three multimodal tasks are prospective-development, not a "
            "canonical test. Six confirmation experiments remain outcome-sealed."
        ),
    }
    return json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))


def write_development_evidence(root: Path, output_path: Path) -> Path:
    freeze_envelope(
        build_development_evidence(root), output_path, require_blinded=False
    )
    return output_path


def verify_development_evidence(root: Path, path: Path) -> dict[str, Any]:
    evidence = verify_envelope(path, require_blinded=False)
    expected = build_development_evidence(root)
    if evidence != expected:
        raise ValueError("development evidence does not replay from pinned artifacts")
    return evidence
