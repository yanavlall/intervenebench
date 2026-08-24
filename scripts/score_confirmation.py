#!/usr/bin/env python3
"""Reveal six authorized outcomes and write aggregate-only confirmation scores."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

from intervenebench.confirmation_scoring import (
    AGGREGATION_PATH,
    CONFIRMATION_IDS,
    HumanArmSummary,
    RawFallbackObservation,
    RevealedOutcomeObservation,
    evaluate_raw_human_fallback,
    participant_bootstrap_score,
    read_confirmation_outcomes,
    score_synthetic_recommendation,
    summarize_human_arms,
    validate_confirmation_reveal_authorization,
    verify_confirmation_scoring_protocol,
)
from intervenebench.eb_fallback import EffectPrior, evaluate_eb_human_fallback
from intervenebench.experiment_statistics import (
    experiment_cluster_bootstrap,
    paired_experiment_cluster_bootstrap,
)
from intervenebench.human_fallback import FallbackObservation
from intervenebench.protocol import freeze_envelope, payload_hash, verify_envelope
from intervenebench.selective_decision import (
    SelectiveDecisionRecord,
    selective_decision_summary,
)


ROOT = Path(__file__).resolve().parents[1]
PREPARATION_PATH = Path("artifacts/confirmation/confirmation_preparation_v1.json")
DEVELOPMENT_EVIDENCE_PATH = Path("artifacts/development/development_evidence_v1.json")
BUDGETS = (0, 10, 25, 50, 100, 250)
POLICIES = (
    "synthetic_only",
    "human_only_balanced",
    "synthetic_plus_balanced_fixed10",
    "synthetic_plus_hedged_fixed10",
    "synthetic_plus_balanced_eb",
    "synthetic_plus_hedged_eb",
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _task_path(experiment_id: str) -> Path:
    name = (
        "tcg8p_continuous_task_candidate.json"
        if experiment_id == "tcg8p"
        else f"{experiment_id}_decision_task_candidate.json"
    )
    return Path("data/manifests/contracts") / name


def _unweighted_sensitivity(
    observations: list[RevealedOutcomeObservation],
    *,
    arm_ids: list[str],
    outcome_unit: str,
) -> dict[str, float]:
    return dict(
        summarize_human_arms(
            observations,
            arm_ids=arm_ids,
            outcome_unit=outcome_unit,
            use_weights=False,
        ).arm_means
    )


def _equal_fold_stratum_sensitivity(
    observations: list[RevealedOutcomeObservation],
    *,
    arm_ids: list[str],
    outcome_unit: str,
) -> dict[str, float]:
    rows = [
        RevealedOutcomeObservation(
            row.participant_id,
            row.arm_id,
            row.raw_value,
            row.decision_score,
            row.weight,
            row.fold_stratum_id,
            row.fold_stratum_id,
        )
        for row in observations
    ]
    return dict(
        summarize_human_arms(
            rows, arm_ids=arm_ids, outcome_unit=outcome_unit
        ).arm_means
    )


def _fallback_observations(
    observations: list[RevealedOutcomeObservation], *, experiment_id: str
) -> list[FallbackObservation]:
    if experiment_id != "pb2rr":
        return [
            FallbackObservation(
                row.participant_id,
                row.arm_id,
                row.decision_score,
                row.weight,
                row.fold_stratum_id,
            )
            for row in observations
        ]
    # The full-score estimand gives each randomized recipient name equal weight.
    # For low-budget pilots, use design weights normalized by the complete-case
    # arm×name weight total, while retaining name-stratified folds. This avoids
    # requiring all 16 names in a 10-person pilot and preserves equal-name mass.
    totals: dict[tuple[str, str], float] = defaultdict(float)
    for row in observations:
        totals[(row.arm_id, row.standardization_cell_id)] += row.weight
    return [
        FallbackObservation(
            row.participant_id,
            row.arm_id,
            row.decision_score,
            row.weight / totals[(row.arm_id, row.standardization_cell_id)],
            row.fold_stratum_id,
        )
        for row in observations
    ]


def _aggregate_fallback(tasks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for budget_index, budget in enumerate(BUDGETS):
        result[str(budget)] = {}
        for policy_index, policy in enumerate(POLICIES):
            rows = {
                experiment_id: task["by_budget"][str(budget)][policy]
                for experiment_id, task in tasks.items()
                if task["by_budget"][str(budget)][policy]["status"] == "estimated"
            }
            if not rows:
                result[str(budget)][policy] = {
                    "status": "not_estimable",
                    "experiment_count": 0,
                }
                continue
            regrets = {
                experiment_id: float(row["mean_regret"])
                for experiment_id, row in rows.items()
            }
            summary: dict[str, Any] = {
                "status": "estimated",
                "experiment_count": len(rows),
                "experiment_ids": sorted(rows),
                "mean_regret": fmean(regrets.values()),
                "mean_exact_choice_rate": fmean(
                    float(row["exact_choice_rate"]) for row in rows.values()
                ),
                "mean_practical_reliability_rate": fmean(
                    float(row["practical_reliability_rate"])
                    for row in rows.values()
                ),
            }
            if len(rows) >= 2:
                summary["experiment_cluster_bootstrap"] = asdict(
                    experiment_cluster_bootstrap(
                        regrets,
                        replicates=10000,
                        seed=2026081404 + budget_index * 100 + policy_index,
                    )
                )
                reference = {
                    experiment_id: float(
                        tasks[experiment_id]["by_budget"][str(budget)][
                            "synthetic_only"
                        ]["mean_regret"]
                    )
                    for experiment_id in rows
                }
                summary["paired_regret_vs_synthetic_bootstrap"] = asdict(
                    paired_experiment_cluster_bootstrap(
                        regrets,
                        reference,
                        replicates=10000,
                        seed=2026081404
                        + 10000
                        + budget_index * 100
                        + policy_index,
                    )
                )
            result[str(budget)][policy] = summary
    return result


def score(*, authorization_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"create-only confirmation score exists: {output_path}")
    aggregation = verify_envelope(ROOT / AGGREGATION_PATH, require_blinded=True)
    protocol = verify_confirmation_scoring_protocol(ROOT)
    evidence = verify_envelope(ROOT / DEVELOPMENT_EVIDENCE_PATH, require_blinded=False)
    authorization = verify_envelope(authorization_path, require_blinded=True)
    validate_confirmation_reveal_authorization(
        authorization,
        aggregation_payload_sha256=payload_hash(aggregation),
        scoring_protocol_payload_sha256=payload_hash(protocol),
        development_evidence_payload_sha256=payload_hash(evidence),
    )
    if aggregation.get("model_calls_made") != 0 or aggregation.get(
        "automatic_next_stage_authorized"
    ) is not False:
        raise PermissionError("frozen aggregation authority drifted")

    tasks = {
        experiment_id: _read_object(ROOT / _task_path(experiment_id))
        for experiment_id in CONFIRMATION_IDS
    }
    aggregation_by_id = {
        row["experiment_id"]: row for row in aggregation["experiment_results"]
    }
    # This is the first operation that opens confirmation target-outcome values.
    outcomes = read_confirmation_outcomes(ROOT, protocol=protocol, tasks=tasks)

    preparation = verify_envelope(ROOT / PREPARATION_PATH, require_blinded=True)
    experiment_scores: dict[str, Any] = {}
    fallback_inputs: dict[str, Any] = {}
    for index, experiment_id in enumerate(CONFIRMATION_IDS):
        task = tasks[experiment_id]
        aggregate = aggregation_by_id[experiment_id]
        arm_ids = [str(arm["arm_id"]) for arm in task["arms"]]
        outcome_unit = (
            "usd_per_month" if experiment_id == "tcg8p" else "normalized_utility"
        )
        human = summarize_human_arms(
            outcomes[experiment_id], arm_ids=arm_ids, outcome_unit=outcome_unit
        )
        practical_tolerance = (
            0.0
            if experiment_id == "tcg8p"
            else float(task["practical_regret_tolerance"])
        )
        model_scores = {}
        for model_id, recommendation in aggregate["model_recommendations"].items():
            model_scores[model_id] = score_synthetic_recommendation(
                arm_ids=arm_ids,
                control_arm_id=str(task["control_arm_id"]),
                human=human,
                synthetic_arm_scores=recommendation["arm_decision_scores"],
                selected_arm_id=str(recommendation["selected_arm_id"]),
                practical_tolerance=practical_tolerance,
            )
        primary_model_id = str(aggregate["primary_model_id"])
        primary = model_scores[primary_model_id]
        primary_recommendation = aggregate["model_recommendations"][primary_model_id]
        bootstrap = participant_bootstrap_score(
            outcomes[experiment_id],
            arm_ids=arm_ids,
            control_arm_id=str(task["control_arm_id"]),
            synthetic_arm_scores=primary_recommendation["arm_decision_scores"],
            selected_arm_id=str(primary_recommendation["selected_arm_id"]),
            practical_tolerance=practical_tolerance,
            outcome_unit=outcome_unit,
            replicates=2000,
            seed=2026081405 + index * 10000,
        )
        sensitivities: dict[str, Any] = {
            "unweighted_complete_case_arm_means": _unweighted_sensitivity(
                outcomes[experiment_id], arm_ids=arm_ids, outcome_unit=outcome_unit
            )
        }
        if experiment_id in {"Blair1131", "KlarS44"}:
            sensitivities["equal_nuisance_stratum_weighted_arm_means"] = (
                _equal_fold_stratum_sensitivity(
                    outcomes[experiment_id],
                    arm_ids=arm_ids,
                    outcome_unit=outcome_unit,
                )
            )
        classical = None
        if experiment_id in preparation["classical_baseline_predictions"]:
            frozen_classical = preparation["classical_baseline_predictions"][
                experiment_id
            ]
            classical = score_synthetic_recommendation(
                arm_ids=arm_ids,
                control_arm_id=str(task["control_arm_id"]),
                human=human,
                synthetic_arm_scores=frozen_classical[
                    "predicted_normalized_utility_effects"
                ],
                selected_arm_id=str(frozen_classical["selected_arm_id"]),
                practical_tolerance=practical_tolerance,
            )
        experiment_scores[experiment_id] = {
            "experiment_id": experiment_id,
            "paradigm_group": task["paradigm_group"],
            "primary_model_id": primary_model_id,
            "primary_score": primary,
            "model_scores": model_scores,
            "classical_baseline_score": classical,
            "participant_bootstrap": bootstrap,
            "prespecified_sensitivities": sensitivities,
            "participant_rows_serialized": 0,
        }
        winner_counts = Counter(
            recommendation["selected_arm_id"]
            for recommendation in aggregate["model_recommendations"].values()
        )
        fallback_inputs[experiment_id] = {
            "arm_ids": arm_ids,
            "control_arm_id": str(task["control_arm_id"]),
            "synthetic_means": primary_recommendation["arm_decision_scores"],
            "winner_votes": {arm: int(winner_counts[arm]) for arm in arm_ids},
            "practical_tolerance": practical_tolerance,
        }

    confidence = aggregation["trust_ranking"]["confidence_by_experiment"]
    exact_records = [
        SelectiveDecisionRecord(
            experiment_id=experiment_id,
            confidence=float(confidence[experiment_id]),
            regret=float(not experiment_scores[experiment_id]["primary_score"]["exact_choice"]),
            exact_correct=bool(
                experiment_scores[experiment_id]["primary_score"]["exact_choice"]
            ),
            practically_reliable=bool(
                experiment_scores[experiment_id]["primary_score"][
                    "practically_reliable"
                ]
            ),
        )
        for experiment_id in CONFIRMATION_IDS
    ]
    normalized_records = [
        SelectiveDecisionRecord(
            experiment_id=experiment_id,
            confidence=float(confidence[experiment_id]),
            regret=float(
                experiment_scores[experiment_id]["primary_score"]["decision_regret"]
            ),
            exact_correct=bool(
                experiment_scores[experiment_id]["primary_score"]["exact_choice"]
            ),
            practically_reliable=bool(
                experiment_scores[experiment_id]["primary_score"][
                    "practically_reliable"
                ]
            ),
        )
        for experiment_id in CONFIRMATION_IDS[1:]
    ]
    trust_evaluation = {
        "exact_error_risk_coverage_all_six": asdict(
            selective_decision_summary(exact_records, minimum_class_count=3)
        ),
        "normalized_regret_risk_coverage_five_tasks": asdict(
            selective_decision_summary(normalized_records, minimum_class_count=3)
        ),
        "learned_threshold": None,
        "accept_abstain_policy": "not_validated_not_deployed",
    }
    ranking_ids = [
        row["experiment_id"] for row in aggregation["trust_ranking"]["ranking"]
    ]
    fixed_coverage = {}
    for label, count in (("50_percent", 3), ("75_percent", 5), ("100_percent", 6)):
        covered = ranking_ids[:count]
        normalized = [eid for eid in covered if eid != "tcg8p"]
        fixed_coverage[label] = {
            "covered_experiment_ids": covered,
            "exact_choice_rate": fmean(
                float(experiment_scores[eid]["primary_score"]["exact_choice"])
                for eid in covered
            ),
            "normalized_regret_experiment_ids": normalized,
            "mean_normalized_regret": (
                fmean(
                    experiment_scores[eid]["primary_score"]["decision_regret"]
                    for eid in normalized
                )
                if normalized
                else None
            ),
            "tcg8p_raw_regret_usd_per_month": (
                experiment_scores["tcg8p"]["primary_score"]["decision_regret"]
                if "tcg8p" in covered
                else None
            ),
        }
    trust_evaluation["fixed_coverage"] = fixed_coverage

    primary = {
        eid: experiment_scores[eid]["primary_score"] for eid in CONFIRMATION_IDS
    }
    normalized_ids = CONFIRMATION_IDS[1:]
    headline = {
        "experiment_count": 6,
        "exact_choice": asdict(
            experiment_cluster_bootstrap(
                {eid: float(primary[eid]["exact_choice"]) for eid in CONFIRMATION_IDS},
                replicates=10000,
                seed=2026081404,
            )
        ),
        "practical_reliability": asdict(
            experiment_cluster_bootstrap(
                {
                    eid: float(primary[eid]["practically_reliable"])
                    for eid in CONFIRMATION_IDS
                },
                replicates=10000,
                seed=2026081405,
            )
        ),
        "mean_normalized_decision_regret": asdict(
            experiment_cluster_bootstrap(
                {eid: primary[eid]["decision_regret"] for eid in normalized_ids},
                replicates=10000,
                seed=2026081406,
            )
        ),
        "mean_normalized_treatment_effect_mae": asdict(
            experiment_cluster_bootstrap(
                {
                    eid: primary[eid]["mean_absolute_treatment_effect_error"]
                    for eid in normalized_ids
                },
                replicates=10000,
                seed=2026081407,
            )
        ),
        "mean_treatment_effect_sign_accuracy": asdict(
            experiment_cluster_bootstrap(
                {
                    eid: primary[eid]["treatment_effect_sign_accuracy"]
                    for eid in CONFIRMATION_IDS
                },
                replicates=10000,
                seed=2026081408,
            )
        ),
        "tcg8p_raw_usd_per_month": {
            "decision_regret": primary["tcg8p"]["decision_regret"],
            "treatment_effect_mae": primary["tcg8p"][
                "mean_absolute_treatment_effect_error"
            ],
        },
    }

    prior_data = protocol["human_fallback"][
        "effect_prior_frozen_on_all_development_experiments"
    ]
    prior = EffectPrior(
        alpha=float(prior_data["alpha"]),
        residual_variance=float(prior_data["residual_variance"]),
        training_experiment_ids=tuple(prior_data["training_experiment_ids"]),
        contrast_count=int(prior_data["contrast_count"]),
        minimum_variance=float(prior_data["minimum_variance"]),
    )
    normalized_fallback: dict[str, Any] = {}
    for index, experiment_id in enumerate(normalized_ids):
        info = fallback_inputs[experiment_id]
        task_budgets = BUDGETS[:-1] if experiment_id == "Blair1131" else BUDGETS
        result = evaluate_eb_human_fallback(
            _fallback_observations(outcomes[experiment_id], experiment_id=experiment_id),
            arm_ids=info["arm_ids"],
            control_arm_id=info["control_arm_id"],
            synthetic_means=info["synthetic_means"],
            winner_votes=info["winner_votes"],
            budgets=task_budgets,
            partitions=20,
            fold_count=10,
            seed=2026081403 + index * 100000,
            pseudocount=10,
            practical_tolerance=info["practical_tolerance"],
            effect_prior=prior,
        )
        if experiment_id == "Blair1131":
            result["budgets"] = list(BUDGETS)
            result["by_budget"]["250"] = {
                policy: {
                    "status": "not_estimable_predeclared_capacity",
                    "human_observations": 250,
                    "reason": protocol["human_fallback"][
                        "predeclared_infeasible_task_budgets"
                    ]["Blair1131"]["250"],
                }
                for policy in POLICIES
            }
        normalized_fallback[experiment_id] = result

    tcg_info = fallback_inputs["tcg8p"]
    raw_fallback = evaluate_raw_human_fallback(
        [
            RawFallbackObservation(
                row.participant_id,
                row.arm_id,
                row.raw_value,
                row.weight,
                row.fold_stratum_id,
            )
            for row in outcomes["tcg8p"]
        ],
        arm_ids=tcg_info["arm_ids"],
        synthetic_locations={
            arm: -float(score)
            for arm, score in tcg_info["synthetic_means"].items()
        },
        budgets=BUDGETS,
        partitions=20,
        fold_count=10,
        seed=2026081403 + 500000,
        practical_tolerance=0.0,
    )
    fallback = {
        "normalized_tasks": normalized_fallback,
        "normalized_aggregate": _aggregate_fallback(normalized_fallback),
        "tcg8p_raw_unit_primary_policies": raw_fallback,
        "tcg8p_negative_ablation_status": "not_applicable_unbounded_raw_unit",
        "participant_rows_serialized": 0,
    }

    payload = {
        "schema_version": "confirmation_score.v1",
        "status": "complete_prospective_confirmation_scoring_stop",
        "evidence_tier": "noncanonical_prospective_confirmation",
        "authorization_payload_sha256": payload_hash(authorization),
        "aggregation_payload_sha256": payload_hash(aggregation),
        "scoring_protocol_payload_sha256": payload_hash(protocol),
        "development_evidence_payload_sha256": payload_hash(evidence),
        "scoring_script_sha256": _file_sha256(Path(__file__)),
        "scoring_module_sha256": _file_sha256(
            ROOT / "src/intervenebench/confirmation_scoring.py"
        ),
        "experiment_ids": list(CONFIRMATION_IDS),
        "experiment_scores": experiment_scores,
        "headline_primary_simulator": headline,
        "trust_evaluation": trust_evaluation,
        "human_fallback": fallback,
        "socrates_tcg8p_status": aggregation["unavailable_model_task_cell"],
        "model_calls_made": 0,
        "modal_compute_used": False,
        "recommendations_changed_after_reveal": False,
        "diagnostics_changed_after_reveal": False,
        "threshold_tuned_after_reveal": False,
        "mechanical_reveal_attempt_log": [
            {
                "attempt": 1,
                "status": "stopped_before_scoring_artifact",
                "reason": "Blair1131 source CSV required deterministic Windows-1252 decoding rather than UTF-8",
                "methodological_changes": False,
            },
            {
                "attempt": 2,
                "status": "this_scoring_run",
                "change": "UTF-8-sig first, Windows-1252 fallback for the same hash-pinned Blair CSV bytes",
                "methodological_changes": False,
            },
        ],
        "confirmation_outcomes_accessed": list(CONFIRMATION_IDS),
        "participant_rows_serialized": 0,
        "automatic_followup_authorized": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    freeze_envelope(payload, output_path, require_blinded=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    score(authorization_path=args.authorization, output_path=args.output)


if __name__ == "__main__":
    main()
