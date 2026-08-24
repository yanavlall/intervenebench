"""Aggregate-only development analyses for the five-task portfolio.

This module never opens participant records. It derives an analytic random-choice
baseline and leave-one-experiment-out effect calibration from the already frozen
development score. The held-out experiment's human effects are excluded when its
calibration coefficient is fitted.
"""

from __future__ import annotations

from math import comb, fsum, isclose
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from .portfolio_development import verify_development_score
from .portfolio_development import verify_development_reveal_authorization
from .protocol import freeze_envelope, payload_hash, verify_envelope


DEFAULT_SCORE_PATH = Path("artifacts/portfolio_pilot/development_score_v2.json")
DEFAULT_ANALYSIS_PATH = Path(
    "artifacts/portfolio_pilot/development_analysis_v4.json"
)


def _random_choice_task(task: Mapping[str, Any]) -> dict[str, Any]:
    human_means = task["human_arm_means"]
    if not isinstance(human_means, Mapping) or len(human_means) < 2:
        raise ValueError("random-choice analysis requires at least two arms")
    best = max(float(value) for value in human_means.values())
    regrets = [best - float(value) for value in human_means.values()]
    return {
        "arm_count": len(human_means),
        "expected_exact_choice_probability": 1.0 / len(human_means),
        "expected_decision_regret": mean(regrets),
    }


def _poisson_binomial_tail(probabilities: list[float], threshold: int) -> float:
    if not probabilities or not 0 <= threshold <= len(probabilities):
        raise ValueError("invalid Poisson-binomial tail request")
    distribution = [1.0] + [0.0] * len(probabilities)
    for probability in probabilities:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("success probabilities must lie in [0,1]")
        updated = [0.0] * len(distribution)
        for successes, mass in enumerate(distribution):
            updated[successes] += mass * (1.0 - probability)
            if successes + 1 < len(distribution):
                updated[successes + 1] += mass * probability
        distribution = updated
    return fsum(distribution[threshold:])


def _paired_regret_comparison(tasks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    improvements = {
        experiment_id: float(task["no_effect_control_tie"]["decision_regret"])
        - float(task["local_llama3_2_3b"]["decision_regret"])
        for experiment_id, task in tasks.items()
    }
    positive = sum(value > 0.0 for value in improvements.values())
    negative = sum(value < 0.0 for value in improvements.values())
    non_ties = positive + negative
    if non_ties == 0:
        one_sided = 1.0
    else:
        one_sided = fsum(
            comb(non_ties, wins) for wins in range(positive, non_ties + 1)
        ) / (2**non_ties)
    return {
        "regret_improvement_by_experiment": improvements,
        "mean_regret_improvement": mean(improvements.values()),
        "experiments_local_lower_regret": positive,
        "experiments_no_effect_lower_regret": negative,
        "exact_one_sided_sign_test_p": one_sided,
        "exact_two_sided_sign_test_p": min(1.0, 2.0 * one_sided),
        "local_practically_reliable_count": sum(
            bool(task["local_llama3_2_3b"]["practically_reliable_at_frozen_tolerance"])
            for task in tasks.values()
        ),
        "no_effect_practically_reliable_count": sum(
            bool(task["no_effect_control_tie"]["practically_reliable_at_frozen_tolerance"])
            for task in tasks.values()
        ),
    }


def load_frozen_recommendations(root: Path) -> dict[str, dict[str, Any]]:
    authorization = verify_development_reveal_authorization(root)
    artifact_dir = (root / authorization["blind_run_manifest_path"]).parent
    return {
        experiment_id: verify_envelope(
            artifact_dir / f"{experiment_id}_recommendation.json",
            require_blinded=True,
        )
        for experiment_id in authorization["experiment_ids"]
    }


def _binary_ranking_auc(
    confidence: Mapping[str, float], correctness: Mapping[str, bool]
) -> float:
    successes = [key for key, value in correctness.items() if value]
    failures = [key for key, value in correctness.items() if not value]
    if not successes or not failures:
        raise ValueError("diagnostic AUC requires both correct and incorrect decisions")
    score = 0.0
    for success in successes:
        for failure in failures:
            if confidence[success] > confidence[failure]:
                score += 1.0
            elif confidence[success] == confidence[failure]:
                score += 0.5
    return score / (len(successes) * len(failures))


def _diagnostic_analysis(
    tasks: Mapping[str, Mapping[str, Any]],
    recommendations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if tasks.keys() != recommendations.keys():
        raise ValueError("diagnostics must cover the frozen development tasks")
    regrets = {
        experiment_id: float(task["local_llama3_2_3b"]["decision_regret"])
        for experiment_id, task in tasks.items()
    }
    correctness = {
        experiment_id: bool(task["local_llama3_2_3b"]["correct_choice"])
        for experiment_id, task in tasks.items()
    }
    practical = {
        experiment_id: bool(
            task["local_llama3_2_3b"]["practically_reliable_at_frozen_tolerance"]
        )
        for experiment_id, task in tasks.items()
    }
    confidence_values = {
        "winner_margin": {
            experiment_id: float(recommendation["diagnostics"]["winner_margin"])
            for experiment_id, recommendation in recommendations.items()
        },
        "winner_stability": {
            experiment_id: float(recommendation["diagnostics"]["winner_stability"])
            for experiment_id, recommendation in recommendations.items()
        },
        "low_response_entropy": {
            experiment_id: 1.0
            - float(
                recommendation["diagnostics"][
                    "mean_normalized_response_entropy"
                ]
            )
            for experiment_id, recommendation in recommendations.items()
        },
    }
    diagnostic_results: dict[str, Any] = {}
    for diagnostic_id, confidence in confidence_values.items():
        ordered = sorted(confidence, key=lambda key: (-confidence[key], key))
        curve = []
        for count in range(1, len(ordered) + 1):
            covered = ordered[:count]
            curve.append(
                {
                    "coverage": count / len(ordered),
                    "covered_experiment_ids": covered,
                    "mean_regret": mean(regrets[key] for key in covered),
                    "exact_choice_rate": mean(
                        float(correctness[key]) for key in covered
                    ),
                    "practical_reliability_rate": mean(
                        float(practical[key]) for key in covered
                    ),
                }
            )
        diagnostic_results[diagnostic_id] = {
            "confidence_by_experiment": confidence,
            "correct_choice_ranking_auc": _binary_ranking_auc(
                confidence, correctness
            ),
            "risk_coverage_curve": curve,
            "mean_risk_across_coverage_levels": mean(
                point["mean_regret"] for point in curve
            ),
        }
    return {
        "diagnostics_frozen_before_outcome_reveal": True,
        "target_human_outcomes_used_as_diagnostic_inputs": False,
        "random_coverage_expected_mean_regret": mean(regrets.values()),
        "by_diagnostic": diagnostic_results,
        "interpretation": "In five development experiments, all three diagnostics ranked the clearest regret failure as most trustworthy and produced worse mean risk across coverage levels than random coverage. This is a negative small-sample diagnostic result, not a validated trust model.",
    }


def _effect_maps(task: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    human = {
        str(arm_id): float(value)
        for arm_id, value in task["human_treatment_effects"].items()
    }
    synthetic = {
        str(arm_id): float(value)
        for arm_id, value in task["local_llama3_2_3b"][
            "synthetic_treatment_effects"
        ].items()
    }
    if not human or human.keys() != synthetic.keys():
        raise ValueError("human and synthetic effects must cover the same arms")
    return human, synthetic


def fit_leave_one_experiment_out_attenuation(
    tasks: Mapping[str, Mapping[str, Any]], held_out_experiment_id: str
) -> float:
    """Fit a zero-intercept effect slope without the target experiment.

    Each training experiment receives equal total weight and each of its arm
    effects receives equal weight within that experiment. The coefficient is
    constrained to [0, 1], so it can only attenuate synthetic effects toward the
    experiment's synthetic control mean.
    """

    if held_out_experiment_id not in tasks:
        raise ValueError("held-out experiment is absent")
    numerator = 0.0
    denominator = 0.0
    training_count = 0
    for experiment_id, task in tasks.items():
        if experiment_id == held_out_experiment_id:
            continue
        human, synthetic = _effect_maps(task)
        within_weight = 1.0 / len(human)
        numerator += within_weight * fsum(
            synthetic[arm_id] * human[arm_id] for arm_id in human
        )
        denominator += within_weight * fsum(
            synthetic[arm_id] ** 2 for arm_id in human
        )
        training_count += 1
    if training_count == 0 or denominator <= 0.0:
        raise ValueError("calibration requires another experiment with nonzero effects")
    return min(1.0, max(0.0, numerator / denominator))


def _calibrated_task(
    tasks: Mapping[str, Mapping[str, Any]], experiment_id: str
) -> dict[str, Any]:
    alpha = fit_leave_one_experiment_out_attenuation(tasks, experiment_id)
    human, synthetic = _effect_maps(tasks[experiment_id])
    raw_errors = [abs(synthetic[arm_id] - human[arm_id]) for arm_id in human]
    calibrated_errors = [
        abs(alpha * synthetic[arm_id] - human[arm_id]) for arm_id in human
    ]
    return {
        "training_experiment_count": len(tasks) - 1,
        "attenuation_coefficient": alpha,
        "raw_treatment_effect_mae": mean(raw_errors),
        "calibrated_treatment_effect_mae": mean(calibrated_errors),
        "choice_preserved": alpha > 0.0,
    }


def build_development_analysis_payload(
    score: Mapping[str, Any],
    recommendations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if score.get("development_only") is not True:
        raise ValueError("development analysis requires a development-only score")
    if score.get("canonical_test_claim") is not False:
        raise ValueError("development analysis cannot consume a canonical-test claim")
    tasks = score.get("tasks")
    if not isinstance(tasks, Mapping) or len(tasks) != 5:
        raise ValueError("development analysis requires the frozen five tasks")

    random_by_task = {
        experiment_id: _random_choice_task(task)
        for experiment_id, task in tasks.items()
    }
    calibrated_by_task = {
        experiment_id: _calibrated_task(tasks, experiment_id)
        for experiment_id in tasks
    }
    random_summary = {
        "expected_correct_intervention_count": fsum(
            task["expected_exact_choice_probability"]
            for task in random_by_task.values()
        ),
        "expected_correct_intervention_rate": mean(
            task["expected_exact_choice_probability"]
            for task in random_by_task.values()
        ),
        "mean_expected_decision_regret": mean(
            task["expected_decision_regret"] for task in random_by_task.values()
        ),
    }
    observed_correct = int(
        score["portfolio_summary"]["local_llama3_2_3b"][
            "correct_intervention_count"
        ]
    )
    random_summary["probability_random_policy_matches_or_exceeds_observed_count"] = (
        _poisson_binomial_tail(
            [
                task["expected_exact_choice_probability"]
                for task in random_by_task.values()
            ],
            observed_correct,
        )
    )
    raw_mae = mean(
        task["raw_treatment_effect_mae"] for task in calibrated_by_task.values()
    )
    calibrated_mae = mean(
        task["calibrated_treatment_effect_mae"]
        for task in calibrated_by_task.values()
    )
    recorded_raw_mae = float(
        score["portfolio_summary"]["local_llama3_2_3b"][
            "mean_treatment_effect_mae"
        ]
    )
    if not isclose(raw_mae, recorded_raw_mae, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("reconstructed raw effect MAE disagrees with frozen score")

    return {
        "schema_version": "portfolio_development_analysis.v4",
        "source_score_sha256": payload_hash(score),
        "development_only": True,
        "canonical_test_claim": False,
        "participant_rows_read": 0,
        "participant_rows_written": 0,
        "random_choice_policy": {
            "definition": "Uniformly select one admissible arm independently in each experiment; report analytic expectations rather than a sampled seed.",
            "by_experiment": random_by_task,
            "portfolio_summary": random_summary,
        },
        "cross_fitted_effect_attenuation": {
            "definition": "For each target experiment, fit a zero-intercept least-squares coefficient on the other four development experiments, with equal total weight per experiment and equal weight per arm effect within experiment; constrain the coefficient to [0,1].",
            "target_outcome_excluded_from_its_fit": True,
            "by_experiment": calibrated_by_task,
            "portfolio_summary": {
                "raw_treatment_effect_mae": raw_mae,
                "calibrated_treatment_effect_mae": calibrated_mae,
                "no_effect_treatment_effect_mae": float(
                    score["portfolio_summary"]["no_effect_control_tie"][
                        "mean_treatment_effect_mae"
                    ]
                ),
                "all_choices_preserved": all(
                    task["choice_preserved"]
                    for task in calibrated_by_task.values()
                ),
            },
        },
        "experiment_level_policy_comparison": _paired_regret_comparison(tasks),
        "outcome_free_diagnostic_evaluation": _diagnostic_analysis(
            tasks, recommendations
        ),
        "claim_boundary": "Random-choice performance is an analytic development baseline. Effect attenuation is leave-one-experiment-out within five revealed development experiments, not validation on a sealed test set.",
    }


def build_development_analysis(
    root: Path,
    *,
    score_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    resolved_score = score_path or root / DEFAULT_SCORE_PATH
    resolved_output = output_path or root / DEFAULT_ANALYSIS_PATH
    score = verify_development_score(root, resolved_score)
    recommendations = load_frozen_recommendations(root)
    payload = build_development_analysis_payload(score, recommendations)
    freeze_envelope(payload, resolved_output, require_blinded=False)
    return resolved_output


def verify_development_analysis(
    root: Path,
    *,
    score_path: Path | None = None,
    analysis_path: Path | None = None,
) -> dict[str, Any]:
    resolved_score = score_path or root / DEFAULT_SCORE_PATH
    resolved_analysis = analysis_path or root / DEFAULT_ANALYSIS_PATH
    score = verify_development_score(root, resolved_score)
    recommendations = load_frozen_recommendations(root)
    analysis = verify_envelope(resolved_analysis, require_blinded=False)
    expected = build_development_analysis_payload(score, recommendations)
    if analysis != expected:
        raise ValueError("development analysis no longer matches the frozen score")
    return analysis
