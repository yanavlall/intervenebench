"""Post-reveal audit of whether confirmation recommendations added decision value.

This module is intentionally mechanical and aggregate-only.  It does not turn
the six-experiment confirmation panel into a second prospective test: the
comparators were formalized after the confirmation outcomes were revealed.
Its purpose is to distinguish useful low-regret choices from results that are
equally easy to obtain by choosing a random arm or always choosing control.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from itertools import product
from math import fsum, isfinite
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from .confirmation_scoring import CONFIRMATION_IDS
from .experiment_statistics import paired_experiment_cluster_bootstrap
from .protocol import canonical_json_bytes, payload_hash


SCORE_PATH = Path("artifacts/confirmation/confirmation_20260814_v1/score_v1.json")
SPEC_PATH = Path("data/manifests/research/confirmation_value_audit_spec_v1.json")
AUDIT_PATH = Path(
    "artifacts/confirmation/confirmation_20260814_v1/value_audit_v1.json"
)
NORMALIZED_EXPERIMENT_IDS = tuple(
    experiment_id for experiment_id in CONFIRMATION_IDS if experiment_id != "tcg8p"
)
TOLERANCE_GRID = (0.0, 0.0025, 0.005, 0.01, 0.025, 0.05)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2026081410
FLOAT_TOLERANCE = 1e-12


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: Sequence[float]) -> float:
    return fsum(values) / len(values)


def _validated_regrets(regrets: Mapping[str, float]) -> dict[str, float]:
    if len(regrets) < 2:
        raise ValueError("a decision task must contain at least two arms")
    result: dict[str, float] = {}
    for arm_id, value in regrets.items():
        if not isinstance(arm_id, str) or not arm_id.strip():
            raise ValueError("arm IDs must be non-empty strings")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or float(value) < -FLOAT_TOLERANCE
        ):
            raise ValueError("arm regrets must be finite and non-negative")
        result[arm_id] = max(0.0, float(value))
    return result


def _uniform_choice_task(
    human_means: Mapping[str, float], *, practical_tolerance: float
) -> dict[str, Any]:
    """Return exact expectations for a uniform draw over admissible arms."""

    if (
        isinstance(practical_tolerance, bool)
        or not isinstance(practical_tolerance, (int, float))
        or not isfinite(float(practical_tolerance))
        or float(practical_tolerance) < 0.0
    ):
        raise ValueError("practical tolerance must be finite and non-negative")
    if len(human_means) < 2:
        raise ValueError("a decision task must contain at least two arms")
    values: dict[str, float] = {}
    for arm_id, value in human_means.items():
        if (
            not isinstance(arm_id, str)
            or not arm_id.strip()
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
        ):
            raise ValueError("human arm means must be finite numeric values")
        values[arm_id] = float(value)
    best = max(values.values())
    regrets = {arm_id: best - value for arm_id, value in values.items()}
    reliable_count = sum(
        regret <= float(practical_tolerance) + FLOAT_TOLERANCE
        for regret in regrets.values()
    )
    arm_count = len(values)
    return {
        "arm_count": arm_count,
        "regret_by_arm": regrets,
        "expected_exact_choice_probability": 1.0 / arm_count,
        "expected_decision_regret": fmean(regrets.values()),
        "practically_reliable_arm_count": reliable_count,
        "expected_practical_reliability_probability": reliable_count / arm_count,
    }


def _poisson_binomial_tail(
    probabilities: Sequence[float], *, threshold: int
) -> float:
    """Exact P(sum Bernoulli(p_i) >= threshold)."""

    if not 0 <= threshold <= len(probabilities):
        raise ValueError("threshold is outside the Bernoulli count support")
    distribution = [1.0] + [0.0] * len(probabilities)
    completed = 0
    for raw_probability in probabilities:
        probability = float(raw_probability)
        if not isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("probabilities must lie in [0, 1]")
        for successes in range(completed + 1, -1, -1):
            stay = distribution[successes] * (1.0 - probability)
            gain = distribution[successes - 1] * probability if successes else 0.0
            distribution[successes] = stay + gain
        completed += 1
    return fsum(distribution[threshold:])


def _exact_uniform_combination_summary(
    tasks: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Enumerate every uniform-arm combination across whole experiments."""

    if not tasks:
        raise ValueError("at least one task is required")
    experiment_ids = tuple(tasks)
    regrets: dict[str, dict[str, float]] = {}
    selected: dict[str, str] = {}
    for experiment_id in experiment_ids:
        task = tasks[experiment_id]
        regrets[experiment_id] = _validated_regrets(task["regret_by_arm"])
        selected_arm_id = task["primary_selected_arm_id"]
        if selected_arm_id not in regrets[experiment_id]:
            raise ValueError("primary selected arm is outside the action set")
        selected[experiment_id] = selected_arm_id

    observed_regrets = [
        regrets[experiment_id][selected[experiment_id]]
        for experiment_id in experiment_ids
    ]
    observed_exact_count = sum(
        value <= FLOAT_TOLERANCE for value in observed_regrets
    )
    observed_mean = _mean(observed_regrets)
    observed_worst = max(observed_regrets)

    exact_at_least = 0
    mean_at_most = 0
    worst_at_most = 0
    combination_count = 0
    arm_lists = [tuple(regrets[experiment_id]) for experiment_id in experiment_ids]
    for arm_combination in product(*arm_lists):
        combination_count += 1
        combination_regrets = [
            regrets[experiment_id][arm_id]
            for experiment_id, arm_id in zip(
                experiment_ids, arm_combination, strict=True
            )
        ]
        exact_count = sum(value <= FLOAT_TOLERANCE for value in combination_regrets)
        exact_at_least += exact_count >= observed_exact_count
        mean_at_most += _mean(combination_regrets) <= observed_mean + FLOAT_TOLERANCE
        worst_at_most += max(combination_regrets) <= observed_worst + FLOAT_TOLERANCE

    return {
        "experiment_ids": list(experiment_ids),
        "combination_count": combination_count,
        "observed_primary_exact_count": observed_exact_count,
        "observed_primary_mean_regret": observed_mean,
        "observed_primary_worst_regret": observed_worst,
        "probability_uniform_exact_count_at_least_observed": exact_at_least
        / combination_count,
        "probability_uniform_mean_regret_at_most_observed": mean_at_most
        / combination_count,
        "probability_uniform_worst_regret_at_most_observed": worst_at_most
        / combination_count,
    }


def _exact_sign_flip_test(
    candidate_by_experiment: Mapping[str, float],
    reference_by_experiment: Mapping[str, float],
) -> dict[str, Any]:
    """Exact paired one-sided sign-flip test over experiment-level effects."""

    if set(candidate_by_experiment) != set(reference_by_experiment):
        raise ValueError("paired comparisons require identical experiment IDs")
    if not candidate_by_experiment:
        raise ValueError("paired comparisons require at least one experiment")
    differences: dict[str, float] = {}
    for experiment_id in candidate_by_experiment:
        candidate = float(candidate_by_experiment[experiment_id])
        reference = float(reference_by_experiment[experiment_id])
        if not isfinite(candidate) or not isfinite(reference):
            raise ValueError("paired values must be finite")
        differences[experiment_id] = candidate - reference
    observed = fmean(differences.values())
    magnitudes = [
        abs(value) for value in differences.values() if abs(value) > FLOAT_TOLERANCE
    ]
    if not magnitudes:
        probability = 1.0
        enumeration_count = 1
    else:
        enumeration_count = 2 ** len(magnitudes)
        at_least_as_favorable = 0
        for signs in product((-1.0, 1.0), repeat=len(magnitudes)):
            randomized_sum = fsum(
                sign * magnitude
                for sign, magnitude in zip(signs, magnitudes, strict=True)
            )
            # Zero paired differences remain zero and the denominator remains
            # the total experiment count, matching the observed mean.
            randomized_mean = randomized_sum / len(differences)
            at_least_as_favorable += randomized_mean <= observed + FLOAT_TOLERANCE
        probability = at_least_as_favorable / enumeration_count
    return {
        "experiment_count": len(differences),
        "non_tied_experiment_count": len(magnitudes),
        "difference_by_experiment": differences,
        "mean_candidate_minus_reference": observed,
        "enumeration_count": enumeration_count,
        "one_sided_probability_under_symmetric_null": probability,
        "direction": "lower_candidate_regret_is_better",
    }


def _contract_path(experiment_id: str) -> Path:
    suffix = (
        "continuous_task_candidate.json"
        if experiment_id == "tcg8p"
        else "decision_task_candidate.json"
    )
    return Path("data/manifests/contracts") / f"{experiment_id}_{suffix}"


def _load_contract(root: Path, experiment_id: str) -> tuple[dict[str, Any], Path]:
    import json

    relative = _contract_path(experiment_id)
    path = root / relative
    with path.open(encoding="utf-8") as stream:
        contract = json.load(stream)
    if contract.get("experiment_id") != experiment_id:
        raise ValueError("task contract experiment identity mismatch")
    arm_ids = [arm.get("arm_id") for arm in contract.get("arms", [])]
    if len(arm_ids) < 2 or len(set(arm_ids)) != len(arm_ids):
        raise ValueError("task contract has an invalid action set")
    if contract.get("control_arm_id") not in arm_ids:
        raise ValueError("task contract control is outside the action set")
    return contract, relative


def build_confirmation_value_audit_spec(
    root: Path, *, score: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the immutable, explicitly post-reveal audit specification."""

    score_path = root / SCORE_PATH
    if tuple(score.get("experiment_ids", ())) != CONFIRMATION_IDS:
        raise ValueError("confirmation score experiment universe changed")
    contracts: dict[str, Any] = {}
    for experiment_id in CONFIRMATION_IDS:
        contract, relative = _load_contract(root, experiment_id)
        contracts[experiment_id] = {
            "path": relative.as_posix(),
            "file_sha256": _file_sha256(root / relative),
            "arm_ids_in_source_order": [arm["arm_id"] for arm in contract["arms"]],
            "control_arm_id": contract["control_arm_id"],
        }
    return {
        "schema_version": "confirmation_value_audit_spec.v1",
        "status": "post_reveal_mechanical_value_audit_spec",
        "outcomes_known_when_specified": True,
        "prospective_status": (
            "secondary_analysis_of_a_prospectively_scored_panel_not_a_new_"
            "prospective_test"
        ),
        "score_path": SCORE_PATH.as_posix(),
        "score_file_sha256": _file_sha256(score_path),
        "score_payload_sha256": payload_hash(score),
        "experiment_ids": list(CONFIRMATION_IDS),
        "normalized_experiment_ids": list(NORMALIZED_EXPERIMENT_IDS),
        "task_contracts": contracts,
        "primary_comparators": [
            "uniform_random_action",
            "no_effect_control_tie",
            "frozen_classical_baseline",
        ],
        "secondary_exploratory_comparators": [
            "frozen_model_plurality_consensus"
        ],
        "primary_tests": [
            "exact_heterogeneous_poisson_binomial_choice_tail",
            "exact_finite_action_space_regret_enumeration",
            "paired_exact_experiment_sign_flip",
            "paired_experiment_cluster_bootstrap",
        ],
        "practical_tolerance_sensitivity_grid": list(TOLERANCE_GRID),
        "bootstrap": {
            "unit": "experiment",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
        },
        "mixed_unit_rule": (
            "tcg8p_raw_usd_regret_is_reported_separately_and_never_pooled_"
            "with_normalized_regret"
        ),
        "simulator_or_threshold_tuning_authorized": False,
        "new_model_calls_authorized": False,
        "participant_row_access_authorized": False,
        "claim_rule": (
            "three_of_six_exact_is_not_evidence_of_added_value_by_itself; "
            "positive decision value requires regret improvement versus simple "
            "comparators with exact finite-choice and experiment-level uncertainty"
        ),
        "compute_rule": (
            "additional_draws_on_the_same_six_tasks_do_not_increase_the_"
            "experiment_level_sample_size"
        ),
    }


def _score_policy(
    *, selected_arm_id: str, human_means: Mapping[str, float]
) -> dict[str, Any]:
    if selected_arm_id not in human_means:
        raise ValueError("policy selection is outside the action set")
    best = max(float(value) for value in human_means.values())
    best_arm = next(
        arm_id
        for arm_id, value in human_means.items()
        if best - float(value) <= FLOAT_TOLERANCE
    )
    regret = best - float(human_means[selected_arm_id])
    return {
        "selected_arm_id": selected_arm_id,
        "human_selected_arm_id": best_arm,
        "exact_choice": regret <= FLOAT_TOLERANCE,
        "decision_regret": regret,
    }


def _paired_comparison(
    candidate: Mapping[str, float],
    reference: Mapping[str, float],
    *,
    seed: int,
) -> dict[str, Any]:
    bootstrap = paired_experiment_cluster_bootstrap(
        candidate,
        reference,
        replicates=BOOTSTRAP_REPLICATES,
        seed=seed,
        confidence_level=0.95,
        lower_is_better=True,
    )
    bootstrap_payload = asdict(bootstrap)
    bootstrap_payload["difference_confidence_interval"] = list(
        bootstrap_payload["difference_confidence_interval"]
    )
    return {
        "cluster_bootstrap": bootstrap_payload,
        "exact_sign_flip": _exact_sign_flip_test(candidate, reference),
    }


def build_confirmation_value_audit_payload(
    root: Path,
    *,
    score: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete value audit from the aggregate score artifact only."""

    if spec.get("score_payload_sha256") != payload_hash(score):
        raise ValueError("audit specification is not bound to the supplied score")
    if spec.get("experiment_ids") != list(CONFIRMATION_IDS):
        raise ValueError("audit specification experiment universe changed")
    experiment_scores = score.get("experiment_scores")
    if not isinstance(experiment_scores, Mapping) or set(experiment_scores) != set(
        CONFIRMATION_IDS
    ):
        raise ValueError("confirmation score experiment set changed")

    experiments: dict[str, Any] = {}
    uniform_tasks: dict[str, dict[str, Any]] = {}
    primary_regret: dict[str, float] = {}
    uniform_regret: dict[str, float] = {}
    control_regret: dict[str, float] = {}
    classical_regret: dict[str, float] = {}
    consensus_regret: dict[str, float] = {}
    primary_bootstrap_optimality: dict[str, float] = {}
    uniform_optimality: dict[str, float] = {}

    for experiment_id in CONFIRMATION_IDS:
        entry = experiment_scores[experiment_id]
        primary = entry["primary_score"]
        human_means = {
            arm_id: float(value)
            for arm_id, value in primary["human_arm_means"].items()
        }
        contract_info = spec["task_contracts"][experiment_id]
        arm_ids = tuple(contract_info["arm_ids_in_source_order"])
        if set(arm_ids) != set(human_means):
            raise ValueError("score and task contract action sets differ")
        practical_tolerance = float(primary["practical_tolerance"])
        uniform = _uniform_choice_task(
            human_means, practical_tolerance=practical_tolerance
        )
        primary_selected = primary["synthetic_selected_arm_id"]
        primary_policy = _score_policy(
            selected_arm_id=primary_selected, human_means=human_means
        )
        control_policy = _score_policy(
            selected_arm_id=contract_info["control_arm_id"],
            human_means=human_means,
        )

        model_scores = entry["model_scores"]
        model_selections = {
            model_id: model_score["synthetic_selected_arm_id"]
            for model_id, model_score in model_scores.items()
        }
        selection_counts = Counter(model_selections.values())
        top_count = max(selection_counts.values())
        consensus_selected = next(
            arm_id for arm_id in arm_ids if selection_counts[arm_id] == top_count
        )
        consensus_policy = _score_policy(
            selected_arm_id=consensus_selected, human_means=human_means
        )
        classical_score = entry.get("classical_baseline_score")
        classical_policy = None
        if classical_score is not None:
            classical_policy = _score_policy(
                selected_arm_id=classical_score["synthetic_selected_arm_id"],
                human_means=human_means,
            )

        uniform_tasks[experiment_id] = {
            "regret_by_arm": uniform["regret_by_arm"],
            "primary_selected_arm_id": primary_selected,
        }
        primary_regret[experiment_id] = primary_policy["decision_regret"]
        uniform_regret[experiment_id] = uniform["expected_decision_regret"]
        control_regret[experiment_id] = control_policy["decision_regret"]
        consensus_regret[experiment_id] = consensus_policy["decision_regret"]
        if classical_policy is not None:
            classical_regret[experiment_id] = classical_policy["decision_regret"]
        primary_bootstrap_optimality[experiment_id] = float(
            entry["participant_bootstrap"]["bootstrap_exact_choice_rate"]
        )
        uniform_optimality[experiment_id] = 1.0 / len(arm_ids)

        experiments[experiment_id] = {
            "outcome_unit": primary["outcome_unit"],
            "arm_ids_in_source_order": list(arm_ids),
            "practical_tolerance": practical_tolerance,
            "primary": primary_policy,
            "uniform_random_action": uniform,
            "no_effect_control_tie": control_policy,
            "frozen_classical_baseline": classical_policy,
            "frozen_model_plurality_consensus": {
                **consensus_policy,
                "model_selection_by_id": model_selections,
                "selection_count_by_arm": {
                    arm_id: selection_counts[arm_id] for arm_id in arm_ids
                },
                "available_model_count": len(model_selections),
                "winner_support_fraction": top_count / len(model_selections),
                "unique_selected_arm_count": len(selection_counts),
                "status": "secondary_post_reveal_defined_comparator",
            },
            "primary_selected_arm_bootstrap_optimality_rate": (
                primary_bootstrap_optimality[experiment_id]
            ),
        }

    normalized_uniform_tasks = {
        experiment_id: uniform_tasks[experiment_id]
        for experiment_id in NORMALIZED_EXPERIMENT_IDS
    }
    normalized_primary = {
        experiment_id: primary_regret[experiment_id]
        for experiment_id in NORMALIZED_EXPERIMENT_IDS
    }
    normalized_uniform = {
        experiment_id: uniform_regret[experiment_id]
        for experiment_id in NORMALIZED_EXPERIMENT_IDS
    }
    normalized_control = {
        experiment_id: control_regret[experiment_id]
        for experiment_id in NORMALIZED_EXPERIMENT_IDS
    }
    normalized_classical = {
        experiment_id: classical_regret[experiment_id]
        for experiment_id in NORMALIZED_EXPERIMENT_IDS
    }
    normalized_consensus = {
        experiment_id: consensus_regret[experiment_id]
        for experiment_id in NORMALIZED_EXPERIMENT_IDS
    }

    observed_exact = sum(
        experiments[experiment_id]["primary"]["exact_choice"]
        for experiment_id in CONFIRMATION_IDS
    )
    exact_probabilities = [
        experiments[experiment_id]["uniform_random_action"]
        ["expected_exact_choice_probability"]
        for experiment_id in CONFIRMATION_IDS
    ]
    observed_practical = sum(
        primary_regret[experiment_id]
        <= experiments[experiment_id]["practical_tolerance"] + FLOAT_TOLERANCE
        for experiment_id in CONFIRMATION_IDS
    )
    practical_probabilities = [
        experiments[experiment_id]["uniform_random_action"]
        ["expected_practical_reliability_probability"]
        for experiment_id in CONFIRMATION_IDS
    ]

    tolerance_sensitivity = []
    for tolerance in TOLERANCE_GRID:
        primary_count = sum(
            normalized_primary[experiment_id] <= tolerance + FLOAT_TOLERANCE
            for experiment_id in NORMALIZED_EXPERIMENT_IDS
        )
        control_count = sum(
            normalized_control[experiment_id] <= tolerance + FLOAT_TOLERANCE
            for experiment_id in NORMALIZED_EXPERIMENT_IDS
        )
        random_probabilities = [
            sum(
                regret <= tolerance + FLOAT_TOLERANCE
                for regret in normalized_uniform_tasks[experiment_id][
                    "regret_by_arm"
                ].values()
            )
            / len(normalized_uniform_tasks[experiment_id]["regret_by_arm"])
            for experiment_id in NORMALIZED_EXPERIMENT_IDS
        ]
        tolerance_sensitivity.append(
            {
                "tolerance": tolerance,
                "primary_reliable_count": primary_count,
                "control_reliable_count": control_count,
                "expected_uniform_reliable_count": fsum(random_probabilities),
                "probability_uniform_reliable_count_at_least_primary": (
                    _poisson_binomial_tail(
                        random_probabilities, threshold=primary_count
                    )
                ),
            }
        )

    return {
        "schema_version": "confirmation_value_audit.v1",
        "status": "post_reveal_mechanical_value_audit_complete",
        "evidence_status": (
            "secondary_analysis_of_prospective_confirmation_not_a_new_"
            "prospective_test"
        ),
        "score_payload_sha256": payload_hash(score),
        "audit_spec_payload_sha256": payload_hash(spec),
        "prospective_experiment_count": len(CONFIRMATION_IDS),
        "normalized_experiment_count": len(NORMALIZED_EXPERIMENT_IDS),
        "tcg8p_reported_separately": True,
        "participant_rows_read": 0,
        "participant_rows_serialized": 0,
        "new_model_calls_made": 0,
        "simulator_or_threshold_changes_made": 0,
        "experiments": experiments,
        "exact_choice_all_six": {
            "primary_exact_count": observed_exact,
            "expected_uniform_exact_count": fsum(exact_probabilities),
            "probability_uniform_exact_count_at_least_primary": (
                _poisson_binomial_tail(
                    exact_probabilities, threshold=observed_exact
                )
            ),
        },
        "frozen_practical_reliability_all_six": {
            "primary_reliable_count": observed_practical,
            "expected_uniform_reliable_count": fsum(practical_probabilities),
            "probability_uniform_reliable_count_at_least_primary": (
                _poisson_binomial_tail(
                    practical_probabilities, threshold=observed_practical
                )
            ),
        },
        "normalized_finite_action_space_enumeration": (
            _exact_uniform_combination_summary(normalized_uniform_tasks)
        ),
        "normalized_regret_comparisons": {
            "primary_vs_uniform_random_action": _paired_comparison(
                normalized_primary,
                normalized_uniform,
                seed=BOOTSTRAP_SEED,
            ),
            "primary_vs_no_effect_control_tie": _paired_comparison(
                normalized_primary,
                normalized_control,
                seed=BOOTSTRAP_SEED + 1,
            ),
            "primary_vs_frozen_classical_baseline": _paired_comparison(
                normalized_primary,
                normalized_classical,
                seed=BOOTSTRAP_SEED + 2,
            ),
            "primary_vs_frozen_model_plurality_consensus_secondary": (
                _paired_comparison(
                    normalized_primary,
                    normalized_consensus,
                    seed=BOOTSTRAP_SEED + 3,
                )
            ),
        },
        "selected_arm_optimality_under_participant_bootstrap": {
            "primary_mean": fmean(primary_bootstrap_optimality.values()),
            "uniform_random_action_mean": fmean(uniform_optimality.values()),
            "comparison": _paired_comparison(
                {key: -value for key, value in primary_bootstrap_optimality.items()},
                {key: -value for key, value in uniform_optimality.items()},
                seed=BOOTSTRAP_SEED + 4,
            ),
            "comparison_sign_note": (
                "probabilities_are_negated_so_lower_is_better_means_higher_"
                "optimality"
            ),
        },
        "normalized_practical_tolerance_sensitivity": tolerance_sensitivity,
        "compute_interpretation": {
            "same_task_draws_increase_experiment_n": False,
            "remaining_modal_credit_should_target": [
                "new_independent_precommitted_experiments",
                "temporal_or_source_replication",
                "frozen_hypothesis_driven_model_contrast",
            ],
            "remaining_modal_credit_should_not_target": (
                "more_draws_on_the_same_six_tasks_to_manufacture_precision"
            ),
        },
    }


def encoded_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Expose the canonical payload digest for command-line provenance."""

    return sha256(canonical_json_bytes(payload)).hexdigest()
