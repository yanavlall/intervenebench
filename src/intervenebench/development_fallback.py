"""One common limited-human fallback evaluation over nine development tasks."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from statistics import fmean, median
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .development_evidence import (
    DEFAULT_DEVELOPMENT_EVIDENCE_PATH,
    DEVELOPMENT_IDS,
    DISCOVERY_DIAGNOSTICS_PATH,
    DISCOVERY_MODEL_ID,
    DISCOVERY_SCORE_PATH,
    PHASE1_RECOMMENDATION_PATH,
    PHASE1_SCORE_PATH,
    PROSPECTIVE_MODEL_ID,
    PROSPECTIVE_RECOMMENDATIONS_PATH,
    PROSPECTIVE_SCORE_PATH,
    SEALED_CONFIRMATION_IDS,
    verify_development_evidence,
)
from .experiment_statistics import (
    experiment_cluster_bootstrap,
    paired_experiment_cluster_bootstrap,
)
from .eb_fallback import (
    EffectCalibrationTask,
    evaluate_eb_human_fallback,
    fit_effect_prior,
)
from .human_fallback import FallbackObservation
from .multimodal_freeze import sha256_file
from .portfolio_development import (
    _read_external_observations,
    verify_development_reveal_authorization,
)
from .prospective_development_protocol import (
    verify_pre_reveal_protocol,
    verify_reveal_authorization,
)
from .prospective_development_score import (
    _read_es4xw,
    verify_prospective_development_score,
)
from .protocol import (
    assert_blinded_payload,
    freeze_envelope,
    payload_hash,
    verify_envelope,
)


FALLBACK_PROTOCOL_PATH = Path(
    "data/manifests/research/development_fallback_protocol_v1.json"
)
DEFAULT_FALLBACK_PATH = Path(
    "artifacts/development/development_fallback_v1.json"
)
SOCSCI_ROOT = Path(
    "data/raw/socsci210/048481111a4425ed83dc0eacf15f8431f252b21a/data"
)
BUDGETS = (0, 10, 25, 50, 100, 250)
PARTITIONS = 20
FOLD_COUNT = 10
SEED = 2026081306
PSEUDOCOUNT = 10
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 2026081307
MINIMUM_PRIOR_VARIANCE = 1e-6
POLICIES = (
    "synthetic_only",
    "human_only_balanced",
    "synthetic_plus_balanced_fixed10",
    "synthetic_plus_hedged_fixed10",
    "synthetic_plus_balanced_eb",
    "synthetic_plus_hedged_eb",
)
TASK_PATHS = {
    "jf46x": Path("data/manifests/contracts/jf46x_decision_task.json"),
    **{
        experiment_id: Path(
            f"data/manifests/contracts/{experiment_id}_decision_task_candidate.json"
        )
        for experiment_id in DEVELOPMENT_IDS
        if experiment_id != "jf46x"
    },
}
SOCSCI_IDS = frozenset(
    {"jf46x", "5vm8g", "xc4yq", "de5hx", "nj5dx", "e2pyb"}
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def build_development_fallback_protocol(root: Path) -> dict[str, Any]:
    evidence = verify_development_evidence(
        root, root / DEFAULT_DEVELOPMENT_EVIDENCE_PATH
    )
    protocol = {
        "schema_version": "development_fallback_protocol.v1",
        "status": "development_only_method_frozen_before_unified_policy_result",
        "development_evidence_payload_sha256": payload_hash(evidence),
        "development_experiment_ids": list(DEVELOPMENT_IDS),
        "development_experiment_count": len(DEVELOPMENT_IDS),
        "confirmation_experiment_ids": list(SEALED_CONFIRMATION_IDS),
        "confirmation_outcome_access_authorized": False,
        "participant_unit": "participant",
        "evaluation": "repeated_arm_stratified_disjoint_ten_fold",
        "partitions": PARTITIONS,
        "fold_count": FOLD_COUNT,
        "budgets": list(BUDGETS),
        "nested_without_replacement_pilot_prefixes": True,
        "same_folds_draws_and_seeds_across_policies": True,
        "seed": SEED,
        "fixed_fusion_pseudocount_per_arm": PSEUDOCOUNT,
        "target_prior_fit": "leave_one_experiment_out",
        "effect_prior": {
            "model": "human_effect_equals_alpha_times_synthetic_effect_plus_residual",
            "intercept": 0.0,
            "alpha_constraint": [0.0, 1.0],
            "experiment_weighting": "equal_total_weight_per_experiment",
            "residual_variance": "mean_within_experiment_contrast_residual_mse",
            "minimum_variance": MINIMUM_PRIOR_VARIANCE,
            "pilot_likelihood": "arm_mean_effects_with_shared_control_covariance",
            "target_human_outcomes_excluded_from_prior": True,
        },
        "allocation": {
            "balanced": "source_order_largest_remainder",
            "hedged": "25_percent_uniform_floor_then_laplace_smoothed_outcome_free_winner_votes",
        },
        "policies": list(POLICIES),
        "headline_metrics": [
            "decision_regret",
            "correct_intervention_choice",
            "practical_reliability",
            "paired_regret_change_vs_synthetic_only",
            "negative_value_rate",
        ],
        "experiment_cluster_bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "paired_policy_resampling": True,
        },
        "stopping_rule": (
            "If balanced EB fails to improve paired mean regret over synthetic-only "
            "at each of budgets 25, 50, and 100 and has nonpositive median paired "
            "improvement across those budgets, stop fusion tuning and retain the "
            "negative result."
        ),
        "source_contract_sha256": {
            experiment_id: _file_sha256(root / TASK_PATHS[experiment_id])
            for experiment_id in DEVELOPMENT_IDS
        },
        "implementation_sha256": {
            "src/intervenebench/human_fallback.py": sha256_file(
                root / "src/intervenebench/human_fallback.py"
            ),
            "src/intervenebench/eb_fallback.py": sha256_file(
                root / "src/intervenebench/eb_fallback.py"
            ),
            "src/intervenebench/development_fallback.py": sha256_file(
                root / "src/intervenebench/development_fallback.py"
            ),
            "scripts/run_development_fallback.py": sha256_file(
                root / "scripts/run_development_fallback.py"
            ),
        },
        "output_path": str(DEFAULT_FALLBACK_PATH),
        "participant_rows_may_be_serialized": False,
        "paid_compute_authorized": False,
        "modal_execution_authorized": False,
        "claim_boundary": (
            "Development-only fallback method selection over nine revealed tasks; "
            "not prospective confirmation or a canonical benchmark result."
        ),
    }
    assert_blinded_payload(protocol)
    return protocol


def write_development_fallback_protocol(root: Path) -> Path:
    path = root / FALLBACK_PROTOCOL_PATH
    freeze_envelope(
        build_development_fallback_protocol(root), path, require_blinded=True
    )
    return path


def verify_development_fallback_protocol(root: Path) -> dict[str, Any]:
    protocol = verify_envelope(root / FALLBACK_PROTOCOL_PATH, require_blinded=True)
    if protocol != build_development_fallback_protocol(root):
        raise ValueError("development fallback protocol does not replay")
    return protocol


def _utility(task: Mapping[str, Any], raw_value: int) -> float:
    mapping = {
        int(option["raw_value"]): float(option["normalized_utility"])
        for option in task["response_options"]
    }
    if raw_value not in mapping:
        raise ValueError("development response lies outside the frozen utility scale")
    return mapping[raw_value]


def _integer(value: Any) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError("development response must be a nonmissing integer")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError("development response must be a finite integer")
    return int(number)


def _read_socsci_development(
    root: Path, *, experiment_id: str, task: Mapping[str, Any]
) -> list[FallbackObservation]:
    if experiment_id not in SOCSCI_IDS:
        raise PermissionError("SocSci fallback reader is restricted to revealed development")
    condition_key = (
        "condition_num"
        if "condition_num" in task["arms"][0]
        else "socsci_condition_num"
    )
    arm_by_condition = {
        int(arm[condition_key]): arm["arm_id"] for arm in task["arms"]
    }
    columns = [
        "study_id",
        "sample_id",
        "participant",
        "condition_num",
        "task_num",
        "response",
    ]
    tables = [
        pq.read_table(
            path,
            columns=columns,
            filters=[
                ("study_id", "=", experiment_id),
                ("task_num", "=", int(task["socsci210_task_num"])),
            ],
        )
        for path in sorted((root / SOCSCI_ROOT).glob("*.parquet"))
    ]
    if not tables:
        raise ValueError("SocSci development shards are unavailable")
    table = pa.concat_tables(tables)
    rows: list[FallbackObservation] = []
    seen: set[str] = set()
    for record in table.to_pylist():
        if record["response"] is None:
            continue
        condition = _integer(record["condition_num"])
        if condition not in arm_by_condition:
            raise ValueError("development condition is outside the action set")
        raw_value = _integer(record["response"])
        participant_id = f"{record['sample_id']}:{record['participant']}"
        if participant_id in seen:
            raise ValueError("development primary outcome has duplicate participants")
        seen.add(participant_id)
        rows.append(
            FallbackObservation(
                participant_id=participant_id,
                arm_id=arm_by_condition[condition],
                utility=_utility(task, raw_value),
                weight=1.0,
            )
        )
    return rows


def _convert_observations(rows: Sequence[Any]) -> list[FallbackObservation]:
    return [
        FallbackObservation(
            participant_id=str(row.participant_id),
            arm_id=str(row.arm_id),
            utility=float(row.utility),
            weight=float(row.weight),
        )
        for row in rows
    ]


def _winner_votes(
    *,
    experiment_id: str,
    arm_ids: Sequence[str],
    phase1_recommendation: Mapping[str, Any],
    discovery_diagnostics: Mapping[str, Any],
    prospective_recommendations: Mapping[str, Any],
) -> dict[str, int]:
    choices: list[str]
    if experiment_id == "jf46x":
        choices = [str(phase1_recommendation["selected_arm_id"])]
    elif experiment_id in DEVELOPMENT_IDS[1:6]:
        row = next(
            row
            for row in discovery_diagnostics["experiment_diagnostics"]
            if row["experiment_id"] == experiment_id
        )
        choices = [str(value) for value in row["model_choices"].values()]
    else:
        choices = [
            str(row["balanced_chosen_arm_id"])
            for row in prospective_recommendations["model_decisions"]
            if row["experiment_id"] == experiment_id
        ]
    if not choices or any(choice not in arm_ids for choice in choices):
        raise ValueError("outcome-free winner votes are invalid")
    counts = Counter(choices)
    return {arm_id: int(counts[arm_id]) for arm_id in arm_ids}


def _load_inputs(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = verify_development_evidence(
        root, root / DEFAULT_DEVELOPMENT_EVIDENCE_PATH
    )
    evidence_by_id = {row["experiment_id"]: row for row in evidence["tasks"]}
    phase1_score = verify_envelope(root / PHASE1_SCORE_PATH, require_blinded=False)
    phase1_recommendation = verify_envelope(
        root / PHASE1_RECOMMENDATION_PATH, require_blinded=True
    )
    discovery_score = verify_envelope(root / DISCOVERY_SCORE_PATH, require_blinded=False)
    discovery_diagnostics = verify_envelope(
        root / DISCOVERY_DIAGNOSTICS_PATH, require_blinded=True
    )
    prospective_score = verify_prospective_development_score(
        root, root / PROSPECTIVE_SCORE_PATH
    )
    prospective_recommendations = verify_envelope(
        root / PROSPECTIVE_RECOMMENDATIONS_PATH, require_blinded=True
    )
    portfolio_authorization = verify_development_reveal_authorization(root)
    prospective_protocol = verify_pre_reveal_protocol(root)
    verify_reveal_authorization(root)
    discovery_primary = {
        row["experiment_id"]: row
        for row in discovery_score["task_scores"]
        if row["model_id"] == DISCOVERY_MODEL_ID
    }
    prospective_primary = {
        experiment_id: task["models"][PROSPECTIVE_MODEL_ID]
        for experiment_id, task in prospective_score["tasks"].items()
    }
    tasks: dict[str, Any] = {}
    for index, experiment_id in enumerate(DEVELOPMENT_IDS):
        task = _read_object(root / TASK_PATHS[experiment_id])
        arm_ids = tuple(arm["arm_id"] for arm in task["arms"])
        control_arm_id = str(task["control_arm_id"])
        if experiment_id in SOCSCI_IDS:
            observations = _read_socsci_development(
                root, experiment_id=experiment_id, task=task
            )
        elif experiment_id in {"turagaS11", "wallaceS12"}:
            observations = _convert_observations(
                _read_external_observations(
                    root,
                    experiment_id=experiment_id,
                    task=task,
                    allowlist=portfolio_authorization["outcome_column_allowlist"][
                        experiment_id
                    ],
                )
            )
        elif experiment_id == "es4xw":
            observations = _convert_observations(
                _read_es4xw(root, task=task, protocol=prospective_protocol)
            )
        else:
            raise PermissionError("fallback input is outside revealed development")
        if experiment_id == "jf46x":
            synthetic_means = phase1_score["synthetic_arm_means"]
        elif experiment_id in discovery_primary:
            synthetic_means = discovery_primary[experiment_id]["synthetic_arm_means"]
        else:
            synthetic_means = prospective_primary[experiment_id][
                "synthetic_arm_means"
            ]
        if set(synthetic_means) != set(arm_ids):
            raise ValueError("primary synthetic action set drifted")
        tasks[experiment_id] = {
            "task": task,
            "arm_ids": arm_ids,
            "control_arm_id": control_arm_id,
            "observations": observations,
            "synthetic_means": {
                arm_id: float(synthetic_means[arm_id]) for arm_id in arm_ids
            },
            "winner_votes": _winner_votes(
                experiment_id=experiment_id,
                arm_ids=arm_ids,
                phase1_recommendation=phase1_recommendation,
                discovery_diagnostics=discovery_diagnostics,
                prospective_recommendations=prospective_recommendations,
            ),
            "human_effects": {
                key: float(value)
                for key, value in evidence_by_id[experiment_id][
                    "human_treatment_effects"
                ].items()
            },
            "synthetic_effects": {
                key: float(value)
                for key, value in evidence_by_id[experiment_id][
                    "synthetic_treatment_effects"
                ].items()
            },
            "seed": SEED + index * 100_000,
        }
    source_hashes = {
        "development_evidence_payload_sha256": payload_hash(evidence),
        "phase1_score_payload_sha256": payload_hash(phase1_score),
        "phase1_recommendation_payload_sha256": payload_hash(
            phase1_recommendation
        ),
        "discovery_score_payload_sha256": payload_hash(discovery_score),
        "discovery_diagnostics_payload_sha256": payload_hash(
            discovery_diagnostics
        ),
        "prospective_score_payload_sha256": payload_hash(prospective_score),
        "prospective_recommendations_payload_sha256": payload_hash(
            prospective_recommendations
        ),
    }
    return tasks, source_hashes


def _json_dataclass(value: Any) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(value), sort_keys=True, allow_nan=False))


def _aggregate(task_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    by_budget: dict[str, Any] = {}
    for budget_index, budget in enumerate(BUDGETS):
        by_budget[str(budget)] = {}
        for policy_index, policy in enumerate(POLICIES):
            rows = {
                experiment_id: task_results[experiment_id]["by_budget"][str(budget)][
                    policy
                ]
                for experiment_id in DEVELOPMENT_IDS
            }
            estimated = {
                experiment_id: row
                for experiment_id, row in rows.items()
                if row["status"] == "estimated"
            }
            if len(estimated) != len(DEVELOPMENT_IDS):
                by_budget[str(budget)][policy] = {
                    "status": "not_estimable_for_all_experiments",
                    "experiment_count": len(estimated),
                }
                continue
            regrets = {
                experiment_id: float(row["mean_regret"])
                for experiment_id, row in estimated.items()
            }
            synthetic_regrets = {
                experiment_id: float(
                    task_results[experiment_id]["by_budget"][str(budget)][
                        "synthetic_only"
                    ]["mean_regret"]
                )
                for experiment_id in DEVELOPMENT_IDS
            }
            mean_bootstrap = _json_dataclass(
                experiment_cluster_bootstrap(
                    regrets,
                    replicates=BOOTSTRAP_REPLICATES,
                    seed=BOOTSTRAP_SEED + budget_index * 100 + policy_index,
                    confidence_level=0.95,
                )
            )
            paired_bootstrap = _json_dataclass(
                paired_experiment_cluster_bootstrap(
                    regrets,
                    synthetic_regrets,
                    replicates=BOOTSTRAP_REPLICATES,
                    seed=BOOTSTRAP_SEED + 10_000 + budget_index * 100 + policy_index,
                    confidence_level=0.95,
                    lower_is_better=True,
                )
            )
            by_budget[str(budget)][policy] = {
                "status": "estimated",
                "experiment_count": len(estimated),
                "mean_regret": fmean(regrets.values()),
                "mean_exact_choice_rate": fmean(
                    float(row["exact_choice_rate"]) for row in estimated.values()
                ),
                "mean_practical_reliability_rate": fmean(
                    float(row["practical_reliability_rate"])
                    for row in estimated.values()
                ),
                "mean_negative_value_rate_vs_synthetic": fmean(
                    float(row["negative_value_rate_vs_synthetic"])
                    for row in estimated.values()
                ),
                "experiment_cluster_bootstrap": mean_bootstrap,
                "paired_regret_vs_synthetic_bootstrap": paired_bootstrap,
            }
    required_budgets = (25, 50, 100)
    improvements = {
        str(budget): -float(
            by_budget[str(budget)]["synthetic_plus_balanced_eb"][
                "paired_regret_vs_synthetic_bootstrap"
            ]["mean_difference"]
        )
        for budget in required_budgets
    }
    paired_experiment_improvements = []
    for budget in required_budgets:
        for experiment_id in DEVELOPMENT_IDS:
            candidate = task_results[experiment_id]["by_budget"][str(budget)][
                "synthetic_plus_balanced_eb"
            ]["mean_regret"]
            reference = task_results[experiment_id]["by_budget"][str(budget)][
                "synthetic_only"
            ]["mean_regret"]
            paired_experiment_improvements.append(float(reference) - float(candidate))
    stop = all(value <= 0.0 for value in improvements.values()) and median(
        paired_experiment_improvements
    ) <= 0.0
    return {
        "by_budget": by_budget,
        "balanced_eb_improvement_over_synthetic_at_required_budgets": improvements,
        "balanced_eb_median_paired_experiment_improvement": median(
            paired_experiment_improvements
        ),
        "fusion_tuning_stop_rule_triggered": stop,
        "fusion_tuning_decision": (
            "stop_and_preserve_negative_result"
            if stop
            else "eligible_for_confirmation_freeze_without_additional_tuning"
        ),
    }


def build_development_fallback_result(root: Path) -> dict[str, Any]:
    protocol = verify_development_fallback_protocol(root)
    inputs, source_hashes = _load_inputs(root)
    calibration = [
        EffectCalibrationTask(
            experiment_id=experiment_id,
            synthetic_effects=inputs[experiment_id]["synthetic_effects"],
            human_effects=inputs[experiment_id]["human_effects"],
        )
        for experiment_id in DEVELOPMENT_IDS
    ]
    task_results: dict[str, Any] = {}
    for experiment_id in DEVELOPMENT_IDS:
        task = inputs[experiment_id]
        prior = fit_effect_prior(
            calibration,
            excluded_experiment_id=experiment_id,
            minimum_variance=MINIMUM_PRIOR_VARIANCE,
        )
        fallback = evaluate_eb_human_fallback(
            task["observations"],
            arm_ids=task["arm_ids"],
            control_arm_id=task["control_arm_id"],
            synthetic_means=task["synthetic_means"],
            winner_votes=task["winner_votes"],
            budgets=BUDGETS,
            partitions=PARTITIONS,
            fold_count=FOLD_COUNT,
            seed=task["seed"],
            pseudocount=PSEUDOCOUNT,
            practical_tolerance=float(task["task"]["practical_regret_tolerance"]),
            effect_prior=prior,
        )
        counts = Counter(row.arm_id for row in task["observations"])
        task_results[experiment_id] = {
            "experiment_id": experiment_id,
            "arm_ids": list(task["arm_ids"]),
            "control_arm_id": task["control_arm_id"],
            "complete_case_count": len(task["observations"]),
            "complete_case_count_by_arm": {
                arm_id: counts[arm_id] for arm_id in task["arm_ids"]
            },
            "winner_votes": task["winner_votes"],
            "effect_prior": fallback["effect_prior"],
            "by_budget": fallback["by_budget"],
            "participant_rows_serialized": 0,
        }
    payload = {
        "schema_version": "development_fallback.v1",
        "status": "complete_common_nine_experiment_development_fallback",
        "development_only": True,
        "canonical_test_claim": False,
        "protocol_payload_sha256": payload_hash(protocol),
        "source_artifact_payload_hashes": source_hashes,
        "experiment_ids": list(DEVELOPMENT_IDS),
        "experiment_count": len(DEVELOPMENT_IDS),
        "tasks": task_results,
        "summary": _aggregate(task_results),
        "participant_rows_serialized": 0,
        "confirmation_outcomes_accessed": [],
        "modal_used": False,
        "paid_cost_usd": 0.0,
        "claim_boundary": protocol["claim_boundary"],
    }
    return json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))


def run_development_fallback(root: Path, output_path: Path) -> Path:
    freeze_envelope(
        build_development_fallback_result(root),
        output_path,
        require_blinded=False,
    )
    return output_path


def verify_development_fallback(root: Path, path: Path) -> dict[str, Any]:
    protocol = verify_development_fallback_protocol(root)
    result = verify_envelope(path, require_blinded=False)
    if (
        result.get("schema_version") != "development_fallback.v1"
        or result.get("protocol_payload_sha256") != payload_hash(protocol)
        or result.get("experiment_ids") != list(DEVELOPMENT_IDS)
        or result.get("participant_rows_serialized") != 0
        or result.get("confirmation_outcomes_accessed") != []
        or result.get("canonical_test_claim") is not False
    ):
        raise ValueError("development fallback artifact is invalid")
    for experiment_id, task in result["tasks"].items():
        if experiment_id in task["effect_prior"]["training_experiment_ids"]:
            raise ValueError("target experiment leaked into its EB prior")
    return result
