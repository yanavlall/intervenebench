"""Aggregate-only replication audit for the frozen human-fallback policies.

The fallback policies were developed on nine revealed experiments and then
frozen before the six-experiment confirmation reveal. Five confirmation tasks
use bounded-normalized utility and are directly comparable with development;
the raw-dollar task remains separate. This module never reads participant rows
or runs models. It verifies and summarizes the already-frozen aggregate files.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from statistics import fmean, median
from typing import Any, Mapping

from .experiment_statistics import paired_experiment_cluster_bootstrap
from .protocol import freeze_envelope, payload_hash, verify_envelope


DEFAULT_DEVELOPMENT_PROTOCOL_PATH = Path(
    "data/manifests/research/development_fallback_protocol_v1.json"
)
DEFAULT_DEVELOPMENT_RESULT_PATH = Path(
    "artifacts/development/development_fallback_v1.json"
)
DEFAULT_CONFIRMATION_PROTOCOL_PATH = Path(
    "data/manifests/research/confirmation_scoring_protocol_v1.json"
)
DEFAULT_CONFIRMATION_SCORE_PATH = Path(
    "artifacts/confirmation/confirmation_20260814_v1/score_v1.json"
)
DEFAULT_FALLBACK_REPLICATION_AUTHORIZATION_PATH = Path(
    "artifacts/confirmation/authorizations/"
    "fallback_replication_audit_20260815_v1.json"
)
DEFAULT_FALLBACK_REPLICATION_AUDIT_PATH = Path(
    "artifacts/confirmation/confirmation_20260814_v1/"
    "fallback_replication_audit_v1.json"
)

EXPECTED_SOURCES = {
    "development_protocol": {
        "path": DEFAULT_DEVELOPMENT_PROTOCOL_PATH,
        "file_sha256": "4b99a5bbe06b931a8ff96628dd70e2f9c72a26ce18a799ffeae7ef09380452e4",
        "payload_sha256": "817fe0aaaefd7a22e0cbdfd85de7ed91b2cd3f65500f0c9ebdc07effff778767",
    },
    "development_result": {
        "path": DEFAULT_DEVELOPMENT_RESULT_PATH,
        "file_sha256": "e833e2dcf5008be0a6c9437353c4fff98e8ac904d16b99f0f523f0d68f2598e0",
        "payload_sha256": "a8f477df11e1a53a705c7c352781f31c0b68b5a9ed1f323b47ea92b1015e0db7",
    },
    "confirmation_protocol": {
        "path": DEFAULT_CONFIRMATION_PROTOCOL_PATH,
        "file_sha256": "45ced71e04ee1167e45c3fc9017dffa596ddb848978ac2f5e59a0b10c74ed5da",
        "payload_sha256": "a5d42c118518ce3275e1ddaf47512f7f181d55b96a13472d83a23f54752a1d6a",
    },
    "confirmation_score": {
        "path": DEFAULT_CONFIRMATION_SCORE_PATH,
        "file_sha256": "8562d148ce04bc44af1481858b94f5a43f62edf94120199b2156c8920a51c2ec",
        "payload_sha256": "fa2acc4661f8397658178a1b4d53e7806b2a35acf032520e625ffdcb79aaf1a7",
    },
}

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
ALL_CONFIRMATION_IDS = (
    "tcg8p",
    "pb2rr",
    "z358z",
    "ShannonS2",
    "Blair1131",
    "KlarS44",
)
NORMALIZED_CONFIRMATION_IDS = (
    "Blair1131",
    "KlarS44",
    "ShannonS2",
    "pb2rr",
    "z358z",
)
BUDGETS = (0, 10, 25, 50, 100, 250)
REQUIRED_REPLICATION_BUDGETS = (25, 50, 100)
POLICIES = (
    "synthetic_only",
    "human_only_balanced",
    "synthetic_plus_balanced_fixed10",
    "synthetic_plus_hedged_fixed10",
    "synthetic_plus_balanced_eb",
    "synthetic_plus_hedged_eb",
)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2026081502

_AUTHORITY = {
    "aggregate_human_outcome_access_authorized": True,
    "participant_row_access_authorized": False,
    "participant_row_serialization_authorized": False,
    "model_calls_authorized": False,
    "model_downloads_authorized": False,
    "modal_compute_authorized": False,
    "new_policy_authorized": False,
    "method_tuning_authorized": False,
    "recommendation_changes_authorized": False,
    "trust_threshold_tuning_authorized": False,
    "automatic_next_stage_authorized": False,
}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_file_sha256(root: Path) -> dict[str, str]:
    files = {
        "audit_module": Path("src/intervenebench/fallback_replication_audit.py"),
        "authorization_builder": Path(
            "scripts/build_fallback_replication_audit_authorization.py"
        ),
        "audit_builder": Path("scripts/build_fallback_replication_audit.py"),
    }
    return {name: _file_sha256(root / path) for name, path in files.items()}


def _load_sources(root: Path) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for name, expected in EXPECTED_SOURCES.items():
        path = root / expected["path"]
        if _file_sha256(path) != expected["file_sha256"]:
            raise ValueError(f"{name} file hash drifted")
        payload = verify_envelope(path)
        if payload_hash(payload) != expected["payload_sha256"]:
            raise ValueError(f"{name} payload hash drifted")
        loaded[name] = payload
    _validate_sources(loaded)
    return loaded


def _validate_sources(sources: Mapping[str, Mapping[str, Any]]) -> None:
    development_protocol = sources["development_protocol"]
    development_result = sources["development_result"]
    confirmation_protocol = sources["confirmation_protocol"]
    confirmation_score = sources["confirmation_score"]

    if (
        development_protocol.get("schema_version")
        != "development_fallback_protocol.v1"
        or development_protocol.get("status")
        != "development_only_method_frozen_before_unified_policy_result"
        or tuple(development_protocol.get("development_experiment_ids", ()))
        != DEVELOPMENT_IDS
        or tuple(development_protocol.get("confirmation_experiment_ids", ()))
        != ALL_CONFIRMATION_IDS
        or development_protocol.get("development_experiment_count") != 9
    ):
        raise ValueError("development fallback protocol identity drifted")
    if (
        tuple(development_protocol.get("budgets", ())) != BUDGETS
        or tuple(development_protocol.get("policies", ())) != POLICIES
        or development_protocol.get("partitions") != 20
        or development_protocol.get("fold_count") != 10
        or development_protocol.get("nested_without_replacement_pilot_prefixes")
        is not True
        or development_protocol.get("same_folds_draws_and_seeds_across_policies")
        is not True
        or development_protocol.get("target_prior_fit")
        != "leave_one_experiment_out"
    ):
        raise ValueError("development fallback design drifted")
    if (
        development_protocol.get("confirmation_outcome_access_authorized") is not False
        or development_protocol.get("participant_rows_may_be_serialized") is not False
        or development_protocol.get("modal_execution_authorized") is not False
        or development_protocol.get("paid_compute_authorized") is not False
    ):
        raise PermissionError("development protocol safety boundary drifted")

    if (
        development_result.get("schema_version") != "development_fallback.v1"
        or development_result.get("status")
        != "complete_common_nine_experiment_development_fallback"
        or tuple(development_result.get("experiment_ids", ())) != DEVELOPMENT_IDS
        or development_result.get("experiment_count") != 9
        or set(development_result.get("tasks", {})) != set(DEVELOPMENT_IDS)
    ):
        raise ValueError("development fallback result identity drifted")
    if (
        development_result.get("participant_rows_serialized") != 0
        or development_result.get("confirmation_outcomes_accessed") != []
        or development_result.get("modal_used") is not False
        or development_result.get("paid_cost_usd") != 0.0
    ):
        raise PermissionError("development fallback result safety boundary drifted")
    if any(
        task.get("participant_rows_serialized") != 0
        for task in development_result["tasks"].values()
    ):
        raise PermissionError("development task serialized participant rows")

    if (
        confirmation_protocol.get("schema_version")
        != "confirmation_scoring_protocol.v1"
        or confirmation_protocol.get("status")
        != "frozen_before_confirmation_outcome_access"
        or tuple(confirmation_protocol.get("experiment_ids", ()))
        != ALL_CONFIRMATION_IDS
    ):
        raise ValueError("confirmation scoring protocol identity drifted")
    fallback = confirmation_protocol.get("human_fallback", {})
    if (
        tuple(fallback.get("budgets", ())) != BUDGETS
        or fallback.get("partitions") != 20
        or fallback.get("fold_count") != 10
        or fallback.get("pilot_evaluation_people_disjoint") is not True
        or fallback.get("sampling_without_replacement") is not True
        or tuple(
            fallback.get("effect_prior_frozen_on_all_development_experiments", {}).get(
                "training_experiment_ids", ()
            )
        )
        != DEVELOPMENT_IDS
    ):
        raise ValueError("confirmation fallback design or prior drifted")
    if (
        confirmation_protocol.get("participant_rows_may_be_serialized") is not False
        or confirmation_protocol.get("model_calls_authorized") is not False
        or confirmation_protocol.get("modal_compute_authorized") is not False
        or confirmation_protocol.get("recommendations_may_change") is not False
        or confirmation_protocol.get("threshold_tuning") != "forbidden"
        or fallback.get("participant_rows_serialized") != 0
    ):
        raise PermissionError("confirmation protocol safety boundary drifted")

    if (
        confirmation_score.get("schema_version") != "confirmation_score.v1"
        or confirmation_score.get("status")
        != "complete_prospective_confirmation_scoring_stop"
        or tuple(confirmation_score.get("experiment_ids", ()))
        != ALL_CONFIRMATION_IDS
        or tuple(confirmation_score.get("confirmation_outcomes_accessed", ()))
        != ALL_CONFIRMATION_IDS
    ):
        raise ValueError("confirmation score identity drifted")
    score_fallback = confirmation_score.get("human_fallback", {})
    if (
        set(score_fallback.get("normalized_tasks", {}))
        != set(NORMALIZED_CONFIRMATION_IDS)
        or score_fallback.get("participant_rows_serialized") != 0
        or confirmation_score.get("participant_rows_serialized") != 0
        or confirmation_score.get("model_calls_made") != 0
        or confirmation_score.get("modal_compute_used") is not False
        or confirmation_score.get("recommendations_changed_after_reveal") is not False
        or confirmation_score.get("threshold_tuned_after_reveal") is not False
        or confirmation_score.get("automatic_followup_authorized") is not False
    ):
        raise PermissionError("confirmation score safety or support drifted")
    if any(
        task.get("participant_rows_serialized") != 0
        for task in score_fallback["normalized_tasks"].values()
    ):
        raise PermissionError("confirmation task serialized participant rows")

    if set(DEVELOPMENT_IDS) & set(ALL_CONFIRMATION_IDS):
        raise ValueError("development and confirmation experiment IDs overlap")


def build_fallback_replication_audit_authorization(root: Path) -> dict[str, Any]:
    sources = _load_sources(root)
    return {
        "schema_version": "intervenebench.fallback_replication_audit_authorization.v1",
        "status": "authorized_existing_aggregate_only_replication_audit",
        "source_payload_sha256": {
            name: payload_hash(payload) for name, payload in sources.items()
        },
        "source_file_sha256": {
            name: expected["file_sha256"]
            for name, expected in EXPECTED_SOURCES.items()
        },
        "authorized_development_experiment_ids": list(DEVELOPMENT_IDS),
        "authorized_confirmation_experiment_ids": list(NORMALIZED_CONFIRMATION_IDS),
        "raw_unit_secondary_experiment_ids": ["tcg8p"],
        "implementation_file_sha256": _implementation_file_sha256(root),
        **_AUTHORITY,
    }


def validate_fallback_replication_audit_authorization(
    authorization: Mapping[str, Any], *, root: Path
) -> None:
    expected = build_fallback_replication_audit_authorization(root)
    if set(authorization) != set(expected):
        raise PermissionError("fallback replication audit authority expanded")
    for key, value in _AUTHORITY.items():
        if authorization.get(key) is not value:
            raise PermissionError("fallback replication audit authority expanded")
    if dict(authorization) != expected:
        raise PermissionError("fallback replication audit authorization binding drifted")


def _estimated_task_values(
    tasks: Mapping[str, Mapping[str, Any]], budget: int, policy: str
) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for experiment_id, task in tasks.items():
        policy_result = task["by_budget"][str(budget)][policy]
        if policy_result.get("status") == "estimated":
            rows[experiment_id] = policy_result
    return rows


def _mean_negative_value_rate(
    tasks: Mapping[str, Mapping[str, Any]], budget: int, policy: str
) -> float | None:
    rows = _estimated_task_values(tasks, budget, policy)
    if not rows:
        return None
    return fmean(float(row["negative_value_rate_vs_synthetic"]) for row in rows.values())


def _panel_table(
    *,
    tasks: Mapping[str, Mapping[str, Any]],
    aggregates: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for budget in BUDGETS:
        budget_rows: dict[str, dict[str, Any]] = {}
        aggregate_budget = aggregates[str(budget)]
        for policy in POLICIES:
            aggregate = aggregate_budget[policy]
            if aggregate.get("status") != "estimated":
                budget_rows[policy] = {
                    "status": aggregate.get("status", "not_estimable"),
                    "experiment_count": int(aggregate.get("experiment_count", 0)),
                }
                continue
            paired = aggregate["paired_regret_vs_synthetic_bootstrap"]
            budget_rows[policy] = {
                "status": "estimated",
                "experiment_count": int(aggregate["experiment_count"]),
                "mean_regret": float(aggregate["mean_regret"]),
                "candidate_minus_synthetic_mean_regret": float(
                    paired["mean_difference"]
                ),
                "paired_95pct_confidence_interval": [
                    float(value) for value in paired["difference_confidence_interval"]
                ],
                "mean_exact_choice_rate": float(aggregate["mean_exact_choice_rate"]),
                "mean_practical_reliability_rate": float(
                    aggregate["mean_practical_reliability_rate"]
                ),
                "mean_negative_value_rate_vs_synthetic": _mean_negative_value_rate(
                    tasks, budget, policy
                ),
            }
        table[str(budget)] = budget_rows
    return table


def _confirmation_balanced_eb_replication(
    development: Mapping[str, Any], confirmation_table: Mapping[str, Any]
) -> dict[str, Any]:
    confirmation_rows: dict[str, Any] = {}
    paired_improvements: list[float] = []
    all_directionally_worse = True
    all_statistically_worse = True
    confirmation_tasks = confirmation_table["_tasks"]
    for budget in REQUIRED_REPLICATION_BUDGETS:
        policy = confirmation_table[str(budget)]["synthetic_plus_balanced_eb"]
        difference = float(policy["candidate_minus_synthetic_mean_regret"])
        interval = policy["paired_95pct_confidence_interval"]
        confirmation_rows[str(budget)] = {
            "candidate_minus_synthetic_mean_regret": difference,
            "candidate_improvement_over_synthetic": -difference,
            "paired_95pct_confidence_interval": interval,
            "statistically_resolved_as_worse_at_95pct": interval[0] > 0.0,
        }
        all_directionally_worse = all_directionally_worse and difference > 0.0
        all_statistically_worse = all_statistically_worse and interval[0] > 0.0
        for task in confirmation_tasks.values():
            by_budget = task["by_budget"][str(budget)]
            paired_improvements.append(
                float(by_budget["synthetic_only"]["mean_regret"])
                - float(by_budget["synthetic_plus_balanced_eb"]["mean_regret"])
            )
    development_summary = development["summary"]
    return {
        "candidate_policy": "synthetic_plus_balanced_eb",
        "reference_policy": "synthetic_only",
        "required_budgets": list(REQUIRED_REPLICATION_BUDGETS),
        "development_stop_rule_triggered": development_summary[
            "fusion_tuning_stop_rule_triggered"
        ],
        "development_decision": development_summary["fusion_tuning_decision"],
        "development_median_paired_experiment_improvement": float(
            development_summary["balanced_eb_median_paired_experiment_improvement"]
        ),
        "confirmation_by_budget": confirmation_rows,
        "confirmation_median_task_budget_improvement": median(paired_improvements),
        "confirmation_directionally_replicated": all_directionally_worse,
        "confirmation_statistically_resolved_at_every_required_budget": (
            all_statistically_worse
        ),
        "decision": (
            "stop_tuning_and_preserve_replicated_negative_result"
            if all_directionally_worse
            else "negative_result_did_not_directionally_replicate"
        ),
        "interpretation": (
            "The frozen EB fusion rule again had higher mean regret at every "
            "required budget. The direction replicated prospectively, while the "
            "five-experiment confirmation intervals do not resolve harm at every "
            "individual budget."
        ),
    }


def _human_only_confirmation_result(confirmation_table: Mapping[str, Any]) -> dict[str, Any]:
    worse_budgets: list[int] = []
    all_point_worse = True
    rows: dict[str, Any] = {}
    for budget in BUDGETS[1:]:
        row = confirmation_table[str(budget)]["human_only_balanced"]
        difference = float(row["candidate_minus_synthetic_mean_regret"])
        interval = row["paired_95pct_confidence_interval"]
        if interval[0] > 0.0:
            worse_budgets.append(budget)
        all_point_worse = all_point_worse and difference > 0.0
        rows[str(budget)] = {
            "candidate_minus_synthetic_mean_regret": difference,
            "paired_95pct_confidence_interval": interval,
            "mean_negative_value_rate_vs_synthetic": row[
                "mean_negative_value_rate_vs_synthetic"
            ],
        }
    return {
        "policy": "human_only_balanced",
        "all_nonzero_budget_point_estimates_worse_than_synthetic": all_point_worse,
        "budgets_with_95pct_paired_ci_entirely_worse_than_synthetic": worse_budgets,
        "by_budget": rows,
        "interpretation": (
            "In this panel, balanced pilots of 10 to 100 people produced higher "
            "mean regret than the frozen synthetic-only decision with paired "
            "experiment-bootstrap intervals entirely above zero. Budget 250 had "
            "only four feasible normalized tasks and was not statistically resolved."
        ),
    }


def _hedged_allocation_result(confirmation_table: Mapping[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    any_better = False
    for family, balanced, hedged in (
        (
            "fixed10",
            "synthetic_plus_balanced_fixed10",
            "synthetic_plus_hedged_fixed10",
        ),
        ("eb", "synthetic_plus_balanced_eb", "synthetic_plus_hedged_eb"),
    ):
        rows: dict[str, float] = {}
        for budget in BUDGETS[1:]:
            difference = (
                float(confirmation_table[str(budget)][hedged]["mean_regret"])
                - float(confirmation_table[str(budget)][balanced]["mean_regret"])
            )
            rows[str(budget)] = difference
            any_better = any_better or difference < 0.0
        comparisons[family] = rows
    return {
        "comparison_metric": "hedged_mean_regret_minus_matching_balanced_mean_regret",
        "by_fusion_family_and_budget": comparisons,
        "hedged_beats_matching_balanced_policy_at_any_confirmation_budget": any_better,
        "decision": "drop_intelligent_label_preserve_as_negative_ablation",
    }


def _secondary_pooled_summary(
    development_tasks: Mapping[str, Mapping[str, Any]],
    confirmation_tasks: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    pooled: dict[str, Any] = {}
    for index, budget in enumerate(BUDGETS[1:]):
        candidate: dict[str, float] = {}
        reference: dict[str, float] = {}
        for tasks in (development_tasks, confirmation_tasks):
            for experiment_id, task in tasks.items():
                by_budget = task["by_budget"][str(budget)]
                candidate_row = by_budget["synthetic_plus_balanced_eb"]
                reference_row = by_budget["synthetic_only"]
                if (
                    candidate_row.get("status") == "estimated"
                    and reference_row.get("status") == "estimated"
                ):
                    candidate[experiment_id] = float(candidate_row["mean_regret"])
                    reference[experiment_id] = float(reference_row["mean_regret"])
        result = paired_experiment_cluster_bootstrap(
            candidate,
            reference,
            replicates=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED + index,
            confidence_level=0.95,
            lower_is_better=True,
        )
        row = asdict(result)
        row["candidate_minus_synthetic_mean_regret"] = row.pop("mean_difference")
        row["difference_confidence_interval"] = list(
            row["difference_confidence_interval"]
        )
        pooled[str(budget)] = row
    return {
        "role": "secondary_descriptive_only_panels_reported_separately_first",
        "warning": (
            "The development policies were selected using development results; "
            "this pooled summary is not a fourteen-experiment prospective estimate."
        ),
        "experiment_is_resampling_unit": True,
        "development_experiment_count": len(development_tasks),
        "confirmation_experiment_count": len(confirmation_tasks),
        "balanced_eb_by_budget": pooled,
    }


def build_fallback_replication_audit(
    root: Path, *, authorization: Mapping[str, Any]
) -> dict[str, Any]:
    validate_fallback_replication_audit_authorization(authorization, root=root)
    sources = _load_sources(root)
    development = sources["development_result"]
    confirmation_score = sources["confirmation_score"]
    confirmation_fallback = confirmation_score["human_fallback"]
    development_table = _panel_table(
        tasks=development["tasks"], aggregates=development["summary"]["by_budget"]
    )
    confirmation_table = _panel_table(
        tasks=confirmation_fallback["normalized_tasks"],
        aggregates=confirmation_fallback["normalized_aggregate"],
    )
    confirmation_table_for_result = dict(confirmation_table)
    confirmation_table_for_result["_tasks"] = confirmation_fallback[
        "normalized_tasks"
    ]

    return {
        "schema_version": "intervenebench.fallback_replication_audit.v1",
        "status": "complete_existing_aggregate_only_replication_audit_stop",
        "analysis_role": (
            "post_reveal_aggregate_only_replication_audit_of_a_pre_reveal_fallback_policy"
        ),
        "source_payload_sha256": {
            name: payload_hash(payload) for name, payload in sources.items()
        },
        "source_file_sha256": {
            name: expected["file_sha256"]
            for name, expected in EXPECTED_SOURCES.items()
        },
        "authorization_payload_sha256": payload_hash(authorization),
        "implementation_file_sha256": _implementation_file_sha256(root),
        "development_panel": {
            "experiment_count": len(DEVELOPMENT_IDS),
            "experiment_ids": list(DEVELOPMENT_IDS),
            "role": "method_development_and_stopping_rule_discovery",
            "target_prior_fit": "leave_one_experiment_out",
            "was_prospective_for_fallback_policy": False,
        },
        "confirmation_panel": {
            "experiment_count": len(NORMALIZED_CONFIRMATION_IDS),
            "experiment_ids": list(NORMALIZED_CONFIRMATION_IDS),
            "role": "prospective_confirmation_of_unchanged_fallback_policies",
            "effect_prior_fit": "all_nine_development_experiments_only",
            "was_prospective_for_fallback_policy": True,
            "budget_250_experiment_count": 4,
            "budget_250_exclusion": "Blair1131_predeclared_infeasible",
        },
        "panels_are_experiment_disjoint": True,
        "design_compatibility": {
            "budgets": list(BUDGETS),
            "policies": list(POLICIES),
            "partitions": 20,
            "fold_count": 10,
            "pilot_and_evaluation_people_disjoint": True,
            "sampling_without_replacement": True,
            "pilot_prefixes_nested_within_policy": True,
            "same_folds_draws_and_seeds_across_policies": True,
            "unit_of_generalization_and_uncertainty": "experiment",
        },
        "development_normalized_results": development_table,
        "confirmation_normalized_results": confirmation_table,
        "balanced_eb_replication_result": _confirmation_balanced_eb_replication(
            development, confirmation_table_for_result
        ),
        "human_only_confirmation_result": _human_only_confirmation_result(
            confirmation_table
        ),
        "hedged_allocation_result": _hedged_allocation_result(confirmation_table),
        "secondary_pooled_descriptive_summary": _secondary_pooled_summary(
            development["tasks"], confirmation_fallback["normalized_tasks"]
        ),
        "tcg8p_raw_unit_secondary": {
            "experiment_id": "tcg8p",
            "unit": "source_USD",
            "pooled_with_normalized_tasks": False,
            "direction": (
                "human_only_mean_regret_exceeded_synthetic_only_at_every_nonzero_budget"
            ),
            "reason_separate": (
                "Its uncapped raw-dollar outcome is not commensurate with bounded "
                "normalized decision regret or the development EB prior."
            ),
        },
        "claim_boundary": {
            "supported_claim": (
                "Across these five prospective bounded-normalized confirmation "
                "experiments, small pilots and the tested frozen fusion/allocation "
                "rules did not reduce mean decision regret relative to the frozen "
                "synthetic-only policy at any tested nonzero budget."
            ),
            "strongest_resolved_subclaim": (
                "Balanced human-only pilots had significantly higher mean regret "
                "than synthetic-only at budgets 10, 25, 50, and 100 under paired "
                "experiment-bootstrap intervals."
            ),
            "forbidden_claims": [
                "humans are useless",
                "synthetic evidence is generally superior to human experiments",
                "the result is a canonical benchmark estimate",
                "the five confirmation experiments establish universal fallback policy",
                "budget 250 is a five-experiment comparison",
                "models, calls, folds, arms, or participants increase experiment N",
            ],
        },
        "participant_rows_accessed": 0,
        "participant_rows_serialized": 0,
        "model_calls_made": 0,
        "model_downloads_made": 0,
        "modal_compute_used": False,
        "new_policy_created": False,
        "method_tuning_performed": False,
        "recommendations_changed": False,
        "trust_threshold_tuned": False,
        "automatic_next_stage": False,
    }


def freeze_fallback_replication_audit(
    root: Path,
    *,
    authorization: Mapping[str, Any],
    destination: Path | None = None,
) -> str:
    path = destination or root / DEFAULT_FALLBACK_REPLICATION_AUDIT_PATH
    payload = build_fallback_replication_audit(root, authorization=authorization)
    return freeze_envelope(payload, path)
