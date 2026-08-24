"""Post-reveal, aggregate-only audit of human-fallback failure patterns.

This module characterizes reproducible patterns that may explain why the frozen
limited-human policies failed. It does not identify a causal mechanism: the
stored artifacts contain task/policy aggregates, not replicate-level paired
decision transitions or participant rows.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

from .protocol import freeze_envelope, payload_hash, verify_envelope


DEFAULT_REPLICATION_AUDIT_PATH = Path(
    "artifacts/confirmation/confirmation_20260814_v1/"
    "fallback_replication_audit_v1.json"
)
DEFAULT_CONFIRMATION_SCORE_PATH = Path(
    "artifacts/confirmation/confirmation_20260814_v1/score_v1.json"
)
DEFAULT_MECHANISM_AUTHORIZATION_PATH = Path(
    "artifacts/confirmation/authorizations/"
    "fallback_failure_mechanism_20260815_v1.json"
)
DEFAULT_MECHANISM_AUDIT_PATH = Path(
    "artifacts/confirmation/confirmation_20260814_v1/"
    "fallback_failure_mechanism_v1.json"
)

EXPECTED_REPLICATION_AUDIT_FILE_SHA256 = (
    "53814b5578950bf0038387750bbd7a432afee2cfffd1b41de4de99707b88fcf1"
)
EXPECTED_REPLICATION_AUDIT_PAYLOAD_SHA256 = (
    "5fadef43f6bb5d35a0fbc338cfebaf5d490deda564125f618adfe6002f60d932"
)
EXPECTED_CONFIRMATION_SCORE_FILE_SHA256 = (
    "8562d148ce04bc44af1481858b94f5a43f62edf94120199b2156c8920a51c2ec"
)
EXPECTED_CONFIRMATION_SCORE_PAYLOAD_SHA256 = (
    "fa2acc4661f8397658178a1b4d53e7806b2a35acf032520e625ffdcb79aaf1a7"
)

EXPERIMENT_IDS = ("Blair1131", "KlarS44", "ShannonS2", "pb2rr", "z358z")
NONZERO_BUDGETS = (10, 25, 50, 100, 250)
REQUIRED_BUDGETS = (25, 50, 100)
EPSILON = 1e-12

_AUTHORITY = {
    "aggregate_human_outcome_access_authorized": True,
    "participant_row_access_authorized": False,
    "participant_row_serialization_authorized": False,
    "model_calls_authorized": False,
    "model_downloads_authorized": False,
    "modal_compute_authorized": False,
    "new_policy_authorized": False,
    "method_tuning_authorized": False,
    "causal_mechanism_claim_authorized": False,
    "recommendation_changes_authorized": False,
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
        "mechanism_module": Path(
            "src/intervenebench/fallback_failure_mechanism.py"
        ),
        "authorization_builder": Path(
            "scripts/build_fallback_failure_mechanism_authorization.py"
        ),
        "audit_builder": Path("scripts/build_fallback_failure_mechanism.py"),
    }
    return {name: _file_sha256(root / path) for name, path in paths.items()}


def _load_sources(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    replication_path = root / DEFAULT_REPLICATION_AUDIT_PATH
    score_path = root / DEFAULT_CONFIRMATION_SCORE_PATH
    if _file_sha256(replication_path) != EXPECTED_REPLICATION_AUDIT_FILE_SHA256:
        raise ValueError("fallback replication audit file hash drifted")
    if _file_sha256(score_path) != EXPECTED_CONFIRMATION_SCORE_FILE_SHA256:
        raise ValueError("confirmation score file hash drifted")
    replication = verify_envelope(replication_path)
    score = verify_envelope(score_path)
    if payload_hash(replication) != EXPECTED_REPLICATION_AUDIT_PAYLOAD_SHA256:
        raise ValueError("fallback replication audit payload hash drifted")
    if payload_hash(score) != EXPECTED_CONFIRMATION_SCORE_PAYLOAD_SHA256:
        raise ValueError("confirmation score payload hash drifted")
    _validate_sources(replication, score)
    return replication, score


def _validate_sources(
    replication: Mapping[str, Any], score: Mapping[str, Any]
) -> None:
    if (
        replication.get("schema_version")
        != "intervenebench.fallback_replication_audit.v1"
        or replication.get("status")
        != "complete_existing_aggregate_only_replication_audit_stop"
        or tuple(replication.get("confirmation_panel", {}).get("experiment_ids", ()))
        != EXPERIMENT_IDS
        or replication.get("confirmation_panel", {}).get("experiment_count") != 5
        or replication.get("participant_rows_accessed") != 0
        or replication.get("participant_rows_serialized") != 0
        or replication.get("model_calls_made") != 0
        or replication.get("method_tuning_performed") is not False
        or replication.get("automatic_next_stage") is not False
    ):
        raise PermissionError("fallback replication audit identity or safety drifted")
    if (
        score.get("schema_version") != "confirmation_score.v1"
        or score.get("status") != "complete_prospective_confirmation_scoring_stop"
        or score.get("participant_rows_serialized") != 0
        or score.get("model_calls_made") != 0
        or score.get("modal_compute_used") is not False
        or score.get("recommendations_changed_after_reveal") is not False
        or score.get("threshold_tuned_after_reveal") is not False
        or score.get("automatic_followup_authorized") is not False
    ):
        raise PermissionError("confirmation score identity or safety drifted")
    tasks = score.get("human_fallback", {}).get("normalized_tasks", {})
    if set(tasks) != set(EXPERIMENT_IDS):
        raise ValueError("confirmation fallback experiment support drifted")
    if any(task.get("participant_rows_serialized") != 0 for task in tasks.values()):
        raise PermissionError("confirmation task serialized participant rows")


def build_mechanism_audit_authorization(root: Path) -> dict[str, Any]:
    replication, score = _load_sources(root)
    return {
        "schema_version": "intervenebench.fallback_failure_mechanism_authorization.v1",
        "status": "authorized_existing_aggregate_only_failure_pattern_audit",
        "replication_audit_payload_sha256": payload_hash(replication),
        "confirmation_score_payload_sha256": payload_hash(score),
        "replication_audit_file_sha256": EXPECTED_REPLICATION_AUDIT_FILE_SHA256,
        "confirmation_score_file_sha256": EXPECTED_CONFIRMATION_SCORE_FILE_SHA256,
        "authorized_experiment_ids": list(EXPERIMENT_IDS),
        "authorized_budgets": list(NONZERO_BUDGETS),
        "implementation_file_sha256": _implementation_file_sha256(root),
        **_AUTHORITY,
    }


def validate_mechanism_audit_authorization(
    authorization: Mapping[str, Any], *, root: Path
) -> None:
    expected = build_mechanism_audit_authorization(root)
    if set(authorization) != set(expected):
        raise PermissionError("fallback mechanism authority expanded")
    for key, value in _AUTHORITY.items():
        if authorization.get(key) is not value:
            raise PermissionError("fallback mechanism authority expanded")
    if dict(authorization) != expected:
        raise PermissionError("fallback mechanism authorization binding drifted")


def _task_delta(
    task: Mapping[str, Any], *, budget: int, policy: str
) -> float:
    row = task["by_budget"][str(budget)][policy]
    if row.get("status") != "estimated":
        raise ValueError("required task-policy-budget cell is not estimated")
    return float(row["paired_mean_regret_change_vs_synthetic"])


def _task_patterns(score: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = score["human_fallback"]["normalized_tasks"]
    experiment_scores = score["experiment_scores"]
    rows: list[dict[str, Any]] = []
    for experiment_id in EXPERIMENT_IDS:
        task = tasks[experiment_id]
        primary = experiment_scores[experiment_id]["primary_score"]
        eb_deltas = [
            _task_delta(
                task, budget=budget, policy="synthetic_plus_balanced_eb"
            )
            for budget in REQUIRED_BUDGETS
        ]
        human_deltas = [
            _task_delta(task, budget=budget, policy="human_only_balanced")
            for budget in REQUIRED_BUDGETS
        ]
        eb_negative_rates = [
            float(
                task["by_budget"][str(budget)]["synthetic_plus_balanced_eb"][
                    "negative_value_rate_vs_synthetic"
                ]
            )
            for budget in REQUIRED_BUDGETS
        ]
        mean_eb = fmean(eb_deltas)
        classification = (
            "worsened"
            if mean_eb > EPSILON
            else "improved"
            if mean_eb < -EPSILON
            else "unchanged"
        )
        rows.append(
            {
                "experiment_id": experiment_id,
                "arm_count": len(primary["human_arm_means"]),
                "full_sample_primary_exact_choice": bool(primary["exact_choice"]),
                "full_sample_primary_decision_regret": float(
                    primary["decision_regret"]
                ),
                "crossfold_synthetic_mean_regret": float(
                    task["by_budget"]["25"]["synthetic_only"]["mean_regret"]
                ),
                "balanced_eb_deltas_by_budget": {
                    str(budget): value
                    for budget, value in zip(REQUIRED_BUDGETS, eb_deltas)
                },
                "human_only_deltas_by_budget": {
                    str(budget): value
                    for budget, value in zip(REQUIRED_BUDGETS, human_deltas)
                },
                "balanced_eb_mean_delta_required_budgets": mean_eb,
                "human_only_mean_delta_required_budgets": fmean(human_deltas),
                "balanced_eb_mean_negative_value_rate_required_budgets": fmean(
                    eb_negative_rates
                ),
                "balanced_eb_pattern": classification,
            }
        )
    return rows


def _harm_correction_asymmetry(
    score: Mapping[str, Any], task_patterns: list[Mapping[str, Any]]
) -> dict[str, Any]:
    tasks = score["human_fallback"]["normalized_tasks"]
    harms: list[float] = []
    corrections: list[float] = []
    unchanged = 0
    cells: list[dict[str, Any]] = []
    leave_one_out: dict[str, dict[str, float]] = {}
    all_loo_worse = True
    for budget in REQUIRED_BUDGETS:
        deltas = {
            experiment_id: _task_delta(
                task,
                budget=budget,
                policy="synthetic_plus_balanced_eb",
            )
            for experiment_id, task in tasks.items()
        }
        leave_one_out[str(budget)] = {}
        for omitted in EXPERIMENT_IDS:
            value = fmean(
                delta for experiment_id, delta in deltas.items() if experiment_id != omitted
            )
            leave_one_out[str(budget)][omitted] = value
            all_loo_worse = all_loo_worse and value > 0.0
        for experiment_id in EXPERIMENT_IDS:
            delta = deltas[experiment_id]
            if delta > EPSILON:
                harms.append(delta)
                direction = "worsened"
            elif delta < -EPSILON:
                corrections.append(-delta)
                direction = "improved"
            else:
                unchanged += 1
                direction = "unchanged"
            cells.append(
                {
                    "experiment_id": experiment_id,
                    "budget": budget,
                    "candidate_minus_synthetic_mean_regret": delta,
                    "direction": direction,
                }
            )
    mean_harm = fmean(harms)
    mean_correction = fmean(corrections)
    return {
        "policy": "synthetic_plus_balanced_eb",
        "required_budgets": list(REQUIRED_BUDGETS),
        "task_budget_cell_count": len(cells),
        "worsened_cell_count": len(harms),
        "improved_cell_count": len(corrections),
        "unchanged_cell_count": unchanged,
        "mean_harm_magnitude": mean_harm,
        "mean_correction_magnitude": mean_correction,
        "harm_to_correction_magnitude_ratio": mean_harm / mean_correction,
        "total_harm_magnitude": sum(harms),
        "total_correction_magnitude": sum(corrections),
        "all_required_budget_leave_one_task_out_means_worse": all_loo_worse,
        "leave_one_task_out_mean_deltas_by_budget": leave_one_out,
        "cells": cells,
        "interpretation": (
            "The EB rule made small recurring corrections on z358z, no net change "
            "on Blair1131 and pb2rr, and much larger harmful moves on KlarS44 and "
            "ShannonS2. Removing any one task leaves positive mean harm at every "
            "required budget."
        ),
    }


def _budget_attenuation(replication: Mapping[str, Any]) -> dict[str, Any]:
    table = replication["confirmation_normalized_results"]

    def row(policy: str) -> dict[str, Any]:
        harm_10 = float(table["10"][policy]["candidate_minus_synthetic_mean_regret"])
        harm_100 = float(
            table["100"][policy]["candidate_minus_synthetic_mean_regret"]
        )
        return {
            "candidate_minus_synthetic_regret_at_10": harm_10,
            "candidate_minus_synthetic_regret_at_100": harm_100,
            "regret_harm_reduction_10_to_100": (harm_10 - harm_100) / harm_10,
            "mean_exact_choice_change_at_10": float(
                table["10"][policy]["mean_exact_choice_rate"]
                - table["10"]["synthetic_only"]["mean_exact_choice_rate"]
            ),
            "mean_exact_choice_change_at_100": float(
                table["100"][policy]["mean_exact_choice_rate"]
                - table["100"]["synthetic_only"]["mean_exact_choice_rate"]
            ),
        }

    human = row("human_only_balanced")
    eb = row("synthetic_plus_balanced_eb")
    return {
        "human_only": human,
        "balanced_eb": eb,
        "harm_remains_positive_at_100_for_both": (
            human["candidate_minus_synthetic_regret_at_100"] > 0.0
            and eb["candidate_minus_synthetic_regret_at_100"] > 0.0
        ),
        "interpretation": (
            "Increasing the budget from 10 to 100 attenuated mean harm by roughly "
            "half, consistent with pilot noise mattering, but did not reverse the "
            "decision disadvantage. Exact-choice rates also remained below the "
            "synthetic-only reference."
        ),
    }


def _regularization_result(replication: Mapping[str, Any]) -> dict[str, Any]:
    table = replication["confirmation_normalized_results"]
    negative_rate_reductions: dict[str, float] = {}
    regret_harm_reductions: dict[str, float] = {}
    reverses = False
    for budget in REQUIRED_BUDGETS:
        human = table[str(budget)]["human_only_balanced"]
        eb = table[str(budget)]["synthetic_plus_balanced_eb"]
        human_negative = float(human["mean_negative_value_rate_vs_synthetic"])
        eb_negative = float(eb["mean_negative_value_rate_vs_synthetic"])
        human_harm = float(human["candidate_minus_synthetic_mean_regret"])
        eb_harm = float(eb["candidate_minus_synthetic_mean_regret"])
        negative_rate_reductions[str(budget)] = (
            human_negative - eb_negative
        ) / human_negative
        regret_harm_reductions[str(budget)] = (human_harm - eb_harm) / human_harm
        reverses = reverses or eb_harm <= 0.0
    return {
        "comparison": "balanced_eb_vs_human_only",
        "negative_value_rate_reduction_eb_vs_human_only_by_budget": (
            negative_rate_reductions
        ),
        "mean_regret_harm_reduction_eb_vs_human_only_by_budget": (
            regret_harm_reductions
        ),
        "regularization_reverses_mean_harm_at_any_required_budget": reverses,
        "interpretation": (
            "EB shrinkage reduced the frequency and magnitude of harmful pilot "
            "updates, but the remaining harmful moves were larger than its "
            "corrections, so aggregate regret stayed above synthetic-only."
        ),
    }


def _figure_data(
    replication: Mapping[str, Any], task_patterns: list[Mapping[str, Any]]
) -> dict[str, Any]:
    table = replication["confirmation_normalized_results"]
    policy_labels = {
        "human_only_balanced": "Humans only",
        "synthetic_plus_balanced_fixed10": "Balanced fixed-10",
        "synthetic_plus_balanced_eb": "Balanced EB",
        "synthetic_plus_hedged_eb": "Hedged EB",
    }
    curves: list[dict[str, Any]] = []
    for policy, label in policy_labels.items():
        curves.append(
            {
                "policy": policy,
                "label": label,
                "points": [
                    {
                        "budget": budget,
                        "mean_delta_regret": float(
                            table[str(budget)][policy][
                                "candidate_minus_synthetic_mean_regret"
                            ]
                        ),
                        "confidence_interval": list(
                            table[str(budget)][policy][
                                "paired_95pct_confidence_interval"
                            ]
                        ),
                        "experiment_count": int(
                            table[str(budget)][policy]["experiment_count"]
                        ),
                    }
                    for budget in NONZERO_BUDGETS
                ],
            }
        )
    heatmap = [
        {
            "experiment_id": row["experiment_id"],
            "values": row["balanced_eb_deltas_by_budget"],
        }
        for row in task_patterns
    ]
    return {
        "cost_regret_curves": curves,
        "eb_task_budget_heatmap": {
            "budgets": list(REQUIRED_BUDGETS),
            "rows": heatmap,
        },
    }


def build_mechanism_audit(
    root: Path, *, authorization: Mapping[str, Any]
) -> dict[str, Any]:
    validate_mechanism_audit_authorization(authorization, root=root)
    replication, score = _load_sources(root)
    task_patterns = _task_patterns(score)
    return {
        "schema_version": "intervenebench.fallback_failure_mechanism.v1",
        "status": "complete_post_reveal_aggregate_only_failure_pattern_audit_stop",
        "analysis_role": "post_reveal_exploratory_failure_pattern_audit",
        "causal_mechanism_identified": False,
        "evidence_scope": (
            "five prospective bounded-normalized confirmation experiments; "
            "task-policy-budget aggregate acquisition summaries only"
        ),
        "replication_audit_payload_sha256": payload_hash(replication),
        "confirmation_score_payload_sha256": payload_hash(score),
        "authorization_payload_sha256": payload_hash(authorization),
        "source_file_sha256": {
            "replication_audit": EXPECTED_REPLICATION_AUDIT_FILE_SHA256,
            "confirmation_score": EXPECTED_CONFIRMATION_SCORE_FILE_SHA256,
        },
        "implementation_file_sha256": _implementation_file_sha256(root),
        "experiment_count": len(EXPERIMENT_IDS),
        "experiment_ids": list(EXPERIMENT_IDS),
        "task_patterns": task_patterns,
        "eb_harm_correction_asymmetry": _harm_correction_asymmetry(
            score, task_patterns
        ),
        "budget_attenuation": _budget_attenuation(replication),
        "regularization_result": _regularization_result(replication),
        "allocation_result": replication["hedged_allocation_result"],
        "transition_accounting": {
            "gross_harmful_flip_rate_recoverable": False,
            "gross_corrective_flip_rate_recoverable": False,
            "net_exact_choice_change_recoverable": True,
            "negative_value_rate_recoverable": True,
            "reason": "replicate_level_transition_pairs_were_not_serialized",
            "consequence": (
                "Do not claim exact counts of correct-to-incorrect or "
                "incorrect-to-correct decision flips."
            ),
        },
        "failure_pattern_synthesis": {
            "supported": (
                "Pilot noise matters: larger budgets and EB shrinkage attenuated "
                "harm. They did not eliminate an asymmetric downside in which "
                "large harmful moves on two tasks outweighed small corrections "
                "on one task."
            ),
            "not_identified": (
                "A causal decomposition into sampling variance, model bias, arm "
                "count, effect size, or allocation error cannot be identified "
                "from five task-level aggregate summaries."
            ),
            "operational_implication": (
                "Fallback policies require explicit value-of-information testing "
                "and a safeguard against overriding a low-regret synthetic choice "
                "with noisy pilot estimates."
            ),
        },
        "figure_data": _figure_data(replication, task_patterns),
        "participant_rows_accessed": 0,
        "participant_rows_serialized": 0,
        "model_calls_made": 0,
        "model_downloads_made": 0,
        "modal_compute_used": False,
        "new_policy_created": False,
        "method_tuning_performed": False,
        "recommendations_changed": False,
        "automatic_next_stage": False,
    }


def freeze_mechanism_audit(
    root: Path,
    *,
    authorization: Mapping[str, Any],
    destination: Path | None = None,
) -> str:
    path = destination or root / DEFAULT_MECHANISM_AUDIT_PATH
    payload = build_mechanism_audit(root, authorization=authorization)
    return freeze_envelope(payload, path)
