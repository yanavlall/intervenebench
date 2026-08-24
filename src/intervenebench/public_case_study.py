"""Aggregate-only public evidence bundle for the prospective case study.

This module deliberately publishes only benchmark-level summaries. It never
serializes participant rows, experiment-level human scores, arm means, or
treatment effects. Release decisions are recomputed from those summaries when
the bundle is loaded; they are not trusted as prose in the artifact.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from .protocol import verify_envelope
from .release_decision import (
    BehavioralEvaluationSummary,
    ReleaseThresholds,
    evaluate_release_decision,
)


DEFAULT_PUBLIC_CASE_STUDY_PATH = Path(
    "data/public/confirmation_case_study_v1.json"
)
_SCORE_PATH = Path(
    "artifacts/confirmation/confirmation_20260814_v1/score_v1.json"
)
_VALUE_AUDIT_PATH = Path(
    "artifacts/confirmation/confirmation_20260814_v1/value_audit_v1.json"
)
_STRICT_PARSE_PATH = Path(
    "artifacts/confirmation/confirmation_20260814_v1/strict_parse_audit.json"
)
_AGGREGATION_PATH = Path(
    "artifacts/confirmation/confirmation_20260814_v1/aggregation_v1.json"
)
_PROGRAM_PATH = Path(
    "data/manifests/research/role_focused_evaluation_program_v1.json"
)

_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "participant_id",
        "participant_ids",
        "participant_row",
        "participant_rows",
        "experiment_scores",
        "human_arm_means",
        "human_treatment_effects",
        "human_outcomes",
        "tau_h",
    }
)


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _walk_keys(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            folded = str(key).casefold()
            if folded in _FORBIDDEN_PUBLIC_KEYS:
                found.append(str(key))
            found.extend(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_keys(child))
    return found


def _fallback_improved(score: Mapping[str, Any]) -> bool:
    aggregate = score["human_fallback"]["normalized_aggregate"]
    for budget, policies in aggregate.items():
        if int(budget) == 0:
            continue
        synthetic = policies["synthetic_only"]["mean_regret"]
        for policy_name, policy in policies.items():
            if policy_name.startswith("synthetic_plus"):
                if policy.get("status") == "estimated" and policy["mean_regret"] < synthetic:
                    return True
    return False


def _source_record(root: Path, relative: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    return {
        "path": relative.as_posix(),
        "file_sha256": _file_sha256(root / relative),
        "payload_sha256": sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }


def build_public_case_study_payload(root: Path) -> dict[str, Any]:
    """Derive the public summary from frozen aggregate artifacts only."""

    score = verify_envelope(root / _SCORE_PATH)
    audit = verify_envelope(root / _VALUE_AUDIT_PATH)
    strict = verify_envelope(root / _STRICT_PARSE_PATH)
    aggregation = verify_envelope(root / _AGGREGATION_PATH)
    program = json.loads((root / _PROGRAM_PATH).read_text(encoding="utf-8"))

    if audit["score_payload_sha256"] != sha256(
        json.dumps(
            score,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest():
        raise ValueError("value audit is not bound to the confirmation score")
    if strict["strict_parseable_call_count"] != aggregation["strict_output_count"]:
        raise ValueError("strict-parse and aggregation output counts disagree")
    if score["participant_rows_serialized"] != 0:
        raise ValueError("confirmation score serialized participant rows")

    case_study = program["evidence_case_study"]
    exact = audit["exact_choice_all_six"]
    finite = audit["normalized_finite_action_space_enumeration"]
    comparisons = audit["normalized_regret_comparisons"]
    uniform = comparisons["primary_vs_uniform_random_action"]["cluster_bootstrap"]
    control = comparisons["primary_vs_no_effect_control_tie"]["cluster_bootstrap"]
    classical = comparisons["primary_vs_frozen_classical_baseline"]["cluster_bootstrap"]
    reliability = audit["frozen_practical_reliability_all_six"]
    trust = score["trust_evaluation"]["exact_error_risk_coverage_all_six"]

    payload: dict[str, Any] = {
        "schema_version": "intervenebench.public_case_study.v1",
        "status": "aggregate_only_public_evidence",
        "title": "Prospective behavioral-simulator decision evaluation",
        "evidence_scope": {
            "development_experiment_count": case_study["development_experiment_count"],
            "prospective_confirmation_experiment_count": audit[
                "prospective_experiment_count"
            ],
            "normalized_confirmation_task_count": audit[
                "normalized_experiment_count"
            ],
        },
        "run_integrity": {
            "planned_model_outputs": strict["planned_call_count"],
            "schema_valid_model_outputs": strict["strict_parseable_call_count"],
            "unavailable_model_outputs": strict["strict_unparseable_call_count"],
            "model_experiment_recommendation_count": aggregation[
                "model_experiment_recommendation_count"
            ],
            "primary_recommendations_frozen_before_reveal": (
                score["recommendations_changed_after_reveal"] is False
            ),
            "trust_diagnostics_frozen_before_reveal": (
                score["diagnostics_changed_after_reveal"] is False
            ),
            "semantic_repairs_made": 0,
            "automatic_reruns_made": 0,
            "participant_rows_serialized": score["participant_rows_serialized"],
            "unavailable_model_task_cell": "Socrates x tcg8p",
        },
        "decision_evidence": {
            "exact_choice": {
                "count": exact["primary_exact_count"],
                "experiment_count": audit["prospective_experiment_count"],
                "uniform_random_tail_probability": exact[
                    "probability_uniform_exact_count_at_least_primary"
                ],
            },
            "normalized_regret": {
                "primary_mean": finite["observed_primary_mean_regret"],
                "primary_worst": finite["observed_primary_worst_regret"],
                "uniform_mean": uniform["reference_mean"],
                "uniform_random_action_tail_probability": finite[
                    "probability_uniform_mean_regret_at_most_observed"
                ],
                "primary_minus_uniform_mean": uniform["mean_difference"],
                "primary_minus_uniform_confidence_interval": uniform[
                    "difference_confidence_interval"
                ],
                "control_mean": control["reference_mean"],
                "primary_minus_control_confidence_interval": control[
                    "difference_confidence_interval"
                ],
                "classical_mean": classical["reference_mean"],
            },
            "practical_reliability": {
                "count": reliability["primary_reliable_count"],
                "experiment_count": audit["prospective_experiment_count"],
            },
            "trust_diagnostics": {
                "exact_choice_auroc": trust["exact_choice_ranking"]["auroc"],
                "aurc": trust["discrete_aurc"],
                "random_abstention_expected_aurc": trust[
                    "random_abstention_expected_aurc"
                ],
                "validated_threshold": score["trust_evaluation"][
                    "learned_threshold"
                ] is not None,
            },
            "limited_human_fallback": {
                "improved_over_synthetic_only": _fallback_improved(score),
                "budgets": [0, 10, 25, 50, 100, 250],
            },
        },
        "claim_boundary": {
            "supported": "limited research-stage candidate screening",
            "not_supported": [
                "autonomous intervention selection",
                "validated confidence-based abstention",
                "reliable small-sample human correction",
                "universal simulator trust",
            ],
            "panel_status": "small noncanonical prospective confirmation panel",
        },
        "privacy": {
            "contains_participant_rows": False,
            "contains_experiment_level_human_scores": False,
            "contains_human_arm_means": False,
            "contains_human_treatment_effects": False,
        },
        "provenance": {
            "source_artifacts": [
                _source_record(root, _SCORE_PATH, score),
                _source_record(root, _VALUE_AUDIT_PATH, audit),
                _source_record(root, _STRICT_PARSE_PATH, strict),
                _source_record(root, _AGGREGATION_PATH, aggregation),
            ],
            "role_program_path": _PROGRAM_PATH.as_posix(),
            "role_program_file_sha256": _file_sha256(root / _PROGRAM_PATH),
        },
    }
    violations = _walk_keys(payload)
    if violations:
        raise ValueError(f"public bundle contains forbidden detailed fields: {violations}")
    return payload


def _summary(payload: Mapping[str, Any]) -> BehavioralEvaluationSummary:
    scope = payload["evidence_scope"]
    integrity = payload["run_integrity"]
    evidence = payload["decision_evidence"]
    exact = evidence["exact_choice"]
    regret = evidence["normalized_regret"]
    reliable = evidence["practical_reliability"]
    trust = evidence["trust_diagnostics"]
    return BehavioralEvaluationSummary(
        prospective_experiment_count=scope["prospective_confirmation_experiment_count"],
        normalized_experiment_count=scope["normalized_confirmation_task_count"],
        exact_choice_count=exact["count"],
        exact_choice_random_tail_probability=exact["uniform_random_tail_probability"],
        practically_reliable_count=reliable["count"],
        mean_normalized_regret=regret["primary_mean"],
        worst_normalized_regret=regret["primary_worst"],
        uniform_mean_normalized_regret=regret["uniform_mean"],
        uniform_regret_tail_probability=regret[
            "uniform_random_action_tail_probability"
        ],
        control_mean_normalized_regret=regret["control_mean"],
        control_difference_interval=tuple(
            regret["primary_minus_control_confidence_interval"]
        ),
        classical_mean_normalized_regret=regret["classical_mean"],
        trust_ranking_better_than_random=(
            trust["aurc"] < trust["random_abstention_expected_aurc"]
        ),
        validated_trust_threshold=trust["validated_threshold"],
        human_fallback_improved=evidence["limited_human_fallback"][
            "improved_over_synthetic_only"
        ],
        schema_valid_output_count=integrity["schema_valid_model_outputs"],
        planned_output_count=integrity["planned_model_outputs"],
    )


def verify_public_case_study(path: Path) -> dict[str, Any]:
    payload = verify_envelope(path)
    if payload.get("schema_version") != "intervenebench.public_case_study.v1":
        raise ValueError("unsupported public case-study schema")
    violations = _walk_keys(payload)
    if violations:
        raise ValueError(f"public bundle contains forbidden detailed fields: {violations}")
    if payload.get("privacy") != {
        "contains_participant_rows": False,
        "contains_experiment_level_human_scores": False,
        "contains_human_arm_means": False,
        "contains_human_treatment_effects": False,
    }:
        raise ValueError("public privacy declaration is absent or invalid")
    return {
        "payload": payload,
        "release_decisions": evaluate_release_decision(
            _summary(payload), ReleaseThresholds()
        ),
    }


def render_public_case_study(report: Mapping[str, Any]) -> str:
    payload = report["payload"]
    decisions = report["release_decisions"]
    scope = payload["evidence_scope"]
    exact = payload["decision_evidence"]["exact_choice"]
    regret = payload["decision_evidence"]["normalized_regret"]
    integrity = payload["run_integrity"]

    labels = (
        ("Candidate screening", "candidate_screening"),
        ("Autonomous intervention selection", "autonomous_intervention_selection"),
        ("Confidence-based abstention", "confidence_based_abstention"),
        ("Small-sample human fallback", "small_sample_human_fallback"),
    )
    lines = [
        "InterveneBench prospective case study",
        "====================================",
        (
            f"Evidence: {scope['prospective_confirmation_experiment_count']} prospective "
            f"experiments ({scope['normalized_confirmation_task_count']} normalized tasks)"
        ),
        (
            f"Integrity: {integrity['schema_valid_model_outputs']}/"
            f"{integrity['planned_model_outputs']} schema-valid model outputs"
        ),
        (
            f"Exact human-best intervention: {exact['count']}/"
            f"{exact['experiment_count']}"
        ),
        (
            f"Mean normalized regret: {regret['primary_mean']:.4f} "
            f"(uniform action: {regret['uniform_mean']:.4f})"
        ),
        "",
        "Scoped release decisions",
        "------------------------",
    ]
    for label, key in labels:
        decision = decisions[key]
        lines.append(f"{label}: {decision['decision'].replace('_', ' ').upper()}")
        for reason in decision["reasons"]:
            lines.append(f"  - {reason}")
    lines.extend(
        [
            "",
            "Claim boundary",
            "--------------",
            f"Supported: {payload['claim_boundary']['supported']}",
            "Not supported: "
            + "; ".join(payload["claim_boundary"]["not_supported"]),
        ]
    )
    return "\n".join(lines) + "\n"
