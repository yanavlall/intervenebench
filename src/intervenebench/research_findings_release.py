"""Aggregate-only synthesis of the completed InterveneBench evidence.

The synthesis is intentionally narrower than the underlying score artifacts. It
contains benchmark-level findings, release decisions, claim boundaries, and
hashes—but no participant rows or experiment-level human results.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from .protocol import payload_hash, verify_envelope
from .public_case_study import verify_public_case_study


DEFAULT_RESEARCH_FINDINGS_PATH = Path("data/public/research_findings_v1.json")
DEFAULT_RESEARCH_FINDINGS_REPORT_PATH = Path(
    "docs/reports/research_findings_v1.md"
)

_SOURCES = {
    "public_confirmation": {
        "path": Path("data/public/confirmation_case_study_v1.json"),
        "file_sha256": "50d12c04266d30ea482ad7fc798464dc5d02477cfa356232f44f25aeffe30e7a",
        "payload_sha256": "e52dbb86f7df8350d70c683ea62c65a40c850ada7e53749cbe3aba19452602b1",
    },
    "development_evidence": {
        "path": Path("artifacts/development/development_evidence_v1.json"),
        "file_sha256": "6e899dc4af135920ed92437f1adfb12a9033e186a72b69df777148af35c83fb4",
        "payload_sha256": "e2411b32d922dc6af0869e0b5dbf4d0b8fbc80b8f7487b34137d2395dcffd0e8",
    },
    "fallback_replication": {
        "path": Path(
            "artifacts/confirmation/confirmation_20260814_v1/"
            "fallback_replication_audit_v1.json"
        ),
        "file_sha256": "53814b5578950bf0038387750bbd7a432afee2cfffd1b41de4de99707b88fcf1",
        "payload_sha256": "5fadef43f6bb5d35a0fbc338cfebaf5d490deda564125f618adfe6002f60d932",
    },
    "fallback_mechanism": {
        "path": Path(
            "artifacts/confirmation/confirmation_20260814_v1/"
            "fallback_failure_mechanism_v1.json"
        ),
        "file_sha256": "6bcc9205c9115059c08c81d3a58466a939172fd5ea03d461e6da658f160f2065",
        "payload_sha256": "aad756e0b47aff0470b4f94f4223a892fe6b7a5e365a8eaaaf649c7307cf5b55",
    },
    "cross_family": {
        "path": Path(
            "artifacts/cross_family_target/"
            "target_run_20260815_v1_continuation_seedfix_v2/"
            "retrospective_cross_family_score_v2.json"
        ),
        "file_sha256": "33a760f0ab6253cba918ea93511a1447f8efb8a0f9f9ed360d5820a240bb3fe5",
        "payload_sha256": "0539f471b9dcff3eb97ee0b553c59322e847681fefaa55c4c03fdfe3d5b8e86d",
    },
}

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
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_forbidden(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in _FORBIDDEN_PUBLIC_KEYS:
                found.append(str(key))
            found.extend(_walk_forbidden(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_forbidden(child))
    return found


def _load_sources(root: Path) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for name, expected in _SOURCES.items():
        path = root / expected["path"]
        if _file_sha256(path) != expected["file_sha256"]:
            raise ValueError(f"{name} file hash drifted")
        payload = verify_envelope(path)
        if payload_hash(payload) != expected["payload_sha256"]:
            raise ValueError(f"{name} payload hash drifted")
        loaded[name] = payload
    return loaded


def _validate_sources(sources: Mapping[str, Mapping[str, Any]]) -> None:
    development = sources["development_evidence"]
    replication = sources["fallback_replication"]
    mechanism = sources["fallback_mechanism"]
    cross_family = sources["cross_family"]

    if (
        development.get("schema_version") != "development_evidence.v1"
        or development.get("experiment_count") != 9
        or development.get("development_only") is not True
        or development.get("canonical_test_claim") is not False
        or development.get("participant_rows_read") != 0
        or development.get("participant_rows_serialized") != 0
    ):
        raise ValueError("development evidence identity or safety boundary drifted")
    if (
        replication.get("schema_version")
        != "intervenebench.fallback_replication_audit.v1"
        or replication.get("participant_rows_accessed") != 0
        or replication.get("participant_rows_serialized") != 0
        or replication.get("model_calls_made") != 0
        or replication.get("method_tuning_performed") is not False
    ):
        raise ValueError("fallback replication identity or safety boundary drifted")
    if (
        mechanism.get("schema_version")
        != "intervenebench.fallback_failure_mechanism.v1"
        or mechanism.get("participant_rows_accessed") != 0
        or mechanism.get("participant_rows_serialized") != 0
        or mechanism.get("model_calls_made") != 0
        or mechanism.get("causal_mechanism_identified") is not False
    ):
        raise ValueError("fallback mechanism identity or safety boundary drifted")
    if (
        cross_family.get("schema_version")
        != "intervenebench.cross_family_retrospective_score.v2"
        or cross_family.get("independent_experiment_n") != 5
        or cross_family.get("primary_policy_changed") is not False
        or cross_family.get("trust_threshold_tuned") is not False
        or cross_family.get("participant_rows_accessed") != 0
        or cross_family.get("participant_rows_serialized") != 0
    ):
        raise ValueError("cross-family identity or safety boundary drifted")


def build_research_findings_payload(root: Path) -> dict[str, Any]:
    sources = _load_sources(root)
    _validate_sources(sources)

    public_report = verify_public_case_study(root / _SOURCES["public_confirmation"]["path"])
    public = public_report["payload"]
    development = sources["development_evidence"]
    replication = sources["fallback_replication"]
    mechanism = sources["fallback_mechanism"]
    cross_family = sources["cross_family"]

    decision = public["decision_evidence"]
    exact = decision["exact_choice"]
    regret = decision["normalized_regret"]
    trust = decision["trust_diagnostics"]
    reliable = decision["practical_reliability"]
    primary = development["primary_summary"]
    eb = replication["balanced_eb_replication_result"]
    human_only = replication["human_only_confirmation_result"]
    mechanism_asymmetry = mechanism["eb_harm_correction_asymmetry"]

    payload: dict[str, Any] = {
        "schema_version": "intervenebench.research_findings.v1",
        "status": "complete_aggregate_only_research_synthesis",
        "thesis": (
            "Evaluate simulated humans by the interventions they choose, the utility "
            "lost when they are wrong, and whether failure can be detected before "
            "human outcomes are revealed."
        ),
        "evidence": {
            "prospective_confirmation": {
                "evidence_tier": "noncanonical_prospective_confirmation",
                "experiment_count": exact["experiment_count"],
                "normalized_experiment_count": public["evidence_scope"][
                    "normalized_confirmation_task_count"
                ],
                "exact_choice_count": exact["count"],
                "exact_choice_uniform_tail_probability": exact[
                    "uniform_random_tail_probability"
                ],
                "practically_reliable_count": reliable["count"],
                "mean_normalized_regret": regret["primary_mean"],
                "worst_normalized_regret": regret["primary_worst"],
                "uniform_mean_normalized_regret": regret["uniform_mean"],
                "uniform_mean_regret_tail_probability": regret[
                    "uniform_random_action_tail_probability"
                ],
                "primary_minus_uniform_mean_regret": regret[
                    "primary_minus_uniform_mean"
                ],
            },
            "prospective_trust_diagnostics": {
                "exact_choice_auroc": trust["exact_choice_auroc"],
                "aurc": trust["aurc"],
                "random_abstention_expected_aurc": trust[
                    "random_abstention_expected_aurc"
                ],
                "ranking_better_than_random_abstention": (
                    trust["aurc"] < trust["random_abstention_expected_aurc"]
                ),
                "validated_threshold": trust["validated_threshold"],
                "conclusion": "not_validated_do_not_deploy",
            },
            "prospective_human_fallback": {
                "experiment_count": replication["confirmation_panel"][
                    "experiment_count"
                ],
                "any_tested_policy_improved_at_any_nonzero_budget": False,
                "human_only_all_point_estimates_worse": human_only[
                    "all_nonzero_budget_point_estimates_worse_than_synthetic"
                ],
                "human_only_budgets_resolved_worse_at_95pct": human_only[
                    "budgets_with_95pct_paired_ci_entirely_worse_than_synthetic"
                ],
                "balanced_eb_directionally_replicated": eb[
                    "confirmation_directionally_replicated"
                ],
                "balanced_eb_decision": eb["decision"],
                "hedged_allocation_beat_balanced": replication[
                    "hedged_allocation_result"
                ]["hedged_beats_matching_balanced_policy_at_any_confirmation_budget"],
                "harm_to_correction_magnitude_ratio": mechanism_asymmetry[
                    "harm_to_correction_magnitude_ratio"
                ],
                "harmful_task_budget_cells": mechanism_asymmetry[
                    "worsened_cell_count"
                ],
                "corrective_task_budget_cells": mechanism_asymmetry[
                    "improved_cell_count"
                ],
                "unchanged_task_budget_cells": mechanism_asymmetry[
                    "unchanged_cell_count"
                ],
                "causal_mechanism_identified": False,
            },
            "retrospective_cross_family": {
                "evidence_tier": "post_reveal_architecture_robustness",
                "experiment_count": cross_family["independent_experiment_n"],
                "primary_exact_choice_rate": cross_family[
                    "primary_exact_choice_rate"
                ],
                "mistral_exact_choice_rate": cross_family[
                    "candidate_exact_choice_rate"
                ],
                "primary_mean_regret": cross_family[
                    "primary_mean_decision_regret"
                ],
                "mistral_mean_regret": cross_family[
                    "candidate_mean_decision_regret"
                ],
                "primary_treatment_effect_mae": cross_family[
                    "primary_mean_treatment_effect_mae"
                ],
                "mistral_treatment_effect_mae": cross_family[
                    "candidate_mean_treatment_effect_mae"
                ],
                "model_disagreement_predictive_signal": cross_family[
                    "diagnostic_evaluation"
                ]["winner_disagreement"]["positive_signal_under_frozen_direction"],
                "adds_prospective_experiment_n": False,
            },
            "development_context": {
                "evidence_tier": "mixed_development_only",
                "experiment_count": primary["experiment_count"],
                "exact_choice_count": primary["correct_intervention_count"],
                "practically_reliable_count": primary[
                    "practically_reliable_count"
                ],
                "mean_normalized_regret": primary["mean_decision_regret"],
                "worst_normalized_regret": primary["worst_case_decision_regret"],
                "canonical_test_claim": False,
            },
        },
        "contributions": [
            "source-verified decision-task corpus construction with explicit utilities",
            "prospective freeze-reveal evaluation of intervention choice and regret",
            "outcome-free trust diagnostics preserved when they fail",
            "disjoint limited-human fallback evaluation with negative-value analysis",
            "hash-bound aggregate artifacts and deterministic replay",
        ],
        "release_decisions": public_report["release_decisions"],
        "scientific_conclusion": (
            "The simulator showed a preliminary low-regret intervention-selection "
            "signal on a small prospective panel, while exact choice remained "
            "chance-compatible. The tested trust ranking and limited-human fallback "
            "did not work and must not be deployed."
        ),
        "claim_boundary": {
            "supported": [
                "limited research-stage candidate screening",
                "prospective low-regret signal on this six-experiment panel",
                "prospectively replicated negative result for the tested fallback family",
            ],
            "not_supported": [
                "autonomous intervention selection",
                "universal simulator trust",
                "validated confidence-based abstention",
                "reliable small-sample human correction",
                "a canonical benchmark or calibrated trust model",
            ],
        },
        "privacy": {
            "participant_rows_accessed": 0,
            "participant_rows_serialized": 0,
            "contains_experiment_level_human_scores": False,
            "contains_human_arm_means": False,
            "contains_human_treatment_effects": False,
        },
        "provenance": {
            name: {
                "path": expected["path"].as_posix(),
                "file_sha256": expected["file_sha256"],
                "payload_sha256": expected["payload_sha256"],
            }
            for name, expected in _SOURCES.items()
        },
    }
    violations = _walk_forbidden(payload)
    if violations:
        raise ValueError(f"research findings contain forbidden detailed fields: {violations}")
    return payload


def verify_research_findings(path: Path) -> dict[str, Any]:
    payload = verify_envelope(path)
    if payload.get("schema_version") != "intervenebench.research_findings.v1":
        raise ValueError("unsupported research-findings schema")
    violations = _walk_forbidden(payload)
    if violations:
        raise ValueError(f"research findings contain forbidden detailed fields: {violations}")
    if payload.get("privacy") != {
        "participant_rows_accessed": 0,
        "participant_rows_serialized": 0,
        "contains_experiment_level_human_scores": False,
        "contains_human_arm_means": False,
        "contains_human_treatment_effects": False,
    }:
        raise ValueError("research-findings privacy declaration is invalid")
    return payload


def render_research_findings_markdown(payload: Mapping[str, Any]) -> str:
    prospective = payload["evidence"]["prospective_confirmation"]
    trust = payload["evidence"]["prospective_trust_diagnostics"]
    fallback = payload["evidence"]["prospective_human_fallback"]
    cross = payload["evidence"]["retrospective_cross_family"]
    development = payload["evidence"]["development_context"]
    decisions = payload["release_decisions"]
    lines = [
        "# InterveneBench: Authoritative Research Findings",
        "",
        "## Bottom line",
        "",
        payload["scientific_conclusion"],
        "",
        "## Evidence at a glance",
        "",
        "| Evidence tier | Exact choice | Mean regret | Interpretation |",
        "|---|---:|---:|---|",
        (
            f"| Prospective confirmation | {prospective['exact_choice_count']}/"
            f"{prospective['experiment_count']} | {prospective['mean_normalized_regret']:.4f} "
            "(5 normalized tasks) | Low regret; exact choice remains chance-compatible |"
        ),
        (
            f"| Development context | {development['exact_choice_count']}/"
            f"{development['experiment_count']} | {development['mean_normalized_regret']:.4f} "
            "| Method development only; not a held-out benchmark |"
        ),
        (
            f"| Mistral retrospective comparator | {int(cross['mistral_exact_choice_rate'] * cross['experiment_count'])}/"
            f"{cross['experiment_count']} | {cross['mistral_mean_regret']:.4f} "
            "| Architecture sensitivity only; adds no prospective N |"
        ),
        "",
        "## 1. Decision value",
        "",
        (
            f"The frozen primary simulator chose {prospective['exact_choice_count']} of "
            f"{prospective['experiment_count']} exact sample winners. Uniform random "
            f"choice would meet or exceed that count with probability "
            f"`{prospective['exact_choice_uniform_tail_probability']:.3f}`."
        ),
        "",
        (
            f"The more decision-relevant signal was regret. Across five normalized "
            f"tasks, mean regret was `{prospective['mean_normalized_regret']:.4f}` "
            f"versus `{prospective['uniform_mean_normalized_regret']:.4f}` for exact "
            f"uniform action. The finite action-space tail probability was "
            f"`{prospective['uniform_mean_regret_tail_probability']:.4f}`. All six "
            "frozen choices satisfied their predeclared practical-reliability rule."
        ),
        "",
        "## 2. Trust diagnostics failed",
        "",
        (
            f"The prespecified confidence ranking was worse than random abstention: "
            f"AURC `{trust['aurc']:.3f}` versus `{trust['random_abstention_expected_aurc']:.3f}`. "
            f"Exact-choice AUROC was `{trust['exact_choice_auroc']:.3f}`. No threshold "
            "was fitted after reveal, and no abstention policy is authorized."
        ),
        "",
        "## 3. Limited-human fallback also failed",
        "",
        (
            "Every tested nonzero-budget policy had higher point regret than the frozen "
            "synthetic-only decision. Human-only pilots were resolved as worse at the "
            f"95% level for budgets {fallback['human_only_budgets_resolved_worse_at_95pct']}. "
            "The development-fitted balanced empirical-Bayes rule repeated the same "
            "negative direction at budgets 25, 50, and 100."
        ),
        "",
        (
            f"Across the 15 balanced-EB task×budget cells used for failure analysis, "
            f"there were {fallback['harmful_task_budget_cells']} harmful, "
            f"{fallback['corrective_task_budget_cells']} corrective, and "
            f"{fallback['unchanged_task_budget_cells']} unchanged cells. The average "
            f"harm was `{fallback['harm_to_correction_magnitude_ratio']:.1f}×` the "
            "average correction. This is an exploratory pattern, not a causal mechanism."
        ),
        "",
        "## 4. Cross-family robustness did not rescue trust",
        "",
        (
            f"On five previously revealed tasks, both the original primary and Mistral "
            f"selected {int(cross['primary_exact_choice_rate'] * cross['experiment_count'])}/"
            f"{cross['experiment_count']} exact winners. Mistral improved treatment-effect "
            f"MAE (`{cross['mistral_treatment_effect_mae']:.4f}` vs "
            f"`{cross['primary_treatment_effect_mae']:.4f}`) but slightly worsened regret "
            f"(`{cross['mistral_mean_regret']:.4f}` vs `{cross['primary_mean_regret']:.4f}`). "
            "Model disagreement did not predict which choice was wrong."
        ),
        "",
        "## Scoped release decision",
        "",
        "| Scope | Decision |",
        "|---|---|",
    ]
    labels = (
        ("Candidate screening", "candidate_screening"),
        ("Autonomous intervention selection", "autonomous_intervention_selection"),
        ("Confidence-based abstention", "confidence_based_abstention"),
        ("Small-sample human fallback", "small_sample_human_fallback"),
    )
    for label, key in labels:
        lines.append(
            f"| {label} | {decisions[key]['decision'].replace('_', ' ').upper()} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "Supported:",
            "",
            *[f"- {item}" for item in payload["claim_boundary"]["supported"]],
            "",
            "Not supported:",
            "",
            *[f"- {item}" for item in payload["claim_boundary"]["not_supported"]],
            "",
            "## Reproducibility",
            "",
            "Run:",
            "",
            "```bash",
            "PYTHONPATH=src python3 -m intervenebench.public_cli verify --root .",
            "```",
            "",
            (
                "The portable verification path needs no local run artifacts, makes "
                "no model calls, and reads no participant rows. Maintainers can add "
                "`--deep-replay` to rebuild the synthesis from restricted aggregate "
                "provenance."
            ),
            "",
        ]
    )
    return "\n".join(lines)
