from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/research/role_focused_evaluation_program_v1.json"


def test_role_focused_program_closes_corpus_expansion_and_retains_zero_authority() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "intervenebench.role_focused_program.v1"
    assert payload["status"] == "evaluation_product_build_frozen"
    assert payload["corpus_policy"] == {
        "open_ended_search_closed": True,
        "independent_replication_panel_required_for_completion": False,
        "existing_completion_queue_status": "closed_without_execution",
        "one_bounded_behavioral_extension": "wr7jg_schema_mapping_only",
    }
    assert payload["primary_artifact"] == (
        "behavioral_simulator_evaluation_and_release_gating_system"
    )
    assert payload["evidence_case_study"]["prospective_experiment_count"] == 6
    assert payload["evidence_case_study"]["recommendations_frozen_before_outcomes"] is True
    assert payload["deliverables"] == [
        "unified_evaluation_lifecycle",
        "scoped_release_gate",
        "aggregate_only_public_demo",
        "model_version_regression_suite",
        "results_explorer",
        "technical_report_and_reproducible_repository",
    ]
    assert payload["authority"] == {
        "authorized_spend_usd": 0,
        "model_calls_authorized": False,
        "human_outcome_reveal_authorized": False,
        "participant_row_access_authorized": False,
        "schema_only_mapping_authorized": False,
    }


def test_governing_documents_name_the_role_focused_program_as_active() -> None:
    required = {
        ROOT / "PROJECT_SPEC.md": "The active scope is the role-focused evaluation product",
        ROOT / "PHASE_1.md": "The current milestone is the role-focused evaluation product",
        ROOT / "docs/InterveneBench_Project_Plan.tex": (
            "The active program is now a role-focused evaluation product"
        ),
        ROOT / "docs/audits/data_audit.md": (
            "2026-08-20: role-focused evaluation-product pivot"
        ),
    }
    for path, marker in required.items():
        assert marker in path.read_text(encoding="utf-8")


def test_historical_replication_stage_grants_no_current_completion_authority() -> None:
    project_spec = " ".join(
        (ROOT / "PROJECT_SPEC.md").read_text(encoding="utf-8").split()
    )
    phase_1 = " ".join(
        (ROOT / "PHASE_1.md").read_text(encoding="utf-8").split()
    )
    assert "not current execution authority or a completion dependency" in project_spec
    assert "no longer required for project completion" in phase_1
