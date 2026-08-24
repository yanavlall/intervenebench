"""Minimal command-line entry points for Phase 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .phase1 import (
    render_smoke_report,
    replay_score,
    run_local_ollama_simulator,
    score_frozen_validation_recommendation,
)
from .portfolio_pilot import run_local_portfolio_pilot, verify_portfolio_run
from .portfolio_development import (
    score_development_portfolio,
    verify_development_score,
)
from .development_analysis import (
    build_development_analysis,
    verify_development_analysis,
)
from .research_program import verify_research_program
from .research_progress import evaluate_contract_progress, load_contract_batch
from .simulator_suite import verify_development_scope
from .modal_freeze import verify_modal_preflight_freeze
from .balanced_forced_choice import verify_full_action_freeze
from .prospective_development_score import (
    DEFAULT_SCORE_PATH as PROSPECTIVE_DEVELOPMENT_SCORE_PATH,
    verify_prospective_development_score,
)
from .development_evidence import (
    DEFAULT_DEVELOPMENT_EVIDENCE_PATH,
    verify_development_evidence,
)
from .development_fallback import (
    DEFAULT_FALLBACK_PATH,
    verify_development_fallback,
)
from .classical_development import (
    DEFAULT_CLASSICAL_DEVELOPMENT_PATH,
    DEFAULT_CLASSICAL_MODEL_PATH,
    verify_classical_development,
    verify_classical_model,
)
from .confirmation_preparation import (
    DEFAULT_CONFIRMATION_PREPARATION_PATH,
    verify_confirmation_preparation,
)
from .confirmation_calls import (
    DEFAULT_CONFIRMATION_CALL_PLAN_PATH,
    verify_confirmation_call_plan,
)
from .confirmation_execution import (
    DEFAULT_CONFIRMATION_EXECUTION_FREEZE_PATH,
    verify_confirmation_execution_freeze,
)
from .public_case_study import (
    DEFAULT_PUBLIC_CASE_STUDY_PATH,
    render_public_case_study,
    verify_public_case_study,
)
from .model_regression import (
    ModelVersionRegressionThresholds,
    compare_model_versions,
    load_model_version_evaluation,
)
from .evaluation_lifecycle import (
    evaluate_confirmation_lifecycle,
    render_confirmation_lifecycle,
)


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/raw/socsci210/048481111a4425ed83dc0eacf15f8431f252b21a/data"


def main() -> None:
    parser = argparse.ArgumentParser(prog="intervenebench")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("simulate-phase1")
    commands.add_parser("score-phase1")
    commands.add_parser("replay-phase1")
    commands.add_parser("report-phase1")
    commands.add_parser("simulate-portfolio-pilot")
    commands.add_parser("verify-portfolio-pilot")
    commands.add_parser("score-portfolio-development")
    commands.add_parser("verify-portfolio-development")
    commands.add_parser("analyze-portfolio-development")
    commands.add_parser("verify-portfolio-analysis")
    commands.add_parser("verify-research-program")
    commands.add_parser("verify-contract-progress")
    commands.add_parser("verify-simulator-development-plan")
    commands.add_parser("verify-modal-preflight-freeze")
    commands.add_parser("verify-balanced-full-action")
    commands.add_parser("verify-prospective-development")
    commands.add_parser("verify-development-evidence")
    commands.add_parser("verify-development-fallback")
    commands.add_parser("verify-classical-development")
    commands.add_parser("verify-classical-model")
    commands.add_parser("verify-confirmation-preparation")
    commands.add_parser("verify-confirmation-call-plan")
    commands.add_parser("verify-confirmation-execution-freeze")
    commands.add_parser("public-case-study")
    regression = commands.add_parser("compare-model-versions")
    regression.add_argument("--candidate", type=Path, required=True)
    regression.add_argument("--reference", type=Path, required=True)
    regression.add_argument("--bootstrap-replicates", type=int, default=10000)
    regression.add_argument("--bootstrap-seed", type=int, default=2026081401)
    commands.add_parser("evaluation-status")
    args = parser.parse_args()

    contracts = ROOT / "data/manifests/contracts"
    splits = ROOT / "data/manifests/splits"
    artifacts = ROOT / "artifacts/phase1"
    bundle = contracts / "jf46x_blinded_bundle.json"
    task = contracts / "jf46x_decision_task.json"
    split = splits / "phase1_split.json"
    raw = artifacts / "jf46x_ollama_outputs.json"
    recommendation = artifacts / "jf46x_recommendation.json"
    score = artifacts / "jf46x_score.json"

    if args.command == "evaluation-status":
        print(
            render_confirmation_lifecycle(
                evaluate_confirmation_lifecycle(ROOT)
            ),
            end="",
        )
    elif args.command == "compare-model-versions":
        print(
            json.dumps(
                compare_model_versions(
                    load_model_version_evaluation(args.candidate),
                    load_model_version_evaluation(args.reference),
                    thresholds=ModelVersionRegressionThresholds(),
                    bootstrap_replicates=args.bootstrap_replicates,
                    bootstrap_seed=args.bootstrap_seed,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "public-case-study":
        print(
            render_public_case_study(
                verify_public_case_study(ROOT / DEFAULT_PUBLIC_CASE_STUDY_PATH)
            ),
            end="",
        )
    elif args.command == "verify-confirmation-execution-freeze":
        print(
            verify_confirmation_execution_freeze(
                ROOT, ROOT / DEFAULT_CONFIRMATION_EXECUTION_FREEZE_PATH
            )
        )
    elif args.command == "verify-confirmation-call-plan":
        print(
            verify_confirmation_call_plan(
                ROOT, ROOT / DEFAULT_CONFIRMATION_CALL_PLAN_PATH
            )
        )
    elif args.command == "verify-confirmation-preparation":
        print(
            verify_confirmation_preparation(
                ROOT, ROOT / DEFAULT_CONFIRMATION_PREPARATION_PATH
            )
        )
    elif args.command == "verify-classical-model":
        print(
            verify_classical_model(
                ROOT, ROOT / DEFAULT_CLASSICAL_MODEL_PATH
            )
        )
    elif args.command == "verify-classical-development":
        print(
            verify_classical_development(
                ROOT, ROOT / DEFAULT_CLASSICAL_DEVELOPMENT_PATH
            )
        )
    elif args.command == "verify-development-fallback":
        print(
            verify_development_fallback(
                ROOT, ROOT / DEFAULT_FALLBACK_PATH
            )
        )
    elif args.command == "verify-development-evidence":
        print(
            verify_development_evidence(
                ROOT, ROOT / DEFAULT_DEVELOPMENT_EVIDENCE_PATH
            )
        )
    elif args.command == "verify-prospective-development":
        print(
            verify_prospective_development_score(
                ROOT, ROOT / PROSPECTIVE_DEVELOPMENT_SCORE_PATH
            )
        )
    elif args.command == "verify-balanced-full-action":
        print(
            verify_full_action_freeze(
                ROOT,
                freeze_path=(
                    ROOT / "configs/simulators/balanced_full_action_v1.json"
                ),
                plan_path=(
                    ROOT
                    / "data/manifests/simulators/balanced_full_action_plan_v1.json"
                ),
            )
        )
    elif args.command == "verify-modal-preflight-freeze":
        print(
            verify_modal_preflight_freeze(
                ROOT,
                freeze_path=(
                    ROOT
                    / "configs/simulators/modal_discovery_preflight_v2.json"
                ),
                call_plan_path=(
                    ROOT
                    / "data/manifests/simulators/modal_preflight_call_plan_v1.json"
                ),
            )
        )
    elif args.command == "verify-simulator-development-plan":
        print(
            verify_development_scope(
                ROOT,
                scope_path=(
                    ROOT
                    / "data/manifests/benchmark/simulator_development_scope.json"
                ),
                config_path=ROOT / "configs/simulators/development_v1.json",
            )
        )
    elif args.command == "verify-contract-progress":
        rows = load_contract_batch(
            ROOT / "data/manifests/audits/depth_first_contract_batches.csv"
        )
        print(evaluate_contract_progress(rows))
    elif args.command == "verify-research-program":
        print(verify_research_program(ROOT))
    elif args.command == "analyze-portfolio-development":
        print(build_development_analysis(ROOT))
    elif args.command == "verify-portfolio-analysis":
        print(verify_development_analysis(ROOT))
    elif args.command == "score-portfolio-development":
        print(
            score_development_portfolio(
                ROOT,
                output_path=(
                    ROOT / "artifacts/portfolio_pilot/development_score_v2.json"
                ),
            )
        )
    elif args.command == "verify-portfolio-development":
        print(
            verify_development_score(
                ROOT,
                ROOT / "artifacts/portfolio_pilot/development_score_v2.json",
            )
        )
    elif args.command == "simulate-portfolio-pilot":
        print(
            run_local_portfolio_pilot(
                ROOT,
                artifact_dir=(
                    ROOT
                    / "artifacts/portfolio_pilot/local_llama3_2_3b_manual"
                ),
            )
        )
    elif args.command == "verify-portfolio-pilot":
        print(
            verify_portfolio_run(
                ROOT,
                ROOT
                / "artifacts/portfolio_pilot/local_llama3_2_3b_20260813_v2/run_manifest.json",
            )
        )
    elif args.command == "simulate-phase1":
        print(
            run_local_ollama_simulator(
                bundle_path=bundle,
                split_path=split,
                decision_task_path=task,
                raw_output_path=raw,
                recommendation_path=recommendation,
            )
        )
    elif args.command == "score-phase1":
        print(
            score_frozen_validation_recommendation(
                parquet_paths=tuple(sorted(DATASET.glob("*.parquet"))),
                decision_task_path=task,
                split_manifest_path=split,
                recommendation_path=recommendation,
                raw_output_path=raw,
                score_path=score,
            )
        )
    elif args.command == "replay-phase1":
        print(
            replay_score(
                score_path=score,
                recommendation_path=recommendation,
                raw_output_path=raw,
            )
        )
    else:
        print(
            render_smoke_report(
                score_path=score,
                recommendation_path=recommendation,
                raw_output_path=raw,
                split_path=split,
                decision_task_path=task,
                bundle_path=bundle,
            )
        )


if __name__ == "__main__":
    main()
