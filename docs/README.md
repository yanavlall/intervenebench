# Documentation map

InterveneBench separates normative protocol, audit evidence, decisions, and reports so that historical observations do not silently become methodological rules.

## Start here

For a quick review:

1. [`reports/research_findings_v1.md`](reports/research_findings_v1.md) — authoritative synthesis of the positive and negative findings.
2. [`SIMILE_EVALS_CASE_STUDY.md`](SIMILE_EVALS_CASE_STUDY.md) — role-focused technical case study.
3. [`PORTFOLIO_BRIEF.md`](PORTFOLIO_BRIEF.md) — one-page problem, method, result, and claim boundary.
4. [`results/index.html`](results/index.html) — portable aggregate-only results and scoped release decisions.
5. [`reports/confirmation_results_v1.md`](reports/confirmation_results_v1.md) — six-experiment prospective decision, trust, and fallback results.
6. [`reports/fallback_replication_audit_v1.md`](reports/fallback_replication_audit_v1.md) — development-to-confirmation replication of the fallback result.
7. [`reports/fallback_failure_mechanism_v1.md`](reports/fallback_failure_mechanism_v1.md) — aggregate-only failure-pattern analysis.
8. [`reports/cross_family_retrospective_results.md`](reports/cross_family_retrospective_results.md) — Mistral architecture-sensitivity result and boundary.
9. [`MODEL_VERSION_REGRESSION.md`](MODEL_VERSION_REGRESSION.md) — experiment-paired simulator regression interface and release boundary.
10. [`../README.md`](../README.md) — project overview, repository map, and one-command verification.
11. [`protocol/EVIDENCE_REPORT_EVAL_PROTOCOL.md`](protocol/EVIDENCE_REPORT_EVAL_PROTOCOL.md) — reusable, held-out evaluation of evidence-grounded research reports.

The active stage extends the role-focused evaluation product with a reusable
evidence-to-report grading workflow. Open-ended
independent-replication corpus search is closed; its manifests, audit reports,
and stopping decisions remain available as a reproducibility trail but do not
authorize further execution.

For implementation or methodological review:

1. [`../AGENTS.md`](../AGENTS.md) — hard project constraints.
2. [`protocol/BENCHMARK_PROTOCOL.md`](protocol/BENCHMARK_PROTOCOL.md) — normative freeze, reveal, and scoring protocol.
3. [`protocol/SEQUENCE_SIMULATION.md`](protocol/SEQUENCE_SIMULATION.md) — paired source-programmed survey-sequence simulation.
4. [`../PROJECT_SPEC.md`](../PROJECT_SPEC.md) — full research specification.
5. [`../PHASE_1.md`](../PHASE_1.md) — completed first pipeline milestone and acceptance criteria.
6. [`InterveneBench_Project_Plan.tex`](InterveneBench_Project_Plan.tex) — printable synchronized plan.

`../PERSONAL_CONTEXT.md` is private strategic context and is intentionally not linked from public-facing project materials.

## Sections

- `protocol/`: rules that govern future work.
- `audits/`: source traces, dataset qualification, and leakage records.
- `decisions/`: frozen scope and corpus decisions.
- `reports/`: completed run and viability reports.

Machine-readable benchmark scope, readiness, contracts, and split records live in `../data/manifests/`; see `../data/manifests/README.md`. The readiness map is an outcome-blind work/dependency ordering, not a canonical split. The supported-ordinal pilot split is also explicitly noncanonical and keeps all real outcomes sealed.

The LaTeX source is canonical. Its PDF is a rendered convenience copy and should be regenerated and visually inspected after material edits.

The completed first expansion is governed by [`decisions/depth_first_research_program_v1.md`](decisions/depth_first_research_program_v1.md). It defined the 15-task discovery, prospective-development, and sealed-confirmation study plus explicit stopping and pivot gates. The full study is now complete; the nine-experiment development evidence and six-experiment prospective confirmation are reported separately. It is historical evidence for the active evaluation-product build, not current execution authority.

The project-fine-tune decision is recorded in [`decisions/lora_development_gate_v1.md`](decisions/lora_development_gate_v1.md). Cloud LoRA remains unauthorized and scientifically unjustified for the current small aggregate training support; the gate lists exact reopening conditions.

The final outcome-blind confirmation plan is recorded in [`decisions/confirmation_preparation_v1.md`](decisions/confirmation_preparation_v1.md). It bound six sealed tasks, all adapters and public stimulus assets, the model/checkpoint matrix, exact call and cost ceilings, a no-threshold trust ranking, and frozen fallback comparisons. The separately authorized reveal and aggregate-only score are documented in [`reports/confirmation_results_v1.md`](reports/confirmation_results_v1.md).

The current simulator-interface evidence is in [`reports/forced_choice_discovery_screen.md`](reports/forced_choice_discovery_screen.md) and [`reports/answer_order_canary.md`](reports/answer_order_canary.md). The latter records an outcome-blind negative result and the frozen pivot from single-order inference to balanced source/reverse answer-order averaging.

The implemented balanced estimator, verified artifact reuse, completed 56-call all-action run, and 20 outcome-blind full recommendations are documented in [`reports/balanced_full_action_freeze.md`](reports/balanced_full_action_freeze.md).

The three-experiment multimodal freeze and prospective-development result are in [`decisions/prospective_multimodal_freeze_v4.md`](decisions/prospective_multimodal_freeze_v4.md) and [`reports/prospective_multimodal_development_results_v1.md`](reports/prospective_multimodal_development_results_v1.md).
