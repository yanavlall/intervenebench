# Evidence-to-Report Evaluation Protocol

**Status:** frozen before report generation or human labeling  
**Protocol artifact:** `data/manifests/research/evidence_report_eval_v1.json`  
**Evidence packet:** `data/manifests/qualitative_eval/intervenebench_report_evidence_packet_v1.json`

## Purpose

Behavioral-simulation products do not end at a model prediction. Their evidence
is summarized in research reports, validation readouts, model-comparison memos,
and release recommendations. A report can preserve every numeric output and
still be unsafe if it hides uncertainty, merges retrospective and prospective
evidence, reverses a HOLD decision, or implies an unsupported deployment scope.

This extension evaluates that output layer. It is not a second attempt to enlarge
the InterveneBench experiment count, and it does not change any core behavioral
simulation result. InterveneBench supplies the first real evidence packet; the
software accepts other evidence packets and report scenarios without changing
the evaluation logic.

## Evaluation object

The scored object is one structured evidence-grounded report generated for a
specific audience and decision request. Each report must contain:

- an executive summary;
- an explicit scoped recommendation;
- evidence claims linked to allowlisted fact identifiers;
- limitations linked to frozen claim-boundary identifiers.

The structure makes provenance and missing evidence mechanically inspectable,
while the natural-language fields preserve the qualitative work that human and
automated graders must evaluate.

## Frozen report panel

The panel crosses:

- eight decision scenarios;
- two meaning-preserving prompt formats;
- three generator roles.

This produces 48 reports. Six scenarios are development-only. Two complete
scenario groups—autonomous deployment and limited-human fallback—are held out
from grader-threshold selection. Model identity is hidden from annotators and
automated judges. Calls, seeds, checkpoint identities, and runtime limits will
be frozen separately before execution.

## Rubric

Humans and automated judges score five dimensions from 0 to 3:

1. factual fidelity;
2. uncertainty calibration;
3. claim-boundary compliance;
4. decision traceability;
5. clarity and actionability.

A report passes only when every dimension is at least 2, total score is at least
12 of 15, and no critical error is present. Critical errors include fabricated
numbers, evidence-tier confusion, unsupported operational claims, reversed HOLD
decisions, omitted material negative findings, invented causal mechanisms, and
misstated prospective status.

## Human labels

One primary annotator labels all reports through a self-contained offline
interface. At least 12 reports, selected by the frozen blinding rule, require a
second independent annotator before any automated-grader release claim. Model
identity and internal report identifiers are stored in a separate blinding key.

If the second-rater minimum is not met, the result remains a single-rater
exploratory case study. The requirement is not weakened after labels are seen.

## Automated graders

Two cross-family rubric judges independently score every report. They receive
the locked evidence packet, scenario, report, and rubric, but not generator
identity or human labels. Outputs must satisfy a strict schema; invalid outputs
are preserved and are not repaired semantically.

Any threshold or combination rule is selected using development scenarios only.
The held-out scenario groups are opened once after the rule is frozen. The
operational rule is fail-closed: deterministic evidence checks must pass and
both independent judges must pass the report. One judge failure blocks release.

## Primary evaluation and release rule

The primary metric is held-out false-pass rate because incorrectly approving an
unsafe report is the most consequential grader error. Automated grading is not
released unless all of the following hold:

- zero held-out false passes;
- held-out pass/fail balanced accuracy at least 0.80;
- mean absolute dimension-score error at most 0.75;
- at least 12 reports have second-rater labels.

Secondary metrics include quadratic-weighted kappa, critical-error precision
and recall, dimension-level error, and generator-ranking agreement. Scenarios,
not individual rubric dimensions or judge calls, are the grouping unit.

## Claim boundary

Passing would support a small grouped case study showing that the frozen grader
workflow can screen evidence-grounded reports under this rubric. It would not
establish a universal qualitative judge, replace expert review, or validate the
grader for unrelated domains without new labeled evidence.

Failing is a useful result. The workflow must preserve false passes, grader
disagreement, rubric ambiguity, and human-review requirements rather than tune
them away on held-out scenarios.

## Current authority

The protocol authorizes no model calls, no spending, no human labels, and no
automated-grader release. A later execution authorization must bind the exact
protocol, evidence packet, prompts, checkpoints, call count, seeds, and cost cap.
Before any generation authority can validate, each materialized image must also
pass a separately authorized, zero-inference remote import smoke that verifies
its image ID, execution-freeze hash, source hash, and complete embedded-file
inventory. The smoke result is hash-bound into the later generation authority.
