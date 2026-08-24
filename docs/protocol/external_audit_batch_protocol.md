# External Audit Batch Protocol

External source qualification may run in bounded parallel batches. Parallelism changes retrieval speed, not the scientific ordering, eligibility rules, or leakage boundary.

## Batch construction

- Take the next three unaudited `pending_source_audit` candidates in ascending `freeze_order`.
- Record the batch before retrieving any source materials.
- Do not select candidates for likely eligibility, source convenience, topic, or expected results.
- A later batch does not begin until the current batch has a central leakage review and registry commit.

## Worker boundary

Each candidate has one read-only evidence worker. A worker may retrieve and hash design materials but may not edit the audit registry, source-bundle manifest, workbook, eligibility tests, or project-status documents.

Workers use a two-stage evidence funnel. Stage 1 is the default and Stage 2 is entered only
when the candidate survives every hard structural gate.

### Stage 1: rapid structural screen

Inspect only stable source metadata and the final questionnaire or programmed instrument.
The candidate stops at Stage 1 as soon as the inspected evidence proves any one of these
conditions:

1. no verified random assignment;
2. duplicate fielding or unresolved experiment identity;
3. measurement wording or order without a downstream intervention outcome;
4. target identity, biography, scenario, or market attributes presented as selectable actions;
5. cells that change the underlying factual world, evidence, or expert position rather than a
   truthful action under a fixed world;
6. no coherent action set available to one stable decision-maker;
7. no source-grounded monotone utility and stable beneficiary;
8. the candidate outcome is observed only after treatment-induced selection; or
9. the exact treatment and outcome cannot be reconstructed from permitted source evidence.

A Stage-1 exclusion packet needs only stable identifiers, the exact instrument evidence that
proves the exclusion, retained design-file hashes, a precise reason code, and an outcome-access
statement. Do not retrieve weights, missing-value codes, every auxiliary outcome, field-allocation
details, response-rate reports, proposals, or manuscripts after a hard exclusion is established.

### Stage 2: full decision-contract audit

Only a Stage-1 survivor receives the full audit. Inspect, as needed and in this order:

1. codebook without participant rows;
2. randomization, programming, and fielding documentation;
3. preregistration or proposal design sections; and
4. manuscript methods only when the earlier materials cannot resolve a required design fact.

Stage 2 must recover the exact action set and reference arm, primary-outcome provenance and
scale, utility orientation, randomization and allocation details, weights and missingness, order or
carryover, personalization, factorial estimand, source-data mapping, and simulator reconstruction
requirements. Any unresolved fact that affects the decision contract remains a scoring blocker.

Retrieve through frozen direct source links, publisher or repository APIs, and link metadata. Generic search-result pages and snippets are not permitted because they can expose reported findings before the design packet is isolated.

Workers must avoid participant response files, result summaries, arm statistics, treatment effects, significance, winners, and manuscript results or discussion. An accidental result-text encounter is reported without transcribing the result and ends result-bearing inspection.

## Required evidence packet

Every Stage-2 worker returns:

- stable source identifiers and URLs;
- retrieved filenames and SHA-256 hashes;
- randomization unit and experimental factors;
- nominal cells and the proposed admissible action set;
- personalization, order, carryover, pooling, or repeated-measure complications;
- candidate primary outcomes, source response scale, and utility direction from instruments or codebooks;
- proposed decision-maker and stable beneficiary;
- proposed eligibility track and design-based reason;
- unresolved assumptions; and
- an explicit outcome-access and accidental-exposure statement.

The worker recommendation is advisory. It is not a registry decision.

The first 16 completed audits were retrospectively back-tested against this funnel in
`data/manifests/audits/external_audit_funnel_backtest.csv`. All five scientifically usable rows would
have escalated to Stage 2 and all eleven design exclusions or duplicates had sufficient Stage-1
evidence. This is a protocol-consistency check, not a prospective estimate of future yield or time
savings.

## Central adjudication and commit

The primary auditor reviews all evidence packets under the same decision-task and leakage rules, resolves inconsistencies, and commits rows in ascending `freeze_order`. No result can change candidate ordering. An eligible canonical-test row requires `outcome_access=sealed`. A candidate with result-text exposure cannot enter an untouched evaluation stratum; if it remains scientifically eligible, it must be explicitly marked `development_only` before any further work. Source-level duplicate checks use stable source nodes, DOIs, sample/fielding identity, and stimuli—not title matching alone.

The batch is complete only when:

- audit and source-bundle rows form a contiguous prefix of the frozen clear-candidate order;
- source hashes and exposure states agree across manifests;
- the workbook and status documents agree with the manifests;
- focused leakage/order tests and the full test suite pass; and
- no participant outcome file was opened for source qualification.

## Scaling rule

The default batch size is three because it fits the available review capacity while leaving one central auditor responsible for consistency. Increase the batch size only if source retrieval remains the bottleneck and central review can still occur before another batch starts.
