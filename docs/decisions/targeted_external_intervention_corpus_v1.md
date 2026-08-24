# Targeted External Intervention Corpus v1

**Status:** closed at the frozen order-9 no-progress checkpoint. The lane
produced one strict scientific survivor (`dvwu7`), below the minimum of two.
Orders 10--12 remain unopened. This decision authorized zero model calls, zero
paid compute, zero participant-row access, and zero human-outcome reveal.

## Why the lane changes

The prior public-root lane closed at its order-20 checkpoint with zero clean
passes. Its dominant failures were predictable for an unfiltered archive:
measurement experiments, target-attribute manipulations, incompatible factual
worlds, missing instruments, and result-bearing ambiguous files.

The replacement is a transparent design-prior filter, not hand-picking. Before
opening another source, it selects projects using only OSF node ID, public
title, and public source URL. Human outcomes, publication results, simulator
outputs, and prior effect directions are absent from selection.

## Frozen selection rule

HTML-unescape each title, apply Unicode NFKC case-folding, and collapse
non-alphanumeric characters to spaces. A title must match one of the exact
regular-expression families frozen in the manifest. They cover:

```text
message or messaging; communication; advertising; disclosure or transparency;
fact-check; rhetoric; narrative; portrayal; plural appeals; frame or reframing;
self-affirmation; implementation intentions; mobilization; invocation;
reminder; warning; feedback; nudge; request; donation; deliberation; policy
justification; and the phrases "an intervention" or "intervention(s) to/for"
```

Exclude titles containing:

```text
measurement; question wording/order/context; order effects; list experiments;
item counts; anchoring vignettes; survey instruments/estimation/mode; phone
modules; data-collection procedures; panel conditioning; over-reporting;
self-report; test performance; response quality/scale; and scale direction/order
```

Remove every already-audited OSF ID and every ID in the global exposure log.
Preserve the original source-universe `selection_sha256` ordering instead of
introducing a new seed. Deduplicate normalized titles by retaining the row with
the smallest original selection hash. The rule finds 38 title matches in the
233-root frozen source universe. Four were already audited, leaving 34 eligible
targeted candidates. The first 12 rows are the only authorized audit batch.

## No-progress checkpoints

A scientific survivor must already pass randomization, fixed-world deployable
actions, bounded utility, exact stimuli, and fielding independence. A
conditional row counts only if its remaining work is mechanical human mapping
or adapter implementation.

- Audit complete three-candidate blocks in frozen order.
- After order 6, close the lane if there are zero scientific survivors.
- After order 9, close the lane if fewer than 2 scientific survivors exist.
- Continue to the final block only if the order-9 checkpoint passes.
- Finish the current block after the fourth survivor, and never continue beyond
  order 12 under v1.

The checkpoints use source structure and reconstructability only. They never
use human outcomes or simulator performance.

## Source boundary

Use the same tightened source protocol as the closed lane: open only files
explicitly identified by filename or direct repository metadata as a final
questionnaire, survey, programming questionnaire, or fielded instrument.
Generic PDFs, manuscripts, modules, descriptions, reports, and output files are
not opened. Mixed archives may be listed, but participant members are never
extracted or opened. Any incidental result-text exposure is logged and makes
the row a prospective failure.

The authoritative manifest is
`data/manifests/audits/tess_targeted_intervention_universe_v1.json`.

The high-precision rule was finalized after two independent local-metadata
reviews and before any source in this lane was downloaded or opened. No manual
candidate addition or deletion is permitted after this freeze.

## Closure

Orders 1--6 produced one survivor, so the first checkpoint passed. Orders 7--9
produced none; the frozen order-9 threshold required two total survivors.
Accordingly, v1 is closed and orders 10--12 may not be opened. The survivor
`dvwu7` moves to a separate mechanical contract-completion queue; it does not
retroactively change the lane's stopping decision.

See `docs/reports/targeted_external_intervention_audit_v1.md` and
`data/manifests/audits/tess_targeted_intervention_lane_closure_v1.json`.
