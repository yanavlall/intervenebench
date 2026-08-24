# Second External TESS Corpus v1

**Status:** frozen ordered source-audit universe. This decision authorizes
public design-material acquisition only. It authorizes zero model calls, zero
paid compute, zero participant-row access, and zero human-outcome reveal.

## Why this corpus

The current pristine SocSci210 pool has one clean task and twelve conditionals;
reaching the twelve-task replication minimum would require eleven of those
twelve conditionals to clear. A second external lane is therefore necessary,
not merely optional, if the project is to test whether the first low-regret
signal replicates rather than reinterpret a small result.

The official OSF account for `TESS-Experiments` (`4547c`) exposes 984 public
nodes, including 457 public root projects. We froze only API metadata: node ID,
title, dates, category, public/root status, and source URL. Descriptions,
abstracts, proposals, results, outcome summaries, and participant files did not
participate in selection.

After exact-node and normalized-title de-duplication against the 199 pinned
SocSci210 nodes and the first 31-study external universe, 233 first-pass root
projects remain. Fielding-level de-duplication is still mandatory during source
audit.

## Frozen selection

Each candidate is ordered by:

```text
SHA256("external-tess-root-v1:20260814:" + osf_node_id)
```

The first 30 rows form the only eligible audit batch. Audit them contiguously.
Do not skip rows because titles look difficult or retain rows because titles
look promising. A versioned no-progress checkpoint applies after order 20: if
fewer than two clean tasks have passed, close this random-root lane rather than
spending ten more audits on a source with inadequate yield. This stopping rule
uses source-audit dispositions only, never human outcomes or simulator results.
If at least two clean tasks pass by order 20, continue in six-candidate blocks,
finish the current block after four pristine runnable tasks pass, and otherwise
adjudicate all 30. If fewer than four ultimately pass, stop this lane rather
than loosening the gates.

The full universe, page-response hashes, de-duplication record, exact batch,
and zero-execution boundary are in
`data/manifests/audits/tess_external_root_universe_v1.json`.

## Source-access boundary

For active rows, auditors may use OSF API metadata and download public source
containers solely to list members and extract final questionnaires, codebooks,
assignment guides, field reports, and exact stimuli. If a container also holds
participant data, its member names may be listed, but participant CSV, SAV,
DTA, SAS, XLSX, or other response-bearing members must never be extracted or
opened. Temporary mixed containers must be removed after the design-only files
are isolated.

Start from final instruments. Do not use generic web search, publication pages,
abstracts, proposals, or project reports to reconstruct a task when they may
contain findings. Any accidental result-text exposure is logged immediately
and permanently excludes that row from the prospective panel.

After the first audit blocks revealed ambiguous-file contamination, the
instrument gate is tightened prospectively: open a document only when its
filename or direct repository metadata explicitly identifies it as a final
questionnaire, survey, programming questionnaire, or fielded instrument.
Generic study-name PDFs, manuscripts, modules, descriptions, reports, output
documents, and ambiguous DOC/PDF files are not opened. If no explicitly labeled
instrument exists, mark the row source-blocked rather than guessing.

Passing source audit is not panel inclusion. Every row must still satisfy the
same deployable-action, fixed-world, bounded-utility, exact-reconstruction,
human-mapping, fielding-independence, model-exposure, and runnable-adapter gates
before any simulator plan is frozen.
