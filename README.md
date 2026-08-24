# InterveneBench

**When simulated humans recommend an intervention, should we trust the decision?**

InterveneBench evaluates behavioral simulators as decision systems. A model sees
an experiment's population, intervention arms, and outcome definition—but not
the human results—then commits to an intervention. That immutable choice is
scored against a randomized human experiment.

The project asks a stricter question than “do model responses look human?”:

> Does synthetic evidence select a low-regret action, can failure be detected
> before outcomes are revealed, and does a small human pilot improve the choice?

## Result

The current evidence is useful precisely because it is mixed.

| Prospective confirmation | Result |
|---|---:|
| Independent randomized experiments | 6 |
| Exact human-best interventions | 3/6 |
| Practically reliable interventions | 6/6 |
| Mean normalized regret | 0.0035 |
| Uniform-random mean regret | 0.0410 |
| Worst normalized regret | 0.0085 |

Exact-choice accuracy was compatible with uniform random action selection
(`p = 0.373`). The decision-regret result was more promising: across the five
bounded-normalized tasks, only `12/216` complete uniform-random action
combinations achieved mean regret as low as the frozen simulator policy
(`p = 0.0556`). All three exact-choice errors were near-ties.

Two important negative findings survived without post-hoc repair:

- The prespecified trust ranking was worse than random abstention (AURC `0.711`
  versus `0.500`; exact-choice AUROC `0.222`). No threshold was deployed.
- Every tested limited-human policy had higher point regret than synthetic-only
  at every nonzero budget. Human-only pilots were statistically worse at budgets
  10, 25, 50, and 100.

The strongest supported use is **limited research-stage candidate screening**.
Autonomous intervention selection, confidence-based abstention, and small-sample
human fallback remain on hold.

Read the [authoritative findings](docs/reports/research_findings_v1.md) or the
[Simile evals case study](docs/SIMILE_EVALS_CASE_STUDY.md).

## What was built

InterveneBench makes five concrete contributions:

1. A source-audited corpus of randomized experiments converted into explicit
   `DecisionTask` contracts: one experiment, one utility, one admissible action
   set.
2. A leakage-safe freeze/reveal protocol that separates outcome-blind model
   recommendations from human-outcome scoring.
3. Decision-aware metrics: treatment-effect error, exact choice, practical
   reliability, and normalized regret.
4. Outcome-free diagnostics and a release gate that preserve failed trust signals
   instead of fitting a threshold after reveal.
5. A disjoint human-fallback evaluator that tests whether limited human evidence
   actually improves the final action.

```text
source-verified randomized experiment
-> outcome-blind simulator input
-> frozen synthetic effects, recommendation, and diagnostics
-> separately authorized human-outcome reveal
-> experiment-level decision and uncertainty scoring
-> disjoint limited-human fallback evaluation
```

Experiments—not model calls, arms, seeds, or participant rows—are the unit of
generalization and headline uncertainty.

### Reusable evidence-to-report evaluation

The repository also includes a model- and organization-neutral workflow for
evaluating research reports generated from a locked evidence packet. It turns
aggregate findings and release decisions into a 48-report evaluation panel,
then provides:

- strict structured report and rubric-judgment schemas with no semantic repair;
- deterministic checks for missing evidence, claim-boundary violations, and
  reversed release decisions;
- a model-blinded offline human-labeling interface and separate private key;
- two-judge, fail-closed automated release decisions;
- held-out false-pass, balanced-accuracy, and dimension-error gates grouped by
  scenario;
- pinned, network-blocked Modal execution with live progress logs and no model
  download or participant-data path.

InterveneBench is the first evidence packet, not a hard-coded dependency of the
evaluation logic. The protocol is frozen but currently has zero execution
authority, so no report-generation or judge calls are counted in the results
above. See [the evidence-to-report protocol](docs/protocol/EVIDENCE_REPORT_EVAL_PROTOCOL.md).

## Why the negative findings matter

The project separates four properties that are often conflated:

- A model can estimate effects poorly but still rank available actions well.
- Exact sample-winner accuracy can look mediocre while decision regret is tiny.
- Model disagreement can reveal architecture sensitivity without identifying
  which model is right.
- Adding a small human sample can increase regret when noisy pilot estimates
  override a strong low-regret recommendation.

The fallback result replicated directionally from nine development experiments
to five untouched normalized confirmation experiments. Across 15
task-by-budget empirical-Bayes cells, there were six harmful updates, three
corrections, and six unchanged decisions; average harm was `13.6×` average
correction. This is an exploratory failure pattern, not a causal mechanism.

A retrospective cross-family Mistral comparator improved treatment-effect MAE
but did not improve decision accuracy or regret. It adds architecture evidence,
not prospective experiment N.

## Evidence tiers

| Tier | Experiments | Role |
|---|---:|---|
| Development | 9 | Build and debug methods; 8/9 exact, mean regret 0.0023 |
| Prospective confirmation | 6 | Frozen recommendation before outcome reveal; headline evidence |
| Retrospective cross-family | 5 | Architecture sensitivity only; no new prospective N |

The six-experiment panel is deliberately labeled noncanonical. It does not
support a universal trust model or publication-scale benchmark claim.

For historical provenance, the earlier five-task local `llama3.2:3b` pilot
selected `3/5` exact interventions versus `0/5` for the no-effect/control-tie
baseline. Mean regret was `0.0038` versus `0.0308`, worst regret was `0.0166`
versus `0.0519`, and treatment-effect MAE was `0.0467` versus `0.0361`. This
remains a development result and is not pooled with prospective confirmation.

## Verify the release

The public research release is hash-bound and can be checked from a clean
checkout with only Python's standard library:

```bash
PYTHONPATH=src python3 -m intervenebench.public_cli verify --root .
```

This portable mode verifies every released file, both public evidence envelopes,
the scoped release decisions, and the frozen zero-authority program. It makes no
model calls, reads no participant rows, and does not require ignored local run
artifacts.

Maintainers with the restricted aggregate source artifacts can additionally
rebuild the synthesis from its hash-pinned provenance:

```bash
PYTHONPATH=src python3 -m intervenebench.public_cli verify --root . --deep-replay
```

The distinction is deliberate: a clean public checkout can prove release
integrity and recompute the decision gate, while the deep mode also proves that
the checked-in synthesis exactly matches the local aggregate evidence chain.

The portable aggregate-only evaluation artifact can also be inspected with:

```bash
PYTHONPATH=src python3 -m intervenebench.public_cli case-study --root .
```

## Evaluation system

The repository includes:

- source provenance, design audits, and leakage logs;
- typed task and response contracts;
- strict structured-output parsers with no semantic repair;
- immutable recommendation and authorization envelopes;
- local text models and pinned Modal text/vision execution;
- answer-order, prompt, stability, entropy, margin, and model-disagreement
  diagnostics;
- experiment-cluster bootstrap and finite action-space randomization checks;
- experiment-paired model-version regression gates;
- aggregate-only public release decisions and a portable results explorer.

Human outcomes remained local. Participant rows are neither redistributed nor
sent to model APIs.

## Repository map

```text
src/intervenebench/       contracts, simulators, estimators, scoring, release gates
tests/                    leakage, freeze/reveal, metrics, fallback, replay tests
data/manifests/           source audits, task contracts, scope and protocol freezes
data/public/              aggregate-only self-verifying public evidence
data/derived/stimuli/     source-faithful simulator assets
docs/protocol/            normative benchmark protocol
docs/audits/              source evidence and exposure records
docs/decisions/           frozen research and product decisions
docs/reports/             empirical results and limitations
docs/results/             portable aggregate-only results explorer
artifacts/                local immutable run artifacts; excluded from Git
```

Start with the [documentation map](docs/README.md). The governing specification
is [PROJECT_SPEC.md](PROJECT_SPEC.md), and the prospective protocol is
[BENCHMARK_PROTOCOL.md](docs/protocol/BENCHMARK_PROTOCOL.md).

## Claim and data boundary

SocSci210 is the primary dataset, pinned at revision
`048481111a4425ed83dc0eacf15f8431f252b21a`. InterveneBench does not accept its
metadata as causal ground truth automatically; every scored task is checked
against original study materials.

“Outcome-blind” means the target experiment's human results were excluded from
the recommendation and diagnostics. It does not prove that a public study was
absent from a foundation model's pretraining.

The honest conclusion is narrow: this project demonstrates a rigorous behavioral
simulation evaluation system and a preliminary low-regret signal, while showing
that the tested confidence and small-human-fallback methods were not good enough
to deploy.
