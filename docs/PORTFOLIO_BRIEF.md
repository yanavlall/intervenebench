# InterveneBench: Portfolio Brief

InterveneBench asks whether simulated humans are useful for decisions, not just
whether they sound human. A simulator receives a source-verified randomized
experiment without its outcome data, freezes an intervention recommendation,
and is then scored against the human experiment.

## The result

Across six prospectively confirmed experiments, the frozen primary policy chose
3/6 exact sample winners and all six choices were practically reliable. On the
five normalized tasks, mean decision regret was `0.0035` versus `0.0410` for
uniform random choice. Exact accuracy remained chance-compatible; the low-regret
signal was promising but small-sample.

The tests designed to make the system safer did not work:

- the prespecified trust ranking was worse than random abstention;
- no validated confidence threshold could be deployed;
- every limited-human fallback policy had higher point regret than
  synthetic-only at every nonzero budget;
- a retrospective cross-family model improved treatment-effect fidelity without
  improving the intervention decision.

Those failures were kept as results rather than tuned away after reveal.

The earlier five-task local `llama3.2:3b` development pilot remains visible for
provenance: exact choice was `3/5` versus `0/5` for the no-effect/control-tie
baseline; mean regret was `0.0038` versus `0.0308`; worst regret was `0.0166`
versus `0.0519`; and treatment-effect MAE was `0.0467` versus `0.0361`. It is not
pooled with the prospective panel.

## The system

The repository implements source audits, explicit decision-task contracts,
strict parsers, immutable recommendations, separate reveal authorization,
experiment-level uncertainty, model-version regression gates, disjoint
human-fallback evaluation, and a self-verifying aggregate-only public release.

The current release decision permits limited research-stage candidate screening
and holds autonomous intervention selection, trust-based abstention, and
small-sample human correction.

Run the complete public verification with:

```bash
PYTHONPATH=src python3 -m intervenebench.public_cli verify --root .
```

See [Authoritative Research Findings](reports/research_findings_v1.md) for the
full result and [Simile Evals Case Study](SIMILE_EVALS_CASE_STUDY.md) for the
role-focused technical discussion.
