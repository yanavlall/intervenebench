# Model-version regression gate

InterveneBench separates two questions:

1. Did a candidate simulator regress relative to the deployed reference?
2. Is either version approved for a particular operational use?

The first is answered by `compare-model-versions`; the second is answered by
the scoped release gate. A regression pass never expands release authority.

## Input

Candidate and reference JSON files use the same frozen panel hash and identical
experiment IDs:

```json
{
  "schema_version": "intervenebench.model_version_evaluation.v1",
  "model_version": "candidate-checkpoint-id",
  "panel_sha256": "<64 lowercase hex characters>",
  "planned_output_count": 300,
  "schema_valid_output_count": 298,
  "experiments": {
    "experiment-a": {
      "normalized_regret": 0.01,
      "exact_choice": true,
      "practically_reliable": true
    },
    "experiment-b": {
      "normalized_regret": 0.03,
      "exact_choice": false,
      "practically_reliable": true
    }
  }
}
```

Only one aggregate row per experiment is accepted. Simulator draws, arms, and
participant rows cannot inflate the benchmark sample size.

## Run

```bash
PYTHONPATH=src .venv/bin/python -m intervenebench.cli compare-model-versions \
  --candidate candidate.json \
  --reference reference.json \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 2026081401
```

The comparison resamples whole experiments in matched pairs. Promotion is held
if the upper confidence bound on mean regret exceeds the noninferiority margin,
worst-case regret increases materially, exact choice or practical reliability
drops beyond tolerance, schema validity regresses, the panel hash changes, or
experiment coverage differs.
