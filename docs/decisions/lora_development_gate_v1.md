# LoRA Development Gate v1

Status: **do not run before confirmation**.

The planned model-comparison review found that a project LoRA is not currently a
defensible use of the available data or compute. The repository's working
data-use decision permits private local analysis, but it defers cloud
fine-tuning and prohibits uploading participant-level records to a managed
service without a separate decision. The nine decision-eligible development
experiments also do not form an adequate LoRA training corpus: three require
visual stimuli, and the aggregate treatment effects contain only a few dozen
arm-level labels.

This is a methodological stop, not a missing deliverable. The project plan makes
fine-tuning optional until it addresses a specific decision-level hypothesis.
Training a 7B/14B adapter on a tiny aggregate dataset would add apparent scope
without credible evidence that it learned transferable behavior.

The gate can be reopened only after all of the following are true:

1. The exact participant-data transfer and compute environment are permitted by
   a documented data-use decision.
2. An experiment-level training and validation split with adequate eligible
   study support is frozen.
3. The decision-level hypothesis, out-of-fold recommendation procedure, and
   stopping rule are prespecified.
4. Checkpoint, records, seeds, learning rates, data sizes, budget, and artifact
   provenance are frozen before execution.

The immediate simulator priority is therefore the classical cross-experiment
baseline, stronger outcome-blind repeated-generation diagnostics, and one
prospective pass over the six still-sealed confirmation experiments.
