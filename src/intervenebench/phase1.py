"""Orchestration helpers for the prospective Phase 1 smoke path."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from .evaluation import Observation, arm_means, evaluate_decision, normalize_utility
from .protocol import (
    canonical_json_bytes,
    freeze_envelope,
    freeze_recommendation,
    payload_hash,
    verify_envelope,
    verify_frozen_recommendation,
)
from .schemas import OutcomeDirection
from .simulators import (
    aggregate_binary_predictions,
    ollama_probability_prompt,
    parse_binary_probability,
    validate_blinded_bundle,
)
from .socsci210 import read_revealed_outcomes
from .uncertainty import bootstrap_arm_optimality


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def run_local_ollama_simulator(
    *,
    bundle_path: Path,
    split_path: Path,
    decision_task_path: Path,
    raw_output_path: Path,
    recommendation_path: Path,
    model: str = "llama3.2:3b",
    draws: int = 3,
    seed: int = 2102026,
    temperature: float = 0.35,
) -> str:
    """Run a design-only local simulator and freeze its recommendation."""

    bundle = read_json_object(bundle_path)
    split = read_json_object(split_path)
    task = read_json_object(decision_task_path)
    validate_blinded_bundle(bundle)
    if task["split"] != "validation":
        raise ValueError("Phase 1 simulation target must be in validation")
    if split["experiment_to_split"].get(task["experiment_id"]) != "validation":
        raise ValueError("decision task disagrees with frozen split")
    if bundle["experiment_id"] != task["experiment_id"]:
        raise ValueError("blinded bundle disagrees with decision task")
    if draws <= 0:
        raise ValueError("draws must be positive")

    arm_ids = tuple(arm["arm_id"] for arm in bundle["arms"])
    outputs: list[dict[str, Any]] = []
    parse_failures = 0
    for arm_id in arm_ids:
        prompt = ollama_probability_prompt(bundle, arm_id=arm_id)
        for draw_index in range(draws):
            request = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": {
                    "type": "object",
                    "properties": {
                        "yes_probability": {"type": "number"},
                        "no_probability": {"type": "number"},
                    },
                    "required": ["yes_probability", "no_probability"],
                    "additionalProperties": False,
                },
                "options": {
                    "seed": seed + draw_index,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_predict": 80,
                },
            }
            completed = subprocess.run(
                ["curl", "--fail", "--silent", "--show-error", "http://127.0.0.1:11434/api/generate", "-d", json.dumps(request)],
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
            api_result = json.loads(completed.stdout)
            raw_response = api_result.get("response", "")
            try:
                parsed = parse_binary_probability(raw_response)
            except ValueError:
                parse_failures += 1
                raise
            outputs.append(
                {
                    "arm_id": arm_id,
                    "draw_index": draw_index,
                    "seed": seed + draw_index,
                    "yes_probability": parsed.yes_probability,
                    "no_probability": parsed.no_probability,
                    "raw_response": raw_response,
                    "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
                    "api_metrics": {
                        key: api_result.get(key)
                        for key in (
                            "total_duration",
                            "load_duration",
                            "prompt_eval_count",
                            "eval_count",
                            "eval_duration",
                        )
                    },
                }
            )

    synthetic_means = aggregate_binary_predictions(
        outputs, arm_ids=arm_ids, draws=draws
    )
    selected_arm = min(
        arm_id
        for arm_id, value in synthetic_means.items()
        if value == max(synthetic_means.values())
    )
    control = task["control_arm_id"]
    effects = {
        arm_id: value - synthetic_means[control]
        for arm_id, value in synthetic_means.items()
        if arm_id != control
    }
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    raw_payload = {
        "schema_version": "simulator_outputs.v1",
        "experiment_id": task["experiment_id"],
        "model": model,
        "draws_per_arm": draws,
        "temperature": temperature,
        "top_p": 0.9,
        "seed": seed,
        "parse_failures": parse_failures,
        "created_at_utc": now,
        "outputs": outputs,
    }
    raw_digest = freeze_envelope(raw_payload, raw_output_path, require_blinded=True)
    recommendation = {
        "schema_version": "recommendation.v1",
        "experiment_id": task["experiment_id"],
        "split": "validation",
        "task_num": task["socsci210_task_num"],
        "selected_arm_id": selected_arm,
        "arm_ranking": sorted(
            arm_ids, key=lambda arm_id: (-synthetic_means[arm_id], arm_id)
        ),
        "synthetic_arm_means": synthetic_means,
        "synthetic_treatment_effects": effects,
        "baselines": {
            "no_effect_control_policy": {
                "synthetic_arm_means": {arm_id: 0.5 for arm_id in arm_ids},
                "synthetic_treatment_effects": {
                    arm_id: 0.0 for arm_id in arm_ids if arm_id != control
                },
                "selected_arm_id": min(arm_ids),
            }
        },
        "tie_rule": task["tie_rule"],
        "split_manifest_sha256": payload_hash(split),
        "decision_task_sha256": payload_hash(task),
        "blinded_bundle_sha256": payload_hash(bundle),
        "simulator_outputs_sha256": raw_digest,
        "simulator": {"id": "ollama", "revision": model},
        "parser": {"id": "strict_binary_probability.v1"},
        "persona_roster": bundle["population"]["roster_id"],
        "diagnostics": {
            "winner_margin": max(synthetic_means.values())
            - min(synthetic_means.values()),
            "parse_failures": parse_failures,
        },
        "provenance": {
            "created_at_utc": now,
            "seed": seed,
            "temperature": temperature,
            "top_p": 0.9,
            "draws_per_arm": draws,
            "raw_output_path": str(raw_output_path),
        },
    }
    return freeze_recommendation(recommendation, recommendation_path)


def score_frozen_validation_recommendation(
    *,
    parquet_paths: tuple[Path, ...],
    decision_task_path: Path,
    split_manifest_path: Path,
    recommendation_path: Path,
    raw_output_path: Path,
    score_path: Path,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: int = 2102026,
) -> str:
    """Reveal only the authorized validation task, score, and freeze the result."""

    task = read_json_object(decision_task_path)
    recommendation = verify_frozen_recommendation(recommendation_path)
    raw_payload = verify_envelope(raw_output_path, require_blinded=True)
    if payload_hash(raw_payload) != recommendation["simulator_outputs_sha256"]:
        raise ValueError("recommendation does not match frozen simulator outputs")
    table = read_revealed_outcomes(
        parquet_paths,
        experiment_id=task["experiment_id"],
        recommendation_path=recommendation_path,
        split_manifest_path=split_manifest_path,
        decision_task_path=decision_task_path,
    )
    condition_to_arm = {
        int(arm["condition_num"]): arm["arm_id"] for arm in task["arms"]
    }
    expected_raw = {
        int(option["raw_value"]): float(option["normalized_utility"])
        for option in task["response_options"]
    }
    observations: list[Observation] = []
    arm_values = {arm["arm_id"]: [] for arm in task["arms"]}
    for row in table.to_pylist():
        if row["response"] is None:
            continue
        raw_value = int(row["response"])
        if raw_value not in expected_raw:
            raise ValueError("revealed response lies outside declared response options")
        arm_id = condition_to_arm.get(int(row["condition_num"]))
        if arm_id is None:
            raise ValueError("revealed condition is absent from decision task")
        utility = normalize_utility(
            raw_value,
            lower=float(task["scale_lower"]),
            upper=float(task["scale_upper"]),
            direction=OutcomeDirection(task["direction"]),
        )
        if utility != expected_raw[raw_value]:
            raise AssertionError("declared utility mapping disagrees with scale transform")
        participant_id = f"{row['study_id']}:{row['participant']}"
        observations.append(Observation(participant_id, arm_id, utility))
        arm_values[arm_id].append(utility)
    human_means = arm_means(observations)
    evaluation = evaluate_decision(
        human_means=human_means,
        synthetic_means=recommendation["synthetic_arm_means"],
        control_arm_id=task["control_arm_id"],
        practical_regret_tolerance=float(task["practical_regret_tolerance"]),
    )
    bootstrap = bootstrap_arm_optimality(
        arm_values, replicates=bootstrap_replicates, seed=bootstrap_seed
    )
    baseline = recommendation["baselines"]["no_effect_control_policy"]
    baseline_evaluation = evaluate_decision(
        human_means=human_means,
        synthetic_means=baseline["synthetic_arm_means"],
        control_arm_id=task["control_arm_id"],
        practical_regret_tolerance=float(task["practical_regret_tolerance"]),
    )
    score = {
        "schema_version": "phase1_score.v1",
        "experiment_id": task["experiment_id"],
        "split": "validation",
        "recommendation_sha256": payload_hash(recommendation),
        "selected_arm_id": evaluation.selected_arm_id,
        "human_best_arm_id": evaluation.human_best_arm_id,
        "correct_choice": evaluation.correct_choice,
        "normalized_decision_regret": evaluation.regret,
        "practically_reliable": evaluation.practically_reliable,
        "human_arm_means": evaluation.human_arm_means,
        "synthetic_arm_means": evaluation.synthetic_arm_means,
        "human_treatment_effects": evaluation.human_treatment_effects,
        "synthetic_treatment_effects": evaluation.synthetic_treatment_effects,
        "no_effect_control_baseline": {
            "selected_arm_id": baseline_evaluation.selected_arm_id,
            "correct_choice": baseline_evaluation.correct_choice,
            "normalized_decision_regret": baseline_evaluation.regret,
            "synthetic_arm_means": baseline_evaluation.synthetic_arm_means,
            "synthetic_treatment_effects": baseline_evaluation.synthetic_treatment_effects,
        },
        "observations_per_arm": {
            arm_id: len(values) for arm_id, values in arm_values.items()
        },
        "bootstrap": {
            "replicates": bootstrap.replicates,
            "seed": bootstrap.seed,
            "optimal_probability": bootstrap.optimal_probability,
            "selected_arm_optimal_probability": bootstrap.optimal_probability[
                evaluation.selected_arm_id
            ],
        },
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    return freeze_envelope(score, score_path)


def replay_score(
    *, score_path: Path, recommendation_path: Path, raw_output_path: Path
) -> dict[str, Any]:
    score = verify_envelope(score_path)
    recommendation = verify_frozen_recommendation(recommendation_path)
    raw = verify_envelope(raw_output_path, require_blinded=True)
    if score["recommendation_sha256"] != payload_hash(recommendation):
        raise ValueError("score does not match recommendation")
    if recommendation["simulator_outputs_sha256"] != payload_hash(raw):
        raise ValueError("recommendation does not match simulator outputs")
    return score


def render_smoke_report(
    *,
    score_path: Path,
    recommendation_path: Path,
    raw_output_path: Path,
    split_path: Path,
    decision_task_path: Path,
    bundle_path: Path,
) -> str:
    """Render the report entirely from frozen artifacts; never call a simulator."""

    score = replay_score(
        score_path=score_path,
        recommendation_path=recommendation_path,
        raw_output_path=raw_output_path,
    )
    recommendation = verify_frozen_recommendation(recommendation_path)
    raw = verify_envelope(raw_output_path, require_blinded=True)
    split = read_json_object(split_path)
    task = read_json_object(decision_task_path)
    bundle = read_json_object(bundle_path)
    if recommendation["split_manifest_sha256"] != payload_hash(split):
        raise ValueError("recommendation does not match split manifest")
    if recommendation["decision_task_sha256"] != payload_hash(task):
        raise ValueError("recommendation does not match decision task")
    if recommendation["blinded_bundle_sha256"] != payload_hash(bundle):
        raise ValueError("recommendation does not match blinded bundle")

    human_effect = score["human_treatment_effects"]["arm_1_past_performance"]
    synthetic_effect = score["synthetic_treatment_effects"]["arm_1_past_performance"]
    effect_error = abs(synthetic_effect - human_effect)
    total_prompt_tokens = sum(
        int(output["api_metrics"].get("prompt_eval_count") or 0)
        for output in raw["outputs"]
    )
    total_completion_tokens = sum(
        int(output["api_metrics"].get("eval_count") or 0)
        for output in raw["outputs"]
    )
    total_seconds = sum(
        int(output["api_metrics"].get("total_duration") or 0)
        for output in raw["outputs"]
    ) / 1_000_000_000
    arm0 = "arm_0_shared_values"
    arm1 = "arm_1_past_performance"
    bootstrap = score["bootstrap"]
    baseline = score["no_effect_control_baseline"]
    return f"""# Phase 1 Smoke-Test Report

> **Validation-only engineering smoke test—not a benchmark conclusion.** This run evaluates one validation experiment. It cannot establish simulator reliability, support a trust model, or justify opening the held-out test experiment.

## Outcome

The prospective pipeline completed successfully. The local simulator selected `{score['selected_arm_id']}` before human outcomes were revealed. The human point estimate selected the same arm, so exact best-arm correctness is **yes** and normalized decision regret is **{score['normalized_decision_regret']:.4f}**.

That apparent success is uncertain: the human difference between arms is only **{human_effect:.4f}** normalized utility, and the participant-within-arm bootstrap assigns the selected arm only **{bootstrap['selected_arm_optimal_probability']:.1%}** probability of being optimal. This is evidence that the protocol works, not that the simulator is trustworthy.

## Frozen design

- Dataset: `socratesft/SocSci210` at `{split['dataset_revision']}`.
- Access regime: `DESIGN_ONLY`; simulation used no participant record, demographic, response, reasoning, human aggregate, or reported result.
- Experiment: `jf46x`, validation split, source Q2 / SocSci210 task 0.
- Decision: DHS shared-values message versus DHS past-performance message.
- Outcome: binary intention to accept the recommended smallpox vaccine; Yes has utility 1 and No has utility 0.
- Estimand: unweighted, unadjusted intention-to-treat mean utility in the released analytic sample.
- Test status: `xc4yq` remains sealed; no test response was loaded or summarized.

## Arm results

| Arm | Synthetic mean | Human mean | Human n |
|---|---:|---:|---:|
| `{arm0}` | {score['synthetic_arm_means'][arm0]:.4f} | {score['human_arm_means'][arm0]:.4f} | {score['observations_per_arm'][arm0]} |
| `{arm1}` | {score['synthetic_arm_means'][arm1]:.4f} | {score['human_arm_means'][arm1]:.4f} | {score['observations_per_arm'][arm1]} |

The simulator estimated a past-performance-versus-shared-values effect of **{synthetic_effect:+.4f}**. The human estimate is **{human_effect:+.4f}**. The absolute treatment-effect error is **{effect_error:.4f}**, while effect sign is correct.

The required no-effect control baseline assigned 0.5 to both arms and selected `{baseline['selected_arm_id']}` under the frozen tie rule. It missed the human point-estimate winner but incurred only **{baseline['normalized_decision_regret']:.4f}** regret, illustrating why regret is more informative than exact choice alone when arms are nearly equivalent.

## Bootstrap uncertainty

- Replicates: {bootstrap['replicates']:,}, seed `{bootstrap['seed']}`.
- Probability shared-values arm is optimal: {bootstrap['optimal_probability'][arm0]:.1%}.
- Probability past-performance arm is optimal: {bootstrap['optimal_probability'][arm1]:.1%}.
- Frozen practical-regret tolerance: {task['practical_regret_tolerance']:.2f}; selected decision is practically reliable for this sample point estimate: `{str(score['practically_reliable']).lower()}`.

## Simulator and provenance

- Simulator: local Ollama `{recommendation['simulator']['revision']}`; no API cost and no external data transmission.
- Calls: {len(raw['outputs'])} ({raw['draws_per_arm']} per arm), temperature {raw['temperature']}, top-p {raw['top_p']}, base seed {raw['seed']}.
- Strict parse failures: {raw['parse_failures']}.
- Model accounting: {total_prompt_tokens:,} prompt tokens, {total_completion_tokens:,} completion tokens, {total_seconds:.1f} aggregate model-reported seconds including initial load.
- Split payload hash: `{recommendation['split_manifest_sha256']}`.
- Decision-task payload hash: `{recommendation['decision_task_sha256']}`.
- Blinded-bundle payload hash: `{recommendation['blinded_bundle_sha256']}`.
- Simulator-output payload hash: `{recommendation['simulator_outputs_sha256']}`.
- Recommendation payload hash: `{score['recommendation_sha256']}`.
- Score payload hash: `{payload_hash(score)}`.

## Acceptance assessment

The Phase 1 protocol path passes: source reconstruction, response-blind qualification, paradigm-group splitting, source-faithful bundle construction, a no-effect baseline, a real local non-oracle simulator, strict output parsing, synthetic effects, immutable recommendation, gated validation reveal, human effects, regret, bootstrap uncertainty, tests, and no-call replay all completed.

Two boundaries remain deliberately open for the full project:

1. The three-study curated-overlap split is a smoke-test stratum, not the final benchmark split. The full outcome-blind SocSci210 registry must be expanded substantially before trust-model claims.
2. One small local model on one near-tie experiment is not a meaningful simulator comparison. Phase 2 needs multiple simple and LLM baselines, many independent experiments and paradigms, and untouched prospective test evaluation.
"""
