"""Response-free orchestration for the five-experiment portfolio pilot."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from hashlib import sha256
from math import fsum, log
from pathlib import Path
from typing import Any, Mapping

from .compute_budget import verify_bound_budget
from .pilot import PILOT_EXPERIMENTS, build_supported_ordinal_pilot
from .protocol import (
    assert_blinded_payload,
    freeze_envelope,
    freeze_recommendation,
    payload_hash,
    verify_envelope,
    verify_frozen_recommendation,
)
from .simulators import (
    aggregate_ordinal_predictions,
    ordinal_probability_prompt,
    ordinal_variant_contract,
    parse_ordinal_relative_weights,
    validate_ordinal_blinded_bundle,
)


SCOPE_PATH = Path("data/manifests/benchmark/portfolio_pilot_scope.json")
BUDGET_PATH = Path("data/manifests/benchmark/supported_ordinal_compute_budget.json")
SPLIT_PATH = Path("data/manifests/splits/supported_ordinal_pilot.json")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_portfolio_scope(scope: Mapping[str, Any]) -> None:
    assert_blinded_payload(scope)
    if scope.get("schema_version") != "portfolio_pilot_scope.v1":
        raise ValueError("unsupported portfolio-pilot scope schema")
    if scope.get("status") != "response_free_local_execution_authorized":
        raise ValueError("portfolio scope does not authorize the local blind run")
    if tuple(scope.get("experiment_ids", ())) != PILOT_EXPERIMENTS:
        raise ValueError("portfolio scope does not match the frozen five experiments")
    if scope.get("canonical_split_status") != "unassigned":
        raise ValueError("portfolio milestone must not create a canonical split")
    for field in (
        "human_outcome_reveal_authorized",
        "paid_inference_authorized",
        "modal_compute_authorized",
        "fine_tuning_authorized",
        "trust_model_claim_authorized",
    ):
        if scope.get(field) is not False:
            raise ValueError(f"portfolio scope must keep {field} false")
    if scope.get("local_zero_cost_inference_authorized") is not True:
        raise ValueError("portfolio scope must explicitly authorize local inference")


def verify_portfolio_scope(root: Path) -> dict[str, Any]:
    scope = _read_object(root / SCOPE_PATH)
    validate_portfolio_scope(scope)
    split = _read_object(root / SPLIT_PATH)
    if payload_hash(split) != scope.get("engineering_split_sha256"):
        raise ValueError("portfolio scope is not bound to the engineering split")
    if split != build_supported_ordinal_pilot(root):
        raise ValueError("engineering split no longer matches its deterministic build")
    verify_bound_budget(root, root / BUDGET_PATH)
    return scope


def _schema(option_values: tuple[int, ...]) -> dict[str, Any]:
    weight_properties = {
        str(value): {"type": "number", "minimum": 0}
        for value in option_values
    }
    return {
        "type": "object",
        "properties": {
            "relative_weights": {
                "type": "object",
                "properties": weight_properties,
                "required": list(weight_properties),
                "additionalProperties": False,
            }
        },
        "required": ["relative_weights"],
        "additionalProperties": False,
    }


def _normalized_entropy(probabilities: Mapping[str, float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    entropy = -fsum(
        probability * log(probability)
        for probability in probabilities.values()
        if probability > 0.0
    )
    return entropy / log(len(probabilities))


def _winner(means: Mapping[str, float]) -> str:
    maximum = max(means.values())
    return min(arm_id for arm_id, value in means.items() if value == maximum)


def _run_experiment(
    *,
    root: Path,
    experiment_id: str,
    model: str,
    draws: int,
    seed: int,
    temperature: float,
    top_p: float,
    artifact_dir: Path,
    split: Mapping[str, Any],
    scope: Mapping[str, Any],
) -> tuple[str, str]:
    contract_dir = root / "data/manifests/contracts"
    task = _read_object(contract_dir / f"{experiment_id}_decision_task_candidate.json")
    bundle = _read_object(contract_dir / f"{experiment_id}_blinded_bundle.json")
    validate_ordinal_blinded_bundle(bundle)
    if task.get("outcome_access") != "sealed" or task.get("reveal_authorized") is not False:
        raise ValueError(f"task is not sealed: {experiment_id}")
    if payload_hash(task) != split["task_sha256"][experiment_id]:
        raise ValueError(f"task hash changed after pilot freeze: {experiment_id}")
    if payload_hash(bundle) != split["blinded_bundle_sha256"][experiment_id]:
        raise ValueError(f"bundle hash changed after pilot freeze: {experiment_id}")
    option_values = tuple(int(option["value"]) for option in bundle["response_options"])
    outputs: list[dict[str, Any]] = []
    entropies: list[float] = []
    for arm in bundle["arms"]:
        arm_id = arm["arm_id"]
        for variant_id, variant_weight in ordinal_variant_contract(
            bundle, arm_id=arm_id
        ):
            prompt = ordinal_probability_prompt(
                bundle, arm_id=arm_id, variant_id=variant_id
            )
            prompt = prompt.replace(
                'Return only one JSON object with exactly this shape: {"probabilities":{"1":NUMBER,...}}. Include every listed answer value once; probabilities must be between 0 and 1 and sum to 1.',
                'Return only one JSON object with exactly this shape: {"relative_weights":{"1":NUMBER,...}}. Include every listed answer value once. Weights must be finite and non-negative; they express relative likelihood and do not need to sum to 1 because the frozen parser normalizes them. At least one weight must be positive.',
            )
            for draw_index in range(draws):
                request = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": _schema(option_values),
                    "options": {
                        "seed": seed + draw_index,
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_predict": 160,
                    },
                }
                completed = subprocess.run(
                    [
                        "curl",
                        "--fail",
                        "--silent",
                        "--show-error",
                        "http://127.0.0.1:11434/api/generate",
                        "-d",
                        json.dumps(request),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                api_result = json.loads(completed.stdout)
                raw_response = api_result.get("response", "")
                parsed, raw_weights = parse_ordinal_relative_weights(
                    raw_response, option_values=option_values
                )
                probabilities = {
                    str(value): probability for value, probability in parsed.probabilities
                }
                entropies.append(_normalized_entropy(probabilities))
                outputs.append(
                    {
                        "arm_id": arm_id,
                        "variant_id": variant_id,
                        "variant_weight": variant_weight,
                        "draw_index": draw_index,
                        "seed": seed + draw_index,
                        "probabilities": probabilities,
                        "raw_relative_weights": {
                            str(value): weight for value, weight in raw_weights
                        },
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
    synthetic_means, by_draw = aggregate_ordinal_predictions(
        outputs, bundle=bundle, draws=draws
    )
    selected = _winner(synthetic_means)
    draw_winners = [_winner(means) for means in by_draw.values()]
    sorted_means = sorted(synthetic_means.values(), reverse=True)
    control = task["control_arm_id"]
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    raw_payload = {
        "schema_version": "portfolio_ordinal_outputs.v1",
        "portfolio_scope_sha256": payload_hash(scope),
        "experiment_id": experiment_id,
        "model": model,
        "draws_per_arm_variant": draws,
        "temperature": temperature,
        "top_p": top_p,
        "base_seed": seed,
        "parse_failures": 0,
        "human_outcomes_opened": False,
        "created_at_utc": now,
        "outputs": outputs,
    }
    raw_path = artifact_dir / f"{experiment_id}_outputs.json"
    raw_sha = freeze_envelope(raw_payload, raw_path, require_blinded=True)
    recommendation = {
        "schema_version": "portfolio_ordinal_recommendation.v1",
        "portfolio_scope_sha256": payload_hash(scope),
        "experiment_id": experiment_id,
        "engineering_split_label": split["experiment_to_split"][experiment_id],
        "canonical_split_status": "unassigned",
        "selected_arm_id": selected,
        "arm_ranking": sorted(
            synthetic_means, key=lambda arm_id: (-synthetic_means[arm_id], arm_id)
        ),
        "synthetic_arm_means": synthetic_means,
        "synthetic_treatment_effects": {
            arm_id: value - synthetic_means[control]
            for arm_id, value in synthetic_means.items()
            if arm_id != control
        },
        "tie_rule": task["tie_rule"],
        "engineering_split_sha256": payload_hash(split),
        "decision_task_sha256": payload_hash(task),
        "blinded_bundle_sha256": payload_hash(bundle),
        "simulator_outputs_sha256": raw_sha,
        "simulator": {"id": "ollama_local", "revision": model},
        "parser": {"id": "explicit_relative_weight_normalization.v1"},
        "persona_roster": bundle["population"]["roster_id"],
        "diagnostics": {
            "winner_margin": sorted_means[0] - sorted_means[1],
            "mean_normalized_response_entropy": fsum(entropies) / len(entropies),
            "draw_winners": draw_winners,
            "winner_stability": draw_winners.count(selected) / len(draw_winners),
            "parse_failures": 0,
        },
        "human_outcome_reveal_authorized": False,
        "provenance": {
            "created_at_utc": now,
            "base_seed": seed,
            "temperature": temperature,
            "top_p": top_p,
            "draws_per_arm_variant": draws,
            "raw_output_path": str(raw_path.relative_to(root)),
            "local_zero_cost_inference": True,
            "paid_cost_usd": 0.0,
            "modal_used": False,
        },
    }
    recommendation_path = artifact_dir / f"{experiment_id}_recommendation.json"
    recommendation_sha = freeze_recommendation(recommendation, recommendation_path)
    return raw_sha, recommendation_sha


def run_local_portfolio_pilot(
    root: Path,
    *,
    model: str = "llama3.2:3b",
    draws: int = 3,
    seed: int = 2102026,
    temperature: float = 0.35,
    top_p: float = 0.9,
    artifact_dir: Path | None = None,
) -> Path:
    """Run and freeze all five local predictions without loading human outcomes."""

    if draws <= 0:
        raise ValueError("draws must be positive")
    scope = verify_portfolio_scope(root)
    split = _read_object(root / SPLIT_PATH)
    target = artifact_dir or root / "artifacts/portfolio_pilot/local_llama3_2_3b"
    target.mkdir(parents=True, exist_ok=True)
    artifact_hashes: dict[str, Any] = {}
    for experiment_id in PILOT_EXPERIMENTS:
        raw_sha, recommendation_sha = _run_experiment(
            root=root,
            experiment_id=experiment_id,
            model=model,
            draws=draws,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            artifact_dir=target,
            split=split,
            scope=scope,
        )
        artifact_hashes[experiment_id] = {
            "outputs_sha256": raw_sha,
            "recommendation_sha256": recommendation_sha,
        }
    manifest_payload = {
        "schema_version": "portfolio_pilot_run_manifest.v1",
        "portfolio_scope_sha256": payload_hash(scope),
        "engineering_split_sha256": payload_hash(split),
        "simulator": {"id": "ollama_local", "revision": model},
        "experiment_ids": list(PILOT_EXPERIMENTS),
        "artifact_hashes": artifact_hashes,
        "human_outcomes_opened": False,
        "human_outcome_reveal_authorized": False,
        "paid_cost_usd": 0.0,
        "modal_used": False,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    manifest_path = target / "run_manifest.json"
    freeze_envelope(manifest_payload, manifest_path, require_blinded=True)
    return manifest_path


def verify_portfolio_run(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = verify_envelope(manifest_path, require_blinded=True)
    scope = verify_portfolio_scope(root)
    split = _read_object(root / SPLIT_PATH)
    if manifest.get("portfolio_scope_sha256") != payload_hash(scope):
        raise ValueError("portfolio run is not bound to the frozen scope")
    if manifest.get("engineering_split_sha256") != payload_hash(split):
        raise ValueError("portfolio run is not bound to the frozen engineering split")
    if tuple(manifest.get("experiment_ids", ())) != PILOT_EXPERIMENTS:
        raise ValueError("portfolio run does not cover the frozen five experiments")
    if manifest.get("human_outcomes_opened") is not False:
        raise ValueError("response-free run cannot report opened human outcomes")
    artifact_dir = manifest_path.parent
    for experiment_id in PILOT_EXPERIMENTS:
        raw = verify_envelope(
            artifact_dir / f"{experiment_id}_outputs.json", require_blinded=True
        )
        recommendation = verify_frozen_recommendation(
            artifact_dir / f"{experiment_id}_recommendation.json"
        )
        expected = manifest["artifact_hashes"][experiment_id]
        if payload_hash(raw) != expected["outputs_sha256"]:
            raise ValueError("portfolio output hash mismatch")
        if payload_hash(recommendation) != expected["recommendation_sha256"]:
            raise ValueError("portfolio recommendation hash mismatch")
        if recommendation["simulator_outputs_sha256"] != payload_hash(raw):
            raise ValueError("recommendation is not bound to simulator output")
    return manifest
