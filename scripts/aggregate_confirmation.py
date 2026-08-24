#!/usr/bin/env python3
"""Freeze confirmation recommendations and trust diagnostics without outcomes."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any, Mapping

from intervenebench.confirmation_aggregation import (
    aggregate_experiment,
    build_trust_ranking,
    continuous_call_summary,
    probability_call_summary,
    validate_aggregation_authorization,
)
from intervenebench.confirmation_calls import (
    DEFAULT_CONFIRMATION_CALL_PLAN_PATH,
    prepare_confirmation_requests,
    verify_confirmation_call_plan,
)
from intervenebench.confirmation_preparation import (
    CONFIRMATION_IDS,
    DEFAULT_CONFIRMATION_PREPARATION_PATH,
    verify_confirmation_preparation,
)
from intervenebench.protocol import (
    assert_blinded_payload,
    freeze_envelope,
    payload_hash,
    verify_envelope,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = Path("data/manifests/research/confirmation_inference_protocol_v1.json")
ADJUDICATION_PATH = Path(
    "artifacts/confirmation/confirmation_20260814_v1/adjudicated_v1/final_manifest.json"
)
STRICT_ROOT = Path(
    "artifacts/confirmation/confirmation_20260814_v1/adjudicated_v1/strict"
)
BOOTSTRAP_SEED = 2026081402
BOOTSTRAP_RESAMPLES = 2000


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _value_to_utility(
    candidate: Mapping[str, Any], bundle: Mapping[str, Any]
) -> dict[int, float] | None:
    if candidate["experiment_id"] == "tcg8p":
        return None
    options = bundle.get("response_options")
    if isinstance(options, list):
        return {
            int(option["value"]): float(option["normalized_utility"])
            for option in options
        }
    if candidate["experiment_id"] == "pb2rr":
        lower = float(candidate["scale_lower"])
        upper = float(candidate["scale_upper"])
        return {
            int(value): (float(value) - lower) / (upper - lower)
            for value in candidate["response_options"]
        }
    raise ValueError("confirmation task lacks a frozen utility mapping")


def _call_row(
    request: Mapping[str, Any],
    output: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    exact = (
        "call_id",
        "model_id",
        "experiment_id",
        "arm_id",
        "nuisance_id",
        "answer_order",
        "stage",
        "prompt_variant",
        "prompt_sha256",
    )
    if any(output.get(key) != request.get(key) for key in exact):
        raise ValueError("strict output identity does not match frozen call")
    mapping = _value_to_utility(candidate, bundle)
    if mapping is None:
        summary = continuous_call_summary(
            output["predicted_value"],
            lower_is_better=candidate["direction"] == "lower_is_better",
        )
    else:
        summary = probability_call_summary(
            output["probabilities_by_source_value"], value_to_utility=mapping
        )
    return {
        "call_id": request["call_id"],
        "model_id": request["model_id"],
        "arm_id": request["arm_id"],
        "nuisance_id": request["nuisance_id"],
        "answer_order": request["answer_order"],
        "stage": request["stage"],
        **summary,
    }


def aggregate(
    *,
    authorization_path: Path,
    output_path: Path,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"create-only aggregation exists: {output_path}")
    preparation = verify_confirmation_preparation(
        ROOT, ROOT / DEFAULT_CONFIRMATION_PREPARATION_PATH
    )
    plan = verify_confirmation_call_plan(
        ROOT, ROOT / DEFAULT_CONFIRMATION_CALL_PLAN_PATH
    )
    protocol = _read_object(ROOT / PROTOCOL_PATH)
    adjudication = verify_envelope(ROOT / ADJUDICATION_PATH, require_blinded=True)
    authorization = verify_envelope(authorization_path, require_blinded=True)
    strict_map = adjudication.get("strict_output_sha256_by_call")
    if not isinstance(strict_map, dict) or len(strict_map) != 1404:
        raise ValueError("adjudicated strict-output map drifted")
    validate_aggregation_authorization(
        authorization,
        run_id=str(adjudication["run_id"]),
        adjudication_manifest_payload_sha256=payload_hash(adjudication),
        call_plan_payload_sha256=payload_hash(plan),
        preparation_payload_sha256=payload_hash(preparation),
        protocol_payload_sha256=payload_hash(protocol),
        strict_output_map_sha256=payload_hash(strict_map),
    )
    if adjudication.get("status") != "confirmation_no_rerun_adjudication_complete_stop":
        raise ValueError("adjudication is not aggregation-ready")
    if adjudication.get("unavailable_call_count") != 60:
        raise ValueError("unavailable confirmation cell drifted")
    unavailable_ids = set(adjudication["unavailable_call_ids"])
    if unavailable_ids.intersection(strict_map):
        raise ValueError("strict and unavailable partitions overlap")

    requests = prepare_confirmation_requests(ROOT, plan=plan, include_reserve=False)
    if len(requests) != 1464:
        raise ValueError("frozen non-reserve request count drifted")
    by_call = {str(request["call_id"]): request for request in requests}
    if set(strict_map).union(unavailable_ids) != set(by_call):
        raise ValueError("adjudicated calls do not cover frozen requests")

    task_protocol = {
        str(task["experiment_id"]): task for task in preparation["tasks"]
    }
    task_data: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for experiment_id in CONFIRMATION_IDS:
        task = task_protocol[experiment_id]
        candidate = _read_object(ROOT / task["candidate_path"])
        bundle = _read_object(ROOT / task["blinded_bundle_path"])
        if payload_hash(candidate) != task["candidate_payload_sha256"] or payload_hash(
            bundle
        ) != task["blinded_bundle_payload_sha256"]:
            raise ValueError("confirmation contract or bundle hash drifted")
        if (
            candidate.get("outcome_access") != "sealed"
            or candidate.get("reveal_authorized") is not False
            or bundle.get("outcome_access") != "sealed"
            or bundle.get("reveal_authorized") is not False
        ):
            raise PermissionError("confirmation task is not outcome sealed")
        task_data[experiment_id] = (candidate, bundle)

    rows_by_experiment: dict[str, list[dict[str, Any]]] = {
        experiment_id: [] for experiment_id in CONFIRMATION_IDS
    }
    strict_hashes_by_experiment: dict[str, dict[str, str]] = {
        experiment_id: {} for experiment_id in CONFIRMATION_IDS
    }
    for call_id, expected_hash in strict_map.items():
        request = by_call[call_id]
        relative = Path(str(request["artifact_relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("strict output path escapes adjudication root")
        output = verify_envelope(ROOT / STRICT_ROOT / relative, require_blinded=True)
        if payload_hash(output) != expected_hash:
            raise ValueError("adjudicated strict output hash drifted")
        experiment_id = str(request["experiment_id"])
        candidate, bundle = task_data[experiment_id]
        rows_by_experiment[experiment_id].append(
            _call_row(request, output, candidate=candidate, bundle=bundle)
        )
        strict_hashes_by_experiment[experiment_id][call_id] = expected_hash

    rng = random.Random(BOOTSTRAP_SEED)
    experiment_results: list[dict[str, Any]] = []
    for experiment_id in CONFIRMATION_IDS:
        task = task_protocol[experiment_id]
        candidate, bundle = task_data[experiment_id]
        arm_ids = [str(arm["arm_id"]) for arm in bundle["arms"]]
        result = aggregate_experiment(
            experiment_id=experiment_id,
            rows=rows_by_experiment[experiment_id],
            arm_ids=arm_ids,
            control_arm_id=str(candidate["control_arm_id"]),
            primary_model_id=str(task["primary_model_id"]),
            bootstrap_resamples=BOOTSTRAP_RESAMPLES,
            rng=rng,
            continuous_unbounded=experiment_id == "tcg8p",
        )
        available_models = sorted(result["model_recommendations"])
        expected_models = set(task["base_model_ids"])
        if experiment_id == "tcg8p":
            expected_models.remove("socrates_qwen2_5_14b_sft")
        if set(available_models) != expected_models:
            raise ValueError("available model-task grid drifted")
        result.update(
            {
                "arm_order": arm_ids,
                "direction": candidate["direction"],
                "outcome_family": candidate["outcome_family"],
                "outcome_unit": candidate.get("outcome_unit"),
                "normalized_for_pooled_regret": experiment_id != "tcg8p",
                "candidate_payload_sha256": task["candidate_payload_sha256"],
                "blinded_bundle_payload_sha256": task[
                    "blinded_bundle_payload_sha256"
                ],
                "strict_output_count": len(rows_by_experiment[experiment_id]),
                "strict_output_map_sha256": payload_hash(
                    strict_hashes_by_experiment[experiment_id]
                ),
                "available_model_ids": available_models,
            }
        )
        experiment_results.append(result)

    diagnostic_rows = [
        {"experiment_id": row["experiment_id"], **row["diagnostics"]}
        for row in experiment_results
    ]
    trust = build_trust_ranking(diagnostic_rows)
    ranking_ids = [row["experiment_id"] for row in trust["ranking"]]
    trust["fixed_coverage_sets"] = {
        "50_percent": ranking_ids[:3],
        "75_percent": ranking_ids[:5],
        "100_percent": ranking_ids,
    }
    model_experiment_count = sum(
        len(row["model_recommendations"]) for row in experiment_results
    )
    if model_experiment_count != 21:
        raise ValueError("confirmation model-experiment recommendation count drifted")
    stage_counts = Counter(
        row["stage"] for rows in rows_by_experiment.values() for row in rows
    )
    if stage_counts != Counter(
        {"base": 1092, "primary_prompt_perturbation": 312}
    ):
        raise ValueError("strict aggregation stage counts drifted")

    payload = {
        "schema_version": "confirmation_outcome_blind_aggregation.v1",
        "run_id": adjudication["run_id"],
        "status": "complete_frozen_outcome_blind_confirmation_aggregation_stop",
        "authorization_payload_sha256": payload_hash(authorization),
        "adjudication_manifest_payload_sha256": payload_hash(adjudication),
        "call_plan_payload_sha256": payload_hash(plan),
        "preparation_payload_sha256": payload_hash(preparation),
        "protocol_payload_sha256": payload_hash(protocol),
        "strict_output_map_sha256": payload_hash(strict_map),
        "aggregation_script_sha256": _file_sha256(Path(__file__)),
        "aggregation_module_sha256": _file_sha256(
            ROOT / "src/intervenebench/confirmation_aggregation.py"
        ),
        "experiment_order": list(CONFIRMATION_IDS),
        "strict_output_count": len(strict_map),
        "strict_output_count_by_stage": dict(sorted(stage_counts.items())),
        "unavailable_call_count": len(unavailable_ids),
        "unavailable_model_task_cell": adjudication[
            "unavailable_model_task_cell"
        ],
        "model_experiment_recommendation_count": model_experiment_count,
        "primary_recommendation_count": len(experiment_results),
        "diagnostic_definition": {
            "primary_normalized_top_two_margin": (
                "bounded tasks: gap between top two mean normalized utilities; "
                "tcg8p: absolute top-two mean-location gap divided by "
                "max(1, abs(best), abs(second))"
            ),
            "primary_resampled_winner_stability": (
                "fraction of 2000 paired nuisance-level bootstrap resamples selecting "
                "the full-sample primary winner; response orders are averaged within "
                "each nuisance level before resampling"
            ),
            "primary_prompt_interface_sensitivity": (
                "maximum absolute base-versus-alternate-format arm mean utility shift; "
                "for tcg8p each arm shift is divided by max(1, abs(base), abs(alternate))"
            ),
            "cross_model_winner_agreement": (
                "fraction of available frozen models selecting the primary-model winner"
            ),
            "cross_model_arm_rank_dispersion": (
                "mean across arms of the population standard deviation across models "
                "of within-model 0-best-to-1-worst midranks"
            ),
            "primary_chosen_arm_normalized_response_entropy": (
                "mean normalized categorical entropy over primary-model base calls in "
                "the selected arm; unavailable for uncapped scalar generation"
            ),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_stream": "single deterministic PRNG stream in frozen experiment order",
        },
        "experiment_results": experiment_results,
        "trust_ranking": trust,
        "adaptive_reserve": "not_run_not_authorized",
        "learned_trust_threshold": None,
        "accept_abstain_policy": "not_validated_not_deployed",
        "model_calls_made": 0,
        "modal_compute_used": False,
        "model_downloads_made": 0,
        "confirmation_outcomes_accessed": False,
        "participant_rows_accessed": 0,
        "human_outcome_scoring_performed": False,
        "automatic_next_stage_authorized": False,
    }
    assert_blinded_payload(payload)
    freeze_envelope(payload, output_path, require_blinded=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate(authorization_path=args.authorization, output_path=args.output)


if __name__ == "__main__":
    main()
