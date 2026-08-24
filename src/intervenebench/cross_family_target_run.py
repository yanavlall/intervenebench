"""One-time target execution, strict parsing, and recommendation freeze helpers."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from math import fsum, isfinite
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from .confirmation_aggregation import (
    aggregate_experiment,
    continuous_call_summary,
    probability_call_summary,
)
from .confirmation_preparation import (
    CONFIRMATION_IDS,
    DEFAULT_CONFIRMATION_PREPARATION_PATH,
    verify_confirmation_preparation,
)
from .cross_family_adjudication import DEFAULT_CACHE_ATTESTATION_PATH
from .cross_family_execution import (
    DEFAULT_EXECUTION_FREEZE_PATH,
    DEFAULT_MATERIALIZATION_PATH,
    prepare_cross_family_target_requests,
    verify_cross_family_execution_freeze,
)
from .cross_family_json_canary import (
    DEFAULT_JSON_CANARY_RESULT_PATH,
    validate_json_canary_completion,
)
from .cross_family_regression import (
    CANDIDATE_MODEL_ID,
    EXPECTED_CALL_COUNT_BY_EXPERIMENT,
)
from .protocol import assert_blinded_payload, payload_hash, verify_envelope


DEFAULT_TARGET_AUTHORIZATION_PATH = Path(
    "artifacts/cross_family_target/authorizations/target_execution_20260815_v1.json"
)
DEFAULT_TARGET_RUN_ROOT = Path(
    "artifacts/cross_family_target/target_run_20260815_v1"
)
DEFAULT_TARGET_PROGRESS_PATH = Path(
    "artifacts/cross_family_target/progress/target_run_20260815_v1.jsonl"
)
BOOTSTRAP_SEED = 2026081501
BOOTSTRAP_RESAMPLES = 2000

_AUTHORITY_FIELDS = frozenset(
    {
        "modal_image_materialization_authorized",
        "modal_compute_authorized",
        "model_download_authorized",
        "paid_inference_authorized",
        "json_canary_authorized",
        "target_inference_authorized",
        "target_call_authorized",
        "strict_parse_authorized",
        "recommendation_aggregation_authorized",
        "automatic_retry_authorized",
        "reserve_call_authorized",
        "human_outcome_access_authorized",
        "participant_row_access_authorized",
        "participant_row_serialization_authorized",
        "regression_scoring_authorized",
        "automatic_next_stage_authorized",
    }
)


def _authority(*, execute: bool) -> dict[str, bool]:
    value = {field: False for field in sorted(_AUTHORITY_FIELDS)}
    if execute:
        for field in (
            "modal_compute_authorized",
            "paid_inference_authorized",
            "target_inference_authorized",
            "target_call_authorized",
            "strict_parse_authorized",
            "recommendation_aggregation_authorized",
        ):
            value[field] = True
    return value


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_target_bindings(root: Path) -> dict[str, Mapping[str, Any]]:
    freeze = verify_cross_family_execution_freeze(
        root, root / DEFAULT_EXECUTION_FREEZE_PATH
    )
    materialization = verify_envelope(
        root / DEFAULT_MATERIALIZATION_PATH, require_blinded=True
    )
    cache = verify_envelope(
        root / DEFAULT_CACHE_ATTESTATION_PATH, require_blinded=True
    )
    json_canary = verify_envelope(
        root / DEFAULT_JSON_CANARY_RESULT_PATH, require_blinded=True
    )
    if materialization.get("freeze_payload_sha256") != payload_hash(freeze):
        raise ValueError("target materialization/freeze binding drifted")
    if payload_hash(cache) != freeze["model"]["cache_attestation_payload_sha256"]:
        raise ValueError("target cache/freeze binding drifted")
    # The target-free canary's own authorization is embedded by hash inside its
    # completion.  Its result validator proves one call and zero target access.
    canary_authorization = verify_envelope(
        root
        / "artifacts/cross_family_target/authorizations/json_canary_20260815_v1.json",
        require_blinded=True,
    )
    validate_json_canary_completion(
        json_canary,
        freeze=freeze,
        authorization=canary_authorization,
        materialization=materialization,
    )
    return {
        "freeze": freeze,
        "materialization": materialization,
        "cache": cache,
        "json_canary": json_canary,
    }


def build_target_authorization(root: Path) -> dict[str, Any]:
    bindings = load_target_bindings(root)
    freeze = bindings["freeze"]
    materialization = bindings["materialization"]
    cache = bindings["cache"]
    canary = bindings["json_canary"]
    requests = prepare_cross_family_target_requests(root)
    call_ids = [row["call_id"] for row in requests]
    value = {
        "schema_version": "intervenebench.cross_family_target_authorization.v1",
        "status": "authorized_exact_624_outcome_blind_calls_parse_and_recommend_only",
        "freeze_payload_sha256": payload_hash(freeze),
        "call_plan_payload_sha256": freeze["call_plan"]["payload_sha256"],
        "call_ids_sha256": payload_hash(call_ids),
        "request_payload_hashes_sha256": freeze["call_plan"][
            "request_payload_hashes_sha256"
        ],
        "materialization_payload_sha256": payload_hash(materialization),
        "modal_image_id": materialization["modal_image_id"],
        "cache_attestation_payload_sha256": payload_hash(cache),
        "json_canary_completion_payload_sha256": payload_hash(canary),
        "checkpoint_commit": freeze["model"]["checkpoint_commit"],
        "planned_call_count": 624,
        "maximum_attempt_count": 624,
        "experiment_group_order": list(CONFIRMATION_IDS),
        "call_count_by_experiment": dict(EXPECTED_CALL_COUNT_BY_EXPERIMENT),
        "maximum_gpu_seconds": freeze["limits"]["maximum_gpu_seconds"],
        "maximum_wall_clock_seconds": freeze["limits"][
            "maximum_wall_clock_seconds"
        ],
        "hard_incremental_cost_cap_usd": freeze["limits"][
            "hard_incremental_cost_cap_usd"
        ],
        **_authority(execute=True),
    }
    assert_blinded_payload(value)
    return value


def validate_target_authorization(
    authorization: Mapping[str, Any],
    *,
    root: Path,
    freeze: Mapping[str, Any],
    materialization: Mapping[str, Any],
    cache: Mapping[str, Any],
    json_canary: Mapping[str, Any],
) -> None:
    assert_blinded_payload(authorization)
    binding_fields = {
        "schema_version",
        "status",
        "freeze_payload_sha256",
        "call_plan_payload_sha256",
        "call_ids_sha256",
        "request_payload_hashes_sha256",
        "materialization_payload_sha256",
        "modal_image_id",
        "cache_attestation_payload_sha256",
        "json_canary_completion_payload_sha256",
        "checkpoint_commit",
        "planned_call_count",
        "maximum_attempt_count",
        "experiment_group_order",
        "call_count_by_experiment",
        "maximum_gpu_seconds",
        "maximum_wall_clock_seconds",
        "hard_incremental_cost_cap_usd",
    }
    if set(authorization) != binding_fields | _AUTHORITY_FIELDS:
        raise PermissionError("target authorization fields drifted")
    expected_authority = _authority(execute=True)
    if any(
        authorization.get(field) is not value
        for field, value in expected_authority.items()
    ):
        raise PermissionError("target authorization scope widened")
    requests = prepare_cross_family_target_requests(root)
    expected = {
        "schema_version": "intervenebench.cross_family_target_authorization.v1",
        "status": "authorized_exact_624_outcome_blind_calls_parse_and_recommend_only",
        "freeze_payload_sha256": payload_hash(freeze),
        "call_plan_payload_sha256": freeze["call_plan"]["payload_sha256"],
        "call_ids_sha256": payload_hash([row["call_id"] for row in requests]),
        "request_payload_hashes_sha256": freeze["call_plan"][
            "request_payload_hashes_sha256"
        ],
        "materialization_payload_sha256": payload_hash(materialization),
        "modal_image_id": materialization["modal_image_id"],
        "cache_attestation_payload_sha256": payload_hash(cache),
        "json_canary_completion_payload_sha256": payload_hash(json_canary),
        "checkpoint_commit": freeze["model"]["checkpoint_commit"],
        "planned_call_count": 624,
        "maximum_attempt_count": 624,
        "experiment_group_order": list(CONFIRMATION_IDS),
        "call_count_by_experiment": dict(EXPECTED_CALL_COUNT_BY_EXPERIMENT),
        "maximum_gpu_seconds": 100_000,
        "maximum_wall_clock_seconds": 10_800,
        "hard_incremental_cost_cap_usd": 90.0,
    }
    if any(authorization.get(field) != value for field, value in expected.items()):
        raise PermissionError("target authorization binding or ceiling drifted")


def _strict_json_integer(raw_text: Any) -> int:
    try:
        value = json.loads(raw_text)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("target output is not valid JSON") from error
    if (
        not isinstance(value, dict)
        or set(value) != {"predicted_value"}
        or not isinstance(value["predicted_value"], int)
        or isinstance(value["predicted_value"], bool)
        or value["predicted_value"] < 0
    ):
        raise ValueError("target output must be exactly one non-negative integer field")
    return value["predicted_value"]


def strict_parse_target_result(
    request: Mapping[str, Any],
    raw: Mapping[str, Any],
    *,
    freeze: Mapping[str, Any],
    expected_modal_image_id: str,
) -> dict[str, Any]:
    if (
        raw.get("call_id") != request["call_id"]
        or raw.get("model_id") != CANDIDATE_MODEL_ID
        or raw.get("request_payload_sha256") != payload_hash(request)
    ):
        raise ValueError("target raw result identity mismatch")
    runtime = raw.get("runtime_attestation")
    result = raw.get("result")
    if not isinstance(runtime, Mapping) or not isinstance(result, Mapping):
        raise ValueError("target raw result lacks runtime or result payload")
    expected_runtime = {
        "modal_image_id": expected_modal_image_id,
        "checkpoint_commit": freeze["model"]["checkpoint_commit"],
        "cache_attestation_sha256": freeze["model"][
            "cache_attestation_payload_sha256"
        ],
        "call_id": request["call_id"],
        "prompt_sha256": request["source_prompt_sha256"],
        "asset_sha256": request["asset_sha256"],
        "method_id": request["method_id"],
    }
    for field, expected in expected_runtime.items():
        if runtime.get(field) != expected:
            raise ValueError(f"target raw runtime binding mismatch: {field}")
    verified: dict[str, Any] = {
        "schema_version": "intervenebench.cross_family_strict_call.v1",
        "call_id": request["call_id"],
        "model_id": CANDIDATE_MODEL_ID,
        "experiment_id": request["experiment_id"],
        "arm_id": request["arm_id"],
        "nuisance_id": request["nuisance_id"],
        "answer_order": request["answer_order"],
        "stage": request["stage"],
        "prompt_variant": request["prompt_variant"],
        "prompt_sha256": request["source_prompt_sha256"],
        "request_payload_sha256": payload_hash(request),
        "runtime_attestation": dict(runtime),
        "semantic_repair_used": False,
    }
    if request["method_id"] == "forced_choice_next_token_softmax.v1":
        probabilities = result.get("probabilities_by_code")
        codes = request["answer_codes"]
        token_ids = result.get("candidate_token_ids")
        if (
            not isinstance(probabilities, Mapping)
            or set(probabilities) != set(codes)
            or not isinstance(token_ids, list)
            or len(token_ids) != len(codes)
            or len(set(token_ids)) != len(token_ids)
            or any(not isinstance(token, int) or isinstance(token, bool) for token in token_ids)
        ):
            raise ValueError("target forced-choice support or token IDs drifted")
        parsed = {code: probabilities[code] for code in codes}
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(float(value))
            or float(value) < 0.0
            for value in parsed.values()
        ):
            raise ValueError("target forced-choice probabilities are invalid")
        total = fsum(float(value) for value in parsed.values())
        if abs(total - 1.0) > 1e-8:
            raise ValueError("target forced-choice probabilities are not normalized")
        by_source = {
            str(source_value): 0.0 for source_value in request["source_option_values"]
        }
        for code, display_value in zip(
            codes, request["display_option_values"], strict=True
        ):
            by_source[str(display_value)] = float(parsed[code]) / total
        verified["probabilities_by_source_value"] = by_source
        verified["candidate_token_ids"] = list(token_ids)
    elif request["method_id"] == "continuous_constrained_integer_generation.v1":
        raw_text = result.get("raw_text")
        verified["raw_text"] = raw_text
        verified["predicted_value"] = _strict_json_integer(raw_text)
        verified["generation_seed"] = request["generation_seed"]
    else:
        raise ValueError("target result method is not allowlisted")
    assert_blinded_payload(verified)
    return verified


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
    raise ValueError("cross-family task lacks a frozen utility mapping")


def _aggregation_row(
    request: Mapping[str, Any],
    output: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        "call_id": request["call_id"],
        "model_id": CANDIDATE_MODEL_ID,
        "experiment_id": request["experiment_id"],
        "arm_id": request["arm_id"],
        "nuisance_id": request["nuisance_id"],
        "answer_order": request["answer_order"],
        "stage": request["stage"],
        "prompt_variant": request["prompt_variant"],
        "prompt_sha256": request["source_prompt_sha256"],
    }
    if any(output.get(field) != value for field, value in expected.items()):
        raise ValueError("strict target output identity drifted before aggregation")
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
        "model_id": CANDIDATE_MODEL_ID,
        "arm_id": request["arm_id"],
        "nuisance_id": request["nuisance_id"],
        "answer_order": request["answer_order"],
        "stage": request["stage"],
        **summary,
    }


def build_recommendations(
    root: Path,
    *,
    requests: Sequence[Mapping[str, Any]],
    strict_outputs: Mapping[str, Mapping[str, Any]],
    strict_output_hashes: Mapping[str, str],
    parse_failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    preparation = verify_confirmation_preparation(
        root, root / DEFAULT_CONFIRMATION_PREPARATION_PATH
    )
    task_protocol = {
        str(task["experiment_id"]): task for task in preparation["tasks"]
    }
    by_id = {str(request["call_id"]): request for request in requests}
    failures_by_experiment = Counter(
        str(failure["experiment_id"]) for failure in parse_failures
    )
    rng = random.Random(BOOTSTRAP_SEED)
    experiment_results: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    for experiment_id in CONFIRMATION_IDS:
        experiment_requests = [
            request for request in requests if request["experiment_id"] == experiment_id
        ]
        if failures_by_experiment[experiment_id]:
            unavailable.append(
                {
                    "experiment_id": experiment_id,
                    "reason": "one_or_more_strict_parse_failures_no_rerun",
                    "strict_parse_failure_count": failures_by_experiment[experiment_id],
                }
            )
            continue
        task = task_protocol[experiment_id]
        candidate = _read_object(root / task["candidate_path"])
        bundle = _read_object(root / task["blinded_bundle_path"])
        if (
            payload_hash(candidate) != task["candidate_payload_sha256"]
            or payload_hash(bundle) != task["blinded_bundle_payload_sha256"]
        ):
            raise ValueError("cross-family candidate or bundle hash drifted")
        if (
            candidate.get("outcome_access") != "sealed"
            or candidate.get("reveal_authorized") is not False
            or bundle.get("outcome_access") != "sealed"
            or bundle.get("reveal_authorized") is not False
        ):
            raise PermissionError("cross-family aggregation task is not outcome sealed")
        rows = [
            _aggregation_row(
                request,
                strict_outputs[str(request["call_id"])],
                candidate=candidate,
                bundle=bundle,
            )
            for request in experiment_requests
        ]
        arm_ids = [str(arm["arm_id"]) for arm in bundle["arms"]]
        result = aggregate_experiment(
            experiment_id=experiment_id,
            rows=rows,
            arm_ids=arm_ids,
            control_arm_id=str(candidate["control_arm_id"]),
            primary_model_id=CANDIDATE_MODEL_ID,
            bootstrap_resamples=BOOTSTRAP_RESAMPLES,
            rng=rng,
            continuous_unbounded=experiment_id == "tcg8p",
        )
        result.update(
            {
                "arm_order": arm_ids,
                "direction": candidate["direction"],
                "outcome_family": candidate["outcome_family"],
                "outcome_unit": candidate.get("outcome_unit"),
                "candidate_payload_sha256": task["candidate_payload_sha256"],
                "blinded_bundle_payload_sha256": task[
                    "blinded_bundle_payload_sha256"
                ],
                "strict_output_count": len(rows),
                "strict_output_map_sha256": payload_hash(
                    {
                        request["call_id"]: strict_output_hashes[request["call_id"]]
                        for request in experiment_requests
                    }
                ),
            }
        )
        experiment_results.append(result)
    return {
        "schema_version": "intervenebench.cross_family_recommendations.v1",
        "status": "frozen_outcome_blind_recommendations_stop",
        "candidate_model_id": CANDIDATE_MODEL_ID,
        "experiment_order": list(CONFIRMATION_IDS),
        "planned_call_count": len(requests),
        "strict_output_count": len(strict_outputs),
        "strict_parse_failure_count": len(parse_failures),
        "strict_output_map_sha256": payload_hash(dict(strict_output_hashes)),
        "parse_failure_map_sha256": payload_hash(list(parse_failures)),
        "recommendation_count": len(experiment_results),
        "unavailable_experiment_count": len(unavailable),
        "unavailable_experiments": unavailable,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "experiment_results": experiment_results,
        "source_model_calls_made": len(requests),
        "additional_model_calls_made_during_aggregation": 0,
        "semantic_repairs_made": 0,
        "reruns_made": 0,
        "human_outcomes_accessed": False,
        "participant_rows_accessed": 0,
        "human_outcome_scoring_performed": False,
        "automatic_next_stage": False,
    }
