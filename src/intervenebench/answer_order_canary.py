"""Outcome-blind reverse-order canary for the parser-free discovery screen."""

from __future__ import annotations

from hashlib import sha256
from math import fsum, isfinite
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .forced_choice_screen import (
    EXPERIMENT_IDS,
    answer_codes,
    read_json_object,
    sha256_file,
)
from .modal_forced_choice import MODEL_IDS
from .protocol import assert_blinded_payload, payload_hash
from .simulators import validate_ordinal_blinded_bundle


def reverse_order_prompt(
    bundle: Mapping[str, Any], *, arm_id: str, variant_id: str | None
) -> str:
    validate_ordinal_blinded_bundle(bundle)
    options = list(reversed(bundle["response_options"]))
    codes = answer_codes(len(options))
    arms = {arm["arm_id"]: arm for arm in bundle["arms"]}
    if arm_id not in arms:
        raise ValueError("requested reverse-order arm is absent")
    arm = arms[arm_id]
    if "message" in arm:
        if variant_id not in {None, "direct"}:
            raise ValueError("direct reverse-order arm does not accept a variant")
        message = arm["message"]
    else:
        variants = {item["variant_id"]: item for item in arm["message_variants"]}
        if variant_id not in variants:
            raise ValueError("requested reverse-order variant is absent")
        message = variants[str(variant_id)]["message"]
    code_list = ", ".join(codes[:-1]) + f", or {codes[-1]}"
    answers = "\n".join(
        f"{code}: {item['label']}" for code, item in zip(codes, options)
    )
    return (
        "Predict the answer of one randomly selected adult from the population "
        f"below. Do not answer as yourself. Choose exactly one answer code from {code_list}. "
        "Do not explain your choice.\n\n"
        f"Population: {bundle['population']['description']}\n\n"
        f"Context: {bundle['common_context']}\n\n"
        f"Intervention: {message}\n\n"
        f"Question: {bundle['outcome_question']}\n"
        f"Answer codes:\n{answers}\n\n"
        "Return only the answer code."
    )


def build_call_plan(root: Path) -> dict[str, Any]:
    source_plan = read_json_object(
        root / "data/manifests/simulators/forced_choice_screen_plan_v1.json"
    )
    calls = []
    for source_call in source_plan["calls"]:
        bundle = read_json_object(
            root
            / f"data/manifests/contracts/{source_call['experiment_id']}_blinded_bundle.json"
        )
        prompt = reverse_order_prompt(
            bundle, arm_id=source_call["arm_id"],
            variant_id=source_call["variant_id"]
        )
        source_values = [item["value"] for item in bundle["response_options"]]
        call = dict(source_call)
        call.update(
            {
                "call_id": source_call["call_id"].replace(
                    "screen--", "reverse-order--", 1
                ),
                "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
                "display_option_values": list(reversed(source_values)),
                "source_option_values": source_values,
                "order_variant": "full_reverse",
                "source_order_call_id": source_call["call_id"],
                "artifact_relative_path": (
                    f"{source_call['model_id']}/{source_call['experiment_id']}/"
                    f"{source_call['arm_id']}.json"
                ),
            }
        )
        call.pop("option_values")
        calls.append(call)
    return {
        "schema_version": "modal_answer_order_canary_plan.v1",
        "plan_id": "intervenebench-reverse-answer-order-canary-20260813-v1",
        "freeze_date": "2026-08-13",
        "status": "frozen_nonexecuting_zero_authority",
        "experiment_ids": list(EXPERIMENT_IDS),
        "model_ids": list(MODEL_IDS),
        "order_variant": "full_reverse_with_inverse_mapping_to_source_values",
        "source_screen_plan_payload_sha256": payload_hash(source_plan),
        "calls": calls,
    }


def verify_freeze(root: Path, *, freeze_path: Path, plan_path: Path) -> dict[str, Any]:
    freeze = read_json_object(freeze_path)
    plan = read_json_object(plan_path)
    if freeze.get("schema_version") != "modal_answer_order_canary_freeze.v1":
        raise ValueError("unsupported answer-order freeze")
    if freeze.get("status") != "frozen_nonexecuting_zero_authority":
        raise ValueError("answer-order freeze must have zero authority")
    if any(freeze["authority"].values()):
        raise PermissionError("answer-order freeze embeds expanded authority")
    if plan != build_call_plan(root):
        raise ValueError("answer-order plan does not replay exactly")
    if len(plan["calls"]) != 40 or freeze["call_plan_payload_sha256"] != payload_hash(plan):
        raise ValueError("answer-order freeze/plan binding mismatch")
    for entry in freeze["implementation_hashes"]:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("answer-order implementation path escapes repository")
        if sha256_file(root / relative) != entry["file_sha256"]:
            raise ValueError(f"answer-order implementation hash mismatch: {entry['path']}")
    lock = freeze["dependency_lock"]
    for path_key, hash_key in (
        ("input_path", "input_file_sha256"), ("lock_path", "lock_file_sha256")
    ):
        if sha256_file(root / lock[path_key]) != lock[hash_key]:
            raise ValueError("answer-order dependency hash mismatch")
    for entry in freeze["task_scope"]["packaged_files"]:
        bundle = read_json_object(root / entry["path"])
        if sha256_file(root / entry["path"]) != entry["file_sha256"]:
            raise ValueError("answer-order bundle file hash mismatch")
        if payload_hash(bundle) != entry["payload_sha256"]:
            raise ValueError("answer-order bundle payload hash mismatch")
        if bundle["outcome_access"] != "sealed" or bundle["reveal_authorized"] is not False:
            raise PermissionError("answer-order bundle is not outcome sealed")
    if tuple(item["model_id"] for item in freeze["models"]) != MODEL_IDS:
        raise ValueError("answer-order model allowlist drifted")
    limits = freeze["limits"]
    if (limits["maximum_planned_calls"], limits["maximum_model_attempts"]) != (40, 40):
        raise ValueError("answer-order attempt ceiling drifted")
    if float(limits["hard_incremental_cost_cap_usd"]) != 1.75:
        raise ValueError("answer-order cost cap drifted")
    thresholds = freeze["robustness_gate"]
    if thresholds != {
        "maximum_median_total_variation": 0.10,
        "maximum_nearest_rank_p90_total_variation": 0.25,
        "minimum_modal_response_stability": 0.75,
        "minimum_screened_pair_choice_stability": 0.80,
        "required_to_scale_single_order_method": "all_thresholds_pass",
        "failure_pivot": "do_not_scale_single_order; develop_balanced_permutation_average",
    }:
        raise ValueError("answer-order robustness gate drifted")
    assert_blinded_payload(freeze)
    assert_blinded_payload(plan)
    return {
        "call_count": 40,
        "incremental_cost_cap_usd": 1.75,
        "freeze_payload_sha256": payload_hash(freeze),
        "call_plan_payload_sha256": payload_hash(plan),
    }


def prepare_requests(root: Path, *, freeze_path: Path, plan_path: Path) -> tuple[dict[str, Any], ...]:
    verify_freeze(root, freeze_path=freeze_path, plan_path=plan_path)
    plan = read_json_object(plan_path)
    requests = []
    for call in plan["calls"]:
        bundle = read_json_object(
            root / f"data/manifests/contracts/{call['experiment_id']}_blinded_bundle.json"
        )
        request = dict(call)
        request["prompt"] = reverse_order_prompt(
            bundle, arm_id=call["arm_id"], variant_id=call["variant_id"]
        )
        assert_blinded_payload(request)
        requests.append(request)
    return tuple(requests)


def verify_result(request: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "call_id", "model_id", "probabilities_by_code", "candidate_token_ids",
        "candidate_token_strings", "runtime_attestation"
    }
    if set(result) != required:
        raise ValueError("answer-order result fields differ from schema")
    if result["call_id"] != request["call_id"] or result["model_id"] != request["model_id"]:
        raise ValueError("answer-order result identity mismatch")
    probabilities = result["probabilities_by_code"]
    codes = tuple(request["answer_codes"])
    if not isinstance(probabilities, Mapping) or tuple(probabilities) != codes:
        raise ValueError("answer-order probability code order mismatch")
    values = list(probabilities.values())
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not isfinite(value) or not 0.0 <= value <= 1.0
        for value in values
    ) or abs(fsum(values) - 1.0) > 1e-6:
        raise ValueError("answer-order probabilities are invalid")
    token_ids = result["candidate_token_ids"]
    if (
        not isinstance(token_ids, list) or len(token_ids) != len(codes)
        or len(set(token_ids)) != len(token_ids)
        or not all(isinstance(item, int) for item in token_ids)
    ):
        raise ValueError("answer-order token contract failed")
    if result["candidate_token_strings"] != list(codes):
        raise ValueError("answer-order token strings mismatch")
    attestation = result["runtime_attestation"]
    if not isinstance(attestation, Mapping):
        raise ValueError("answer-order runtime attestation missing")
    return {
        "call_id": request["call_id"],
        "source_order_call_id": request["source_order_call_id"],
        "model_id": request["model_id"],
        "experiment_id": request["experiment_id"],
        "arm_id": request["arm_id"],
        "order_variant": "full_reverse",
        "prompt_sha256": request["prompt_sha256"],
        "method_id": "forced_choice_next_token_softmax.v1",
        "answer_code_probabilities": dict(probabilities),
        "probabilities": {
            value: float(probabilities[code])
            for code, value in zip(codes, request["display_option_values"])
        },
        "source_option_values": request["source_option_values"],
        "display_option_values": request["display_option_values"],
        "candidate_token_ids": token_ids,
        "runtime_attestation": dict(attestation),
    }


def build_materialization_authorization(
    *, freeze: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": "modal_answer_order_materialization_authorization.v1",
        "authorization_id": "intervenebench-answer-order-materialization-20260813-v1",
        "scope": "materialize_exact_reverse_order_image_only",
        "freeze_payload_sha256": payload_hash(freeze),
        "call_plan_payload_sha256": payload_hash(plan),
        "modal_profile": "yanav",
        "image_materialization_authorized": True,
        "model_download_authorized": False,
        "paid_inference_authorized": False,
        "sealed_task_inference_authorized": False,
        "outcome_access_authorized": False,
        "fine_tuning_authorized": False,
        "next_stage_authorized": False,
        "maximum_incremental_cost_usd": 1.75,
        "status": "answer_order_image_only_authorized",
    }
    assert_blinded_payload(payload)
    return payload


def validate_materialization_authorization(
    authorization: Mapping[str, Any], *, freeze: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    if dict(authorization) != build_materialization_authorization(freeze=freeze, plan=plan):
        raise PermissionError("answer-order materialization authorization mismatch")


def build_execution_authorization(
    *, freeze: Mapping[str, Any], plan: Mapping[str, Any], modal_image_id: str,
    cache_hashes: Mapping[str, str]
) -> dict[str, Any]:
    if not modal_image_id or set(cache_hashes) != set(MODEL_IDS):
        raise ValueError("answer-order image/cache binding incomplete")
    payload = {
        "schema_version": "modal_answer_order_execution_authorization.v1",
        "authorization_id": "intervenebench-answer-order-execution-20260813-v1",
        "scope": "exact_40_call_reverse_order_outcome_blind_canary_only",
        "freeze_payload_sha256": payload_hash(freeze),
        "call_plan_payload_sha256": payload_hash(plan),
        "modal_profile": "yanav",
        "modal_image_id": modal_image_id,
        "cache_attestation_sha256_by_model": dict(sorted(cache_hashes.items())),
        "model_download_authorized": False,
        "modal_execution_authorized": True,
        "paid_inference_authorized": True,
        "sealed_task_inference_authorized": False,
        "outcome_access_authorized": False,
        "fine_tuning_authorized": False,
        "next_stage_authorized": False,
        "maximum_planned_calls": 40,
        "maximum_model_attempts": 40,
        "maximum_incremental_cost_usd": 1.75,
        "status": "single_answer_order_canary_authorized",
    }
    assert_blinded_payload(payload)
    return payload


def validate_execution_authorization(
    authorization: Mapping[str, Any], *, freeze: Mapping[str, Any], plan: Mapping[str, Any],
    modal_image_id: str, cache_hashes: Mapping[str, str]
) -> None:
    expected = build_execution_authorization(
        freeze=freeze, plan=plan, modal_image_id=modal_image_id,
        cache_hashes=cache_hashes
    )
    if dict(authorization) != expected:
        raise PermissionError("answer-order execution authorization mismatch")


def validate_runtime_attestation(
    request: Mapping[str, Any], attestation: Mapping[str, Any], *,
    freeze: Mapping[str, Any], authorization: Mapping[str, Any]
) -> None:
    model = {item["model_id"]: item for item in freeze["models"]}[request["model_id"]]
    expected = {
        "modal_sdk_version": "1.5.4",
        "modal_image_id": authorization["modal_image_id"],
        "image_recipe_sha256": payload_hash(freeze["runtime"]["image_recipe"]),
        "dependency_lock_sha256": freeze["dependency_lock"]["lock_file_sha256"],
        "transformers_version": "4.57.6", "cuda_runtime_version": "12.8",
        "checkpoint_commit": model["checkpoint_commit"],
        "weight_manifest_sha256": model["weight_file_manifest_sha256"],
        "tokenizer_manifest_sha256": model["tokenizer_manifest_sha256"],
        "cache_attestation_sha256": authorization[
            "cache_attestation_sha256_by_model"
        ][request["model_id"]],
        "call_id": request["call_id"], "prompt_sha256": request["prompt_sha256"],
        "method_id": "forced_choice_next_token_softmax.v1",
    }
    for field, value in expected.items():
        if attestation.get(field) != value:
            raise ValueError(f"answer-order runtime mismatch: {field}")
    if str(attestation.get("torch_version", "")).split("+")[0] != "2.9.1":
        raise ValueError("answer-order runtime mismatch: torch_version")
    if str(attestation.get("python_version", "")).split(".")[:2] != ["3", "11"]:
        raise ValueError("answer-order runtime mismatch: python_version")
    if "L40S" not in str(attestation.get("gpu_name", "")):
        raise ValueError("answer-order runtime mismatch: gpu_name")
