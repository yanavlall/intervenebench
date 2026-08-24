"""Local, hash-bound helpers for the separately authorized Modal preflight."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from .modal_freeze import verify_modal_preflight_freeze
from .protocol import assert_blinded_payload, payload_hash
from .simulators import ordinal_probability_prompt, parse_ordinal_distribution


@dataclass(frozen=True, slots=True)
class PreparedModalCall:
    call_id: str
    model_id: str
    experiment_id: str
    option_values: tuple[int, ...]
    prompt: str
    request: dict[str, Any]


MATERIALIZATION_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema_version",
        "authorization_id",
        "scope",
        "freeze_payload_sha256",
        "call_plan_payload_sha256",
        "modal_profile",
        "image_materialization_authorized",
        "model_download_authorized",
        "paid_inference_authorized",
        "sealed_task_inference_authorized",
        "outcome_access_authorized",
        "fine_tuning_authorized",
        "next_stage_authorized",
        "maximum_total_cost_usd",
        "status",
    }
)
CACHE_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema_version",
        "authorization_id",
        "scope",
        "freeze_payload_sha256",
        "call_plan_payload_sha256",
        "modal_profile",
        "modal_image_id",
        "image_recipe_sha256",
        "dependency_lock_sha256",
        "model_ids",
        "model_download_authorized",
        "paid_inference_authorized",
        "sealed_task_inference_authorized",
        "outcome_access_authorized",
        "fine_tuning_authorized",
        "next_stage_authorized",
        "maximum_total_cost_usd",
        "status",
    }
)
EXECUTION_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema_version",
        "authorization_id",
        "scope",
        "freeze_payload_sha256",
        "call_plan_payload_sha256",
        "modal_profile",
        "modal_image_id",
        "image_recipe_sha256",
        "dependency_lock_sha256",
        "cache_attestation_sha256_by_model",
        "model_download_authorized",
        "modal_execution_authorized",
        "paid_inference_authorized",
        "sealed_task_inference_authorized",
        "outcome_access_authorized",
        "fine_tuning_authorized",
        "next_stage_authorized",
        "maximum_planned_calls",
        "maximum_model_attempts",
        "maximum_total_cost_usd",
        "status",
    }
)


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def prepare_modal_calls(
    root: Path, *, freeze_path: Path, call_plan_path: Path
) -> tuple[PreparedModalCall, ...]:
    """Reconstruct all 40 prompts locally from allowlisted blinded bundles."""

    verify_modal_preflight_freeze(
        root, freeze_path=freeze_path, call_plan_path=call_plan_path
    )
    freeze = read_json_object(freeze_path)
    plan = read_json_object(call_plan_path)
    bundle_paths = {
        entry["experiment_id"]: root / entry["path"]
        for entry in freeze["task_scope"]["packaged_files"]
    }
    generation = freeze["generation"]
    prepared: list[PreparedModalCall] = []
    for call in plan["calls"]:
        bundle = read_json_object(bundle_paths[call["experiment_id"]])
        prompt = ordinal_probability_prompt(
            bundle, arm_id=call["arm_id"], variant_id=call["variant_id"]
        )
        request = {
            "call_id": call["call_id"],
            "model_id": call["model_id"],
            "experiment_id": call["experiment_id"],
            "arm_id": call["arm_id"],
            "variant_id": call["variant_id"],
            "prompt": prompt,
            "prompt_sha256": call["prompt_sha256"],
            "bundle_payload_sha256": call["bundle_payload_sha256"],
            "json_schema_sha256": call["json_schema_sha256"],
            "parser_id": call["parser_id"],
            "seed": call["seed"],
            "temperature": call["temperature"],
            "top_p": call["top_p"],
            "maximum_output_tokens": call["maximum_output_tokens"],
            "option_values": [option["value"] for option in bundle["response_options"]],
            "artifact_relative_path": call["artifact_relative_path"],
        }
        assert_blinded_payload(request)
        prepared.append(
            PreparedModalCall(
                call_id=call["call_id"],
                model_id=call["model_id"],
                experiment_id=call["experiment_id"],
                option_values=tuple(request["option_values"]),
                prompt=prompt,
                request=request,
            )
        )
    return tuple(prepared)


def verify_remote_result(call: PreparedModalCall, result: Mapping[str, Any]) -> dict:
    """Validate one raw remote response locally without repair or reprompting."""

    allowed = {"call_id", "model_id", "raw_text", "runtime_attestation"}
    if set(result) != allowed:
        raise ValueError("remote result fields do not match the frozen schema")
    if result["call_id"] != call.call_id or result["model_id"] != call.model_id:
        raise ValueError("remote result identity mismatch")
    raw_text = result["raw_text"]
    if not isinstance(raw_text, str) or not raw_text:
        raise ValueError("remote result raw text is empty")
    attestation = result["runtime_attestation"]
    if not isinstance(attestation, Mapping):
        raise ValueError("runtime attestation is missing")
    distribution = parse_ordinal_distribution(
        raw_text, option_values=call.option_values
    )
    return {
        "call_id": call.call_id,
        "model_id": call.model_id,
        "experiment_id": call.experiment_id,
        "prompt_sha256": sha256(call.prompt.encode("utf-8")).hexdigest(),
        "raw_text": raw_text,
        "probabilities": dict(distribution.probabilities),
        "runtime_attestation": dict(attestation),
    }


def validate_runtime_attestation(
    call: PreparedModalCall,
    attestation: Mapping[str, Any],
    *,
    freeze: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> None:
    """Fail closed on any runtime identity or call-binding mismatch."""

    required = set(freeze["runtime"]["required_runtime_attestation"]) | {
        "call_id",
        "prompt_sha256",
        "torch_version",
        "transformers_version",
        "cache_attestation_sha256",
    }
    missing = sorted(required - set(attestation))
    if missing:
        raise ValueError(f"runtime attestation is missing fields: {missing}")
    models = {model["model_id"]: model for model in freeze["models"]}
    model = models[call.model_id]
    package_versions = {
        package.split("==", 1)[0]: package.split("==", 1)[1]
        for package in freeze["runtime"]["packages"]
    }
    expected = {
        "modal_sdk_version": freeze["runtime"]["modal_sdk_version"],
        "modal_image_id": authorization["modal_image_id"],
        "image_recipe_sha256": payload_hash(freeze["runtime"]["image_recipe"]),
        "dependency_lock_sha256": freeze["dependency_lock"]["lock_file_sha256"],
        "torch_version": package_versions["torch"],
        "transformers_version": package_versions["transformers"],
        "cuda_runtime_version": freeze["runtime"]["expected_cuda_runtime_version"],
        "checkpoint_commit": model["checkpoint_commit"],
        "weight_manifest_sha256": model["weight_file_manifest_sha256"],
        "tokenizer_manifest_sha256": model["tokenizer_manifest_sha256"],
        "cache_attestation_sha256": authorization[
            "cache_attestation_sha256_by_model"
        ][call.model_id],
        "call_id": call.call_id,
        "prompt_sha256": call.request["prompt_sha256"],
    }
    for field, value in expected.items():
        actual = attestation[field]
        if field == "torch_version":
            actual = str(actual).split("+")[0]
        if actual != value:
            raise ValueError(f"runtime attestation mismatch: {field}")
    if str(attestation["python_version"]).split(".")[:2] != ["3", "11"]:
        raise ValueError("runtime attestation mismatch: python_version")
    if "L40S" not in str(attestation["gpu_name"]):
        raise ValueError("runtime attestation mismatch: gpu_name")


def build_materialization_authorization_payload(
    *,
    freeze: Mapping[str, Any],
    call_plan: Mapping[str, Any],
    modal_profile: str,
    maximum_total_cost_usd: float,
) -> dict[str, Any]:
    """Authorize image hydration only, before an immutable image ID exists."""

    payload = {
        "schema_version": "modal_preflight_materialization_authorization.v1",
        "authorization_id": (
            "intervenebench-modal-preflight-materialization-20260813-v1"
        ),
        "scope": "materialize_exact_locked_modal_image_only",
        "freeze_payload_sha256": payload_hash(freeze),
        "call_plan_payload_sha256": payload_hash(call_plan),
        "modal_profile": modal_profile,
        "image_materialization_authorized": True,
        "model_download_authorized": False,
        "paid_inference_authorized": False,
        "sealed_task_inference_authorized": False,
        "outcome_access_authorized": False,
        "fine_tuning_authorized": False,
        "next_stage_authorized": False,
        "maximum_total_cost_usd": maximum_total_cost_usd,
        "status": "image_materialization_only_authorized",
    }
    assert_blinded_payload(payload)
    return payload


def build_cache_authorization_payload(
    *,
    freeze: Mapping[str, Any],
    call_plan: Mapping[str, Any],
    modal_profile: str,
    modal_image_id: str,
    maximum_total_cost_usd: float,
) -> dict[str, Any]:
    """Authorize exact checkpoint caching, while keeping GPU inference denied."""

    if not modal_image_id.strip():
        raise ValueError("materialized Modal image ID is required")
    payload = {
        "schema_version": "modal_preflight_cache_authorization.v1",
        "authorization_id": "intervenebench-modal-preflight-cache-20260813-v1",
        "scope": "cache_and_verify_exact_four_pinned_public_checkpoints_only",
        "freeze_payload_sha256": payload_hash(freeze),
        "call_plan_payload_sha256": payload_hash(call_plan),
        "modal_profile": modal_profile,
        "modal_image_id": modal_image_id,
        "image_recipe_sha256": payload_hash(freeze["runtime"]["image_recipe"]),
        "dependency_lock_sha256": freeze["dependency_lock"]["lock_file_sha256"],
        "model_ids": [model["model_id"] for model in freeze["models"]],
        "model_download_authorized": True,
        "paid_inference_authorized": False,
        "sealed_task_inference_authorized": False,
        "outcome_access_authorized": False,
        "fine_tuning_authorized": False,
        "next_stage_authorized": False,
        "maximum_total_cost_usd": maximum_total_cost_usd,
        "status": "checkpoint_cache_only_authorized",
    }
    assert_blinded_payload(payload)
    return payload


def build_execution_authorization_payload(
    *,
    freeze: Mapping[str, Any],
    call_plan: Mapping[str, Any],
    modal_profile: str,
    modal_image_id: str,
    cache_attestation_sha256_by_model: Mapping[str, str],
    maximum_total_cost_usd: float,
) -> dict[str, Any]:
    """Create the separate one-stage authority object; callers freeze it create-only."""

    if not modal_image_id.strip():
        raise ValueError("materialized Modal image ID is required")
    model_ids = [model["model_id"] for model in freeze["models"]]
    if set(cache_attestation_sha256_by_model) != set(model_ids):
        raise ValueError("cache attestations must cover the exact four frozen models")
    for digest in cache_attestation_sha256_by_model.values():
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("cache attestation hashes must be SHA-256 digests")
        int(digest, 16)
    payload = {
        "schema_version": "modal_preflight_execution_authorization.v1",
        "authorization_id": "intervenebench-modal-preflight-authority-20260813-v2",
        "scope": "run_exact_40_call_discovery_parser_preflight_only",
        "freeze_payload_sha256": payload_hash(freeze),
        "call_plan_payload_sha256": payload_hash(call_plan),
        "modal_profile": modal_profile,
        "modal_image_id": modal_image_id,
        "image_recipe_sha256": payload_hash(freeze["runtime"]["image_recipe"]),
        "dependency_lock_sha256": freeze["dependency_lock"]["lock_file_sha256"],
        "cache_attestation_sha256_by_model": dict(
            sorted(cache_attestation_sha256_by_model.items())
        ),
        "model_download_authorized": False,
        "modal_execution_authorized": True,
        "paid_inference_authorized": True,
        "sealed_task_inference_authorized": False,
        "outcome_access_authorized": False,
        "fine_tuning_authorized": False,
        "next_stage_authorized": False,
        "maximum_planned_calls": 40,
        "maximum_model_attempts": 44,
        "maximum_total_cost_usd": maximum_total_cost_usd,
        "status": "single_stage_authorized",
    }
    assert_blinded_payload(payload)
    return payload


def validate_materialization_authorization(
    authorization: Mapping[str, Any],
    *,
    freeze: Mapping[str, Any],
    call_plan: Mapping[str, Any],
) -> None:
    if set(authorization) != MATERIALIZATION_AUTHORIZATION_FIELDS:
        raise ValueError("materialization authorization fields do not match schema")
    if authorization["schema_version"] != (
        "modal_preflight_materialization_authorization.v1"
    ) or authorization["status"] != "image_materialization_only_authorized":
        raise ValueError("unsupported materialization authorization")
    _validate_common_authorization_bindings(
        authorization, freeze=freeze, call_plan=call_plan
    )
    if authorization["image_materialization_authorized"] is not True:
        raise PermissionError("image materialization is not authorized")
    for field in (
        "model_download_authorized",
        "paid_inference_authorized",
        "sealed_task_inference_authorized",
        "outcome_access_authorized",
        "fine_tuning_authorized",
        "next_stage_authorized",
    ):
        if authorization[field] is not False:
            raise PermissionError(f"materialization authority expanded: {field}")


def validate_cache_authorization(
    authorization: Mapping[str, Any],
    *,
    freeze: Mapping[str, Any],
    call_plan: Mapping[str, Any],
    modal_image_id: str,
) -> None:
    if set(authorization) != CACHE_AUTHORIZATION_FIELDS:
        raise ValueError("cache authorization fields do not match schema")
    if authorization["schema_version"] != "modal_preflight_cache_authorization.v1" or (
        authorization["status"] != "checkpoint_cache_only_authorized"
    ):
        raise ValueError("unsupported cache authorization")
    _validate_common_authorization_bindings(
        authorization, freeze=freeze, call_plan=call_plan
    )
    _validate_runtime_bindings(
        authorization, freeze=freeze, modal_image_id=modal_image_id
    )
    model_ids = [model["model_id"] for model in freeze["models"]]
    if authorization["model_ids"] != model_ids:
        raise ValueError("cache model allowlist differs from the freeze")
    if authorization["model_download_authorized"] is not True:
        raise PermissionError("model caching is not authorized")
    for field in (
        "paid_inference_authorized",
        "sealed_task_inference_authorized",
        "outcome_access_authorized",
        "fine_tuning_authorized",
        "next_stage_authorized",
    ):
        if authorization[field] is not False:
            raise PermissionError(f"cache authority expanded: {field}")


def validate_execution_authorization(
    authorization: Mapping[str, Any],
    *,
    freeze: Mapping[str, Any],
    call_plan: Mapping[str, Any],
    modal_image_id: str,
    cache_attestation_sha256_by_model: Mapping[str, str],
) -> None:
    if set(authorization) != EXECUTION_AUTHORIZATION_FIELDS:
        raise ValueError("execution authorization fields do not match schema")
    if authorization["schema_version"] != (
        "modal_preflight_execution_authorization.v1"
    ) or authorization["status"] != "single_stage_authorized":
        raise ValueError("unsupported execution authorization")
    _validate_common_authorization_bindings(
        authorization, freeze=freeze, call_plan=call_plan
    )
    _validate_runtime_bindings(
        authorization, freeze=freeze, modal_image_id=modal_image_id
    )
    if authorization["cache_attestation_sha256_by_model"] != dict(
        sorted(cache_attestation_sha256_by_model.items())
    ):
        raise ValueError("execution authority has different cache attestations")
    if authorization["model_download_authorized"] is not False:
        raise PermissionError("execution authority must not permit model downloads")
    for field in ("modal_execution_authorized", "paid_inference_authorized"):
        if authorization[field] is not True:
            raise PermissionError(f"execution is not authorized: {field}")
    for field in (
        "sealed_task_inference_authorized",
        "outcome_access_authorized",
        "fine_tuning_authorized",
        "next_stage_authorized",
    ):
        if authorization[field] is not False:
            raise PermissionError(f"execution authority expanded: {field}")
    if authorization["maximum_planned_calls"] != 40 or authorization[
        "maximum_model_attempts"
    ] != 44:
        raise ValueError("execution call or attempt limit drifted")


def _validate_common_authorization_bindings(
    authorization: Mapping[str, Any],
    *,
    freeze: Mapping[str, Any],
    call_plan: Mapping[str, Any],
) -> None:
    assert_blinded_payload(authorization)
    if authorization["freeze_payload_sha256"] != payload_hash(freeze):
        raise ValueError("authorization is not bound to the freeze")
    if authorization["call_plan_payload_sha256"] != payload_hash(call_plan):
        raise ValueError("authorization is not bound to the call plan")
    if authorization["modal_profile"] != "yanav":
        raise ValueError("authorization uses a different Modal profile")
    cap = float(freeze["limits"]["hard_total_cost_cap_usd"])
    if float(authorization["maximum_total_cost_usd"]) != cap:
        raise ValueError("authorization cost cap differs from the freeze")


def _validate_runtime_bindings(
    authorization: Mapping[str, Any],
    *,
    freeze: Mapping[str, Any],
    modal_image_id: str,
) -> None:
    if authorization["modal_image_id"] != modal_image_id:
        raise ValueError("authorization is bound to a different Modal image")
    if authorization["image_recipe_sha256"] != payload_hash(
        freeze["runtime"]["image_recipe"]
    ):
        raise ValueError("authorization image recipe hash mismatch")
    if authorization["dependency_lock_sha256"] != freeze["dependency_lock"][
        "lock_file_sha256"
    ]:
        raise ValueError("authorization dependency lock hash mismatch")
