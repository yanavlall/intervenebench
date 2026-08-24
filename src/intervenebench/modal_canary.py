"""Outcome-blind helpers for the four-call grammar-constrained Modal canary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .protocol import assert_blinded_payload, payload_hash
from .simulators import (
    ordinal_variant_contract,
    parse_ordinal_relative_weights,
    validate_ordinal_blinded_bundle,
)


MODEL_IDS = (
    "qwen3_8b_generic",
    "qwen3_14b_generic",
    "qwen2_5_14b_generic",
    "socrates_qwen2_5_14b_sft",
)
EXPERIMENT_ID = "5vm8g"
BASE_SEED = 21_026_000
WEIGHT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["relative_weights"],
    "properties": {
        "relative_weights": {
            "type": "object",
            "additionalProperties": False,
            "required": ["1", "2", "3", "4", "5"],
            "properties": {
                str(value): {"type": "integer", "minimum": 1, "maximum": 100}
                for value in range(1, 6)
            },
        }
    },
}


@dataclass(frozen=True, slots=True)
class PreparedCanaryCall:
    call_id: str
    model_id: str
    option_values: tuple[int, ...]
    prompt: str
    request: dict[str, Any]


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_weights_prompt(
    bundle: Mapping[str, Any], *, arm_id: str, variant_id: str | None
) -> str:
    """Build the frozen positive-integer response contract for the canary."""

    validate_ordinal_blinded_bundle(bundle)
    arms = {arm["arm_id"]: arm for arm in bundle["arms"]}
    if arm_id not in arms:
        raise ValueError("requested arm is absent from ordinal bundle")
    arm = arms[arm_id]
    if "message" in arm:
        if variant_id not in {None, "direct"}:
            raise ValueError("direct ordinal arm does not accept a message variant")
        message = arm["message"]
    else:
        variants = {item["variant_id"]: item for item in arm["message_variants"]}
        if variant_id not in variants:
            raise ValueError("requested message variant is absent from ordinal arm")
        message = variants[str(variant_id)]["message"]
    answers = ", ".join(
        f"{item['value']}={item['label']}" for item in bundle["response_options"]
    )
    keys = ",".join(
        f'"{item["value"]}":INTEGER' for item in bundle["response_options"]
    )
    return (
        "Estimate the response distribution for a random adult in the population "
        "below. Do not answer as yourself. Return only one JSON object with exactly "
        f'this shape: {{"relative_weights":{{{keys}}}}}. Include every listed '
        "answer value once. Each weight must be a positive integer from 1 through "
        "100. The weights do not need to sum to 100; they will be normalized.\n\n"
        f"Population: {bundle['population']['description']}\n\n"
        f"Context: {bundle['common_context']}\n\n"
        f"Intervention: {message}\n\n"
        f"Question: {bundle['outcome_question']}\n"
        f"Answers: {answers}"
    )


def build_canary_call_plan(root: Path) -> dict[str, Any]:
    bundle_path = root / "data/manifests/contracts/5vm8g_blinded_bundle.json"
    bundle = read_json_object(bundle_path)
    arm = bundle["arms"][0]
    arm_id = arm["arm_id"]
    variant_id = ordinal_variant_contract(bundle, arm_id=arm_id)[0][0]
    prompt = relative_weights_prompt(
        bundle, arm_id=arm_id, variant_id=variant_id
    )
    prompt_digest = sha256(prompt.encode("utf-8")).hexdigest()
    schema_digest = payload_hash(WEIGHT_SCHEMA)
    calls = []
    for index, model_id in enumerate(MODEL_IDS):
        calls.append(
            {
                "call_id": f"canary--{model_id}--5vm8g--{arm_id}",
                "model_id": model_id,
                "experiment_id": EXPERIMENT_ID,
                "bundle_payload_sha256": payload_hash(bundle),
                "arm_id": arm_id,
                "variant_id": variant_id,
                "prompt_sha256": prompt_digest,
                "json_schema_sha256": schema_digest,
                "parser_id": "parse_ordinal_relative_weights.v1",
                "constraint_backend": "outlines_core_json_schema",
                "seed": BASE_SEED + 10 * index,
                "temperature": 0.2,
                "top_p": 0.95,
                "maximum_output_tokens": 128,
                "artifact_relative_path": f"{model_id}/5vm8g/{arm_id}.json",
            }
        )
    return {
        "schema_version": "modal_constrained_canary_call_plan.v1",
        "plan_id": "intervenebench-four-model-constrained-canary-20260813-v1",
        "freeze_date": "2026-08-13",
        "status": "frozen_nonexecuting_zero_authority",
        "experiment_ids": [EXPERIMENT_ID],
        "model_ids": list(MODEL_IDS),
        "selection_rule": "same first source-order arm of 5vm8g for every model",
        "json_schema": WEIGHT_SCHEMA,
        "calls": calls,
    }


def verify_canary_freeze(
    root: Path, *, freeze_path: Path, call_plan_path: Path
) -> dict[str, Any]:
    freeze = read_json_object(freeze_path)
    plan = read_json_object(call_plan_path)
    if freeze.get("schema_version") != "modal_constrained_canary_freeze.v1":
        raise ValueError("unsupported canary freeze")
    if freeze.get("status") != "frozen_nonexecuting_zero_authority":
        raise ValueError("canary freeze must carry zero execution authority")
    if any(freeze["authority"].values()):
        raise PermissionError("canary freeze embeds expanded authority")
    if plan != build_canary_call_plan(root):
        raise ValueError("canary call plan does not replay exactly")
    if len(plan["calls"]) != 4 or tuple(plan["model_ids"]) != MODEL_IDS:
        raise ValueError("canary must contain exactly one call per frozen model")
    if plan["experiment_ids"] != [EXPERIMENT_ID]:
        raise ValueError("canary task scope drifted")
    if freeze["call_plan_payload_sha256"] != payload_hash(plan):
        raise ValueError("canary freeze is not bound to its call plan")
    for entry in freeze["implementation_hashes"]:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("implementation path escapes repository")
        if sha256_file(root / relative) != entry["file_sha256"]:
            raise ValueError(f"implementation hash mismatch: {entry['path']}")
    for path_key, hash_key in (
        ("input_path", "input_file_sha256"),
        ("lock_path", "lock_file_sha256"),
    ):
        if sha256_file(root / freeze["dependency_lock"][path_key]) != freeze[
            "dependency_lock"
        ][hash_key]:
            raise ValueError("canary dependency lock hash mismatch")
    bundle_entry = freeze["task_scope"]["packaged_file"]
    bundle = read_json_object(root / bundle_entry["path"])
    if sha256_file(root / bundle_entry["path"]) != bundle_entry["file_sha256"]:
        raise ValueError("canary bundle file hash mismatch")
    if payload_hash(bundle) != bundle_entry["payload_sha256"]:
        raise ValueError("canary bundle payload hash mismatch")
    if bundle["outcome_access"] != "sealed" or bundle["reveal_authorized"] is not False:
        raise PermissionError("canary bundle is not outcome sealed")
    models = freeze["models"]
    if tuple(model["model_id"] for model in models) != MODEL_IDS:
        raise ValueError("canary model allowlist drifted")
    limits = freeze["limits"]
    if limits["maximum_planned_calls"] != 4 or limits["maximum_model_attempts"] != 4:
        raise ValueError("canary attempt limit drifted")
    if float(limits["hard_incremental_cost_cap_usd"]) != 1.25:
        raise ValueError("canary incremental cost cap drifted")
    assert_blinded_payload(freeze)
    assert_blinded_payload(plan)
    return {
        "freeze_payload_sha256": payload_hash(freeze),
        "call_plan_payload_sha256": payload_hash(plan),
        "call_count": 4,
        "model_count": 4,
        "incremental_cost_cap_usd": 1.25,
    }


def prepare_canary_calls(
    root: Path, *, freeze_path: Path, call_plan_path: Path
) -> tuple[PreparedCanaryCall, ...]:
    verify_canary_freeze(root, freeze_path=freeze_path, call_plan_path=call_plan_path)
    freeze = read_json_object(freeze_path)
    plan = read_json_object(call_plan_path)
    bundle = read_json_object(root / freeze["task_scope"]["packaged_file"]["path"])
    option_values = tuple(item["value"] for item in bundle["response_options"])
    prepared = []
    for call in plan["calls"]:
        prompt = relative_weights_prompt(
            bundle, arm_id=call["arm_id"], variant_id=call["variant_id"]
        )
        request = dict(call)
        request["prompt"] = prompt
        request["option_values"] = list(option_values)
        request["json_schema"] = plan["json_schema"]
        assert_blinded_payload(request)
        prepared.append(
            PreparedCanaryCall(
                call_id=call["call_id"],
                model_id=call["model_id"],
                option_values=option_values,
                prompt=prompt,
                request=request,
            )
        )
    return tuple(prepared)


def verify_canary_result(call: PreparedCanaryCall, result: Mapping[str, Any]) -> dict[str, Any]:
    if set(result) != {"call_id", "model_id", "raw_text", "runtime_attestation"}:
        raise ValueError("canary result fields differ from the frozen schema")
    if result["call_id"] != call.call_id or result["model_id"] != call.model_id:
        raise ValueError("canary result identity mismatch")
    raw_text = result["raw_text"]
    if not isinstance(raw_text, str) or not raw_text:
        raise ValueError("canary output is empty")
    decoded = json.loads(raw_text)
    if not isinstance(decoded, dict) or set(decoded) != {"relative_weights"}:
        raise ValueError("canary output is outside the constrained contract")
    weights = decoded["relative_weights"]
    expected = {str(value) for value in call.option_values}
    if not isinstance(weights, dict) or set(weights) != expected:
        raise ValueError("canary weights do not cover the exact option set")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100
        for value in weights.values()
    ):
        raise ValueError("canary weights must be integers from 1 through 100")
    distribution, raw_weights = parse_ordinal_relative_weights(
        raw_text, option_values=call.option_values
    )
    attestation = result["runtime_attestation"]
    if not isinstance(attestation, Mapping):
        raise ValueError("canary runtime attestation is missing")
    return {
        "call_id": call.call_id,
        "model_id": call.model_id,
        "experiment_id": EXPERIMENT_ID,
        "prompt_sha256": sha256(call.prompt.encode("utf-8")).hexdigest(),
        "raw_text": raw_text,
        "relative_weights": {str(key): value for key, value in raw_weights},
        "probabilities": dict(distribution.probabilities),
        "runtime_attestation": dict(attestation),
    }


def build_execution_authorization(
    *, freeze: Mapping[str, Any], plan: Mapping[str, Any], modal_image_id: str,
    cache_hashes: Mapping[str, str]
) -> dict[str, Any]:
    if not modal_image_id:
        raise ValueError("Modal image ID is required")
    if set(cache_hashes) != set(MODEL_IDS):
        raise ValueError("cache hashes must cover all four models")
    payload = {
        "schema_version": "modal_constrained_canary_execution_authorization.v1",
        "authorization_id": "intervenebench-constrained-canary-20260813-v1",
        "scope": "exact_four_call_outcome_blind_grammar_canary_only",
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
        "maximum_planned_calls": 4,
        "maximum_model_attempts": 4,
        "maximum_incremental_cost_usd": 1.25,
        "status": "single_canary_stage_authorized",
    }
    assert_blinded_payload(payload)
    return payload


def build_materialization_authorization(
    *, freeze: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": "modal_constrained_canary_materialization_authorization.v1",
        "authorization_id": "intervenebench-constrained-canary-materialization-20260813-v1",
        "scope": "materialize_exact_canary_image_only",
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
        "maximum_incremental_cost_usd": 1.25,
        "status": "canary_image_materialization_only_authorized",
    }
    assert_blinded_payload(payload)
    return payload


def validate_materialization_authorization(
    authorization: Mapping[str, Any], *, freeze: Mapping[str, Any],
    plan: Mapping[str, Any]
) -> None:
    expected = build_materialization_authorization(freeze=freeze, plan=plan)
    if dict(authorization) != expected:
        raise PermissionError("canary materialization authorization does not match exactly")


def validate_execution_authorization(
    authorization: Mapping[str, Any], *, freeze: Mapping[str, Any],
    plan: Mapping[str, Any], modal_image_id: str, cache_hashes: Mapping[str, str]
) -> None:
    expected = build_execution_authorization(
        freeze=freeze, plan=plan, modal_image_id=modal_image_id,
        cache_hashes=cache_hashes
    )
    if dict(authorization) != expected:
        raise PermissionError("canary execution authorization does not match exactly")


def validate_runtime_attestation(
    call: PreparedCanaryCall, attestation: Mapping[str, Any], *,
    freeze: Mapping[str, Any], authorization: Mapping[str, Any]
) -> None:
    model = {item["model_id"]: item for item in freeze["models"]}[call.model_id]
    expected = {
        "modal_sdk_version": "1.5.4",
        "modal_image_id": authorization["modal_image_id"],
        "image_recipe_sha256": payload_hash(freeze["runtime"]["image_recipe"]),
        "dependency_lock_sha256": freeze["dependency_lock"]["lock_file_sha256"],
        "transformers_version": "4.57.6",
        "outlines_version": "1.2.11",
        "outlines_core_version": "0.2.14",
        "cuda_runtime_version": "12.8",
        "checkpoint_commit": model["checkpoint_commit"],
        "weight_manifest_sha256": model["weight_file_manifest_sha256"],
        "tokenizer_manifest_sha256": model["tokenizer_manifest_sha256"],
        "cache_attestation_sha256": authorization[
            "cache_attestation_sha256_by_model"
        ][call.model_id],
        "call_id": call.call_id,
        "prompt_sha256": call.request["prompt_sha256"],
        "constraint_backend": "outlines_core_json_schema",
    }
    for field, value in expected.items():
        if attestation.get(field) != value:
            raise ValueError(f"canary runtime attestation mismatch: {field}")
    if str(attestation.get("torch_version", "")).split("+")[0] != "2.9.1":
        raise ValueError("canary runtime attestation mismatch: torch_version")
    if str(attestation.get("python_version", "")).split(".")[:2] != ["3", "11"]:
        raise ValueError("canary runtime attestation mismatch: python_version")
    if "L40S" not in str(attestation.get("gpu_name", "")):
        raise ValueError("canary runtime attestation mismatch: gpu_name")
