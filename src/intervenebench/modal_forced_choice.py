"""Outcome-blind helpers for the four-model forced-choice logit canary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from math import fsum, isfinite
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .protocol import assert_blinded_payload, payload_hash
from .simulators import ordinal_variant_contract, validate_ordinal_blinded_bundle


MODEL_IDS = (
    "qwen3_8b_generic",
    "qwen3_14b_generic",
    "qwen2_5_14b_generic",
    "socrates_qwen2_5_14b_sft",
)
EXPERIMENT_ID = "5vm8g"
ANSWER_CODES = ("A", "B", "C", "D", "E")


@dataclass(frozen=True, slots=True)
class PreparedForcedChoiceCall:
    call_id: str
    model_id: str
    option_values: tuple[int, ...]
    answer_codes: tuple[str, ...]
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


def forced_choice_prompt(
    bundle: Mapping[str, Any], *, arm_id: str, variant_id: str | None
) -> str:
    """Build one prompt whose next token is restricted analytically to A--E."""

    validate_ordinal_blinded_bundle(bundle)
    options = bundle["response_options"]
    if len(options) != len(ANSWER_CODES):
        raise ValueError("forced-choice canary requires exactly five response options")
    arms = {arm["arm_id"]: arm for arm in bundle["arms"]}
    if arm_id not in arms:
        raise ValueError("requested arm is absent from ordinal bundle")
    arm = arms[arm_id]
    if "message" in arm:
        if variant_id not in {None, "direct"}:
            raise ValueError("direct forced-choice arm does not accept a variant")
        message = arm["message"]
    else:
        variants = {item["variant_id"]: item for item in arm["message_variants"]}
        if variant_id not in variants:
            raise ValueError("requested message variant is absent")
        message = variants[str(variant_id)]["message"]
    answers = "\n".join(
        f"{code}: {item['label']}" for code, item in zip(ANSWER_CODES, options)
    )
    return (
        "Predict the answer of one randomly selected adult from the population "
        "below. Do not answer as yourself. Choose exactly one answer code from "
        "A, B, C, D, or E. Do not explain your choice.\n\n"
        f"Population: {bundle['population']['description']}\n\n"
        f"Context: {bundle['common_context']}\n\n"
        f"Intervention: {message}\n\n"
        f"Question: {bundle['outcome_question']}\n"
        f"Answer codes:\n{answers}\n\n"
        "Return only the answer code."
    )


def build_call_plan(root: Path) -> dict[str, Any]:
    bundle = read_json_object(root / "data/manifests/contracts/5vm8g_blinded_bundle.json")
    arm_id = bundle["arms"][0]["arm_id"]
    variant_id = ordinal_variant_contract(bundle, arm_id=arm_id)[0][0]
    prompt = forced_choice_prompt(bundle, arm_id=arm_id, variant_id=variant_id)
    prompt_digest = sha256(prompt.encode("utf-8")).hexdigest()
    option_values = [item["value"] for item in bundle["response_options"]]
    calls = [
        {
            "call_id": f"forced-choice--{model_id}--5vm8g--{arm_id}",
            "model_id": model_id,
            "experiment_id": EXPERIMENT_ID,
            "bundle_payload_sha256": payload_hash(bundle),
            "arm_id": arm_id,
            "variant_id": variant_id,
            "prompt_sha256": prompt_digest,
            "answer_codes": list(ANSWER_CODES),
            "option_values": option_values,
            "method_id": "forced_choice_next_token_softmax.v1",
            "temperature": 1.0,
            "artifact_relative_path": f"{model_id}/5vm8g/{arm_id}.json",
        }
        for model_id in MODEL_IDS
    ]
    return {
        "schema_version": "modal_forced_choice_call_plan.v1",
        "plan_id": "intervenebench-four-model-forced-choice-canary-20260813-v1",
        "freeze_date": "2026-08-13",
        "status": "frozen_nonexecuting_zero_authority",
        "experiment_ids": [EXPERIMENT_ID],
        "model_ids": list(MODEL_IDS),
        "selection_rule": "same first source-order arm of 5vm8g for every model",
        "calls": calls,
    }


def verify_freeze(root: Path, *, freeze_path: Path, call_plan_path: Path) -> dict[str, Any]:
    freeze = read_json_object(freeze_path)
    plan = read_json_object(call_plan_path)
    if freeze.get("schema_version") != "modal_forced_choice_freeze.v1":
        raise ValueError("unsupported forced-choice freeze")
    if freeze.get("status") != "frozen_nonexecuting_zero_authority":
        raise ValueError("forced-choice freeze must have zero authority")
    if any(freeze["authority"].values()):
        raise PermissionError("forced-choice freeze embeds expanded authority")
    if plan != build_call_plan(root):
        raise ValueError("forced-choice call plan does not replay exactly")
    if freeze["call_plan_payload_sha256"] != payload_hash(plan):
        raise ValueError("forced-choice freeze is not bound to its plan")
    for entry in freeze["implementation_hashes"]:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("implementation path escapes repository")
        if sha256_file(root / relative) != entry["file_sha256"]:
            raise ValueError(f"implementation hash mismatch: {entry['path']}")
    dependency = freeze["dependency_lock"]
    for path_key, hash_key in (
        ("input_path", "input_file_sha256"), ("lock_path", "lock_file_sha256")
    ):
        if sha256_file(root / dependency[path_key]) != dependency[hash_key]:
            raise ValueError("forced-choice dependency hash mismatch")
    bundle_entry = freeze["task_scope"]["packaged_file"]
    bundle = read_json_object(root / bundle_entry["path"])
    if sha256_file(root / bundle_entry["path"]) != bundle_entry["file_sha256"]:
        raise ValueError("forced-choice bundle file hash mismatch")
    if payload_hash(bundle) != bundle_entry["payload_sha256"]:
        raise ValueError("forced-choice bundle payload hash mismatch")
    if bundle["outcome_access"] != "sealed" or bundle["reveal_authorized"] is not False:
        raise PermissionError("forced-choice bundle is not outcome sealed")
    if tuple(model["model_id"] for model in freeze["models"]) != MODEL_IDS:
        raise ValueError("forced-choice model allowlist drifted")
    limits = freeze["limits"]
    if (limits["maximum_planned_calls"], limits["maximum_model_attempts"]) != (4, 4):
        raise ValueError("forced-choice attempt ceiling drifted")
    if float(limits["hard_incremental_cost_cap_usd"]) != 0.90:
        raise ValueError("forced-choice cost cap drifted")
    assert_blinded_payload(freeze)
    assert_blinded_payload(plan)
    return {
        "call_count": 4,
        "model_count": 4,
        "incremental_cost_cap_usd": 0.90,
        "freeze_payload_sha256": payload_hash(freeze),
        "call_plan_payload_sha256": payload_hash(plan),
    }


def prepare_calls(
    root: Path, *, freeze_path: Path, call_plan_path: Path
) -> tuple[PreparedForcedChoiceCall, ...]:
    verify_freeze(root, freeze_path=freeze_path, call_plan_path=call_plan_path)
    freeze = read_json_object(freeze_path)
    plan = read_json_object(call_plan_path)
    bundle = read_json_object(root / freeze["task_scope"]["packaged_file"]["path"])
    prepared = []
    for call in plan["calls"]:
        prompt = forced_choice_prompt(
            bundle, arm_id=call["arm_id"], variant_id=call["variant_id"]
        )
        request = dict(call)
        request["prompt"] = prompt
        assert_blinded_payload(request)
        prepared.append(
            PreparedForcedChoiceCall(
                call_id=call["call_id"], model_id=call["model_id"],
                option_values=tuple(call["option_values"]),
                answer_codes=tuple(call["answer_codes"]), prompt=prompt,
                request=request
            )
        )
    return tuple(prepared)


def verify_result(
    call: PreparedForcedChoiceCall, result: Mapping[str, Any]
) -> dict[str, Any]:
    required = {
        "call_id", "model_id", "probabilities_by_code", "candidate_token_ids",
        "candidate_token_strings", "runtime_attestation"
    }
    if set(result) != required:
        raise ValueError("forced-choice result fields differ from frozen schema")
    if result["call_id"] != call.call_id or result["model_id"] != call.model_id:
        raise ValueError("forced-choice result identity mismatch")
    probabilities = result["probabilities_by_code"]
    if not isinstance(probabilities, Mapping) or tuple(probabilities) != call.answer_codes:
        raise ValueError("forced-choice probabilities have wrong code order")
    values = list(probabilities.values())
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not isfinite(value) or value < 0.0 or value > 1.0
        for value in values
    ) or abs(fsum(values) - 1.0) > 1e-6:
        raise ValueError("forced-choice probabilities are invalid")
    token_ids = result["candidate_token_ids"]
    token_strings = result["candidate_token_strings"]
    if (
        not isinstance(token_ids, list) or len(token_ids) != 5
        or len(set(token_ids)) != 5 or not all(isinstance(item, int) for item in token_ids)
    ):
        raise ValueError("forced-choice candidate tokens are not five distinct IDs")
    if token_strings != list(call.answer_codes):
        raise ValueError("forced-choice token strings do not match answer codes")
    option_probabilities = {
        value: float(probabilities[code])
        for code, value in zip(call.answer_codes, call.option_values)
    }
    attestation = result["runtime_attestation"]
    if not isinstance(attestation, Mapping):
        raise ValueError("forced-choice runtime attestation is missing")
    return {
        "call_id": call.call_id,
        "model_id": call.model_id,
        "experiment_id": EXPERIMENT_ID,
        "prompt_sha256": sha256(call.prompt.encode("utf-8")).hexdigest(),
        "method_id": "forced_choice_next_token_softmax.v1",
        "answer_code_probabilities": dict(probabilities),
        "probabilities": option_probabilities,
        "candidate_token_ids": token_ids,
        "runtime_attestation": dict(attestation),
    }


def build_materialization_authorization(
    *, freeze: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": "modal_forced_choice_materialization_authorization.v1",
        "authorization_id": "intervenebench-forced-choice-materialization-20260813-v1",
        "scope": "materialize_exact_forced_choice_image_only",
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
        "maximum_incremental_cost_usd": 0.90,
        "status": "forced_choice_image_only_authorized",
    }
    assert_blinded_payload(payload)
    return payload


def validate_materialization_authorization(
    authorization: Mapping[str, Any], *, freeze: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    if dict(authorization) != build_materialization_authorization(freeze=freeze, plan=plan):
        raise PermissionError("forced-choice materialization authorization mismatch")


def build_execution_authorization(
    *, freeze: Mapping[str, Any], plan: Mapping[str, Any], modal_image_id: str,
    cache_hashes: Mapping[str, str]
) -> dict[str, Any]:
    if not modal_image_id or set(cache_hashes) != set(MODEL_IDS):
        raise ValueError("forced-choice image/cache binding is incomplete")
    payload = {
        "schema_version": "modal_forced_choice_execution_authorization.v1",
        "authorization_id": "intervenebench-forced-choice-execution-20260813-v1",
        "scope": "exact_four_forward_pass_forced_choice_canary_only",
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
        "maximum_incremental_cost_usd": 0.90,
        "status": "single_forced_choice_canary_authorized",
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
        raise PermissionError("forced-choice execution authorization mismatch")


def validate_runtime_attestation(
    call: PreparedForcedChoiceCall, attestation: Mapping[str, Any], *,
    freeze: Mapping[str, Any], authorization: Mapping[str, Any]
) -> None:
    model = {item["model_id"]: item for item in freeze["models"]}[call.model_id]
    expected = {
        "modal_sdk_version": "1.5.4",
        "modal_image_id": authorization["modal_image_id"],
        "image_recipe_sha256": payload_hash(freeze["runtime"]["image_recipe"]),
        "dependency_lock_sha256": freeze["dependency_lock"]["lock_file_sha256"],
        "transformers_version": "4.57.6",
        "cuda_runtime_version": "12.8",
        "checkpoint_commit": model["checkpoint_commit"],
        "weight_manifest_sha256": model["weight_file_manifest_sha256"],
        "tokenizer_manifest_sha256": model["tokenizer_manifest_sha256"],
        "cache_attestation_sha256": authorization[
            "cache_attestation_sha256_by_model"
        ][call.model_id],
        "call_id": call.call_id,
        "prompt_sha256": call.request["prompt_sha256"],
        "method_id": "forced_choice_next_token_softmax.v1",
    }
    for field, value in expected.items():
        if attestation.get(field) != value:
            raise ValueError(f"forced-choice runtime mismatch: {field}")
    if str(attestation.get("torch_version", "")).split("+")[0] != "2.9.1":
        raise ValueError("forced-choice runtime mismatch: torch_version")
    if str(attestation.get("python_version", "")).split(".")[:2] != ["3", "11"]:
        raise ValueError("forced-choice runtime mismatch: python_version")
    if "L40S" not in str(attestation.get("gpu_name", "")):
        raise ValueError("forced-choice runtime mismatch: gpu_name")
