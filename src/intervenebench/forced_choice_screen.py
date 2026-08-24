"""Outcome-blind contracts for the 40-call parser-free discovery screen."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from math import fsum, isfinite
from pathlib import Path, PurePosixPath
from string import ascii_uppercase
from typing import Any, Mapping

from .modal_forced_choice import MODEL_IDS, read_json_object, sha256_file
from .protocol import assert_blinded_payload, payload_hash
from .simulators import ordinal_variant_contract, validate_ordinal_blinded_bundle


EXPERIMENT_IDS = ("5vm8g", "xc4yq", "de5hx", "turagaS11", "wallaceS12")
MAXIMUM_CODES = tuple(ascii_uppercase[:8])


@dataclass(frozen=True, slots=True)
class PreparedScreenCall:
    call_id: str
    model_id: str
    experiment_id: str
    option_values: tuple[int, ...]
    answer_codes: tuple[str, ...]
    prompt: str
    request: dict[str, Any]


def answer_codes(option_count: int) -> tuple[str, ...]:
    if not 2 <= option_count <= len(MAXIMUM_CODES):
        raise ValueError("forced-choice screen supports two through eight options")
    return MAXIMUM_CODES[:option_count]


def screen_prompt(
    bundle: Mapping[str, Any], *, arm_id: str, variant_id: str | None
) -> str:
    validate_ordinal_blinded_bundle(bundle)
    codes = answer_codes(len(bundle["response_options"]))
    arms = {arm["arm_id"]: arm for arm in bundle["arms"]}
    if arm_id not in arms:
        raise ValueError("requested screen arm is absent")
    arm = arms[arm_id]
    if "message" in arm:
        if variant_id not in {None, "direct"}:
            raise ValueError("direct screen arm does not accept a variant")
        message = arm["message"]
    else:
        variants = {item["variant_id"]: item for item in arm["message_variants"]}
        if variant_id not in variants:
            raise ValueError("requested screen message variant is absent")
        message = variants[str(variant_id)]["message"]
    code_list = ", ".join(codes[:-1]) + f", or {codes[-1]}"
    answers = "\n".join(
        f"{code}: {item['label']}"
        for code, item in zip(codes, bundle["response_options"])
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
    templates: list[dict[str, Any]] = []
    for experiment_id in EXPERIMENT_IDS:
        bundle = read_json_object(
            root / f"data/manifests/contracts/{experiment_id}_blinded_bundle.json"
        )
        codes = answer_codes(len(bundle["response_options"]))
        for arm in (bundle["arms"][0], bundle["arms"][-1]):
            arm_id = arm["arm_id"]
            variant_id = ordinal_variant_contract(bundle, arm_id=arm_id)[0][0]
            prompt = screen_prompt(bundle, arm_id=arm_id, variant_id=variant_id)
            templates.append(
                {
                    "experiment_id": experiment_id,
                    "bundle_payload_sha256": payload_hash(bundle),
                    "arm_id": arm_id,
                    "variant_id": variant_id,
                    "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
                    "answer_codes": list(codes),
                    "option_values": [item["value"] for item in bundle["response_options"]],
                }
            )
    calls = []
    for model_id in MODEL_IDS:
        for template in templates:
            call = dict(template)
            call.update(
                {
                    "call_id": (
                        f"screen--{model_id}--{template['experiment_id']}--"
                        f"{template['arm_id']}"
                    ),
                    "model_id": model_id,
                    "method_id": "forced_choice_next_token_softmax.v1",
                    "temperature": 1.0,
                    "artifact_relative_path": (
                        f"{model_id}/{template['experiment_id']}/"
                        f"{template['arm_id']}.json"
                    ),
                }
            )
            calls.append(call)
    return {
        "schema_version": "modal_forced_choice_screen_plan.v1",
        "plan_id": "intervenebench-40-call-forced-choice-discovery-screen-20260813-v1",
        "freeze_date": "2026-08-13",
        "status": "frozen_nonexecuting_zero_authority",
        "experiment_ids": list(EXPERIMENT_IDS),
        "model_ids": list(MODEL_IDS),
        "selection_rule": "first_and_last_source_order_arm; first declared nuisance variant",
        "calls": calls,
    }


def verify_freeze(root: Path, *, freeze_path: Path, plan_path: Path) -> dict[str, Any]:
    freeze = read_json_object(freeze_path)
    plan = read_json_object(plan_path)
    if freeze.get("schema_version") != "modal_forced_choice_screen_freeze.v1":
        raise ValueError("unsupported forced-choice screen freeze")
    if freeze.get("status") != "frozen_nonexecuting_zero_authority":
        raise ValueError("screen freeze must have zero authority")
    if any(freeze["authority"].values()):
        raise PermissionError("screen freeze embeds expanded authority")
    if plan != build_call_plan(root):
        raise ValueError("screen call plan does not replay exactly")
    if freeze["call_plan_payload_sha256"] != payload_hash(plan):
        raise ValueError("screen freeze is not bound to its plan")
    if len(plan["calls"]) != 40:
        raise ValueError("screen must contain exactly forty calls")
    for entry in freeze["implementation_hashes"]:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("screen implementation path escapes repository")
        if sha256_file(root / relative) != entry["file_sha256"]:
            raise ValueError(f"screen implementation hash mismatch: {entry['path']}")
    lock = freeze["dependency_lock"]
    for path_key, hash_key in (
        ("input_path", "input_file_sha256"), ("lock_path", "lock_file_sha256")
    ):
        if sha256_file(root / lock[path_key]) != lock[hash_key]:
            raise ValueError("screen dependency lock hash mismatch")
    for entry in freeze["task_scope"]["packaged_files"]:
        bundle = read_json_object(root / entry["path"])
        if sha256_file(root / entry["path"]) != entry["file_sha256"]:
            raise ValueError("screen bundle file hash mismatch")
        if payload_hash(bundle) != entry["payload_sha256"]:
            raise ValueError("screen bundle payload hash mismatch")
        if bundle["outcome_access"] != "sealed" or bundle["reveal_authorized"] is not False:
            raise PermissionError("screen bundle is not outcome sealed")
    if tuple(model["model_id"] for model in freeze["models"]) != MODEL_IDS:
        raise ValueError("screen model allowlist drifted")
    limits = freeze["limits"]
    if (limits["maximum_planned_calls"], limits["maximum_model_attempts"]) != (40, 40):
        raise ValueError("screen attempt ceiling drifted")
    if float(limits["hard_incremental_cost_cap_usd"]) != 1.75:
        raise ValueError("screen cost cap drifted")
    assert_blinded_payload(freeze)
    assert_blinded_payload(plan)
    return {
        "call_count": 40, "model_count": 4, "experiment_count": 5,
        "incremental_cost_cap_usd": 1.75,
        "freeze_payload_sha256": payload_hash(freeze),
        "call_plan_payload_sha256": payload_hash(plan),
    }


def prepare_calls(
    root: Path, *, freeze_path: Path, plan_path: Path
) -> tuple[PreparedScreenCall, ...]:
    verify_freeze(root, freeze_path=freeze_path, plan_path=plan_path)
    freeze = read_json_object(freeze_path)
    plan = read_json_object(plan_path)
    bundle_paths = {
        item["experiment_id"]: root / item["path"]
        for item in freeze["task_scope"]["packaged_files"]
    }
    bundles = {key: read_json_object(path) for key, path in bundle_paths.items()}
    prepared = []
    for call in plan["calls"]:
        prompt = screen_prompt(
            bundles[call["experiment_id"]], arm_id=call["arm_id"],
            variant_id=call["variant_id"]
        )
        request = dict(call)
        request["prompt"] = prompt
        assert_blinded_payload(request)
        prepared.append(
            PreparedScreenCall(
                call_id=call["call_id"], model_id=call["model_id"],
                experiment_id=call["experiment_id"],
                option_values=tuple(call["option_values"]),
                answer_codes=tuple(call["answer_codes"]), prompt=prompt,
                request=request
            )
        )
    return tuple(prepared)


def verify_result(call: PreparedScreenCall, result: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "call_id", "model_id", "probabilities_by_code", "candidate_token_ids",
        "candidate_token_strings", "runtime_attestation"
    }
    if set(result) != required:
        raise ValueError("screen result fields differ from schema")
    if result["call_id"] != call.call_id or result["model_id"] != call.model_id:
        raise ValueError("screen result identity mismatch")
    probabilities = result["probabilities_by_code"]
    if not isinstance(probabilities, Mapping) or tuple(probabilities) != call.answer_codes:
        raise ValueError("screen probability code order mismatch")
    values = list(probabilities.values())
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not isfinite(value) or not 0.0 <= value <= 1.0
        for value in values
    ) or abs(fsum(values) - 1.0) > 1e-6:
        raise ValueError("screen probabilities are invalid")
    token_ids = result["candidate_token_ids"]
    if (
        not isinstance(token_ids, list) or len(token_ids) != len(call.answer_codes)
        or len(set(token_ids)) != len(token_ids)
        or not all(isinstance(item, int) for item in token_ids)
    ):
        raise ValueError("screen candidate token IDs are invalid")
    if result["candidate_token_strings"] != list(call.answer_codes):
        raise ValueError("screen candidate token strings mismatch")
    attestation = result["runtime_attestation"]
    if not isinstance(attestation, Mapping):
        raise ValueError("screen runtime attestation is missing")
    return {
        "call_id": call.call_id,
        "model_id": call.model_id,
        "experiment_id": call.experiment_id,
        "prompt_sha256": sha256(call.prompt.encode("utf-8")).hexdigest(),
        "method_id": "forced_choice_next_token_softmax.v1",
        "answer_code_probabilities": dict(probabilities),
        "probabilities": {
            value: float(probabilities[code])
            for code, value in zip(call.answer_codes, call.option_values)
        },
        "candidate_token_ids": token_ids,
        "runtime_attestation": dict(attestation),
    }


def build_materialization_authorization(
    *, freeze: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": "modal_forced_choice_screen_materialization_authorization.v1",
        "authorization_id": "intervenebench-forced-choice-screen-materialization-20260813-v1",
        "scope": "materialize_exact_40_call_screen_image_only",
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
        "status": "screen_image_only_authorized",
    }
    assert_blinded_payload(payload)
    return payload


def validate_materialization_authorization(
    authorization: Mapping[str, Any], *, freeze: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    if dict(authorization) != build_materialization_authorization(freeze=freeze, plan=plan):
        raise PermissionError("screen materialization authorization mismatch")


def build_execution_authorization(
    *, freeze: Mapping[str, Any], plan: Mapping[str, Any], modal_image_id: str,
    cache_hashes: Mapping[str, str]
) -> dict[str, Any]:
    if not modal_image_id or set(cache_hashes) != set(MODEL_IDS):
        raise ValueError("screen image/cache binding is incomplete")
    payload = {
        "schema_version": "modal_forced_choice_screen_execution_authorization.v1",
        "authorization_id": "intervenebench-forced-choice-screen-execution-20260813-v1",
        "scope": "exact_40_call_outcome_blind_discovery_screen_only",
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
        "status": "single_discovery_screen_authorized",
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
        raise PermissionError("screen execution authorization mismatch")


def validate_runtime_attestation(
    call: PreparedScreenCall, attestation: Mapping[str, Any], *,
    freeze: Mapping[str, Any], authorization: Mapping[str, Any]
) -> None:
    model = {item["model_id"]: item for item in freeze["models"]}[call.model_id]
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
        ][call.model_id],
        "call_id": call.call_id, "prompt_sha256": call.request["prompt_sha256"],
        "method_id": "forced_choice_next_token_softmax.v1",
    }
    for field, value in expected.items():
        if attestation.get(field) != value:
            raise ValueError(f"screen runtime mismatch: {field}")
    if str(attestation.get("torch_version", "")).split("+")[0] != "2.9.1":
        raise ValueError("screen runtime mismatch: torch_version")
    if str(attestation.get("python_version", "")).split(".")[:2] != ["3", "11"]:
        raise ValueError("screen runtime mismatch: python_version")
    if "L40S" not in str(attestation.get("gpu_name", "")):
        raise ValueError("screen runtime mismatch: gpu_name")
