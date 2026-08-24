"""Balanced source/reverse forced-choice estimators and full-action freeze."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from math import fsum, isfinite
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .answer_order_canary import reverse_order_prompt
from .forced_choice_screen import (
    EXPERIMENT_IDS,
    answer_codes,
    read_json_object,
    screen_prompt,
    sha256_file,
)
from .modal_forced_choice import MODEL_IDS
from .protocol import assert_blinded_payload, payload_hash, verify_envelope
from .simulators import ordinal_variant_contract


SOURCE_RUN_RELATIVE = Path(
    "artifacts/forced_choice_screen/discovery_screen_20260813_v1"
)
REVERSE_RUN_RELATIVE = Path(
    "artifacts/answer_order_canary/answer_order_canary_20260813_v1"
)


def _normalized_probabilities(values: Mapping[Any, Any]) -> dict[int, float]:
    probabilities = {int(key): float(value) for key, value in values.items()}
    if (
        not probabilities
        or any(not isfinite(value) or value < 0.0 for value in probabilities.values())
        or abs(fsum(probabilities.values()) - 1.0) > 1e-6
    ):
        raise ValueError("probability distribution is not finite and normalized")
    return probabilities


def weighted_distribution_average(
    source: Mapping[Any, Any], reverse: Mapping[Any, Any]
) -> dict[int, float]:
    """Return the frozen equal-weight average on shared source-value support."""

    first = _normalized_probabilities(source)
    second = _normalized_probabilities(reverse)
    if set(first) != set(second):
        raise ValueError("balanced distributions have different support")
    first_total = fsum(first.values())
    second_total = fsum(second.values())
    averaged = {
        key: 0.5 * first[key] / first_total + 0.5 * second[key] / second_total
        for key in first
    }
    if abs(fsum(averaged.values()) - 1.0) > 1e-12:
        raise ValueError("balanced distribution lost normalization")
    return averaged


def _verified_output(
    *, root: Path, run_relative: Path, relative_path: str,
    call_id: str, expected_hash: str
) -> dict[str, Any]:
    path = root / run_relative / relative_path
    envelope = read_json_object(path)
    if payload_hash(envelope["payload"]) != expected_hash:
        raise ValueError(f"balanced input hash mismatch: {call_id}")
    output = verify_envelope(path, require_blinded=True)
    if output["call_id"] != call_id:
        raise ValueError("balanced input identity mismatch")
    return output


def build_balanced_discovery_artifact(
    root: Path, *, source_run_root: Path, reverse_run_root: Path
) -> dict[str, Any]:
    """Average the existing paired calls and derive screened-pair choices."""

    source_final = verify_envelope(
        source_run_root / "final_manifest.json", require_blinded=True
    )
    reverse_final = verify_envelope(
        reverse_run_root / "final_manifest.json", require_blinded=True
    )
    source_plan = read_json_object(
        root / "data/manifests/simulators/forced_choice_screen_plan_v1.json"
    )
    reverse_plan = read_json_object(
        root / "data/manifests/simulators/answer_order_canary_plan_v1.json"
    )
    source_calls = {call["call_id"]: call for call in source_plan["calls"]}
    if set(source_calls) != set(source_final["call_output_sha256"]):
        raise ValueError("source-order artifact set is incomplete")
    if {call["call_id"] for call in reverse_plan["calls"]} != set(
        reverse_final["call_output_sha256"]
    ):
        raise ValueError("reverse-order artifact set is incomplete")
    bundles = {
        experiment_id: read_json_object(
            root / f"data/manifests/contracts/{experiment_id}_blinded_bundle.json"
        )
        for experiment_id in EXPERIMENT_IDS
    }
    models = {
        model["model_id"]: model
        for model in read_json_object(
            root / "configs/simulators/forced_choice_screen_v1.json"
        )["models"]
    }
    rows: list[dict[str, Any]] = []
    expected_by_arm: dict[tuple[str, str, str], float] = {}
    for reverse_call in reverse_plan["calls"]:
        source_call = source_calls[reverse_call["source_order_call_id"]]
        source_output = _verified_output(
            root=root,
            run_relative=source_run_root.relative_to(root),
            relative_path=source_call["artifact_relative_path"],
            call_id=source_call["call_id"],
            expected_hash=source_final["call_output_sha256"][source_call["call_id"]],
        )
        reverse_output = _verified_output(
            root=root,
            run_relative=reverse_run_root.relative_to(root),
            relative_path=reverse_call["artifact_relative_path"],
            call_id=reverse_call["call_id"],
            expected_hash=reverse_final["call_output_sha256"][reverse_call["call_id"]],
        )
        balanced = weighted_distribution_average(
            source_output["probabilities"], reverse_output["probabilities"]
        )
        bundle = bundles[reverse_call["experiment_id"]]
        source_values = [int(item["value"]) for item in bundle["response_options"]]
        utility = {
            int(item["value"]): float(item["normalized_utility"])
            for item in bundle["response_options"]
        }
        modal_value = max(
            source_values,
            key=lambda value: (balanced[value], -source_values.index(value)),
        )
        expected_utility = fsum(
            balanced[value] * utility[value] for value in balanced
        )
        model_id = reverse_call["model_id"]
        experiment_id = reverse_call["experiment_id"]
        arm_id = reverse_call["arm_id"]
        row = {
            "model_id": model_id,
            "experiment_id": experiment_id,
            "arm_id": arm_id,
            "variant_id": reverse_call["variant_id"],
            "source_order_call_id": source_call["call_id"],
            "reverse_order_call_id": reverse_call["call_id"],
            "source_order_weight": 0.5,
            "reverse_order_weight": 0.5,
            "balanced_probabilities": balanced,
            "balanced_modal_response_value": modal_value,
            "balanced_expected_normalized_utility": expected_utility,
            "model_exposure": models[model_id]["exposure_by_experiment"][
                experiment_id
            ],
        }
        rows.append(row)
        expected_by_arm[(model_id, experiment_id, arm_id)] = expected_utility

    recommendations = []
    for model_id in MODEL_IDS:
        for experiment_id in EXPERIMENT_IDS:
            arm_order = [
                call["arm_id"]
                for call in source_plan["calls"]
                if call["model_id"] == model_id
                and call["experiment_id"] == experiment_id
            ]
            if len(arm_order) != 2:
                raise ValueError("balanced discovery pair is not exactly two arms")
            chosen = max(
                arm_order,
                key=lambda arm_id: (
                    expected_by_arm[(model_id, experiment_id, arm_id)],
                    -arm_order.index(arm_id),
                ),
            )
            recommendations.append(
                {
                    "model_id": model_id,
                    "experiment_id": experiment_id,
                    "screened_arm_ids": arm_order,
                    "balanced_screened_pair_choice": chosen,
                    "balanced_choice_expected_normalized_utility": expected_by_arm[
                        (model_id, experiment_id, chosen)
                    ],
                    "full_action_set_recommendation": False,
                }
            )
    result = {
        "schema_version": "balanced_forced_choice_discovery.v1",
        "source_run_manifest_payload_sha256": payload_hash(source_final),
        "reverse_run_manifest_payload_sha256": payload_hash(reverse_final),
        "source_call_output_sha256": source_final["call_output_sha256"],
        "reverse_call_output_sha256": reverse_final["call_output_sha256"],
        "outcome_access": "not_accessed",
        "scope": "development_discovery_only_no_human_scoring",
        "order_aggregation": (
            "equal_weight_source_and_full_reverse_after_inverse_mapping"
        ),
        "balanced_arm_prediction_count": len(rows),
        "screened_pair_recommendation_count": len(recommendations),
        "balanced_arm_predictions": rows,
        "screened_pair_recommendations": recommendations,
        "interpretation_boundary": (
            "Equal source/reverse averaging is invariant to exchanging those two "
            "orders, but does not prove invariance to every answer permutation or "
            "fidelity to human outcomes. Screened pairs remain incomplete action sets."
        ),
        "status": "balanced_discovery_artifact_complete_outcome_blind",
    }
    assert_blinded_payload(result)
    return result


def _existing_call_lookup(root: Path) -> tuple[dict[tuple[str, str, str, str], dict[str, Any]], dict[str, Any]]:
    source_plan = read_json_object(
        root / "data/manifests/simulators/forced_choice_screen_plan_v1.json"
    )
    reverse_plan = read_json_object(
        root / "data/manifests/simulators/answer_order_canary_plan_v1.json"
    )
    source_final = verify_envelope(
        root / SOURCE_RUN_RELATIVE / "final_manifest.json", require_blinded=True
    )
    reverse_final = verify_envelope(
        root / REVERSE_RUN_RELATIVE / "final_manifest.json", require_blinded=True
    )
    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for call in source_plan["calls"]:
        _verified_output(
            root=root,
            run_relative=SOURCE_RUN_RELATIVE,
            relative_path=call["artifact_relative_path"],
            call_id=call["call_id"],
            expected_hash=source_final["call_output_sha256"][call["call_id"]],
        )
        lookup[(call["model_id"], call["experiment_id"], call["arm_id"], "source")] = {
            **call,
            "acquisition": "reuse_verified_existing",
            "order_variant": "source",
            "repository_artifact_path": str(
                SOURCE_RUN_RELATIVE / call["artifact_relative_path"]
            ),
            "artifact_payload_sha256": source_final["call_output_sha256"][
                call["call_id"]
            ],
        }
    for call in reverse_plan["calls"]:
        _verified_output(
            root=root,
            run_relative=REVERSE_RUN_RELATIVE,
            relative_path=call["artifact_relative_path"],
            call_id=call["call_id"],
            expected_hash=reverse_final["call_output_sha256"][call["call_id"]],
        )
        lookup[(call["model_id"], call["experiment_id"], call["arm_id"], "reverse")] = {
            **call,
            "acquisition": "reuse_verified_existing",
            "order_variant": "reverse",
            "repository_artifact_path": str(
                REVERSE_RUN_RELATIVE / call["artifact_relative_path"]
            ),
            "artifact_payload_sha256": reverse_final["call_output_sha256"][
                call["call_id"]
            ],
        }
    return lookup, {
        "source_final_payload_sha256": payload_hash(source_final),
        "source_call_output_sha256": source_final["call_output_sha256"],
        "reverse_final_payload_sha256": payload_hash(reverse_final),
        "reverse_call_output_sha256": reverse_final["call_output_sha256"],
    }


def build_full_action_plan(root: Path) -> dict[str, Any]:
    """Build 136 logical calls while scheduling only the 56 missing calls."""

    existing, existing_hashes = _existing_call_lookup(root)
    model_specs = {
        model["model_id"]: model
        for model in read_json_object(
            root / "configs/simulators/forced_choice_screen_v1.json"
        )["models"]
    }
    logical_calls: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        for experiment_id in EXPERIMENT_IDS:
            bundle = read_json_object(
                root / f"data/manifests/contracts/{experiment_id}_blinded_bundle.json"
            )
            source_values = [int(item["value"]) for item in bundle["response_options"]]
            codes = list(answer_codes(len(source_values)))
            for arm in bundle["arms"]:
                arm_id = arm["arm_id"]
                variant_id = ordinal_variant_contract(bundle, arm_id=arm_id)[0][0]
                for order_variant in ("source", "reverse"):
                    prompt = (
                        screen_prompt(bundle, arm_id=arm_id, variant_id=variant_id)
                        if order_variant == "source"
                        else reverse_order_prompt(
                            bundle, arm_id=arm_id, variant_id=variant_id
                        )
                    )
                    key = (model_id, experiment_id, arm_id, order_variant)
                    if key in existing:
                        entry = dict(existing[key])
                        if entry["prompt_sha256"] != sha256(
                            prompt.encode("utf-8")
                        ).hexdigest():
                            raise ValueError("reused prompt differs from full-action prompt")
                    else:
                        call_id = (
                            f"balanced-full--{order_variant}--{model_id}--"
                            f"{experiment_id}--{arm_id}"
                        )
                        entry = {
                            "call_id": call_id,
                            "model_id": model_id,
                            "experiment_id": experiment_id,
                            "bundle_payload_sha256": payload_hash(bundle),
                            "arm_id": arm_id,
                            "variant_id": variant_id,
                            "order_variant": order_variant,
                            "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
                            "answer_codes": codes,
                            "source_option_values": source_values,
                            "display_option_values": (
                                source_values
                                if order_variant == "source"
                                else list(reversed(source_values))
                            ),
                            "method_id": "forced_choice_next_token_softmax.v1",
                            "temperature": 1.0,
                            "acquisition": "new_authorization_required",
                            "artifact_relative_path": (
                                f"new/{model_id}/{experiment_id}/{arm_id}/"
                                f"{order_variant}.json"
                            ),
                        }
                    entry["source_option_values"] = source_values
                    entry["display_option_values"] = (
                        source_values
                        if order_variant == "source"
                        else list(reversed(source_values))
                    )
                    entry.pop("option_values", None)
                    entry["model_exposure"] = model_specs[model_id][
                        "exposure_by_experiment"
                    ][experiment_id]
                    entry["full_action_pair_key"] = (
                        f"{model_id}--{experiment_id}--{arm_id}"
                    )
                    logical_calls.append(entry)
    new_calls = [
        call for call in logical_calls
        if call["acquisition"] == "new_authorization_required"
    ]
    arm_count = sum(
        len(
            read_json_object(
                root / f"data/manifests/contracts/{experiment_id}_blinded_bundle.json"
            )["arms"]
        )
        for experiment_id in EXPERIMENT_IDS
    )
    result = {
        "schema_version": "balanced_full_action_plan.v1",
        "plan_id": "intervenebench-balanced-full-action-20260813-v1",
        "freeze_date": "2026-08-13",
        "status": "frozen_nonexecuting_zero_authority",
        "experiment_ids": list(EXPERIMENT_IDS),
        "model_ids": list(MODEL_IDS),
        "arm_rule": "all source-declared admissible arms in bundle order",
        "nuisance_variant_rule": "first source-declared variant for every arm",
        "answer_order_rule": "source and full reverse for every arm",
        "order_aggregation": (
            "equal weight after reverse outputs are inverse-mapped to source values"
        ),
        "existing_artifact_hashes": existing_hashes,
        "counts": {
            "experiment_count": len(EXPERIMENT_IDS),
            "model_count": len(MODEL_IDS),
            "unique_arm_count": arm_count,
            "balanced_arm_prediction_count": arm_count * len(MODEL_IDS),
            "logical_ordered_call_count": len(logical_calls),
            "reused_ordered_call_count": len(logical_calls) - len(new_calls),
            "new_ordered_call_count": len(new_calls),
            "new_calls_per_model": len(new_calls) // len(MODEL_IDS),
            "full_action_recommendation_count": len(EXPERIMENT_IDS) * len(MODEL_IDS),
        },
        "logical_calls": logical_calls,
        "new_calls": new_calls,
    }
    assert_blinded_payload(result)
    return result


def verify_full_action_freeze(
    root: Path, *, freeze_path: Path, plan_path: Path
) -> dict[str, Any]:
    freeze = read_json_object(freeze_path)
    plan = read_json_object(plan_path)
    if freeze.get("schema_version") != "balanced_full_action_freeze.v1":
        raise ValueError("unsupported balanced full-action freeze")
    if freeze.get("status") != "frozen_nonexecuting_zero_authority":
        raise ValueError("full-action freeze must have zero authority")
    if any(freeze["authority"].values()):
        raise PermissionError("full-action freeze embeds expanded authority")
    if plan != build_full_action_plan(root):
        raise ValueError("full-action plan does not replay exactly")
    if freeze["call_plan_payload_sha256"] != payload_hash(plan):
        raise ValueError("full-action freeze/plan binding mismatch")
    for entry in freeze["implementation_hashes"]:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("full-action implementation path escapes repository")
        if sha256_file(root / relative) != entry["file_sha256"]:
            raise ValueError(f"full-action implementation hash mismatch: {entry['path']}")
    for entry in freeze["task_scope"]["packaged_files"]:
        bundle = read_json_object(root / entry["path"])
        if sha256_file(root / entry["path"]) != entry["file_sha256"]:
            raise ValueError("full-action bundle file hash mismatch")
        if payload_hash(bundle) != entry["payload_sha256"]:
            raise ValueError("full-action bundle payload hash mismatch")
        if bundle["outcome_access"] != "sealed" or bundle["reveal_authorized"] is not False:
            raise PermissionError("full-action bundle is not outcome sealed")
    counts = plan["counts"]
    expected = (136, 80, 56, 14)
    observed = (
        counts["logical_ordered_call_count"],
        counts["reused_ordered_call_count"],
        counts["new_ordered_call_count"],
        counts["new_calls_per_model"],
    )
    if observed != expected:
        raise ValueError("full-action call counts drifted")
    limits = freeze["limits"]
    if (limits["maximum_planned_new_calls"], limits["maximum_model_attempts"]) != (56, 56):
        raise ValueError("full-action attempt ceiling drifted")
    if float(limits["hard_incremental_cost_cap_usd"]) != 1.75:
        raise ValueError("full-action cost cap drifted")
    assert_blinded_payload(freeze)
    assert_blinded_payload(plan)
    return {
        "logical_call_count": 136,
        "reused_call_count": 80,
        "new_call_count": 56,
        "incremental_cost_cap_usd": 1.75,
        "freeze_payload_sha256": payload_hash(freeze),
        "call_plan_payload_sha256": payload_hash(plan),
    }


def prepare_new_requests(
    root: Path, *, freeze_path: Path, plan_path: Path
) -> tuple[dict[str, Any], ...]:
    """Reconstruct the exact 56 missing prompts after verifying the freeze."""

    verify_full_action_freeze(root, freeze_path=freeze_path, plan_path=plan_path)
    plan = read_json_object(plan_path)
    requests: list[dict[str, Any]] = []
    for call in plan["new_calls"]:
        bundle = read_json_object(
            root / f"data/manifests/contracts/{call['experiment_id']}_blinded_bundle.json"
        )
        prompt = (
            screen_prompt(
                bundle, arm_id=call["arm_id"], variant_id=call["variant_id"]
            )
            if call["order_variant"] == "source"
            else reverse_order_prompt(
                bundle, arm_id=call["arm_id"], variant_id=call["variant_id"]
            )
        )
        if sha256(prompt.encode("utf-8")).hexdigest() != call["prompt_sha256"]:
            raise ValueError("full-action prompt does not match frozen plan")
        request = dict(call)
        request["prompt"] = prompt
        assert_blinded_payload(request)
        requests.append(request)
    return tuple(requests)


def verify_new_result(
    request: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    required = {
        "call_id", "model_id", "probabilities_by_code", "candidate_token_ids",
        "candidate_token_strings", "runtime_attestation"
    }
    if set(result) != required:
        raise ValueError("full-action result fields differ from schema")
    if result["call_id"] != request["call_id"] or result["model_id"] != request["model_id"]:
        raise ValueError("full-action result identity mismatch")
    probabilities = result["probabilities_by_code"]
    codes = tuple(request["answer_codes"])
    if not isinstance(probabilities, Mapping) or tuple(probabilities) != codes:
        raise ValueError("full-action probability code order mismatch")
    probability_values = list(probabilities.values())
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or not 0.0 <= float(value) <= 1.0
            for value in probability_values
        )
        or abs(fsum(float(value) for value in probability_values) - 1.0) > 1e-6
    ):
        raise ValueError("full-action probabilities are invalid")
    token_ids = result["candidate_token_ids"]
    if (
        not isinstance(token_ids, list)
        or len(token_ids) != len(codes)
        or len(set(token_ids)) != len(token_ids)
        or not all(isinstance(item, int) for item in token_ids)
    ):
        raise ValueError("full-action token contract failed")
    if result["candidate_token_strings"] != list(codes):
        raise ValueError("full-action token strings mismatch")
    if not isinstance(result["runtime_attestation"], Mapping):
        raise ValueError("full-action runtime attestation missing")
    return {
        "call_id": request["call_id"],
        "model_id": request["model_id"],
        "experiment_id": request["experiment_id"],
        "arm_id": request["arm_id"],
        "variant_id": request["variant_id"],
        "order_variant": request["order_variant"],
        "full_action_pair_key": request["full_action_pair_key"],
        "model_exposure": request["model_exposure"],
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
        "runtime_attestation": dict(result["runtime_attestation"]),
    }


def build_materialization_authorization(
    *, freeze: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": "balanced_full_action_materialization_authorization.v1",
        "authorization_id": "intervenebench-balanced-full-action-materialization-20260813-v1",
        "scope": "materialize_exact_balanced_full_action_image_only",
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
        "status": "balanced_full_action_image_only_authorized",
    }
    assert_blinded_payload(payload)
    return payload


def validate_materialization_authorization(
    authorization: Mapping[str, Any], *, freeze: Mapping[str, Any],
    plan: Mapping[str, Any]
) -> None:
    if dict(authorization) != build_materialization_authorization(
        freeze=freeze, plan=plan
    ):
        raise PermissionError("full-action materialization authorization mismatch")


def build_execution_authorization(
    *, freeze: Mapping[str, Any], plan: Mapping[str, Any], modal_image_id: str,
    cache_hashes: Mapping[str, str]
) -> dict[str, Any]:
    if not modal_image_id or set(cache_hashes) != set(MODEL_IDS):
        raise ValueError("full-action image/cache binding incomplete")
    payload = {
        "schema_version": "balanced_full_action_execution_authorization.v1",
        "authorization_id": "intervenebench-balanced-full-action-execution-20260813-v1",
        "scope": "exact_56_missing_balanced_full_action_calls_only",
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
        "maximum_planned_calls": 56,
        "maximum_model_attempts": 56,
        "maximum_incremental_cost_usd": 1.75,
        "status": "single_balanced_full_action_completion_authorized",
    }
    assert_blinded_payload(payload)
    return payload


def validate_execution_authorization(
    authorization: Mapping[str, Any], *, freeze: Mapping[str, Any],
    plan: Mapping[str, Any], modal_image_id: str,
    cache_hashes: Mapping[str, str]
) -> None:
    expected = build_execution_authorization(
        freeze=freeze, plan=plan, modal_image_id=modal_image_id,
        cache_hashes=cache_hashes
    )
    if dict(authorization) != expected:
        raise PermissionError("full-action execution authorization mismatch")


def validate_runtime_attestation(
    request: Mapping[str, Any], attestation: Mapping[str, Any], *,
    freeze: Mapping[str, Any], authorization: Mapping[str, Any]
) -> None:
    model = {item["model_id"]: item for item in freeze["models"]}[
        request["model_id"]
    ]
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
        ][request["model_id"]],
        "call_id": request["call_id"],
        "prompt_sha256": request["prompt_sha256"],
        "method_id": "forced_choice_next_token_softmax.v1",
    }
    for field, value in expected.items():
        if attestation.get(field) != value:
            raise ValueError(f"full-action runtime mismatch: {field}")
    if str(attestation.get("torch_version", "")).split("+")[0] != "2.9.1":
        raise ValueError("full-action runtime mismatch: torch_version")
    if str(attestation.get("python_version", "")).split(".")[:2] != ["3", "11"]:
        raise ValueError("full-action runtime mismatch: python_version")
    if "L40S" not in str(attestation.get("gpu_name", "")):
        raise ValueError("full-action runtime mismatch: gpu_name")


def build_completed_full_action_artifact(
    root: Path, *, new_run_root: Path
) -> dict[str, Any]:
    """Combine 80 reused and 56 new calls into 20 full recommendations."""

    plan = read_json_object(
        root / "data/manifests/simulators/balanced_full_action_plan_v1.json"
    )
    final = verify_envelope(new_run_root / "final_manifest.json", require_blinded=True)
    if final.get("status") != "balanced_full_action_completion_passed_56_of_56_stop":
        raise ValueError("full-action completion run is not complete")
    new_ids = {call["call_id"] for call in plan["new_calls"]}
    if new_ids != set(final["call_output_sha256"]):
        raise ValueError("full-action completion outputs differ from plan")
    bundles = {
        experiment_id: read_json_object(
            root / f"data/manifests/contracts/{experiment_id}_blinded_bundle.json"
        )
        for experiment_id in EXPERIMENT_IDS
    }
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    component_hashes: dict[str, str] = {}
    for call in plan["logical_calls"]:
        if call["acquisition"] == "reuse_verified_existing":
            path = root / call["repository_artifact_path"]
            expected_hash = call["artifact_payload_sha256"]
        else:
            path = new_run_root / call["artifact_relative_path"]
            expected_hash = final["call_output_sha256"][call["call_id"]]
        envelope = read_json_object(path)
        if payload_hash(envelope["payload"]) != expected_hash:
            raise ValueError("full-action component hash mismatch")
        output = verify_envelope(path, require_blinded=True)
        if output["call_id"] != call["call_id"]:
            raise ValueError("full-action component identity mismatch")
        by_pair[call["full_action_pair_key"]][call["order_variant"]] = output
        component_hashes[call["call_id"]] = expected_hash

    arm_rows: list[dict[str, Any]] = []
    utility_by_arm: dict[tuple[str, str, str], float] = {}
    for pair_key, outputs in sorted(by_pair.items()):
        if set(outputs) != {"source", "reverse"}:
            raise ValueError("full-action arm lacks a paired order")
        model_id, experiment_id, arm_id = pair_key.split("--", 2)
        balanced = weighted_distribution_average(
            outputs["source"]["probabilities"],
            outputs["reverse"]["probabilities"],
        )
        bundle = bundles[experiment_id]
        source_values = [int(item["value"]) for item in bundle["response_options"]]
        utility = {
            int(item["value"]): float(item["normalized_utility"])
            for item in bundle["response_options"]
        }
        expected_utility = fsum(
            balanced[value] * utility[value] for value in balanced
        )
        modal_value = max(
            source_values,
            key=lambda value: (balanced[value], -source_values.index(value)),
        )
        exposure = next(
            call["model_exposure"]
            for call in plan["logical_calls"]
            if call["full_action_pair_key"] == pair_key
        )
        arm_rows.append(
            {
                "model_id": model_id,
                "experiment_id": experiment_id,
                "arm_id": arm_id,
                "model_exposure": exposure,
                "balanced_probabilities": balanced,
                "balanced_modal_response_value": modal_value,
                "balanced_expected_normalized_utility": expected_utility,
                "source_order_call_id": outputs["source"]["call_id"],
                "reverse_order_call_id": outputs["reverse"]["call_id"],
            }
        )
        utility_by_arm[(model_id, experiment_id, arm_id)] = expected_utility

    recommendations = []
    for model_id in MODEL_IDS:
        for experiment_id in EXPERIMENT_IDS:
            arm_order = [
                arm["arm_id"] for arm in bundles[experiment_id]["arms"]
            ]
            chosen = max(
                arm_order,
                key=lambda arm_id: (
                    utility_by_arm[(model_id, experiment_id, arm_id)],
                    -arm_order.index(arm_id),
                ),
            )
            recommendations.append(
                {
                    "model_id": model_id,
                    "experiment_id": experiment_id,
                    "admissible_arm_ids": arm_order,
                    "chosen_arm_id": chosen,
                    "chosen_expected_normalized_utility": utility_by_arm[
                        (model_id, experiment_id, chosen)
                    ],
                    "model_exposure": next(
                        row["model_exposure"]
                        for row in arm_rows
                        if row["model_id"] == model_id
                        and row["experiment_id"] == experiment_id
                    ),
                    "full_action_set_recommendation": True,
                    "human_outcome_accessed": False,
                }
            )
    result = {
        "schema_version": "balanced_full_action_recommendations.v1",
        "completion_run_manifest_payload_sha256": payload_hash(final),
        "call_plan_payload_sha256": payload_hash(plan),
        "component_call_output_sha256": dict(sorted(component_hashes.items())),
        "outcome_access": "not_accessed",
        "scope": "development_discovery_only_full_action_no_human_scoring",
        "ordered_component_call_count": len(component_hashes),
        "balanced_arm_prediction_count": len(arm_rows),
        "full_action_recommendation_count": len(recommendations),
        "order_aggregation": (
            "equal_weight_source_and_full_reverse_after_inverse_mapping"
        ),
        "balanced_arm_predictions": arm_rows,
        "full_action_recommendations": recommendations,
        "interpretation_boundary": (
            "These are complete synthetic recommendations but not human accuracy, "
            "treatment-effect fidelity, or regret results. Exposure labels constrain "
            "which comparisons may be primary."
        ),
        "status": "complete_balanced_full_action_recommendations_outcome_blind",
    }
    assert_blinded_payload(result)
    return result
