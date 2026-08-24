"""Outcome-sealed call planning for the three prospective image experiments."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from string import ascii_uppercase
from typing import Any, Mapping

from .balanced_forced_choice import read_json_object
from .protocol import assert_blinded_payload, payload_hash
from .simulators import (
    ordinal_png_multimodal_prompt,
    validate_ordinal_png_multimodal_bundle,
)


EXPERIMENT_IDS = ("nj5dx", "es4xw", "e2pyb")
MODEL_SPECS = (
    {
        "model_id": "qwen3_vl_8b_primary",
        "hf_repository": "Qwen/Qwen3-VL-8B-Instruct",
        "checkpoint_commit": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "modality": "exact_png_vision",
        "role": "primary_multimodal",
        "license_id": "apache-2.0",
        "training_exposure": "unknown_pretraining_exposure",
    },
    {
        "model_id": "qwen2_5_vl_7b_comparator",
        "hf_repository": "Qwen/Qwen2.5-VL-7B-Instruct",
        "checkpoint_commit": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "modality": "exact_png_vision",
        "role": "multimodal_robustness_comparator",
        "license_id": "apache-2.0",
        "training_exposure": "unknown_pretraining_exposure",
    },
    {
        "model_id": "qwen3_8b_text_ablation",
        "hf_repository": "Qwen/Qwen3-8B",
        "checkpoint_commit": "b968826d9c46dd6066d109eabc6255188de91218",
        "modality": "accessible_text_only",
        "role": "frozen_discovery_winner_modality_ablation",
        "license_id": "apache-2.0",
        "training_exposure": "unknown_pretraining_exposure",
    },
)
MODEL_IDS = tuple(spec["model_id"] for spec in MODEL_SPECS)
VISION_MODEL_IDS = tuple(
    spec["model_id"]
    for spec in MODEL_SPECS
    if spec["modality"] == "exact_png_vision"
)


@dataclass(frozen=True, slots=True)
class MultimodalForcedChoicePrompt:
    text: str
    asset_path: str | None
    asset_sha256: str | None


def _answer_codes(option_count: int) -> tuple[str, ...]:
    if not 2 <= option_count <= 8:
        raise ValueError("multimodal forced choice supports two through eight options")
    return tuple(ascii_uppercase[:option_count])


def multimodal_forced_choice_prompt(
    bundle: Mapping[str, Any],
    *,
    arm_id: str,
    repository_root: Path,
    option_order: str,
    include_exact_image: bool,
) -> MultimodalForcedChoicePrompt:
    """Create one answer-code prompt and verify any exact fielded PNG."""

    validate_ordinal_png_multimodal_bundle(bundle)
    if option_order not in {"source", "reverse"}:
        raise ValueError("option_order must be source or reverse")
    arms = {arm["arm_id"]: arm for arm in bundle["arms"]}
    if arm_id not in arms:
        raise ValueError("requested multimodal arm is absent")
    arm = arms[arm_id]
    options = list(bundle["response_options"])
    if option_order == "reverse":
        options.reverse()
    codes = _answer_codes(len(options))
    code_list = ", ".join(codes[:-1]) + f", or {codes[-1]}"
    answers = "\n".join(
        f"{code}: {option['label']}" for code, option in zip(codes, options)
    )
    accessible_text = arm["accessible_text"]
    if not include_exact_image and bundle["experiment_id"] == "es4xw":
        # Picture numbers carry no semantic content and would create a fake arm cue
        # when the exact team images are deliberately withheld from this ablation.
        _, separator, remainder = accessible_text.partition(":")
        if not separator:
            raise ValueError("es4xw accessible text lacks its expected picture prefix")
        accessible_text = f"Exact source picture:{remainder}"
    modality_note = (
        "The exact fielded PNG is attached."
        if include_exact_image
        else (
            "This is an accessible-text-only ablation. The original image is not "
            "available to this model."
        )
    )
    text = (
        "Predict the answer of one randomly selected adult from the population "
        f"below. Do not answer as yourself. Choose exactly one answer code from "
        f"{code_list}. Do not explain your choice. Do not identify or label any "
        "individual person's protected identity; predict only the population "
        "response to the stated question.\n\n"
        f"Population: {bundle['population']['description']}\n\n"
        f"Context: {bundle['common_context']}\n\n"
        f"Modality: {modality_note}\n\n"
        f"Accessible image description: {accessible_text}\n\n"
        f"Question: {bundle['outcome_question']}\n"
        f"Answer codes:\n{answers}\n\n"
        "Return only the answer code."
    )
    if not include_exact_image:
        return MultimodalForcedChoicePrompt(text, None, None)
    materialized = ordinal_png_multimodal_prompt(
        bundle,
        arm_id=arm_id,
        repository_root=repository_root,
        option_order=option_order,
    )
    return MultimodalForcedChoicePrompt(
        text=text,
        asset_path=materialized.asset_paths[0],
        asset_sha256=materialized.asset_sha256[0],
    )


def build_multimodal_prospective_plan(root: Path) -> dict[str, Any]:
    """Build the complete zero-authority 54-call prospective plan."""

    model_lookup = {spec["model_id"]: spec for spec in MODEL_SPECS}
    calls: list[dict[str, Any]] = []
    task_files: list[dict[str, Any]] = []
    for experiment_id in EXPERIMENT_IDS:
        relative = Path(
            f"data/manifests/contracts/{experiment_id}_blinded_bundle.json"
        )
        bundle = read_json_object(root / relative)
        validate_ordinal_png_multimodal_bundle(bundle)
        task_files.append(
            {
                "experiment_id": experiment_id,
                "path": relative.as_posix(),
                "payload_sha256": payload_hash(bundle),
            }
        )
        source_values = [int(option["value"]) for option in bundle["response_options"]]
        answer_codes = list(_answer_codes(len(source_values)))
        for model_id in MODEL_IDS:
            spec = model_lookup[model_id]
            include_exact_image = spec["modality"] == "exact_png_vision"
            for arm in bundle["arms"]:
                for option_order in ("source", "reverse"):
                    prepared = multimodal_forced_choice_prompt(
                        bundle,
                        arm_id=arm["arm_id"],
                        repository_root=root,
                        option_order=option_order,
                        include_exact_image=include_exact_image,
                    )
                    call_id = (
                        f"prospective-mm--{model_id}--{experiment_id}--"
                        f"{arm['arm_id']}--{option_order}"
                    )
                    calls.append(
                        {
                            "call_id": call_id,
                            "model_id": model_id,
                            "experiment_id": experiment_id,
                            "bundle_payload_sha256": payload_hash(bundle),
                            "arm_id": arm["arm_id"],
                            "option_order": option_order,
                            "source_option_values": source_values,
                            "display_option_values": (
                                source_values
                                if option_order == "source"
                                else list(reversed(source_values))
                            ),
                            "answer_codes": answer_codes,
                            "prompt_sha256": sha256(
                                prepared.text.encode("utf-8")
                            ).hexdigest(),
                            "asset_path": (
                                str(
                                    Path(prepared.asset_path).resolve().relative_to(
                                        root.resolve()
                                    )
                                )
                                if prepared.asset_path is not None
                                else None
                            ),
                            "asset_sha256": prepared.asset_sha256,
                            "modality": spec["modality"],
                            "method_id": "forced_choice_next_token_softmax.v1",
                            "temperature": 1.0,
                            "generation_calls": 0,
                            "artifact_relative_path": (
                                f"calls/{model_id}/{experiment_id}/{arm['arm_id']}/"
                                f"{option_order}.json"
                            ),
                        }
                    )
    plan = {
        "schema_version": "prospective_multimodal_call_plan.v1",
        "plan_id": "intervenebench-prospective-multimodal-20260813-v1",
        "freeze_date": "2026-08-13",
        "status": "frozen_nonexecuting_zero_authority",
        "evidence_tier": "prospective_development_outcome_sealed",
        "experiment_ids": list(EXPERIMENT_IDS),
        "model_specs": list(MODEL_SPECS),
        "primary_model_id": "qwen3_vl_8b_primary",
        "text_baseline_id": "qwen3_8b_text_ablation",
        "task_files": task_files,
        "calls": calls,
        "logical_call_count": len(calls),
        "vision_call_count": sum(
            call["modality"] == "exact_png_vision" for call in calls
        ),
        "text_ablation_call_count": sum(
            call["modality"] == "accessible_text_only" for call in calls
        ),
        "order_aggregation": (
            "equal source/reverse distribution average after inverse mapping"
        ),
        "recommendation_rule": (
            "maximize balanced expected normalized utility across every source arm; "
            "tie by source arm order"
        ),
        "selection_boundary": {
            "target_human_outcomes_accessed": False,
            "target_human_outcomes_used": False,
            "primary_vlm_selected_outcome_blind": True,
            "selection_rationale": (
                "Qwen3-VL-8B is the closest open multimodal continuation of the "
                "Qwen3 family selected on revealed text-only discovery tasks; "
                "Qwen2.5-VL-7B is a pinned same-scale robustness comparator."
            ),
            "socrates_excluded_reason": (
                "text-only and released participant mapping marks all three target "
                "experiments as training-seen"
            ),
        },
        "authority": {
            "model_download_authorized": False,
            "image_materialization_authorized": False,
            "paid_inference_authorized": False,
            "human_outcome_access_authorized": False,
            "outcome_reveal_authorized": False,
            "automatic_next_stage_authorized": False,
        },
    }
    if len(calls) != 54 or plan["vision_call_count"] != 36:
        raise ValueError("prospective multimodal call arithmetic drifted")
    assert_blinded_payload(plan)
    return plan
