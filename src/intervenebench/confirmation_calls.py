"""Deterministic call materialization for the sealed confirmation panel."""

from __future__ import annotations

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from string import ascii_uppercase
from typing import Any, Mapping, Sequence

from .confirmation_preparation import (
    BASE_CALLS,
    CONFIRMATION_IDS,
    DEFAULT_CONFIRMATION_PREPARATION_PATH,
    MAXIMUM_ATTEMPTS,
    PERTURBATION_CALLS,
    PLANNED_CALLS,
    RESERVE_CALLS,
    verify_confirmation_preparation,
)
from .protocol import assert_blinded_payload, freeze_envelope, payload_hash, verify_envelope
from .simulators import (
    materialize_sequence_episode,
    ollama_continuous_prompt,
    ordinal_probability_prompt,
    sequence_probability_prompt,
)


DEFAULT_CONFIRMATION_CALL_PLAN_PATH = Path(
    "data/manifests/simulators/confirmation_call_plan_v1.json"
)
_SEQUENCE_SEED_BASE = 202608140500
_CONTINUOUS_SEED_BASE = 202608140600
_STAGES = (
    "base",
    "primary_prompt_perturbation",
    "outcome_free_adaptive_reserve",
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _paths(root: Path, experiment_id: str) -> tuple[Path, Path]:
    contracts = root / "data/manifests/contracts"
    if experiment_id == "tcg8p":
        return (
            contracts / "tcg8p_continuous_task_candidate.json",
            contracts / "tcg8p_continuous_blinded_bundle.json",
        )
    return (
        contracts / f"{experiment_id}_decision_task_candidate.json",
        contracts / f"{experiment_id}_blinded_bundle.json",
    )


def _answer_codes(count: int) -> list[str]:
    if not 2 <= count <= 12:
        raise ValueError("confirmation forced choice supports two through twelve options")
    return list(ascii_uppercase[:count])


def _options(
    candidate: Mapping[str, Any], bundle: Mapping[str, Any]
) -> tuple[list[int], list[str]]:
    raw = bundle.get("response_options")
    if isinstance(raw, list):
        return (
            [int(option["value"]) for option in raw],
            [str(option["label"]) for option in raw],
        )
    if candidate.get("experiment_id") == "pb2rr":
        values = [int(value) for value in candidate["response_options"]]
        return values, [f"${value}" for value in values]
    raise ValueError("confirmation response options are unavailable")


def _forced_choice_wrapper(
    body: str,
    *,
    values: Sequence[int],
    labels: Sequence[str],
    answer_order: str,
    prompt_variant: str,
) -> tuple[str, list[int], list[str]]:
    if answer_order not in {"source", "reverse"}:
        raise ValueError("forced-choice answer order must be source or reverse")
    if len(values) != len(labels):
        raise ValueError("forced-choice values and labels must align")
    display_values = list(values)
    display_labels = list(labels)
    if answer_order == "reverse":
        display_values.reverse()
        display_labels.reverse()
    codes = _answer_codes(len(values))
    code_list = ", ".join(codes[:-1]) + f", or {codes[-1]}"
    answers = "\n".join(
        f"{code}: {label}" for code, label in zip(codes, display_labels, strict=True)
    )
    if prompt_variant == "standard":
        instruction = (
            "Predict the answer of one randomly selected adult from the population "
            f"below. Do not answer as yourself. Choose exactly one answer code from "
            f"{code_list}. Do not explain your choice."
        )
        closing = "Return only the answer code."
    elif prompt_variant == "alternate_format":
        instruction = (
            "Simulate one randomly selected respondent from the population described "
            f"below. Select exactly one code ({code_list}) for that respondent. Give "
            "no explanation or reasoning."
        )
        closing = "Output exactly the single selected code and nothing else."
    else:
        raise ValueError("unsupported confirmation prompt variant")
    return (
        f"{instruction}\n\n{body}\nAnswer codes:\n{answers}\n\n{closing}",
        display_values,
        codes,
    )


def _sequence_body(
    bundle: Mapping[str, Any], *, arm_id: str, sequence_seed: int
) -> tuple[str, str]:
    episode = materialize_sequence_episode(bundle, seed=sequence_seed)
    probability_prompt = sequence_probability_prompt(
        bundle, arm_id=arm_id, episode=episode
    )
    marker = "Population:"
    if marker not in probability_prompt or "\nTarget answers:" not in probability_prompt:
        raise ValueError("sequence prompt shape drifted")
    body = marker + probability_prompt.split(marker, 1)[1]
    body = body.rsplit("\nTarget answers:", 1)[0] + "\n"
    return body, episode.episode_id


def _ordinal_body(
    bundle: Mapping[str, Any], *, arm_id: str, variant_id: str
) -> str:
    probability_prompt = ordinal_probability_prompt(
        bundle, arm_id=arm_id, variant_id=variant_id
    )
    marker = "Population:"
    if marker not in probability_prompt or "\nAnswers:" not in probability_prompt:
        raise ValueError("ordinal prompt shape drifted")
    body = marker + probability_prompt.split(marker, 1)[1]
    return body.rsplit("\nAnswers:", 1)[0] + "\n"


def _pb2rr_body(
    bundle: Mapping[str, Any], *, arm_id: str, nuisance_id: str, include_image: bool
) -> str:
    arms = {str(arm["arm_id"]): arm for arm in bundle["arms"]}
    nuisance = {
        str(level["nuisance_id"]): level for level in bundle["nuisance_contract"]["levels"]
    }
    arm = arms[arm_id]
    level = nuisance[nuisance_id]
    modality = (
        "The exact source article PNG is attached."
        if include_image
        else "This accessible-text ablation does not receive the source image."
    )
    return (
        f"Population: {bundle['population']['description']}\n\n"
        f"Context: {bundle['common_context']}\n\n"
        f"Randomized recipient context: {level['context']}\n\n"
        f"Modality: {modality}\n\n"
        f"Accessible article description: {arm['accessible_text']}\n\n"
        f"Question: {bundle['outcome_question']}\n"
    )


def _continuous_prompt(
    bundle: Mapping[str, Any], *, arm_id: str, prompt_variant: str
) -> str:
    standard = ollama_continuous_prompt(bundle, arm_id=arm_id)
    if prompt_variant == "standard":
        return standard
    if prompt_variant != "alternate_format" or "Population:" not in standard:
        raise ValueError("unsupported continuous prompt variant")
    body = "Population:" + standard.split("Population:", 1)[1]
    return (
        "Simulate one randomly selected adult from the population below rather than "
        "answering as yourself. Output exactly one JSON object with one integer field "
        'named predicted_value, for example {"predicted_value": 12}. The integer must '
        "be at least zero. Give no explanation.\n\n"
        f"{body}"
    )


def _prompt_for_call(
    *,
    root: Path,
    candidate: Mapping[str, Any],
    bundle: Mapping[str, Any],
    call: Mapping[str, Any],
) -> tuple[str, list[int] | None, list[str] | None]:
    experiment_id = str(call["experiment_id"])
    prompt_variant = str(call["prompt_variant"])
    if experiment_id == "tcg8p":
        return (
            _continuous_prompt(
                bundle, arm_id=str(call["arm_id"]), prompt_variant=prompt_variant
            ),
            None,
            None,
        )
    values, labels = _options(candidate, bundle)
    if experiment_id in {"z358z", "ShannonS2", "KlarS44"}:
        body, episode_id = _sequence_body(
            bundle,
            arm_id=str(call["arm_id"]),
            sequence_seed=int(call["sequence_seed"]),
        )
        if call.get("sequence_episode_id") != episode_id:
            raise ValueError("sequence episode hash drifted")
    elif experiment_id == "Blair1131":
        body = _ordinal_body(
            bundle,
            arm_id=str(call["arm_id"]),
            variant_id=str(call["nuisance_id"]),
        )
    elif experiment_id == "pb2rr":
        body = _pb2rr_body(
            bundle,
            arm_id=str(call["arm_id"]),
            nuisance_id=str(call["nuisance_id"]),
            include_image=call["asset_path"] is not None,
        )
    else:
        raise ValueError("unsupported confirmation experiment")
    return _forced_choice_wrapper(
        body,
        values=values,
        labels=labels,
        answer_order=str(call["answer_order"]),
        prompt_variant=prompt_variant,
    )


def _cell_specs(
    experiment_id: str,
    *,
    stage: str,
    bundle: Mapping[str, Any],
) -> list[dict[str, Any]]:
    prompt_variant = (
        "alternate_format"
        if stage == "primary_prompt_perturbation"
        else "standard"
    )
    cells: list[dict[str, Any]] = []
    if experiment_id == "tcg8p":
        indices = range(20, 40) if stage == "outcome_free_adaptive_reserve" else range(20)
        for index in indices:
            cells.append(
                {
                    "nuisance_id": f"draw_{index:02d}",
                    "answer_order": "not_applicable",
                    "prompt_variant": prompt_variant,
                    "sequence_seed": None,
                    "sequence_episode_id": None,
                    "generation_seed": _CONTINUOUS_SEED_BASE + index,
                }
            )
    elif experiment_id == "pb2rr":
        if stage == "outcome_free_adaptive_reserve":
            return []
        for level in bundle["nuisance_contract"]["levels"]:
            for order in ("source", "reverse"):
                cells.append(
                    {
                        "nuisance_id": str(level["nuisance_id"]),
                        "answer_order": order,
                        "prompt_variant": prompt_variant,
                        "sequence_seed": None,
                        "sequence_episode_id": None,
                        "generation_seed": None,
                    }
                )
    elif experiment_id in {"z358z", "ShannonS2", "KlarS44"}:
        indices = range(8, 16) if stage == "outcome_free_adaptive_reserve" else range(8)
        for index in indices:
            seed = _SEQUENCE_SEED_BASE + index
            episode = materialize_sequence_episode(bundle, seed=seed)
            for order in ("source", "reverse"):
                cells.append(
                    {
                        "nuisance_id": f"episode_{index:02d}",
                        "answer_order": order,
                        "prompt_variant": prompt_variant,
                        "sequence_seed": seed,
                        "sequence_episode_id": episode.episode_id,
                        "generation_seed": None,
                    }
                )
    elif experiment_id == "Blair1131":
        if stage == "outcome_free_adaptive_reserve":
            return []
        first_arm = bundle["arms"][0]
        for variant in first_arm["message_variants"]:
            for order in ("source", "reverse"):
                cells.append(
                    {
                        "nuisance_id": str(variant["variant_id"]),
                        "answer_order": order,
                        "prompt_variant": prompt_variant,
                        "sequence_seed": None,
                        "sequence_episode_id": None,
                        "generation_seed": None,
                    }
                )
    else:
        raise ValueError("unsupported confirmation task")
    return cells


def build_confirmation_call_plan(root: Path) -> dict[str, Any]:
    preparation = verify_confirmation_preparation(
        root, root / DEFAULT_CONFIRMATION_PREPARATION_PATH
    )
    task_protocol = {
        row["experiment_id"]: row for row in preparation["protocol_snapshot"]["tasks"]
    }
    pb_assets = {
        row["arm_id"]: row for row in preparation["pb2rr_modal_assets"]
    }
    calls: list[dict[str, Any]] = []
    task_files: list[dict[str, Any]] = []
    for experiment_id in CONFIRMATION_IDS:
        candidate_path, bundle_path = _paths(root, experiment_id)
        candidate = _read_object(candidate_path)
        bundle = _read_object(bundle_path)
        task_files.append(
            {
                "experiment_id": experiment_id,
                "candidate_path": str(candidate_path.relative_to(root)),
                "candidate_payload_sha256": payload_hash(candidate),
                "bundle_path": str(bundle_path.relative_to(root)),
                "bundle_payload_sha256": payload_hash(bundle),
            }
        )
        protocol = task_protocol[experiment_id]
        arm_ids = [str(arm["arm_id"]) for arm in bundle["arms"]]
        for stage in _STAGES:
            cells = _cell_specs(experiment_id, stage=stage, bundle=bundle)
            model_ids = (
                protocol["base_model_ids"]
                if stage == "base"
                else [protocol["primary_model_id"]]
            )
            for model_id in model_ids:
                for arm_id in arm_ids:
                    for cell in cells:
                        asset_path: str | None = None
                        asset_sha256: str | None = None
                        modality = "text"
                        if experiment_id == "pb2rr":
                            if model_id in {
                                "qwen3_vl_8b_primary",
                                "qwen2_5_vl_7b_comparator",
                            }:
                                asset_path = pb_assets[arm_id]["png_path"]
                                asset_sha256 = pb_assets[arm_id]["png_sha256"]
                                modality = "exact_png_vision"
                            else:
                                modality = "accessible_text_only"
                        call_stub: dict[str, Any] = {
                            "stage": stage,
                            "model_id": model_id,
                            "experiment_id": experiment_id,
                            "arm_id": arm_id,
                            **cell,
                            "asset_path": asset_path,
                            "asset_sha256": asset_sha256,
                        }
                        prompt, display_values, codes = _prompt_for_call(
                            root=root,
                            candidate=candidate,
                            bundle=bundle,
                            call=call_stub,
                        )
                        nuisance_id = cell["nuisance_id"]
                        order = cell["answer_order"]
                        call_id = (
                            f"confirmation--{stage}--{model_id}--{experiment_id}--"
                            f"{arm_id}--{nuisance_id}--{order}"
                        )
                        source_values = None
                        if experiment_id != "tcg8p":
                            source_values, _ = _options(candidate, bundle)
                        calls.append(
                            {
                                "call_id": call_id,
                                **call_stub,
                                "adapter": protocol["adapter"],
                                "bundle_payload_sha256": payload_hash(bundle),
                                "source_option_values": source_values,
                                "display_option_values": display_values,
                                "answer_codes": codes,
                                "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
                                "modality": modality,
                                "method_id": (
                                    "continuous_constrained_integer_generation.v1"
                                    if experiment_id == "tcg8p"
                                    else "forced_choice_next_token_softmax.v1"
                                ),
                                "temperature": 0.7 if experiment_id == "tcg8p" else 1.0,
                                "top_p": 0.9 if experiment_id == "tcg8p" else None,
                                "max_new_tokens": 32 if experiment_id == "tcg8p" else 0,
                                "generation_calls": 1 if experiment_id == "tcg8p" else 0,
                                "artifact_relative_path": (
                                    f"calls/{stage}/{model_id}/{experiment_id}/{arm_id}/"
                                    f"{nuisance_id}--{order}.json"
                                ),
                            }
                        )
    counts = Counter(call["stage"] for call in calls)
    if counts != Counter(
        {
            "base": BASE_CALLS,
            "primary_prompt_perturbation": PERTURBATION_CALLS,
            "outcome_free_adaptive_reserve": RESERVE_CALLS,
        }
    ):
        raise ValueError("confirmation materialized call counts drifted")
    call_ids = [call["call_id"] for call in calls]
    if len(call_ids) != len(set(call_ids)) or len(calls) != MAXIMUM_ATTEMPTS:
        raise ValueError("confirmation call IDs must be complete and unique")
    plan = {
        "schema_version": "confirmation_call_plan.v1",
        "status": "frozen_nonexecuting_zero_authority",
        "freeze_date": "2026-08-14",
        "experiment_ids": list(CONFIRMATION_IDS),
        "preparation_payload_sha256": payload_hash(preparation),
        "implementation_file_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "task_files": task_files,
        "planned_stages": ["base", "primary_prompt_perturbation"],
        "conditional_stage": "outcome_free_adaptive_reserve",
        "planned_call_count": PLANNED_CALLS,
        "maximum_attempt_count": MAXIMUM_ATTEMPTS,
        "call_count_by_stage": dict(sorted(counts.items())),
        "calls": calls,
        "authority": dict(preparation["authority"]),
        "confirmation_outcomes_accessed": False,
        "participant_rows_read": 0,
        "participant_rows_serialized": 0,
    }
    assert_blinded_payload(plan)
    return json.loads(json.dumps(plan, sort_keys=True, allow_nan=False))


def prepare_confirmation_requests(
    root: Path,
    *,
    plan: Mapping[str, Any],
    include_reserve: bool,
) -> tuple[dict[str, Any], ...]:
    if plan != build_confirmation_call_plan(root):
        raise ValueError("confirmation call plan does not replay")
    loaded: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for experiment_id in CONFIRMATION_IDS:
        candidate_path, bundle_path = _paths(root, experiment_id)
        loaded[experiment_id] = (
            _read_object(candidate_path),
            _read_object(bundle_path),
        )
    requests: list[dict[str, Any]] = []
    for frozen in plan["calls"]:
        if (
            frozen["stage"] == "outcome_free_adaptive_reserve"
            and not include_reserve
        ):
            continue
        candidate, bundle = loaded[frozen["experiment_id"]]
        prompt, display_values, codes = _prompt_for_call(
            root=root,
            candidate=candidate,
            bundle=bundle,
            call=frozen,
        )
        if sha256(prompt.encode("utf-8")).hexdigest() != frozen["prompt_sha256"]:
            raise ValueError("confirmation prompt hash does not replay")
        if display_values != frozen["display_option_values"] or codes != frozen[
            "answer_codes"
        ]:
            raise ValueError("confirmation answer mapping does not replay")
        if frozen["asset_path"] is not None:
            asset = root / frozen["asset_path"]
            if sha256(asset.read_bytes()).hexdigest() != frozen["asset_sha256"]:
                raise ValueError("confirmation call asset hash mismatch")
        request = dict(frozen)
        request["prompt"] = prompt
        requests.append(request)
    expected = MAXIMUM_ATTEMPTS if include_reserve else PLANNED_CALLS
    if len(requests) != expected:
        raise ValueError("confirmation request count drifted")
    return tuple(requests)


def write_confirmation_call_plan(root: Path) -> Path:
    path = root / DEFAULT_CONFIRMATION_CALL_PLAN_PATH
    freeze_envelope(build_confirmation_call_plan(root), path, require_blinded=True)
    return path


def verify_confirmation_call_plan(root: Path, path: Path) -> dict[str, Any]:
    payload = verify_envelope(path, require_blinded=True)
    expected = build_confirmation_call_plan(root)
    if payload != expected:
        raise ValueError("confirmation call plan does not replay")
    return payload
