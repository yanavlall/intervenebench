"""Outcome-free freeze for a retrospective cross-family robustness replay.

This module selects the exact logical calls used by the confirmation panel's
primary Qwen policy and binds them to one independent Mistral checkpoint.  It
does not download a model, make an inference call, or authorize either action.
The six panel outcomes were already revealed under the original prospective
protocol, so this study is explicitly retrospective architecture robustness.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isclose
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .model_regression import ModelVersionRegressionThresholds
from .protocol import (
    assert_blinded_payload,
    canonical_json_bytes,
    freeze_envelope,
    payload_hash,
    verify_envelope,
)


CANDIDATE_MODEL_ID = "mistral_small_3_1_24b_cross_family"
EXPERIMENT_IDS = ("tcg8p", "pb2rr", "z358z", "ShannonS2", "Blair1131", "KlarS44")
EXPECTED_CALL_COUNT_BY_EXPERIMENT = {
    "tcg8p": 120,
    "pb2rr": 128,
    "z358z": 64,
    "ShannonS2": 192,
    "Blair1131": 24,
    "KlarS44": 96,
}
EXPECTED_STAGE_COUNTS = {"base": 312, "primary_prompt_perturbation": 312}
INCLUDED_SOURCE_STAGES = tuple(EXPECTED_STAGE_COUNTS)
SYSTEM_INSTRUCTION = (
    "You are a behavioral response simulator. Follow the user prompt exactly and "
    "predict only the requested response for the described population member. Do "
    "not use study titles, authors, reported findings, or human outcome data. Return "
    "only the response schema requested by the user prompt."
)

DEFAULT_PROTOCOL_PATH = Path(
    "data/manifests/research/cross_family_regression_protocol_v1.json"
)
DEFAULT_MODEL_SOURCE_MANIFEST_PATH = Path(
    "data/manifests/simulators/mistral_small_3_1_24b_source_manifest_v1.json"
)
DEFAULT_SOURCE_CALL_PLAN_PATH = Path(
    "data/manifests/simulators/confirmation_call_plan_v1.json"
)
DEFAULT_CALL_PLAN_PATH = Path(
    "data/manifests/simulators/cross_family_call_plan_v1.json"
)

_AUTHORITY_FIELDS = frozenset(
    {
        "model_download_authorized",
        "modal_resource_creation_authorized",
        "modal_compute_authorized",
        "paid_inference_authorized",
        "candidate_inference_authorized",
        "automatic_retry_authorized",
        "reserve_call_authorized",
        "participant_row_access_authorized",
        "participant_row_serialization_authorized",
        "human_outcome_access_authorized",
        "regression_scoring_authorized",
        "automatic_next_stage_authorized",
    }
)
_MODEL_FILE_FIELDS = frozenset(
    {"path", "size_bytes", "git_oid", "storage", "content_sha256"}
)
_SOURCE_CALL_FIELDS = frozenset(
    {
        "adapter",
        "answer_codes",
        "answer_order",
        "arm_id",
        "artifact_relative_path",
        "asset_path",
        "asset_sha256",
        "bundle_payload_sha256",
        "call_id",
        "display_option_values",
        "experiment_id",
        "generation_calls",
        "generation_seed",
        "max_new_tokens",
        "method_id",
        "modality",
        "model_id",
        "nuisance_id",
        "prompt_sha256",
        "prompt_variant",
        "sequence_episode_id",
        "sequence_seed",
        "source_option_values",
        "stage",
        "temperature",
        "top_p",
    }
)
_LOGICAL_COPY_FIELDS = (
    "experiment_id",
    "stage",
    "adapter",
    "method_id",
    "modality",
    "arm_id",
    "nuisance_id",
    "answer_order",
    "prompt_variant",
    "bundle_payload_sha256",
    "asset_path",
    "asset_sha256",
    "temperature",
    "top_p",
    "max_new_tokens",
    "generation_seed",
    "sequence_episode_id",
    "sequence_seed",
    "answer_codes",
    "source_option_values",
    "display_option_values",
)


@dataclass(frozen=True, slots=True)
class CrossFamilyFreezeSummary:
    candidate_model_id: str
    experiment_count: int
    planned_call_count: int
    base_call_count: int
    prompt_perturbation_call_count: int
    maximum_gpu_seconds: int
    maximum_gpu_cost_usd: float
    hard_incremental_cost_cap_usd: float
    retrospective_only: bool


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _require_digest(value: Any, *, field: str, length: int = 64) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{field} must be a {length}-character lowercase hex digest")
    if value != value.casefold() or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a {length}-character lowercase hex digest")
    return value


def _require_no_authority(authority: Any) -> Mapping[str, Any]:
    if not isinstance(authority, Mapping) or set(authority) != _AUTHORITY_FIELDS:
        raise ValueError("cross-family authority fields are invalid")
    if any(value is not False for value in authority.values()):
        raise PermissionError("cross-family freeze must grant zero execution authority")
    return authority


def validate_cross_family_protocol(protocol: Mapping[str, Any]) -> None:
    assert_blinded_payload(protocol)
    if protocol.get("schema_version") != "intervenebench.cross_family_regression_protocol.v1":
        raise ValueError("unsupported cross-family protocol schema")
    if protocol.get("status") != "frozen_no_authority":
        raise ValueError("cross-family protocol must remain a no-authority freeze")
    if protocol.get("study_role") != "retrospective_cross_family_robustness":
        raise ValueError("cross-family study role drifted")
    if tuple(protocol.get("experiment_ids", ())) != EXPERIMENT_IDS:
        raise ValueError("cross-family experiment order drifted")
    if protocol.get("candidate_model_id") != CANDIDATE_MODEL_ID:
        raise ValueError("cross-family candidate model drifted")
    if protocol.get("planned_call_count") != 624:
        raise ValueError("cross-family protocol must freeze exactly 624 calls")
    if protocol.get("source_stages_included") != list(INCLUDED_SOURCE_STAGES):
        raise ValueError("cross-family source stages drifted")
    if protocol.get("source_stage_excluded") != "outcome_free_adaptive_reserve":
        raise ValueError("cross-family reserve exclusion drifted")
    if protocol.get("source_confirmation_call_plan_path") != str(
        DEFAULT_SOURCE_CALL_PLAN_PATH
    ):
        raise ValueError("cross-family source call-plan path drifted")
    _require_digest(
        protocol.get("source_confirmation_call_plan_file_sha256"),
        field="source_confirmation_call_plan_file_sha256",
    )
    _require_digest(
        protocol.get("source_confirmation_call_plan_payload_sha256"),
        field="source_confirmation_call_plan_payload_sha256",
    )
    if protocol.get("system_instruction", {}).get("text") != SYSTEM_INSTRUCTION:
        raise ValueError("cross-family system instruction drifted")
    if protocol.get("system_instruction", {}).get("dynamic_date_allowed") is not False:
        raise ValueError("dynamic system-prompt dates are forbidden")
    if protocol.get("system_instruction", {}).get("use_repository_system_prompt") is not False:
        raise ValueError("repository system prompt must remain disabled")
    if protocol.get("system_instruction", {}).get("sha256") != payload_hash(
        SYSTEM_INSTRUCTION
    ):
        raise ValueError("cross-family system instruction hash drifted")

    boundary = protocol.get("claim_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("cross-family claim boundary is required")
    if any(
        boundary.get(key) is not False
        for key in (
            "prospective_confirmation",
            "adds_independent_experiments",
            "changes_prospective_experiment_count",
        )
    ):
        raise ValueError("retrospective study must not inflate prospective evidence")

    inference = protocol.get("inference_contract")
    if not isinstance(inference, Mapping):
        raise ValueError("cross-family inference contract is required")
    if inference.get("logical_replay_rule") != (
        "one candidate call for every nonreserve primary-policy source call"
    ):
        raise ValueError("cross-family replay rule drifted")
    if inference.get("same_task_arm_nuisance_order_prompt_assets") is not True:
        raise ValueError("cross-family logical cells must remain paired")
    if inference.get("source_prompt_text_mutation_allowed") is not False:
        raise ValueError("source prompt text mutation is forbidden")
    interfaces = inference.get("interfaces")
    if not isinstance(interfaces, Mapping) or set(interfaces) != {
        "forced_choice_next_token",
        "continuous_integer",
    }:
        raise ValueError("cross-family interfaces drifted")
    forced = interfaces["forced_choice_next_token"]
    if (
        forced.get("preflight_requirement")
        != "every answer code is one exact distinct tokenizer token"
        or forced.get("required_schema_validity") != 1.0
        or forced.get("semantic_repair_allowed") is not False
    ):
        raise ValueError("cross-family forced-choice gate drifted")
    continuous = interfaces["continuous_integer"]
    if (
        continuous.get("parser")
        != "strict_nonnegative_integer_no_repair_no_clamp"
        or continuous.get("semantic_repair_allowed") is not False
    ):
        raise ValueError("cross-family continuous parser drifted")

    failure = protocol.get("failure_policy")
    if not isinstance(failure, Mapping):
        raise ValueError("cross-family failure policy is required")
    if (
        failure.get("automatic_retries") != 0
        or failure.get("reserve_calls") != 0
        or failure.get("primary_substitution_allowed") is not False
    ):
        raise ValueError("cross-family failure policy must fail closed")

    budget = protocol.get("budget")
    if not isinstance(budget, Mapping):
        raise ValueError("cross-family budget is required")
    if (
        budget.get("provider") != "modal"
        or budget.get("gpu_type") != "A100-80GB"
        or budget.get("gpu_count") != 1
        or budget.get("maximum_gpu_seconds") != 100_000
        or budget.get("official_gpu_price_usd_per_second") != 0.000694
        or budget.get("hard_incremental_cost_cap_usd") != 90.0
    ):
        raise ValueError("cross-family budget drifted")
    expected_gpu_cost = (
        budget["maximum_gpu_seconds"] * budget["official_gpu_price_usd_per_second"]
    )
    if not isclose(float(budget.get("maximum_gpu_cost_usd", -1)), expected_gpu_cost):
        raise ValueError("cross-family GPU cost arithmetic is invalid")

    expected_thresholds = asdict(ModelVersionRegressionThresholds())
    if protocol.get("regression_thresholds") != expected_thresholds:
        raise ValueError("cross-family regression thresholds drifted")
    _require_no_authority(protocol.get("authority"))


def validate_model_source_manifest(manifest: Mapping[str, Any]) -> None:
    assert_blinded_payload(manifest)
    if manifest.get("schema_version") != "intervenebench.model_source_manifest.v1":
        raise ValueError("unsupported model source manifest schema")
    if manifest.get("model_id") != CANDIDATE_MODEL_ID:
        raise ValueError("model source manifest candidate drifted")
    if manifest.get("hf_repository") != (
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    ):
        raise ValueError("model repository drifted")
    _require_digest(manifest.get("checkpoint_commit"), field="checkpoint_commit", length=40)
    if manifest.get("checkpoint_commit") != "68faf511d618ef198fef186659617cfd2eb8e33a":
        raise ValueError("model checkpoint drifted")
    if (
        manifest.get("license_id") != "apache-2.0"
        or manifest.get("repository_private") is not False
        or manifest.get("repository_gated") is not False
    ):
        raise ValueError("candidate model must remain public, ungated, and Apache-2.0")
    if (
        manifest.get("load_format") != "mistral_original_consolidated"
        or manifest.get("required_gpu") != "A100-80GB"
        or manifest.get("dtype") != "bfloat16"
        or manifest.get("quantization") != "none"
        or manifest.get("trust_remote_code") is not False
    ):
        raise ValueError("candidate runtime shape drifted")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("model source files are required")
    paths: list[str] = []
    for entry in files:
        if not isinstance(entry, Mapping) or set(entry) != _MODEL_FILE_FIELDS:
            raise ValueError("model source file entry is malformed")
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("model source path must be non-empty")
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("model source path escapes its repository")
        paths.append(path)
        size = entry.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("model source file size is invalid")
        _require_digest(entry.get("git_oid"), field=f"{path}.git_oid", length=40)
        storage = entry.get("storage")
        content_digest = entry.get("content_sha256")
        if storage == "git":
            if content_digest is not None:
                raise ValueError("small Git files receive content hashes after cache materialization")
        elif storage == "git_lfs":
            _require_digest(content_digest, field=f"{path}.content_sha256")
        else:
            raise ValueError("model source storage must be git or git_lfs")
    if len(paths) != len(set(paths)):
        raise ValueError("model source paths must be unique")
    if "consolidated.safetensors" not in paths:
        raise ValueError("original consolidated model weight is required")
    if any(path.startswith("model-") for path in paths) or "model.safetensors.index.json" in paths:
        raise ValueError("duplicate Transformers shards must not be downloaded")
    excluded = manifest.get("excluded_repository_paths")
    if not isinstance(excluded, list) or any(path in paths for path in excluded):
        raise ValueError("model source exclusions are invalid")
    _require_no_authority(manifest.get("authority"))


def _load_source_plan(root: Path, protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    path = root / DEFAULT_SOURCE_CALL_PLAN_PATH
    if _file_sha256(path) != protocol["source_confirmation_call_plan_file_sha256"]:
        raise ValueError("source confirmation call-plan file hash drifted")
    source = verify_envelope(path, require_blinded=True)
    if payload_hash(source) != protocol["source_confirmation_call_plan_payload_sha256"]:
        raise ValueError("source confirmation call-plan payload hash drifted")
    return source


def _primary_source_calls(source: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    calls = source.get("calls")
    if not isinstance(calls, list):
        raise ValueError("source confirmation plan calls are unavailable")
    selected: list[Mapping[str, Any]] = []
    for call in calls:
        if not isinstance(call, Mapping) or set(call) != _SOURCE_CALL_FIELDS:
            raise ValueError("source confirmation call schema drifted")
        experiment_id = call.get("experiment_id")
        if experiment_id not in EXPERIMENT_IDS:
            continue
        expected_model = (
            "qwen3_vl_8b_primary" if experiment_id == "pb2rr" else "qwen3_8b_generic"
        )
        if call.get("stage") in INCLUDED_SOURCE_STAGES and call.get("model_id") == expected_model:
            selected.append(call)
    if len(selected) != 624:
        raise ValueError("primary source grid must contain exactly 624 calls")
    if Counter(call["experiment_id"] for call in selected) != Counter(
        EXPECTED_CALL_COUNT_BY_EXPERIMENT
    ):
        raise ValueError("primary source experiment counts drifted")
    if Counter(call["stage"] for call in selected) != Counter(EXPECTED_STAGE_COUNTS):
        raise ValueError("primary source stage counts drifted")
    ids = [str(call["call_id"]) for call in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("primary source call IDs are not unique")
    return selected


def _candidate_call(
    source: Mapping[str, Any],
    *,
    index: int,
    system_instruction_sha256: str,
    chat_template_git_oid: str,
) -> dict[str, Any]:
    copied = {field: source[field] for field in _LOGICAL_COPY_FIELDS}
    request_spec = {
        "schema_version": "intervenebench.cross_family_request_spec.v1",
        "candidate_model_id": CANDIDATE_MODEL_ID,
        "source_call_id": source["call_id"],
        "source_model_id": source["model_id"],
        "source_prompt_sha256": source["prompt_sha256"],
        "bundle_payload_sha256": source["bundle_payload_sha256"],
        "system_instruction_sha256": system_instruction_sha256,
        "chat_template_git_oid": chat_template_git_oid,
        **copied,
    }
    candidate_call_id = f"cross-family--{index:04d}--{source['call_id']}"
    artifact = (
        f"calls/{source['stage']}/{CANDIDATE_MODEL_ID}/"
        f"{source['experiment_id']}/{index:04d}.json"
    )
    return {
        "schema_version": "intervenebench.cross_family_call.v1",
        "candidate_call_id": candidate_call_id,
        "candidate_model_id": CANDIDATE_MODEL_ID,
        "candidate_artifact_relative_path": artifact,
        "candidate_request_spec_sha256": payload_hash(request_spec),
        "source_call_id": source["call_id"],
        "source_model_id": source["model_id"],
        "source_prompt_sha256": source["prompt_sha256"],
        "source_artifact_relative_path": source["artifact_relative_path"],
        "system_instruction_sha256": system_instruction_sha256,
        "chat_template_git_oid": chat_template_git_oid,
        **copied,
    }


def build_cross_family_call_plan(root: Path) -> dict[str, Any]:
    protocol = verify_envelope(root / DEFAULT_PROTOCOL_PATH, require_blinded=True)
    model = verify_envelope(
        root / DEFAULT_MODEL_SOURCE_MANIFEST_PATH, require_blinded=True
    )
    validate_cross_family_protocol(protocol)
    validate_model_source_manifest(model)
    source = _load_source_plan(root, protocol)
    selected = _primary_source_calls(source)
    chat_template = next(
        entry for entry in model["files"] if entry["path"] == "chat_template.json"
    )
    system_hash = protocol["system_instruction"]["sha256"]
    calls = [
        _candidate_call(
            call,
            index=index,
            system_instruction_sha256=system_hash,
            chat_template_git_oid=chat_template["git_oid"],
        )
        for index, call in enumerate(selected, start=1)
    ]
    payload = {
        "schema_version": "intervenebench.cross_family_call_plan.v1",
        "status": "frozen_no_authority",
        "study_role": "retrospective_cross_family_robustness",
        "candidate_model_id": CANDIDATE_MODEL_ID,
        "experiment_ids": list(EXPERIMENT_IDS),
        "protocol_path": str(DEFAULT_PROTOCOL_PATH),
        "protocol_payload_sha256": payload_hash(protocol),
        "model_source_manifest_path": str(DEFAULT_MODEL_SOURCE_MANIFEST_PATH),
        "model_source_manifest_payload_sha256": payload_hash(model),
        "source_confirmation_call_plan_path": str(DEFAULT_SOURCE_CALL_PLAN_PATH),
        "source_confirmation_call_plan_file_sha256": _file_sha256(
            root / DEFAULT_SOURCE_CALL_PLAN_PATH
        ),
        "source_confirmation_call_plan_payload_sha256": payload_hash(source),
        "planned_call_count": len(calls),
        "call_count_by_stage": dict(EXPECTED_STAGE_COUNTS),
        "call_count_by_experiment": dict(EXPECTED_CALL_COUNT_BY_EXPERIMENT),
        "reserve_call_count": 0,
        "source_prompt_reconstruction_required": True,
        "source_prompt_hash_must_match_before_chat_formatting": True,
        "candidate_chat_format_must_match_pinned_template": True,
        "authority": dict(protocol["authority"]),
        "calls": calls,
    }
    assert_blinded_payload(payload)
    return payload


def freeze_cross_family_call_plan(
    root: Path,
    *,
    destination: Path | None = None,
) -> str:
    path = root / (destination or DEFAULT_CALL_PLAN_PATH)
    return freeze_envelope(
        build_cross_family_call_plan(root), path, require_blinded=True
    )


def require_cross_family_execution_authority(protocol: Mapping[str, Any]) -> None:
    """Fail closed because this artifact is deliberately not an authorization."""

    validate_cross_family_protocol(protocol)
    raise PermissionError(
        "a separate execution authorization is required after cache and adapter preflight"
    )


def verify_cross_family_freeze(root: Path) -> CrossFamilyFreezeSummary:
    protocol = verify_envelope(root / DEFAULT_PROTOCOL_PATH, require_blinded=True)
    model = verify_envelope(
        root / DEFAULT_MODEL_SOURCE_MANIFEST_PATH, require_blinded=True
    )
    plan = verify_envelope(root / DEFAULT_CALL_PLAN_PATH, require_blinded=True)
    validate_cross_family_protocol(protocol)
    validate_model_source_manifest(model)
    rebuilt = build_cross_family_call_plan(root)
    if plan != rebuilt:
        raise ValueError("stored cross-family call plan does not match deterministic rebuild")
    if plan["protocol_payload_sha256"] != payload_hash(protocol):
        raise ValueError("cross-family plan is not bound to its protocol")
    if plan["model_source_manifest_payload_sha256"] != payload_hash(model):
        raise ValueError("cross-family plan is not bound to its model source manifest")
    _require_no_authority(plan.get("authority"))
    budget = protocol["budget"]
    return CrossFamilyFreezeSummary(
        candidate_model_id=CANDIDATE_MODEL_ID,
        experiment_count=len(EXPERIMENT_IDS),
        planned_call_count=plan["planned_call_count"],
        base_call_count=plan["call_count_by_stage"]["base"],
        prompt_perturbation_call_count=plan["call_count_by_stage"][
            "primary_prompt_perturbation"
        ],
        maximum_gpu_seconds=budget["maximum_gpu_seconds"],
        maximum_gpu_cost_usd=budget["maximum_gpu_cost_usd"],
        hard_incremental_cost_cap_usd=budget["hard_incremental_cost_cap_usd"],
        retrospective_only=True,
    )
