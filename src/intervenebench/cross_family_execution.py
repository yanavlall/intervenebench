"""Hash-bound, zero-authority execution package for the Mistral replay.

This module reconstructs the 624 outcome-blind target requests and freezes the
runtime needed to execute them.  Building or verifying the package never makes
a model call and never reads human outcomes or participant rows.  Image
materialization has a separate narrow authorization; the one-call JSON canary
and target inference require later, distinct authorizations.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from .confirmation_calls import (
    DEFAULT_CONFIRMATION_CALL_PLAN_PATH,
    prepare_confirmation_requests,
    verify_confirmation_call_plan,
)
from .cross_family_adjudication import (
    DEFAULT_ADJUDICATION_PATH,
    DEFAULT_CACHE_ATTESTATION_PATH,
    DEFAULT_CANARY_RESULT_PATH,
)
from .cross_family_modal import (
    DEFAULT_CROSS_FAMILY_MODAL_FREEZE_PATH,
    verify_cross_family_modal_freeze,
)
from .cross_family_regression import (
    CANDIDATE_MODEL_ID,
    DEFAULT_CALL_PLAN_PATH,
    DEFAULT_MODEL_SOURCE_MANIFEST_PATH,
    DEFAULT_PROTOCOL_PATH,
    SYSTEM_INSTRUCTION,
    verify_cross_family_freeze,
)
from .protocol import (
    assert_blinded_payload,
    payload_hash,
    verify_envelope,
)


DEFAULT_EXECUTION_FREEZE_PATH = Path(
    "configs/simulators/cross_family_execution_v1.json"
)
DEFAULT_CACHE_INVENTORY_PATH = Path(
    "artifacts/cross_family_preflight/cache_inventory_audit_20260815_v3.json"
)
DEFAULT_MATERIALIZATION_AUTHORIZATION_PATH = Path(
    "artifacts/cross_family_target/authorizations/materialization_20260815_v2.json"
)
DEFAULT_MATERIALIZATION_PATH = Path(
    "artifacts/cross_family_target/materialization_20260815_v2.json"
)
DEFAULT_MATERIALIZATION_PROGRESS_PATH = Path(
    "artifacts/cross_family_target/progress/materialization_20260815_v2.jsonl"
)

APP_PATH = Path("infra/modal/cross_family_target_app.py")
WRAPPER_PATH = Path("scripts/run_cross_family_target.py")
BUILDER_PATH = Path("scripts/build_cross_family_execution_freeze.py")
VALIDATOR_PATH = Path("src/intervenebench/cross_family_execution.py")
LOCK_PATH = Path("infra/modal/cross-family-requirements.lock")
ASSET_ROOT = Path("data/derived/stimuli/pb2rr")
MODEL_VOLUME_NAME = "intervenebench-model-cache-v1"
MODEL_CACHE_SUBDIRECTORY = "mistral_small_3_1_24b_cross_family"
EXPECTED_PRELIGHT_IMAGE_ID = "im-rOjQLgFUQXyeAfYVdVVQ8a"
EXPECTED_CALL_COUNT = 624
EXPECTED_METHOD_COUNTS = {
    "continuous_constrained_integer_generation.v1": 120,
    "forced_choice_next_token_softmax.v1": 504,
}
EXPECTED_MODALITY_COUNTS = {"exact_png_vision": 128, "text": 496}

JSON_CANARY_PROMPT = (
    "This is a synthetic response-schema check, not a study. Return only one "
    "JSON object with exactly this key: {\"predicted_value\": INTEGER}. Use a "
    "non-negative whole number and include no other keys, prose, or formatting."
)

_AUTHORITY_FIELDS = frozenset(
    {
        "modal_image_materialization_authorized",
        "modal_compute_authorized",
        "model_download_authorized",
        "paid_inference_authorized",
        "json_canary_authorized",
        "target_inference_authorized",
        "target_call_authorized",
        "automatic_retry_authorized",
        "reserve_call_authorized",
        "human_outcome_access_authorized",
        "participant_row_access_authorized",
        "participant_row_serialization_authorized",
        "regression_scoring_authorized",
        "automatic_next_stage_authorized",
    }
)

_COPIED_FIELDS = (
    "adapter",
    "answer_codes",
    "answer_order",
    "arm_id",
    "asset_path",
    "asset_sha256",
    "bundle_payload_sha256",
    "display_option_values",
    "experiment_id",
    "generation_seed",
    "max_new_tokens",
    "method_id",
    "modality",
    "nuisance_id",
    "prompt_variant",
    "sequence_episode_id",
    "sequence_seed",
    "source_option_values",
    "stage",
    "temperature",
    "top_p",
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _zero_authority() -> dict[str, bool]:
    return {field: False for field in sorted(_AUTHORITY_FIELDS)}


def _require_zero_authority(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != _AUTHORITY_FIELDS:
        raise PermissionError("cross-family target authority fields drifted")
    if any(item is not False for item in value.values()):
        raise PermissionError("cross-family target freeze must grant zero authority")


def _require_digest(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.casefold()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def prepare_cross_family_target_requests(root: Path) -> list[dict[str, Any]]:
    """Reconstruct all 624 prompts without reading human outcomes."""

    verify_cross_family_freeze(root)
    candidate_plan = verify_envelope(
        root / DEFAULT_CALL_PLAN_PATH, require_blinded=True
    )
    source_plan = verify_confirmation_call_plan(
        root, root / DEFAULT_CONFIRMATION_CALL_PLAN_PATH
    )
    source_requests = prepare_confirmation_requests(
        root, plan=source_plan, include_reserve=False
    )
    by_source_id = {request["call_id"]: request for request in source_requests}
    requests: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidate_plan["calls"]:
        source_id = str(candidate["source_call_id"])
        if source_id not in by_source_id:
            raise ValueError(f"missing source request: {source_id}")
        source = by_source_id[source_id]
        for field in _COPIED_FIELDS:
            if candidate[field] != source[field]:
                raise ValueError(
                    f"cross-family/source logical field mismatch: {source_id}.{field}"
                )
        if candidate["source_prompt_sha256"] != source["prompt_sha256"]:
            raise ValueError("cross-family source prompt hash drifted")
        prompt = source["prompt"]
        if sha256(prompt.encode("utf-8")).hexdigest() != candidate[
            "source_prompt_sha256"
        ]:
            raise ValueError("reconstructed cross-family prompt hash drifted")
        call_id = str(candidate["candidate_call_id"])
        if call_id in seen:
            raise ValueError("cross-family target call IDs must be unique")
        seen.add(call_id)
        request = {
            "schema_version": "intervenebench.cross_family_target_request.v1",
            "call_id": call_id,
            "model_id": CANDIDATE_MODEL_ID,
            "candidate_request_spec_sha256": candidate[
                "candidate_request_spec_sha256"
            ],
            "source_call_id": source_id,
            "source_prompt_sha256": candidate["source_prompt_sha256"],
            "prompt": prompt,
            "artifact_relative_path": candidate[
                "candidate_artifact_relative_path"
            ],
            **{field: source[field] for field in _COPIED_FIELDS},
        }
        asset_path = request["asset_path"]
        if asset_path is None:
            if request["asset_sha256"] is not None:
                raise ValueError("text request unexpectedly has an asset digest")
        else:
            asset = (root / str(asset_path)).resolve()
            allowed = (root / ASSET_ROOT).resolve()
            if asset != allowed and allowed not in asset.parents:
                raise ValueError("cross-family asset escapes the public asset root")
            if not asset.is_file() or asset.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
                raise ValueError("cross-family vision asset is not a PNG")
            if _file_sha256(asset) != request["asset_sha256"]:
                raise ValueError("cross-family vision asset hash drifted")
        assert_blinded_payload(request)
        requests.append(request)

    if len(requests) != EXPECTED_CALL_COUNT:
        raise ValueError("cross-family target request count drifted")
    if Counter(row["method_id"] for row in requests) != Counter(
        EXPECTED_METHOD_COUNTS
    ):
        raise ValueError("cross-family target method counts drifted")
    if Counter(row["modality"] for row in requests) != Counter(
        EXPECTED_MODALITY_COUNTS
    ):
        raise ValueError("cross-family target modality counts drifted")
    return requests


def _verify_preflight_artifacts(root: Path) -> dict[str, Mapping[str, Any]]:
    preflight = verify_cross_family_modal_freeze(
        root, root / DEFAULT_CROSS_FAMILY_MODAL_FREEZE_PATH
    )
    cache = verify_envelope(
        root / DEFAULT_CACHE_ATTESTATION_PATH, require_blinded=True
    )
    inventory = verify_envelope(
        root / DEFAULT_CACHE_INVENTORY_PATH, require_blinded=True
    )
    canary = verify_envelope(root / DEFAULT_CANARY_RESULT_PATH, require_blinded=True)
    adjudication = verify_envelope(
        root / DEFAULT_ADJUDICATION_PATH, require_blinded=True
    )
    if cache.get("modal_image_id") != EXPECTED_PRELIGHT_IMAGE_ID:
        raise ValueError("cross-family cache image binding drifted")
    if cache.get("checkpoint_commit") != preflight["model"]["checkpoint_commit"]:
        raise ValueError("cross-family cache checkpoint binding drifted")
    if inventory.get("cache_attestation_payload_sha256") != payload_hash(cache):
        raise ValueError("cross-family inventory/cache binding drifted")
    if (
        inventory.get("status")
        != "passed_expected_payloads_plus_known_downloader_metadata_only"
        or inventory.get("unexpected_model_payload_file_count") != 0
        or inventory.get("staging_entry_count") != 0
        or inventory.get("modal_app_state") != "stopped"
    ):
        raise ValueError("cross-family cache inventory did not pass")
    if adjudication.get("canary_result_payload_sha256") != payload_hash(canary):
        raise ValueError("cross-family canary adjudication binding drifted")
    if (
        adjudication.get("status")
        != "passed_three_of_three_canaries_target_json_gap_stop"
        or adjudication.get("target_package_ready_for_separate_freeze") is not True
        or adjudication.get("target_inference_ready") is not False
    ):
        raise ValueError("cross-family canary adjudication status drifted")
    return {
        "preflight": preflight,
        "cache": cache,
        "inventory": inventory,
        "canary": canary,
        "adjudication": adjudication,
    }


def build_cross_family_execution_freeze(root: Path) -> dict[str, Any]:
    summary = verify_cross_family_freeze(root)
    protocol = verify_envelope(root / DEFAULT_PROTOCOL_PATH, require_blinded=True)
    plan = verify_envelope(root / DEFAULT_CALL_PLAN_PATH, require_blinded=True)
    model = verify_envelope(
        root / DEFAULT_MODEL_SOURCE_MANIFEST_PATH, require_blinded=True
    )
    artifacts = _verify_preflight_artifacts(root)
    requests = prepare_cross_family_target_requests(root)
    request_hashes = {
        row["call_id"]: payload_hash(row)
        for row in requests
    }
    call_ids = [row["call_id"] for row in requests]
    asset_paths = sorted(
        {str(row["asset_path"]) for row in requests if row["asset_path"] is not None}
    )
    assets = [
        {
            "path": path,
            "sha256": _file_sha256(root / path),
            "size_bytes": (root / path).stat().st_size,
            "mime_type": "image/png",
        }
        for path in asset_paths
    ]
    image_recipe = {
        "base": "modal.Image.debian_slim",
        "python": "3.11",
        "uv_version": "0.12.4",
        "requirements_require_hashes": True,
        "apt_packages": ["libgl1", "libglib2.0-0"],
        "embedded_files": [
            str(DEFAULT_EXECUTION_FREEZE_PATH),
            str(DEFAULT_MODEL_SOURCE_MANIFEST_PATH),
            str(LOCK_PATH),
            str(APP_PATH),
            *asset_paths,
        ],
        "checkpoint_volume": "read_only",
        "network_during_inference": "blocked",
        "checkpoint_download_function_present": False,
    }
    value = {
        "schema_version": "intervenebench.cross_family_execution_freeze.v1",
        "status": "frozen_nonexecuting_zero_authority",
        "freeze_date": "2026-08-15",
        "study_role": "retrospective_cross_family_robustness",
        "protocol_payload_sha256": payload_hash(protocol),
        "call_plan": {
            "path": str(DEFAULT_CALL_PLAN_PATH),
            "file_sha256": _file_sha256(root / DEFAULT_CALL_PLAN_PATH),
            "payload_sha256": payload_hash(plan),
            "planned_call_count": EXPECTED_CALL_COUNT,
            "call_ids_sha256": payload_hash(call_ids),
            "request_payload_sha256_by_call_id": request_hashes,
            "request_payload_hashes_sha256": payload_hash(request_hashes),
            "method_counts": dict(EXPECTED_METHOD_COUNTS),
            "modality_counts": dict(EXPECTED_MODALITY_COUNTS),
        },
        "model": {
            "model_id": model["model_id"],
            "hf_repository": model["hf_repository"],
            "checkpoint_commit": model["checkpoint_commit"],
            "source_manifest_payload_sha256": payload_hash(model),
            "source_manifest_file_sha256": _file_sha256(
                root / DEFAULT_MODEL_SOURCE_MANIFEST_PATH
            ),
            "cache_attestation_payload_sha256": payload_hash(artifacts["cache"]),
            "cache_inventory_audit_payload_sha256": payload_hash(
                artifacts["inventory"]
            ),
            "cache_source_image_id": artifacts["cache"]["modal_image_id"],
        },
        "preflight": {
            "modal_preflight_freeze_payload_sha256": payload_hash(
                artifacts["preflight"]
            ),
            "canary_result_payload_sha256": payload_hash(artifacts["canary"]),
            "canary_adjudication_payload_sha256": payload_hash(
                artifacts["adjudication"]
            ),
            "three_interface_canaries_passed": True,
            "target_inference_ready": False,
            "blocking_reason": "exact_target_continuous_json_schema_not_canaried",
        },
        "required_json_canary": {
            "schema_version": "intervenebench.cross_family_json_canary_spec.v1",
            "canary_id": "synthetic_continuous_json_integer_v1",
            "prompt": JSON_CANARY_PROMPT,
            "prompt_sha256": sha256(JSON_CANARY_PROMPT.encode("utf-8")).hexdigest(),
            "adapter": "continuous_json_integer_generation.v1",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 32,
            "seed": 174904,
            "planned_call_count": 1,
            "target_free": True,
            "semantic_repair_allowed": False,
            "authorized": False,
        },
        "system_instruction": {
            "text": SYSTEM_INSTRUCTION,
            "sha256": payload_hash(SYSTEM_INSTRUCTION),
        },
        "public_assets": assets,
        "runtime": {
            "app_name": "intervenebench-cross-family-target-v1",
            "python_version": "3.11",
            "vllm_version": "0.8.5",
            "mistral_common_version": "1.5.4",
            "torch_version": "2.6.0",
            "transformers_version": "4.53.3",
            "gpu": "A100-80GB:1",
            "tokenizer_mode": "mistral",
            "config_format": "mistral",
            "load_format": "mistral",
            "dtype": "bfloat16",
            "maximum_model_length": 16384,
            "model_volume_name": MODEL_VOLUME_NAME,
            "model_cache_subdirectory": MODEL_CACHE_SUBDIRECTORY,
            "dependency_lock_sha256": _file_sha256(root / LOCK_PATH),
            "image_recipe": image_recipe,
            "image_recipe_sha256": payload_hash(image_recipe),
        },
        "limits": {
            "planned_call_count": summary.planned_call_count,
            "maximum_attempt_count": summary.planned_call_count,
            "maximum_gpu_seconds": summary.maximum_gpu_seconds,
            "official_gpu_price_usd_per_second": 0.000694,
            "maximum_gpu_cost_usd": summary.maximum_gpu_cost_usd,
            "hard_incremental_cost_cap_usd": summary.hard_incremental_cost_cap_usd,
            "maximum_wall_clock_seconds": 10_800,
            "automatic_retries": 0,
            "reserve_calls": 0,
            "semantic_repair_allowed": False,
            "completed_call_rerun_forbidden": True,
        },
        "stage_gates": {
            "image_materialization_requires_separate_authorization": True,
            "json_canary_requires_separate_authorization": True,
            "target_inference_requires_passed_json_canary": True,
            "target_inference_requires_separate_authorization": True,
            "human_outcome_scoring_requires_separate_authorization": True,
            "automatic_transition_allowed": False,
        },
        "implementation_hashes": {
            str(APP_PATH): _file_sha256(root / APP_PATH),
            str(WRAPPER_PATH): _file_sha256(root / WRAPPER_PATH),
            str(BUILDER_PATH): _file_sha256(root / BUILDER_PATH),
            str(VALIDATOR_PATH): _file_sha256(root / VALIDATOR_PATH),
            str(LOCK_PATH): _file_sha256(root / LOCK_PATH),
        },
        "authority": _zero_authority(),
        "model_downloaded_by_this_package": False,
        "inference_calls_made_by_this_package": 0,
        "target_calls_made": 0,
        "human_outcomes_accessed": False,
        "participant_rows_read": 0,
        "participant_rows_serialized": 0,
        "automatic_next_stage": False,
    }
    assert_blinded_payload(value)
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def verify_cross_family_execution_freeze(root: Path, path: Path) -> dict[str, Any]:
    actual = json.loads(path.read_text(encoding="utf-8"))
    expected = build_cross_family_execution_freeze(root)
    if actual != expected:
        raise ValueError("cross-family execution freeze does not replay")
    _require_zero_authority(actual.get("authority"))
    return actual


def build_materialization_authorization(
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": "intervenebench.cross_family_target_materialization_authorization.v1",
        "status": "authorized_target_image_build_zero_inference",
        "freeze_payload_sha256": payload_hash(freeze),
        "call_plan_payload_sha256": freeze["call_plan"]["payload_sha256"],
        "planned_call_count": EXPECTED_CALL_COUNT,
        **_zero_authority(),
    }
    value["modal_image_materialization_authorized"] = True
    assert_blinded_payload(value)
    return value


def validate_materialization_authorization(
    authorization: Mapping[str, Any], *, freeze: Mapping[str, Any]
) -> None:
    assert_blinded_payload(authorization)
    expected_fields = {
        "schema_version",
        "status",
        "freeze_payload_sha256",
        "call_plan_payload_sha256",
        "planned_call_count",
        *_AUTHORITY_FIELDS,
    }
    if set(authorization) != expected_fields:
        raise PermissionError("cross-family materialization authority fields drifted")
    expected_authority = _zero_authority()
    expected_authority["modal_image_materialization_authorized"] = True
    if any(
        authorization.get(field) is not value
        for field, value in expected_authority.items()
    ):
        raise PermissionError("cross-family materialization authority widened")
    if (
        authorization.get("schema_version")
        != "intervenebench.cross_family_target_materialization_authorization.v1"
        or authorization.get("status")
        != "authorized_target_image_build_zero_inference"
        or authorization.get("freeze_payload_sha256") != payload_hash(freeze)
        or authorization.get("call_plan_payload_sha256")
        != freeze["call_plan"]["payload_sha256"]
        or authorization.get("planned_call_count") != EXPECTED_CALL_COUNT
    ):
        raise PermissionError("cross-family materialization authority binding drifted")


def validate_json_canary_result(
    result: Mapping[str, Any], *, freeze: Mapping[str, Any], modal_image_id: str
) -> None:
    """Validate a future separately-authorized one-call JSON canary result."""

    expected_fields = {
        "schema_version",
        "status",
        "canary_id",
        "prompt_sha256",
        "raw_text",
        "parsed_value",
        "semantic_repair_used",
        "modal_image_id",
        "runtime_attestation",
        "target_calls_made",
        "human_outcomes_accessed",
        "participant_rows_read",
        "automatic_next_stage",
    }
    if set(result) != expected_fields:
        raise ValueError("cross-family JSON canary result fields drifted")
    spec = freeze["required_json_canary"]
    if (
        result.get("schema_version")
        != "intervenebench.cross_family_json_canary_result.v1"
        or result.get("status") != "passed_target_free_json_schema"
        or result.get("canary_id") != spec["canary_id"]
        or result.get("prompt_sha256") != spec["prompt_sha256"]
        or result.get("modal_image_id") != modal_image_id
        or not isinstance(result.get("runtime_attestation"), Mapping)
        or result.get("semantic_repair_used") is not False
        or result.get("target_calls_made") != 0
        or result.get("human_outcomes_accessed") is not False
        or result.get("participant_rows_read") != 0
        or result.get("automatic_next_stage") is not False
    ):
        raise ValueError("cross-family JSON canary did not pass its frozen contract")
    raw = result.get("raw_text")
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("cross-family JSON canary output is not exact JSON") from error
    if (
        not isinstance(parsed, dict)
        or set(parsed) != {"predicted_value"}
        or not isinstance(parsed["predicted_value"], int)
        or isinstance(parsed["predicted_value"], bool)
        or parsed["predicted_value"] < 0
        or result.get("parsed_value") != parsed["predicted_value"]
    ):
        raise ValueError("cross-family JSON canary output violates the integer schema")
