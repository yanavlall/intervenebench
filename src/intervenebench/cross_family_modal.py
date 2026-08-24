"""Fail-closed Modal preflight for the retrospective Mistral replay.

This module freezes only three preparatory stages: image materialization, an
exact public-checkpoint cache, and three synthetic canaries.  It deliberately
contains no target execution entrypoint or authority.  The 624 target calls
remain bound by hash only until the canary result is separately adjudicated.
"""

from __future__ import annotations

import base64
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import re
from typing import Any, Mapping

from .cross_family_regression import (
    CANDIDATE_MODEL_ID,
    DEFAULT_CALL_PLAN_PATH,
    DEFAULT_MODEL_SOURCE_MANIFEST_PATH,
    DEFAULT_PROTOCOL_PATH,
    verify_cross_family_freeze,
)
from .protocol import assert_blinded_payload, payload_hash, verify_envelope


DEFAULT_CROSS_FAMILY_MODAL_FREEZE_PATH = Path(
    "configs/simulators/cross_family_modal_preflight_v1.json"
)
APP_PATH = Path("infra/modal/cross_family_app.py")
WRAPPER_PATH = Path("scripts/run_cross_family_preflight.py")
BUILDER_PATH = Path("scripts/build_cross_family_modal_freeze.py")
VALIDATOR_PATH = Path("src/intervenebench/cross_family_modal.py")
REQUIREMENTS_INPUT_PATH = Path("infra/modal/cross-family-requirements.in")
DEPENDENCY_LOCK_PATH = Path("infra/modal/cross-family-requirements.lock")
MODEL_VOLUME_NAME = "intervenebench-model-cache-v1"
MODEL_CACHE_SUBDIRECTORY = "mistral_small_3_1_24b_cross_family"

_TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8A"
    "AQUBAScY42YAAAAASUVORK5CYII="
)
_TINY_PNG_SHA256 = "431ced6916a2a21a156e38701afe55bbd7f88969fbbfc56d7fe099d47f265460"
_NONNEGATIVE_INTEGER = re.compile(r"(?:0|[1-9][0-9]*)\Z")

_AUTHORITY_FIELDS = frozenset(
    {
        "modal_image_materialization_authorized",
        "model_download_authorized",
        "modal_compute_authorized",
        "paid_inference_authorized",
        "candidate_inference_authorized",
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


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_digest(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.casefold()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _zero_authority() -> dict[str, bool]:
    return {key: False for key in sorted(_AUTHORITY_FIELDS)}


def _synthetic_canary_manifest() -> dict[str, Any]:
    requests = [
        {
            "canary_id": "synthetic_text_forced_choice_v1",
            "adapter": "forced_choice_next_token_softmax.v1",
            "modality": "text",
            "prompt": (
                "This is a synthetic parser check, not a study. Choose exactly one "
                "response code. A = option one; B = option two; C = option three. "
                "Return one code only."
            ),
            "answer_codes": ["A", "B", "C"],
            "temperature": 1.0,
            "maximum_engine_probe_tokens": 1,
            "seed": 174901,
        },
        {
            "canary_id": "synthetic_png_forced_choice_v1",
            "adapter": "forced_choice_next_token_softmax.v1",
            "modality": "exact_png_vision",
            "prompt": (
                "This is a synthetic image-decoding check, not a study. Inspect the "
                "attached tiny PNG and choose exactly one response code. A = visible; "
                "B = not visible. Return one code only."
            ),
            "answer_codes": ["A", "B"],
            "asset": {
                "mime_type": "image/png",
                "base64": _TINY_PNG_BASE64,
                "sha256": _TINY_PNG_SHA256,
            },
            "temperature": 1.0,
            "maximum_engine_probe_tokens": 1,
            "seed": 174902,
        },
        {
            "canary_id": "synthetic_continuous_integer_v1",
            "adapter": "continuous_constrained_integer_generation.v1",
            "modality": "text",
            "prompt": (
                "This is a synthetic strict-parser check, not a study. Predict a "
                "non-negative whole number. Return digits only, with no punctuation, "
                "spaces, or explanation."
            ),
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 32,
            "seed": 174903,
        },
    ]
    for request in requests:
        request["prompt_sha256"] = sha256(
            request["prompt"].encode("utf-8")
        ).hexdigest()
    manifest = {
        "schema_version": "intervenebench.cross_family_synthetic_canary_manifest.v1",
        "status": "target_free",
        "planned_call_count": 3,
        "requests": requests,
        "target_prompts_included": False,
        "target_assets_included": False,
        "human_data_included": False,
    }
    assert_blinded_payload(manifest)
    return manifest


def _validate_canary_manifest(manifest: Mapping[str, Any]) -> None:
    assert_blinded_payload(manifest)
    if (
        manifest.get("schema_version")
        != "intervenebench.cross_family_synthetic_canary_manifest.v1"
        or manifest.get("status") != "target_free"
        or manifest.get("planned_call_count") != 3
    ):
        raise ValueError("synthetic canary manifest drifted")
    if any(
        manifest.get(key) is not False
        for key in ("target_prompts_included", "target_assets_included", "human_data_included")
    ):
        raise ValueError("synthetic canary manifest contains target or human data")
    requests = manifest.get("requests")
    if not isinstance(requests, list) or len(requests) != 3:
        raise ValueError("synthetic canary request count drifted")
    if {request.get("modality") for request in requests} != {
        "text",
        "exact_png_vision",
    }:
        raise ValueError("synthetic canary modality coverage drifted")
    for request in requests:
        prompt = request.get("prompt")
        if (
            not isinstance(prompt, str)
            or sha256(prompt.encode("utf-8")).hexdigest()
            != request.get("prompt_sha256")
        ):
            raise ValueError("synthetic canary prompt hash drifted")
    vision = next(row for row in requests if row["modality"] == "exact_png_vision")
    data = base64.b64decode(vision["asset"]["base64"], validate=True)
    if data[:8] != b"\x89PNG\r\n\x1a\n" or sha256(data).hexdigest() != _TINY_PNG_SHA256:
        raise ValueError("synthetic canary PNG drifted")


def build_cross_family_modal_freeze(root: Path) -> dict[str, Any]:
    verify_cross_family_freeze(root)
    protocol = verify_envelope(root / DEFAULT_PROTOCOL_PATH, require_blinded=True)
    model = verify_envelope(
        root / DEFAULT_MODEL_SOURCE_MANIFEST_PATH, require_blinded=True
    )
    plan = verify_envelope(root / DEFAULT_CALL_PLAN_PATH, require_blinded=True)
    canary = _synthetic_canary_manifest()
    _validate_canary_manifest(canary)
    if plan["planned_call_count"] != 624:
        raise ValueError("cross-family target call count drifted")
    if model["model_id"] != CANDIDATE_MODEL_ID:
        raise ValueError("cross-family model binding drifted")
    maximum_download_bytes = sum(int(row["size_bytes"]) for row in model["files"])
    image_recipe = {
        "base": "modal.Image.debian_slim",
        "python": "3.11",
        "uv_version": "0.12.4",
        "requirements_require_hashes": True,
        "dependency_lock_sha256": _file_sha256(root / DEPENDENCY_LOCK_PATH),
        "network_during_canary": "blocked",
        "checkpoint_volume_during_canary": "read_only",
    }
    value = {
        "schema_version": "intervenebench.cross_family_modal_preflight_freeze.v1",
        "status": "frozen_nonexecuting_zero_authority",
        "freeze_date": "2026-08-14",
        "study_role": "retrospective_cross_family_robustness_preflight",
        "protocol_payload_sha256": payload_hash(protocol),
        "system_instruction": dict(protocol["system_instruction"]),
        "model": {
            "model_id": model["model_id"],
            "hf_repository": model["hf_repository"],
            "checkpoint_commit": model["checkpoint_commit"],
            "source_manifest_payload_sha256": payload_hash(model),
            "source_manifest_file_sha256": _file_sha256(
                root / DEFAULT_MODEL_SOURCE_MANIFEST_PATH
            ),
            "load_format": model["load_format"],
            "dtype": model["dtype"],
            "quantization": model["quantization"],
        },
        "runtime": {
            "app_name": "intervenebench-cross-family-preflight-v1",
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
            "tensor_parallel_size": 1,
            "trust_remote_code": False,
            "maximum_model_length": 16384,
            "model_volume_name": MODEL_VOLUME_NAME,
            "model_cache_subdirectory": MODEL_CACHE_SUBDIRECTORY,
            "requirements_input_sha256": _file_sha256(
                root / REQUIREMENTS_INPUT_PATH
            ),
            "dependency_lock_sha256": _file_sha256(root / DEPENDENCY_LOCK_PATH),
            "image_recipe": image_recipe,
            "image_recipe_sha256": payload_hash(image_recipe),
        },
        "cache": {
            "network_access": "checkpoint_download_only",
            "public_ungated_checkpoint_only": True,
            "maximum_download_bytes": maximum_download_bytes,
            "hard_incremental_cost_cap_usd": 12.0,
            "automatic_retries": 0,
            "unexpected_repository_files_downloaded": False,
            "small_git_files_verified_by_git_blob_sha1": True,
            "all_cached_files_attested_with_content_sha256": True,
            "startup_smoke_required": True,
            "startup_smoke_client_deadline_seconds": 180,
            "local_heartbeat_seconds": 60,
            "cache_client_deadline_seconds": 7200,
        },
        "canary": {
            "manifest_payload_sha256": payload_hash(canary),
            "manifest": canary,
            "planned_call_count": 3,
            "maximum_attempt_count": 3,
            "hard_incremental_cost_cap_usd": 15.0,
            "target_assets_included": False,
            "target_prompts_included": False,
            "automatic_retries": 0,
            "interface_adjudication_required_after_pass": True,
            "target_execution_after_pass_is_automatic": False,
            "forced_choice_engine_probe": {
                "semantic_free_generation_allowed": False,
                "engine_probe_tokens": 1,
                "allowed_token_mask_required": True,
                "all_allowed_token_logprobs_required": True,
                "equivalence_claim_status": "pending_target_free_canary_adjudication",
            },
        },
        "target_execution": {
            "authorized": False,
            "planned_call_count": 624,
            "call_plan_payload_sha256": payload_hash(plan),
            "call_plan_file_sha256": _file_sha256(root / DEFAULT_CALL_PLAN_PATH),
            "target_prompt_or_asset_uploaded_by_this_package": False,
            "requires_separate_interface_adjudication": True,
            "requires_separate_execution_freeze": True,
            "requires_separate_user_authorization": True,
        },
        "implementation_hashes": {
            str(APP_PATH): _file_sha256(root / APP_PATH),
            str(WRAPPER_PATH): _file_sha256(root / WRAPPER_PATH),
            str(BUILDER_PATH): _file_sha256(root / BUILDER_PATH),
            str(VALIDATOR_PATH): _file_sha256(root / VALIDATOR_PATH),
            str(REQUIREMENTS_INPUT_PATH): _file_sha256(root / REQUIREMENTS_INPUT_PATH),
            str(DEPENDENCY_LOCK_PATH): _file_sha256(root / DEPENDENCY_LOCK_PATH),
        },
        "authority": _zero_authority(),
        "human_outcomes_accessed": False,
        "participant_rows_read": 0,
        "model_downloaded": False,
        "modal_resources_created": False,
        "inference_calls_made": 0,
    }
    assert_blinded_payload(value)
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def verify_cross_family_modal_freeze(root: Path, path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value != build_cross_family_modal_freeze(root):
        raise ValueError("cross-family Modal preflight freeze does not replay")
    if set(value.get("authority", {})) != _AUTHORITY_FIELDS or any(
        value["authority"].values()
    ):
        raise PermissionError("cross-family Modal preflight embeds authority")
    return value


def _require_authority_shape(
    authorization: Mapping[str, Any],
    *,
    required: Mapping[str, bool],
    binding_fields: set[str],
) -> None:
    present = {key for key in _AUTHORITY_FIELDS if key in authorization}
    if present != _AUTHORITY_FIELDS or set(required) != _AUTHORITY_FIELDS:
        raise PermissionError("cross-family authority fields drifted")
    exact_fields = {"schema_version", "status", *binding_fields, *_AUTHORITY_FIELDS}
    if set(authorization) != exact_fields:
        raise PermissionError("cross-family authorization fields drifted")
    if any(authorization.get(key) is not value for key, value in required.items()):
        raise PermissionError("cross-family authority values drifted")


def validate_materialization_authorization(
    authorization: Mapping[str, Any], *, freeze: Mapping[str, Any]
) -> None:
    assert_blinded_payload(authorization)
    if (
        authorization.get("schema_version")
        != "intervenebench.cross_family_materialization_authorization.v1"
        or authorization.get("status")
        != "authorized_image_build_zero_download_zero_inference"
    ):
        raise PermissionError("invalid cross-family materialization authority")
    required = {
        "modal_image_materialization_authorized": True,
        "model_download_authorized": False,
        "modal_compute_authorized": False,
        "paid_inference_authorized": False,
        "candidate_inference_authorized": False,
        "target_call_authorized": False,
        "automatic_retry_authorized": False,
        "reserve_call_authorized": False,
        "human_outcome_access_authorized": False,
        "participant_row_access_authorized": False,
        "participant_row_serialization_authorized": False,
        "regression_scoring_authorized": False,
        "automatic_next_stage_authorized": False,
    }
    _require_authority_shape(
        authorization, required=required, binding_fields={"freeze_payload_sha256"}
    )
    if authorization.get("freeze_payload_sha256") != payload_hash(freeze):
        raise PermissionError("materialization authority is bound to another freeze")


def validate_cache_authorization(
    authorization: Mapping[str, Any], *, freeze: Mapping[str, Any], modal_image_id: str
) -> None:
    assert_blinded_payload(authorization)
    if (
        authorization.get("schema_version")
        != "intervenebench.cross_family_cache_authorization.v1"
        or authorization.get("status")
        != "authorized_exact_public_checkpoint_cache_only"
    ):
        raise PermissionError("invalid cross-family cache authority")
    required = {
        "modal_image_materialization_authorized": False,
        "model_download_authorized": True,
        "modal_compute_authorized": True,
        "paid_inference_authorized": False,
        "candidate_inference_authorized": False,
        "target_call_authorized": False,
        "automatic_retry_authorized": False,
        "reserve_call_authorized": False,
        "human_outcome_access_authorized": False,
        "participant_row_access_authorized": False,
        "participant_row_serialization_authorized": False,
        "regression_scoring_authorized": False,
        "automatic_next_stage_authorized": False,
    }
    _require_authority_shape(
        authorization,
        required=required,
        binding_fields={
            "freeze_payload_sha256",
            "modal_image_id",
            "model_id",
            "checkpoint_commit",
            "model_source_manifest_payload_sha256",
            "maximum_download_bytes",
            "hard_incremental_cost_cap_usd",
        },
    )
    expected = {
        "freeze_payload_sha256": payload_hash(freeze),
        "modal_image_id": modal_image_id,
        "model_id": freeze["model"]["model_id"],
        "checkpoint_commit": freeze["model"]["checkpoint_commit"],
        "model_source_manifest_payload_sha256": freeze["model"][
            "source_manifest_payload_sha256"
        ],
        "maximum_download_bytes": freeze["cache"]["maximum_download_bytes"],
        "hard_incremental_cost_cap_usd": freeze["cache"][
            "hard_incremental_cost_cap_usd"
        ],
    }
    if any(authorization.get(key) != value for key, value in expected.items()):
        raise PermissionError("cache authority binding or ceiling drifted")


def validate_canary_authorization(
    authorization: Mapping[str, Any],
    *,
    freeze: Mapping[str, Any],
    modal_image_id: str,
    cache_attestation_sha256: str,
) -> None:
    assert_blinded_payload(authorization)
    if (
        authorization.get("schema_version")
        != "intervenebench.cross_family_canary_authorization.v1"
        or authorization.get("status")
        != "authorized_three_synthetic_canaries_only"
    ):
        raise PermissionError("invalid cross-family canary authority")
    required = {
        "modal_image_materialization_authorized": False,
        "model_download_authorized": False,
        "modal_compute_authorized": True,
        "paid_inference_authorized": True,
        "candidate_inference_authorized": True,
        "target_call_authorized": False,
        "automatic_retry_authorized": False,
        "reserve_call_authorized": False,
        "human_outcome_access_authorized": False,
        "participant_row_access_authorized": False,
        "participant_row_serialization_authorized": False,
        "regression_scoring_authorized": False,
        "automatic_next_stage_authorized": False,
    }
    _require_authority_shape(
        authorization,
        required=required,
        binding_fields={
            "freeze_payload_sha256",
            "modal_image_id",
            "cache_attestation_sha256",
            "canary_manifest_payload_sha256",
            "planned_canary_call_count",
            "maximum_attempt_count",
            "hard_incremental_cost_cap_usd",
        },
    )
    _require_digest(cache_attestation_sha256, field="cache_attestation_sha256")
    expected = {
        "freeze_payload_sha256": payload_hash(freeze),
        "modal_image_id": modal_image_id,
        "cache_attestation_sha256": cache_attestation_sha256,
        "canary_manifest_payload_sha256": freeze["canary"][
            "manifest_payload_sha256"
        ],
        "planned_canary_call_count": 3,
        "maximum_attempt_count": 3,
        "hard_incremental_cost_cap_usd": freeze["canary"][
            "hard_incremental_cost_cap_usd"
        ],
    }
    if any(authorization.get(key) != value for key, value in expected.items()):
        raise PermissionError("canary authority binding or ceiling drifted")


def parse_strict_nonnegative_integer(text: Any) -> int:
    if not isinstance(text, str) or _NONNEGATIVE_INTEGER.fullmatch(text) is None:
        raise ValueError("continuous canary must be one canonical non-negative integer")
    return int(text)


def validate_forced_choice_probe(
    value: Mapping[str, Any], *, expected_codes: list[str]
) -> dict[str, Any]:
    required = {
        "schema_version",
        "answer_codes",
        "token_ids",
        "probabilities",
        "sampled_code",
        "free_generation_used",
        "engine_probe_tokens",
    }
    if set(value) != required:
        raise ValueError("forced-choice probe fields drifted")
    if value.get("schema_version") != "intervenebench.masked_next_token_probe.v1":
        raise ValueError("forced-choice probe schema drifted")
    if value.get("answer_codes") != expected_codes:
        raise ValueError("forced-choice answer codes drifted")
    token_ids = value.get("token_ids")
    if (
        not isinstance(token_ids, list)
        or len(token_ids) != len(expected_codes)
        or any(not isinstance(token, int) or isinstance(token, bool) for token in token_ids)
        or len(set(token_ids)) != len(token_ids)
    ):
        raise ValueError("forced-choice token IDs are not exact and distinct")
    probabilities = value.get("probabilities")
    if not isinstance(probabilities, Mapping) or list(probabilities) != expected_codes:
        raise ValueError("forced-choice probability support drifted")
    if any(
        not isinstance(probability, (int, float))
        or isinstance(probability, bool)
        or probability < 0
        or probability > 1
        for probability in probabilities.values()
    ):
        raise ValueError("forced-choice probabilities are invalid")
    if not isclose(sum(float(x) for x in probabilities.values()), 1.0, abs_tol=1e-8):
        raise ValueError("forced-choice probabilities must sum to one")
    if value.get("sampled_code") not in expected_codes:
        raise ValueError("forced-choice sampled code is outside the allowlist")
    if value.get("free_generation_used") is not False:
        raise ValueError("forced-choice probe must not use free generation")
    if value.get("engine_probe_tokens") != 1:
        raise ValueError("forced-choice engine probe must use exactly one masked token")
    return json.loads(json.dumps(value, sort_keys=False, allow_nan=False))
