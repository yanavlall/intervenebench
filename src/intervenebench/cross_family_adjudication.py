"""Replayable, target-free adjudication of the Mistral interface canaries."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .cross_family_modal import (
    DEFAULT_CROSS_FAMILY_MODAL_FREEZE_PATH,
    parse_strict_nonnegative_integer,
    validate_forced_choice_probe,
    verify_cross_family_modal_freeze,
)
from .protocol import assert_blinded_payload, payload_hash, verify_envelope


DEFAULT_CANARY_RESULT_PATH = Path(
    "artifacts/cross_family_preflight/canary_20260815_v1.json"
)
DEFAULT_CACHE_ATTESTATION_PATH = Path(
    "artifacts/cross_family_preflight/cache_attestation_20260815_v3.json"
)
DEFAULT_ADJUDICATION_PATH = Path(
    "artifacts/cross_family_preflight/canary_adjudication_20260815_v2.json"
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_cross_family_canary_adjudication(root: Path) -> dict[str, Any]:
    freeze_path = root / DEFAULT_CROSS_FAMILY_MODAL_FREEZE_PATH
    result_path = root / DEFAULT_CANARY_RESULT_PATH
    cache_path = root / DEFAULT_CACHE_ATTESTATION_PATH
    freeze = verify_cross_family_modal_freeze(root, freeze_path)
    result = verify_envelope(result_path, require_blinded=True)
    cache = verify_envelope(cache_path, require_blinded=True)

    if (
        result.get("schema_version")
        != "intervenebench.cross_family_canary_run.v1"
        or result.get("status") != "completed_requires_local_adjudication"
        or result.get("canary_manifest_payload_sha256")
        != freeze["canary"]["manifest_payload_sha256"]
    ):
        raise ValueError("cross-family canary result does not match the freeze")
    if any(
        result.get(key) is not expected
        for key, expected in {
            "target_calls_made": 0,
            "target_prompts_or_assets_accessed": False,
            "human_data_accessed": False,
            "automatic_next_stage": False,
        }.items()
    ):
        raise PermissionError("canary result exceeded its target-free authority")

    requests = freeze["canary"]["manifest"]["requests"]
    rows = result.get("canary_results")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("cross-family canary result count drifted")

    checks: list[dict[str, Any]] = []
    for row, request in zip(rows, requests, strict=True):
        if (
            row.get("canary_id") != request["canary_id"]
            or row.get("prompt_sha256") != request["prompt_sha256"]
        ):
            raise ValueError("cross-family canary identity drifted")
        if request["adapter"] == "forced_choice_next_token_softmax.v1":
            probe = validate_forced_choice_probe(
                row["result"], expected_codes=request["answer_codes"]
            )
            checks.append(
                {
                    "canary_id": request["canary_id"],
                    "interface": request["adapter"],
                    "modality": request["modality"],
                    "status": "passed_schema_and_execution",
                    "answer_code_count": len(probe["answer_codes"]),
                    "distinct_single_token_ids": len(set(probe["token_ids"])),
                    "probability_sum": sum(probe["probabilities"].values()),
                    "semantic_expected_label_prespecified": False,
                }
            )
        elif request["adapter"] == "continuous_constrained_integer_generation.v1":
            probe = row["result"]
            if (
                probe.get("schema_version")
                != "intervenebench.strict_nonnegative_integer_probe.v1"
                or probe.get("semantic_repair_used") is not False
                or parse_strict_nonnegative_integer(probe.get("raw_text"))
                != probe.get("parsed_value")
            ):
                raise ValueError("continuous canary failed strict parsing")
            checks.append(
                {
                    "canary_id": request["canary_id"],
                    "interface": request["adapter"],
                    "modality": request["modality"],
                    "status": "passed_schema_and_execution",
                    "semantic_repair_used": False,
                }
            )
        else:
            raise ValueError("unrecognized canary adapter")

    runtime = result.get("runtime_attestation")
    if not isinstance(runtime, dict):
        raise ValueError("canary runtime attestation is absent")
    expected_runtime = {
        "modal_image_id": cache["modal_image_id"],
        "checkpoint_commit": cache["checkpoint_commit"],
        "cache_attestation_sha256": payload_hash(cache),
        "dependency_lock_sha256": freeze["runtime"]["dependency_lock_sha256"],
        "image_recipe_sha256": freeze["runtime"]["image_recipe_sha256"],
    }
    if any(runtime.get(key) != value for key, value in expected_runtime.items()):
        raise ValueError("canary runtime/cache binding failed")
    if runtime.get("package_versions") != {
        "mistral_common": freeze["runtime"]["mistral_common_version"],
        "torch": freeze["runtime"]["torch_version"],
        "transformers": freeze["runtime"]["transformers_version"],
        "vllm": freeze["runtime"]["vllm_version"],
    }:
        raise ValueError("canary dependency versions drifted")

    value = {
        "schema_version": "intervenebench.cross_family_canary_adjudication.v1",
        "status": "passed_three_of_three_canaries_target_json_gap_stop",
        "adjudication_date": "2026-08-15",
        "canary_result_payload_sha256": payload_hash(result),
        "canary_result_file_sha256": _file_sha256(result_path),
        "cache_attestation_payload_sha256": payload_hash(cache),
        "modal_preflight_freeze_payload_sha256": payload_hash(freeze),
        "interface_checks": checks,
        "runtime_binding": expected_runtime,
        "claim_boundary": {
            "established": (
                "The pinned Mistral runtime can execute the frozen masked-token "
                "text/vision and strict-integer interfaces without schema repair."
            ),
            "not_established": [
                "the exact target continuous JSON response schema",
                "semantic accuracy on the synthetic 1x1 image",
                "behavioral or human-response fidelity",
                "target-experiment decision reliability",
                "prospective replication",
            ],
            "vision_note": (
                "The frozen vision canary prespecified transport/decode execution "
                "and schema validity, but no expected semantic answer label."
            ),
            "continuous_note": (
                "The frozen continuous canary required bare digits, while the "
                "target tcg8p prompt requires exactly one JSON object with the "
                "predicted_value key. A target-free JSON canary is required before "
                "target inference."
            ),
        },
        "target_package_ready_for_separate_freeze": True,
        "target_inference_ready": False,
        "required_followup_canary": {
            "adapter": "continuous_json_integer_generation.v1",
            "planned_call_count": 1,
            "target_free": True,
            "semantic_repair_allowed": False,
            "must_pass_before_target_execution_authorization": True,
        },
        "planned_target_call_count": freeze["target_execution"][
            "planned_call_count"
        ],
        "target_execution_authorized": False,
        "human_outcome_access_authorized": False,
        "participant_row_access_authorized": False,
        "automatic_next_stage_authorized": False,
    }
    assert_blinded_payload(value)
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
