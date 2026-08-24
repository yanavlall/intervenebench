"""Hash-bound, zero-authority execution freeze for evidence-report generation."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from .evidence_report_eval import (
    validate_eval_protocol,
    verify_report_generation_plan,
)
from .protocol import payload_hash, verify_envelope


QWEN_CACHE_MANIFEST = Path(
    "artifacts/modal_discovery_preflight/cache_manifest_20260813_v3.json"
)
MISTRAL_CACHE_ATTESTATION = Path(
    "artifacts/cross_family_preflight/cache_attestation_20260815_v3.json"
)
TEXT_MODEL_MANIFEST = Path(
    "data/manifests/simulators/model_file_manifests_v1.json"
)
MISTRAL_MODEL_MANIFEST = Path(
    "data/manifests/simulators/mistral_small_3_1_24b_source_manifest_v1.json"
)
QWEN_LOCK = Path("infra/modal/multimodal-requirements.lock")
MISTRAL_LOCK = Path("infra/modal/cross-family-requirements.lock")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_plain_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report_execution_freeze(
    root: Path,
    packet: Mapping[str, Any],
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind report calls to existing verified public checkpoint caches."""

    validate_eval_protocol(protocol, packet)
    verify_report_generation_plan(plan, packet, protocol)
    if plan["authority"] != {
        "model_calls_authorized": False,
        "automatic_retries_authorized": False,
        "reserve_calls_authorized": False,
        "human_labels_access_authorized": False,
    }:
        raise ValueError("report plan is not frozen with zero authority")

    qwen_cache = verify_envelope(root / QWEN_CACHE_MANIFEST, require_blinded=True)
    mistral_cache = verify_envelope(
        root / MISTRAL_CACHE_ATTESTATION, require_blinded=True
    )
    text_manifest = _load_plain_json(root / TEXT_MODEL_MANIFEST)
    mistral_manifest = verify_envelope(
        root / MISTRAL_MODEL_MANIFEST, require_blinded=True
    )
    text_by_id = {model["model_id"]: model for model in text_manifest["models"]}
    qwen_cache_hashes = qwen_cache["cache_attestation_sha256_by_model"]

    role_specs = [
        {
            "model_role": "qwen3_8b_incumbent",
            "cache_model_id": "qwen3_8b_generic",
            "hf_repository": text_by_id["qwen3_8b_generic"]["repository"],
            "checkpoint_commit": text_by_id["qwen3_8b_generic"]["commit"],
            "cache_attestation_payload_sha256": qwen_cache_hashes[
                "qwen3_8b_generic"
            ],
            "source_manifest_sha256": payload_hash(
                text_by_id["qwen3_8b_generic"]
            ),
            "gpu": "L40S:1",
            "runtime_family": "qwen_transformers",
            "model_download_authorized": False,
        },
        {
            "model_role": "qwen3_14b_scale_candidate",
            "cache_model_id": "qwen3_14b_generic",
            "hf_repository": text_by_id["qwen3_14b_generic"]["repository"],
            "checkpoint_commit": text_by_id["qwen3_14b_generic"]["commit"],
            "cache_attestation_payload_sha256": qwen_cache_hashes[
                "qwen3_14b_generic"
            ],
            "source_manifest_sha256": payload_hash(
                text_by_id["qwen3_14b_generic"]
            ),
            "gpu": "L40S:1",
            "runtime_family": "qwen_transformers",
            "model_download_authorized": False,
        },
        {
            "model_role": "mistral_small_3_1_24b_cross_family",
            "cache_model_id": "mistral_small_3_1_24b_cross_family",
            "hf_repository": mistral_manifest["hf_repository"],
            "checkpoint_commit": mistral_manifest["checkpoint_commit"],
            "cache_attestation_payload_sha256": payload_hash(mistral_cache),
            "source_manifest_sha256": payload_hash(mistral_manifest),
            "gpu": "A100-80GB:1",
            "runtime_family": "mistral_vllm",
            "model_download_authorized": False,
        },
    ]
    role_specs.sort(key=lambda model: model["model_role"])
    calls_by_role = {
        role: sum(call["model_role"] == role for call in plan["calls"])
        for role in protocol["generation"]["model_roles"]
    }
    if set(calls_by_role) != {model["model_role"] for model in role_specs}:
        raise ValueError("report model roles do not match cached execution models")
    if set(calls_by_role.values()) != {16} or sum(calls_by_role.values()) != 48:
        raise ValueError("report execution call grid is not the frozen 48-call panel")

    qwen_recipe = {
        "base": "modal.Image.debian_slim",
        "python": "3.11",
        "dependency_lock_path": str(QWEN_LOCK),
        "dependency_lock_sha256": _file_sha256(root / QWEN_LOCK),
        "requirements_require_hashes": True,
        "uv_version": "0.12.4",
        "checkpoint_volume": "read_only",
        "network_during_inference": "blocked",
        "app_source_sha256": _file_sha256(
            root / "infra/modal/evidence_report_app.py"
        ),
        "embedded_files": [
            "configs/simulators/evidence_report_execution_v1.json",
            "data/manifests/qualitative_eval/report_generation_plan_v1.json",
            str(TEXT_MODEL_MANIFEST),
            str(MISTRAL_MODEL_MANIFEST),
            str(QWEN_LOCK),
            str(MISTRAL_LOCK),
            "infra/modal/evidence_report_app.py",
        ],
    }
    mistral_recipe = {
        "base": "modal.Image.debian_slim",
        "python": "3.11",
        "dependency_lock_path": str(MISTRAL_LOCK),
        "dependency_lock_sha256": _file_sha256(root / MISTRAL_LOCK),
        "requirements_require_hashes": True,
        "uv_version": "0.12.4",
        "checkpoint_volume": "read_only",
        "network_during_inference": "blocked",
        "app_source_sha256": _file_sha256(
            root / "infra/modal/evidence_report_app.py"
        ),
        "embedded_files": [
            "configs/simulators/evidence_report_execution_v1.json",
            "data/manifests/qualitative_eval/report_generation_plan_v1.json",
            str(TEXT_MODEL_MANIFEST),
            str(MISTRAL_MODEL_MANIFEST),
            str(QWEN_LOCK),
            str(MISTRAL_LOCK),
            "infra/modal/evidence_report_app.py",
        ],
    }
    return {
        "schema_version": "intervenebench.evidence_report_execution_freeze.v1",
        "status": "frozen_zero_authority",
        "evaluation_id": protocol["evaluation_id"],
        "evidence_packet_sha256": payload_hash(packet),
        "evaluation_protocol_sha256": payload_hash(protocol),
        "generation_plan_payload_sha256": payload_hash(plan),
        "planned_call_count": plan["call_count"],
        "planned_calls_by_model_role": dict(sorted(calls_by_role.items())),
        "models": role_specs,
        "source_bindings": {
            "qwen_cache_manifest_payload_sha256": payload_hash(qwen_cache),
            "qwen_model_manifest_file_sha256": _file_sha256(
                root / TEXT_MODEL_MANIFEST
            ),
            "mistral_cache_attestation_payload_sha256": payload_hash(
                mistral_cache
            ),
            "mistral_model_manifest_payload_sha256": payload_hash(
                mistral_manifest
            ),
        },
        "runtime": {
            "app_name": "intervenebench-evidence-report-eval-v1",
            "model_volume_name": "intervenebench-model-cache-v1",
            "qwen_image_recipe": qwen_recipe,
            "qwen_image_recipe_sha256": payload_hash(qwen_recipe),
            "mistral_image_recipe": mistral_recipe,
            "mistral_image_recipe_sha256": payload_hash(mistral_recipe),
        },
        "limits": {
            "automatic_retries": 0,
            "reserve_calls": 0,
            "semantic_repair_allowed": False,
            "maximum_wall_clock_seconds": 10_800,
            "maximum_gpu_seconds_by_family": {
                "qwen_transformers": 18_000,
                "mistral_vllm": 9_000,
            },
            "official_price_usd_per_second": {
                "L40S:1": 0.000542,
                "A100-80GB:1": 0.000694,
            },
            "maximum_gpu_cost_usd": 16.002,
            "ancillary_and_failure_reserve_usd": 18.998,
            "hard_incremental_cost_cap_usd": 35.0,
            "completed_call_rerun_forbidden": True,
        },
        "privacy": {
            "public_aggregate_evidence_packet_allowed": True,
            "participant_rows_allowed": False,
            "experiment_level_human_scores_allowed": False,
            "human_labels_visible_to_generators": False,
        },
        "authority": {
            "modal_image_materialization_authorized": False,
            "model_download_authorized": False,
            "inference_authorized": False,
            "automatic_retries_authorized": False,
            "reserve_calls_authorized": False,
            "participant_row_access_authorized": False,
            "experiment_level_human_score_access_authorized": False,
            "automatic_judging_authorized": False,
            "automatic_next_stage_authorized": False,
        },
    }


def validate_report_materialization_authorization(
    authorization: Mapping[str, Any], freeze: Mapping[str, Any]
) -> None:
    expected = {
        "schema_version": "intervenebench.report_eval_materialization_authorization.v1",
        "execution_freeze_payload_sha256": payload_hash(freeze),
        "modal_image_materialization_authorized": True,
        "model_download_authorized": False,
        "inference_authorized": False,
        "participant_row_access_authorized": False,
        "automatic_next_stage_authorized": False,
    }
    if authorization != expected:
        raise ValueError("authorization is not exact frozen materialization authority")


def validate_report_import_smoke_authorization(
    authorization: Mapping[str, Any],
    freeze: Mapping[str, Any],
    *,
    materialized_image_ids: Mapping[str, str],
) -> None:
    image_ids = dict(materialized_image_ids)
    if set(image_ids) != {"qwen", "mistral"} or any(
        not isinstance(image_id, str) or not image_id.startswith("im-")
        for image_id in image_ids.values()
    ):
        raise ValueError("materialized image IDs are invalid")
    expected = {
        "schema_version": "intervenebench.report_eval_import_smoke_authorization.v1",
        "execution_freeze_payload_sha256": payload_hash(freeze),
        "modal_image_ids": image_ids,
        "exact_import_smoke_call_count": 2,
        "import_smoke_authorized": True,
        "model_download_authorized": False,
        "inference_authorized": False,
        "participant_row_access_authorized": False,
        "experiment_level_human_score_access_authorized": False,
        "automatic_next_stage_authorized": False,
    }
    if authorization != expected:
        raise ValueError("authorization is not exact frozen import-smoke authority")


def validate_report_import_smoke_result(
    result: Mapping[str, Any],
    freeze: Mapping[str, Any],
    *,
    materialized_image_ids: Mapping[str, str],
) -> None:
    image_ids = dict(materialized_image_ids)
    expected_paths = {
        "/opt/intervenebench/evidence_report_execution_v1.json",
        "/opt/intervenebench/report_generation_plan_v1.json",
        "/opt/intervenebench/model_file_manifests_v1.json",
        "/opt/intervenebench/mistral_small_3_1_24b_source_manifest_v1.json",
        "/opt/intervenebench/multimodal-requirements.lock",
        "/opt/intervenebench/cross-family-requirements.lock",
        "/root/evidence_report_app.py",
    }
    if (
        result.get("schema_version")
        != "intervenebench.report_eval_import_smoke.v1"
        or result.get("status") != "two_remote_imports_verified_zero_inference"
        or result.get("execution_freeze_payload_sha256") != payload_hash(freeze)
        or result.get("modal_image_ids") != image_ids
        or result.get("import_smoke_call_count") != 2
        or result.get("model_downloads") != 0
        or result.get("inference_calls") != 0
        or result.get("participant_rows_accessed") != 0
        or result.get("experiment_level_human_scores_accessed") is not False
        or result.get("automatic_next_stage") is not False
    ):
        raise ValueError("import-smoke result does not satisfy the frozen gate")
    smokes = result.get("smokes")
    if not isinstance(smokes, Mapping) or set(smokes) != {"qwen", "mistral"}:
        raise ValueError("import-smoke result has incomplete image coverage")
    for image_kind, image_id in image_ids.items():
        smoke = smokes[image_kind]
        expected_source_sha256 = freeze["runtime"][
            f"{image_kind}_image_recipe"
        ]["app_source_sha256"]
        if (
            not isinstance(smoke, Mapping)
            or smoke.get("schema_version")
            != "intervenebench.report_eval_remote_import_smoke.v1"
            or smoke.get("image_kind") != image_kind
            or smoke.get("modal_image_id") != image_id
            or smoke.get("execution_freeze_payload_sha256") != payload_hash(freeze)
            or smoke.get("source_sha256") != expected_source_sha256
            or set(smoke.get("verified_embedded_paths", [])) != expected_paths
            or smoke.get("model_downloaded") is not False
            or smoke.get("inference_performed") is not False
            or smoke.get("participant_rows_accessed") != 0
            or smoke.get("experiment_level_human_scores_accessed") is not False
        ):
            raise ValueError(f"remote import smoke failed for {image_kind}")


def validate_report_execution_authorization(
    authorization: Mapping[str, Any],
    freeze: Mapping[str, Any],
    *,
    materialized_image_ids: Mapping[str, str],
    import_smoke_result: Mapping[str, Any],
) -> None:
    image_ids = dict(materialized_image_ids)
    if set(image_ids) != {"qwen", "mistral"} or any(
        not isinstance(image_id, str) or not image_id.startswith("im-")
        for image_id in image_ids.values()
    ):
        raise ValueError("materialized image IDs are invalid")
    validate_report_import_smoke_result(
        import_smoke_result,
        freeze,
        materialized_image_ids=image_ids,
    )
    expected = {
        "schema_version": "intervenebench.report_eval_execution_authorization.v1",
        "execution_freeze_payload_sha256": payload_hash(freeze),
        "modal_image_ids": image_ids,
        "import_smoke_payload_sha256": payload_hash(import_smoke_result),
        "exact_call_count": freeze["planned_call_count"],
        "hard_incremental_cost_cap_usd": freeze["limits"][
            "hard_incremental_cost_cap_usd"
        ],
        "inference_authorized": True,
        "model_download_authorized": False,
        "automatic_retries_authorized": False,
        "reserve_calls_authorized": False,
        "participant_row_access_authorized": False,
        "experiment_level_human_score_access_authorized": False,
        "automatic_judging_authorized": False,
        "automatic_next_stage_authorized": False,
    }
    if authorization != expected:
        raise ValueError("authorization is not exact frozen execution authority")
