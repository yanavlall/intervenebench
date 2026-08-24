from __future__ import annotations

import json
from pathlib import Path

import pytest

from intervenebench.evidence_report_execution import (
    build_report_execution_freeze,
    validate_report_execution_authorization,
    validate_report_import_smoke_authorization,
    validate_report_import_smoke_result,
    validate_report_materialization_authorization,
)
from intervenebench.protocol import payload_hash, verify_envelope


ROOT = Path(__file__).resolve().parents[1]
PACKET_PATH = ROOT / (
    "data/manifests/qualitative_eval/"
    "intervenebench_report_evidence_packet_v1.json"
)
PROTOCOL_PATH = ROOT / "data/manifests/research/evidence_report_eval_v1.json"
PLAN_PATH = ROOT / (
    "data/manifests/qualitative_eval/report_generation_plan_v1.json"
)
FREEZE_PATH = ROOT / "configs/simulators/evidence_report_execution_v1.json"


def _inputs() -> tuple[dict, dict, dict]:
    packet = json.loads(PACKET_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    plan = verify_envelope(PLAN_PATH, require_blinded=True)
    return packet, protocol, plan


def test_execution_freeze_binds_48_calls_three_cached_models_and_zero_authority() -> None:
    packet, protocol, plan = _inputs()
    freeze = build_report_execution_freeze(ROOT, packet, protocol, plan)

    assert freeze["planned_call_count"] == 48
    assert freeze["planned_calls_by_model_role"] == {
        "mistral_small_3_1_24b_cross_family": 16,
        "qwen3_14b_scale_candidate": 16,
        "qwen3_8b_incumbent": 16,
    }
    assert {model["gpu"] for model in freeze["models"]} == {
        "A100-80GB:1",
        "L40S:1",
    }
    assert all(model["model_download_authorized"] is False for model in freeze["models"])
    assert all(value is False for value in freeze["authority"].values())
    assert freeze["limits"]["hard_incremental_cost_cap_usd"] == 35.0


def test_persisted_execution_freeze_replays_exactly() -> None:
    packet, protocol, plan = _inputs()
    persisted = verify_envelope(FREEZE_PATH, require_blinded=True)
    assert persisted == build_report_execution_freeze(ROOT, packet, protocol, plan)


def test_materialization_and_execution_authorities_are_exact_and_fail_closed() -> None:
    packet, protocol, plan = _inputs()
    freeze = build_report_execution_freeze(ROOT, packet, protocol, plan)
    freeze_sha = payload_hash(freeze)
    materialization = {
        "schema_version": "intervenebench.report_eval_materialization_authorization.v1",
        "execution_freeze_payload_sha256": freeze_sha,
        "modal_image_materialization_authorized": True,
        "model_download_authorized": False,
        "inference_authorized": False,
        "participant_row_access_authorized": False,
        "automatic_next_stage_authorized": False,
    }
    validate_report_materialization_authorization(materialization, freeze)

    image_ids = {
        "qwen": "im-qwen-pinned",
        "mistral": "im-mistral-pinned",
    }
    smoke_authorization = {
        "schema_version": "intervenebench.report_eval_import_smoke_authorization.v1",
        "execution_freeze_payload_sha256": freeze_sha,
        "modal_image_ids": image_ids,
        "exact_import_smoke_call_count": 2,
        "import_smoke_authorized": True,
        "model_download_authorized": False,
        "inference_authorized": False,
        "participant_row_access_authorized": False,
        "experiment_level_human_score_access_authorized": False,
        "automatic_next_stage_authorized": False,
    }
    validate_report_import_smoke_authorization(
        smoke_authorization,
        freeze,
        materialized_image_ids=image_ids,
    )
    paths = [
        "/opt/intervenebench/cross-family-requirements.lock",
        "/opt/intervenebench/evidence_report_execution_v1.json",
        "/opt/intervenebench/mistral_small_3_1_24b_source_manifest_v1.json",
        "/opt/intervenebench/model_file_manifests_v1.json",
        "/opt/intervenebench/multimodal-requirements.lock",
        "/opt/intervenebench/report_generation_plan_v1.json",
        "/root/evidence_report_app.py",
    ]
    smokes = {
        kind: {
            "schema_version": "intervenebench.report_eval_remote_import_smoke.v1",
            "image_kind": kind,
            "modal_image_id": image_id,
            "execution_freeze_payload_sha256": freeze_sha,
            "source_sha256": freeze["runtime"][f"{kind}_image_recipe"][
                "app_source_sha256"
            ],
            "verified_embedded_paths": paths,
            "model_downloaded": False,
            "inference_performed": False,
            "participant_rows_accessed": 0,
            "experiment_level_human_scores_accessed": False,
        }
        for kind, image_id in image_ids.items()
    }
    smoke_result = {
        "schema_version": "intervenebench.report_eval_import_smoke.v1",
        "status": "two_remote_imports_verified_zero_inference",
        "execution_freeze_payload_sha256": freeze_sha,
        "authorization_payload_sha256": "b" * 64,
        "materialization_payload_sha256": "c" * 64,
        "modal_image_ids": image_ids,
        "import_smoke_call_count": 2,
        "smokes": smokes,
        "model_downloads": 0,
        "inference_calls": 0,
        "participant_rows_accessed": 0,
        "experiment_level_human_scores_accessed": False,
        "automatic_next_stage": False,
    }
    validate_report_import_smoke_result(
        smoke_result,
        freeze,
        materialized_image_ids=image_ids,
    )

    execution = {
        "schema_version": "intervenebench.report_eval_execution_authorization.v1",
        "execution_freeze_payload_sha256": freeze_sha,
        "modal_image_ids": image_ids,
        "import_smoke_payload_sha256": payload_hash(smoke_result),
        "exact_call_count": 48,
        "hard_incremental_cost_cap_usd": 35.0,
        "inference_authorized": True,
        "model_download_authorized": False,
        "automatic_retries_authorized": False,
        "reserve_calls_authorized": False,
        "participant_row_access_authorized": False,
        "experiment_level_human_score_access_authorized": False,
        "automatic_judging_authorized": False,
        "automatic_next_stage_authorized": False,
    }
    validate_report_execution_authorization(
        execution,
        freeze,
        materialized_image_ids=execution["modal_image_ids"],
        import_smoke_result=smoke_result,
    )

    drifted = dict(execution)
    drifted["exact_call_count"] = 49
    with pytest.raises(ValueError, match="exact frozen execution authority"):
        validate_report_execution_authorization(
            drifted,
            freeze,
            materialized_image_ids=execution["modal_image_ids"],
            import_smoke_result=smoke_result,
        )

    broken_smoke = dict(smoke_result)
    broken_smoke["inference_calls"] = 1
    with pytest.raises(ValueError, match="frozen gate"):
        validate_report_import_smoke_result(
            broken_smoke,
            freeze,
            materialized_image_ids=image_ids,
        )
