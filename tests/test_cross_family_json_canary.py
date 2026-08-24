from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from intervenebench.cross_family_json_canary import (
    build_json_canary_authorization,
    load_json_canary_bindings,
    validate_json_canary_authorization,
    validate_json_canary_completion,
)
from intervenebench.protocol import payload_hash


ROOT = Path(__file__).resolve().parents[1]


def test_json_canary_authority_is_exactly_one_target_free_call() -> None:
    bindings = load_json_canary_bindings(ROOT)
    authorization = build_json_canary_authorization(ROOT)
    validate_json_canary_authorization(authorization, **bindings)
    assert authorization["planned_call_count"] == 1
    assert authorization["maximum_attempt_count"] == 1
    assert authorization["json_canary_authorized"] is True
    assert authorization["target_inference_authorized"] is False
    assert authorization["target_call_authorized"] is False
    assert authorization["model_download_authorized"] is False
    assert authorization["automatic_retry_authorized"] is False
    assert authorization["automatic_next_stage_authorized"] is False
    widened = deepcopy(authorization)
    widened["target_inference_authorized"] = True
    with pytest.raises(PermissionError, match="widened"):
        validate_json_canary_authorization(widened, **bindings)


def test_json_canary_completion_has_no_target_or_human_access() -> None:
    bindings = load_json_canary_bindings(ROOT)
    authorization = build_json_canary_authorization(ROOT)
    freeze = bindings["freeze"]
    materialization = bindings["materialization"]
    raw = {
        "schema_version": "intervenebench.cross_family_json_canary_result.v1",
        "status": "passed_target_free_json_schema",
        "canary_id": freeze["required_json_canary"]["canary_id"],
        "prompt_sha256": freeze["required_json_canary"]["prompt_sha256"],
        "raw_text": '{"predicted_value": 7}',
        "parsed_value": 7,
        "semantic_repair_used": False,
        "modal_image_id": materialization["modal_image_id"],
        "runtime_attestation": {"modal_image_id": materialization["modal_image_id"]},
        "target_calls_made": 0,
        "human_outcomes_accessed": False,
        "participant_rows_read": 0,
        "automatic_next_stage": False,
    }
    completion = {
        "schema_version": "intervenebench.cross_family_json_canary_completion.v1",
        "status": "passed_target_free_json_schema_stop",
        "freeze_payload_sha256": payload_hash(freeze),
        "authorization_payload_sha256": payload_hash(authorization),
        "materialization_payload_sha256": payload_hash(materialization),
        "modal_image_id": materialization["modal_image_id"],
        "attempt_count": 1,
        "raw_result": raw,
        "target_prompts_or_assets_accessed": False,
        "target_calls_made": 0,
        "model_downloaded": False,
        "human_outcomes_accessed": False,
        "participant_rows_read": 0,
        "automatic_next_stage": False,
    }
    validate_json_canary_completion(
        completion,
        freeze=freeze,
        authorization=authorization,
        materialization=materialization,
    )
    bad = deepcopy(completion)
    bad["target_calls_made"] = 1
    with pytest.raises(ValueError, match="exceeded"):
        validate_json_canary_completion(
            bad,
            freeze=freeze,
            authorization=authorization,
            materialization=materialization,
        )


def test_json_canary_wrapper_has_heartbeats_one_call_and_no_auto_next() -> None:
    path = ROOT / "scripts/run_cross_family_json_canary.py"
    source = path.read_text(encoding="utf-8")
    assert "import modal\n" not in source
    assert source.index("validate_json_canary_authorization") < source.index(
        "_load_app()"
    )
    assert "planned_call_count" not in source or "attempt_count" in source
    assert "maximum_attempt_count" not in source or "3600" in source
    assert "remote_call_active" in source
    assert "call.cancel(terminate_containers=True)" in source
    assert "automatic_next_stage" in source
    assert "run_cross_family_target_group" not in source
    spec = importlib.util.spec_from_file_location("cross_family_json_canary_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
