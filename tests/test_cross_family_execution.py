from __future__ import annotations

from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from intervenebench.cross_family_execution import (
    DEFAULT_EXECUTION_FREEZE_PATH,
    build_cross_family_execution_freeze,
    build_materialization_authorization,
    prepare_cross_family_target_requests,
    validate_json_canary_result,
    validate_materialization_authorization,
    verify_cross_family_execution_freeze,
)


ROOT = Path(__file__).resolve().parents[1]


def test_all_624_target_requests_reconstruct_hash_exactly() -> None:
    requests = prepare_cross_family_target_requests(ROOT)
    assert len(requests) == 624
    assert len({row["call_id"] for row in requests}) == 624
    assert Counter(row["method_id"] for row in requests) == {
        "continuous_constrained_integer_generation.v1": 120,
        "forced_choice_next_token_softmax.v1": 504,
    }
    assert Counter(row["modality"] for row in requests) == {
        "exact_png_vision": 128,
        "text": 496,
    }
    continuous = [
        row for row in requests
        if row["method_id"] == "continuous_constrained_integer_generation.v1"
    ]
    assert all("predicted_value" in row["prompt"] for row in continuous)
    assert all("JSON object" in row["prompt"] for row in continuous)
    assert all(row["experiment_id"] == "tcg8p" for row in continuous)
    vision_assets = {row["asset_path"] for row in requests if row["asset_path"]}
    assert vision_assets == {
        "data/derived/stimuli/pb2rr/hispanic_population_growth_article.png",
        "data/derived/stimuli/pb2rr/iphone_growth_control_article.png",
    }


def test_execution_freeze_is_zero_authority_and_json_canary_blocked() -> None:
    freeze = build_cross_family_execution_freeze(ROOT)
    assert freeze["status"] == "frozen_nonexecuting_zero_authority"
    assert set(freeze["authority"].values()) == {False}
    assert freeze["call_plan"]["planned_call_count"] == 624
    assert len(freeze["call_plan"]["request_payload_sha256_by_call_id"]) == 624
    assert freeze["preflight"]["target_inference_ready"] is False
    assert freeze["required_json_canary"]["authorized"] is False
    assert freeze["stage_gates"]["target_inference_requires_passed_json_canary"] is True
    assert freeze["limits"]["maximum_attempt_count"] == 624
    assert freeze["limits"]["automatic_retries"] == 0
    assert freeze["limits"]["reserve_calls"] == 0
    assert freeze["human_outcomes_accessed"] is False
    assert freeze["participant_rows_read"] == 0


def test_materialization_authority_is_exactly_image_only() -> None:
    freeze = build_cross_family_execution_freeze(ROOT)
    authorization = build_materialization_authorization(freeze)
    validate_materialization_authorization(authorization, freeze=freeze)
    assert authorization["modal_image_materialization_authorized"] is True
    assert authorization["target_inference_authorized"] is False
    assert authorization["json_canary_authorized"] is False
    assert authorization["model_download_authorized"] is False
    widened = deepcopy(authorization)
    widened["json_canary_authorized"] = True
    with pytest.raises(PermissionError, match="widened"):
        validate_materialization_authorization(widened, freeze=freeze)


def test_json_canary_validator_requires_exact_integer_object() -> None:
    freeze = build_cross_family_execution_freeze(ROOT)
    result = {
        "schema_version": "intervenebench.cross_family_json_canary_result.v1",
        "status": "passed_target_free_json_schema",
        "canary_id": freeze["required_json_canary"]["canary_id"],
        "prompt_sha256": freeze["required_json_canary"]["prompt_sha256"],
        "raw_text": '{"predicted_value": 3}',
        "parsed_value": 3,
        "semantic_repair_used": False,
        "modal_image_id": "im-test",
        "runtime_attestation": {"modal_image_id": "im-test"},
        "target_calls_made": 0,
        "human_outcomes_accessed": False,
        "participant_rows_read": 0,
        "automatic_next_stage": False,
    }
    validate_json_canary_result(result, freeze=freeze, modal_image_id="im-test")
    bad = deepcopy(result)
    bad["raw_text"] = '{"predicted_value": 3.0}'
    with pytest.raises(ValueError, match="integer schema"):
        validate_json_canary_result(bad, freeze=freeze, modal_image_id="im-test")
    bad = deepcopy(result)
    bad["raw_text"] = '{"predicted_value": 3, "extra": 1}'
    with pytest.raises(ValueError, match="integer schema"):
        validate_json_canary_result(bad, freeze=freeze, modal_image_id="im-test")


def test_target_app_has_no_download_and_is_fail_closed() -> None:
    source = (ROOT / "infra/modal/cross_family_target_app.py").read_text(
        encoding="utf-8"
    )
    assert "snapshot_download" not in source
    assert "cache_cross_family_checkpoint" not in source
    assert "create_if_missing=False" in source
    assert "with_mount_options(read_only=True)" in source
    assert "block_network=True" in source
    assert "retries=0" in source
    assert "run_cross_family_json_canary" in source
    assert "run_cross_family_target_group" in source
    assert 'Path("/root/infra/modal/cross_family_target_app.py")' in source
    assert ".add_local_file(SOURCE_PATH, str(REMOTE_SOURCE_PATH), copy=True)" in source


def test_wrapper_validates_locally_and_exposes_only_materialization() -> None:
    path = ROOT / "scripts/run_cross_family_target.py"
    source = path.read_text(encoding="utf-8")
    assert "import modal\n" not in source
    assert source.index("validate_materialization_authorization") < source.index(
        "_load_app()"
    )
    assert 'choices=("materialize",)' in source
    assert "run_cross_family_json_canary.remote" not in source
    assert "run_cross_family_target_group.remote" not in source
    spec = importlib.util.spec_from_file_location("cross_family_target_wrapper", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_frozen_execution_package_replays_when_present() -> None:
    path = ROOT / DEFAULT_EXECUTION_FREEZE_PATH
    if not path.exists():
        pytest.skip("execution freeze is generated after implementation audit")
    actual = verify_cross_family_execution_freeze(ROOT, path)
    assert actual == build_cross_family_execution_freeze(ROOT)
