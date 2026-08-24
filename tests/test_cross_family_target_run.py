from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from intervenebench.cross_family_execution import prepare_cross_family_target_requests
from intervenebench.cross_family_target_run import (
    build_recommendations,
    build_target_authorization,
    load_target_bindings,
    strict_parse_target_result,
    validate_target_authorization,
)
from intervenebench.protocol import payload_hash


ROOT = Path(__file__).resolve().parents[1]


def _runtime(request: dict, bindings: dict) -> dict:
    freeze = bindings["freeze"]
    return {
        "modal_image_id": bindings["materialization"]["modal_image_id"],
        "checkpoint_commit": freeze["model"]["checkpoint_commit"],
        "cache_attestation_sha256": freeze["model"][
            "cache_attestation_payload_sha256"
        ],
        "call_id": request["call_id"],
        "prompt_sha256": request["source_prompt_sha256"],
        "asset_sha256": request["asset_sha256"],
        "method_id": request["method_id"],
    }


def _raw_probability(request: dict, bindings: dict) -> dict:
    codes = request["answer_codes"]
    probability = 1.0 / len(codes)
    return {
        "call_id": request["call_id"],
        "model_id": request["model_id"],
        "request_payload_sha256": payload_hash(request),
        "result": {
            "probabilities_by_code": {code: probability for code in codes},
            "candidate_token_ids": list(range(100, 100 + len(codes))),
            "free_generation_used": False,
            "engine_probe_tokens": 1,
        },
        "runtime_attestation": _runtime(request, bindings),
    }


def _raw_continuous(request: dict, bindings: dict, text: str = '{"predicted_value": 7}') -> dict:
    return {
        "call_id": request["call_id"],
        "model_id": request["model_id"],
        "request_payload_sha256": payload_hash(request),
        "result": {
            "raw_text": text,
            "generation_seed": request["generation_seed"],
            "semantic_repair_used": False,
        },
        "runtime_attestation": _runtime(request, bindings),
    }


def test_target_authority_is_exact_624_and_no_retries_or_humans() -> None:
    bindings = load_target_bindings(ROOT)
    authorization = build_target_authorization(ROOT)
    validate_target_authorization(authorization, root=ROOT, **bindings)
    assert authorization["planned_call_count"] == 624
    assert authorization["maximum_attempt_count"] == 624
    assert authorization["target_inference_authorized"] is True
    assert authorization["strict_parse_authorized"] is True
    assert authorization["recommendation_aggregation_authorized"] is True
    assert authorization["model_download_authorized"] is False
    assert authorization["automatic_retry_authorized"] is False
    assert authorization["reserve_call_authorized"] is False
    assert authorization["human_outcome_access_authorized"] is False
    assert authorization["regression_scoring_authorized"] is False
    assert authorization["automatic_next_stage_authorized"] is False
    widened = deepcopy(authorization)
    widened["human_outcome_access_authorized"] = True
    with pytest.raises(PermissionError, match="widened"):
        validate_target_authorization(widened, root=ROOT, **bindings)


def test_strict_parser_accepts_exact_interfaces_and_rejects_repair() -> None:
    bindings = load_target_bindings(ROOT)
    requests = prepare_cross_family_target_requests(ROOT)
    probability_request = next(
        row for row in requests
        if row["method_id"] == "forced_choice_next_token_softmax.v1"
    )
    continuous_request = next(
        row for row in requests
        if row["method_id"] == "continuous_constrained_integer_generation.v1"
    )
    parsed_probability = strict_parse_target_result(
        probability_request,
        _raw_probability(probability_request, bindings),
        freeze=bindings["freeze"],
        expected_modal_image_id=bindings["materialization"]["modal_image_id"],
    )
    assert sum(parsed_probability["probabilities_by_source_value"].values()) == pytest.approx(1.0)
    parsed_continuous = strict_parse_target_result(
        continuous_request,
        _raw_continuous(continuous_request, bindings),
        freeze=bindings["freeze"],
        expected_modal_image_id=bindings["materialization"]["modal_image_id"],
    )
    assert parsed_continuous["predicted_value"] == 7
    for invalid in (
        "7",
        '{"predicted_value": 7.0}',
        '{"predicted_value": -1}',
        '{"predicted_value": 7, "extra": 1}',
        "```json\n{\"predicted_value\": 7}\n```",
    ):
        with pytest.raises(ValueError):
            strict_parse_target_result(
                continuous_request,
                _raw_continuous(continuous_request, bindings, invalid),
                freeze=bindings["freeze"],
                expected_modal_image_id=bindings["materialization"]["modal_image_id"],
            )


def test_uniform_toy_outputs_freeze_six_complete_recommendations() -> None:
    bindings = load_target_bindings(ROOT)
    requests = prepare_cross_family_target_requests(ROOT)
    strict_outputs = {}
    strict_hashes = {}
    for request in requests:
        raw = (
            _raw_continuous(request, bindings)
            if request["method_id"] == "continuous_constrained_integer_generation.v1"
            else _raw_probability(request, bindings)
        )
        parsed = strict_parse_target_result(
            request,
            raw,
            freeze=bindings["freeze"],
            expected_modal_image_id=bindings["materialization"]["modal_image_id"],
        )
        strict_outputs[request["call_id"]] = parsed
        strict_hashes[request["call_id"]] = payload_hash(parsed)
    recommendations = build_recommendations(
        ROOT,
        requests=requests,
        strict_outputs=strict_outputs,
        strict_output_hashes=strict_hashes,
        parse_failures=[],
    )
    assert recommendations["planned_call_count"] == 624
    assert recommendations["strict_output_count"] == 624
    assert recommendations["recommendation_count"] == 6
    assert recommendations["unavailable_experiment_count"] == 0
    assert recommendations["semantic_repairs_made"] == 0
    assert recommendations["reruns_made"] == 0
    assert recommendations["human_outcomes_accessed"] is False
    assert recommendations["human_outcome_scoring_performed"] is False


def test_execution_wrapper_has_sequential_groups_heartbeats_and_no_scoring() -> None:
    path = ROOT / "scripts/run_cross_family_target_execution.py"
    source = path.read_text(encoding="utf-8")
    assert "import modal\n" not in source
    assert source.index("validate_target_authorization") < source.index("_load_app()")
    assert "experiment_group_active" in source
    assert "call.cancel(terminate_containers=True)" in source
    assert "automatic_retries\": 0" in source
    assert "reserve_calls\": 0" in source
    assert "semantic_repairs\": 0" in source
    assert "human_outcomes_accessed\": False" in source
    assert "human_outcome_scoring_performed\": False" in source
    assert "score_confirmation" not in source
    assert "ThreadPoolExecutor" not in source
    spec = importlib.util.spec_from_file_location("cross_family_target_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
