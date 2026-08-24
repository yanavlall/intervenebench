from __future__ import annotations

from pathlib import Path

from intervenebench.cross_family_seedfix import (
    build_seedfix_freeze,
    build_seedfix_materialization_authorization,
    validate_seedfix_materialization_authorization,
    verify_seedfix_freeze,
    build_seedfix_v2_materialization_authorization,
    verify_seedfix_v2_freeze,
)


ROOT = Path(__file__).resolve().parents[1]


def test_seedfix_is_one_runtime_correction_with_zero_prior_inferences() -> None:
    freeze = verify_seedfix_freeze(ROOT)
    assert freeze == build_seedfix_freeze(ROOT)
    assert freeze["completed_target_inference_count_before_correction"] == 0
    assert freeze["preserved_target_output_count_before_correction"] == 0
    assert freeze["remaining_call_count"] == 504
    assert freeze["unavailable_experiment_id"] == "tcg8p"
    assert freeze["unavailable_call_count"] == 120
    assert freeze["correction"] == {
        "request_payloads_changed": False,
        "request_hashes_changed": False,
        "prompt_or_asset_changed": False,
        "affected_method_id": "forced_choice_next_token_softmax.v1",
        "source_generation_seed": None,
        "effective_engine_seed": 0,
        "application_point": "after_request_hash_validation_before_sampling_params",
        "continuous_interface_changed": False,
        "semantic_repair": False,
    }


def test_materialization_authority_is_zero_inference_only() -> None:
    authorization = build_seedfix_materialization_authorization(ROOT)
    validate_seedfix_materialization_authorization(authorization, root=ROOT)
    assert authorization["modal_image_materialization_authorized"] is True
    assert authorization["modal_compute_authorized"] is False
    assert authorization["paid_inference_authorized"] is False
    assert authorization["seedfix_canary_authorized"] is False
    assert authorization["remaining_target_calls_authorized"] is False
    assert authorization["model_download_authorized"] is False
    assert authorization["human_outcome_access_authorized"] is False
    assert authorization["regression_scoring_authorized"] is False
    assert authorization["automatic_next_stage_authorized"] is False


def test_seedfix_worker_validates_original_hash_before_null_seed_mapping() -> None:
    source = (ROOT / "infra/modal/cross_family_target_seedfix_app.py").read_text(
        encoding="utf-8"
    )
    request_hash_check = source.index(
        'base._payload_hash(request) != base.REQUEST_HASHES[call_id]'
    )
    seedfix_call = source.index("raw = _forced_choice_seedfix(llm, request)")
    assert 'fixed_request["generation_seed"] = 0' in source
    assert request_hash_check < seedfix_call
    assert 'block_network=True' in source
    assert 'restrict_modal_access=True' in source
    assert 'retries=0' in source
    assert 'max_containers=1' in source
    assert 'model_volume.with_mount_options(read_only=True)' in source


def test_seedfix_stages_remain_separate_and_no_scoring() -> None:
    materialization = (
        ROOT / "scripts/run_cross_family_seedfix_materialization.py"
    ).read_text(encoding="utf-8")
    canary = (ROOT / "scripts/run_cross_family_seedfix_canary.py").read_text(
        encoding="utf-8"
    )
    continuation = (ROOT / "scripts/run_cross_family_continuation.py").read_text(
        encoding="utf-8"
    )
    assert "remote_function_calls_made\": 0" in materialization
    assert "inference_calls_made\": 0" in materialization
    assert "target_calls_made\": 0" in canary
    assert "validate_seedfix_continuation_authorization" in continuation
    assert "score_confirmation" not in materialization + canary + continuation


def test_seedfix_v2_preserves_requests_and_only_widens_logprob_window() -> None:
    freeze = verify_seedfix_v2_freeze(ROOT)
    assert freeze["completed_target_inference_count_before_v2"] == 0
    assert freeze["preserved_target_output_count_before_v2"] == 0
    assert freeze["remaining_call_count"] == 504
    assert freeze["correction"]["request_payloads_changed"] is False
    assert freeze["correction"]["request_hashes_changed"] is False
    assert freeze["correction"]["prompt_or_asset_changed"] is False
    assert freeze["correction"]["effective_engine_seed"] == 0
    assert freeze["correction"]["requested_logprob_count"] == 20
    assert freeze["correction"]["maximum_target_answer_code_count"] == 11
    authorization = build_seedfix_v2_materialization_authorization(ROOT)
    assert authorization["modal_image_materialization_authorized"] is True
    assert authorization["paid_inference_authorized"] is False
    assert authorization["remaining_target_calls_authorized"] is False
    source = (
        ROOT / "infra/modal/cross_family_target_seedfix_v2_app.py"
    ).read_text(encoding="utf-8")
    assert "logprobs=max(20, len(token_ids))" in source
    assert source.index(
        'base._payload_hash(request) != base.REQUEST_HASHES[call_id]'
    ) < source.index("raw = _forced_choice_v2(llm, request)")
