from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from intervenebench.cross_family_regression import (
    CANDIDATE_MODEL_ID,
    DEFAULT_CALL_PLAN_PATH,
    DEFAULT_MODEL_SOURCE_MANIFEST_PATH,
    DEFAULT_PROTOCOL_PATH,
    EXPECTED_CALL_COUNT_BY_EXPERIMENT,
    CrossFamilyFreezeSummary,
    build_cross_family_call_plan,
    require_cross_family_execution_authority,
    verify_cross_family_freeze,
)
from intervenebench.model_regression import ModelVersionRegressionThresholds
from intervenebench.protocol import assert_blinded_payload, payload_hash, verify_envelope


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PLAN_PATH = ROOT / "data/manifests/simulators/confirmation_call_plan_v1.json"


def _source_primary_calls() -> list[dict[str, object]]:
    source = verify_envelope(SOURCE_PLAN_PATH, require_blinded=True)
    selected: list[dict[str, object]] = []
    for call in source["calls"]:
        expected_model = (
            "qwen3_vl_8b_primary"
            if call["experiment_id"] == "pb2rr"
            else "qwen3_8b_generic"
        )
        if (
            call["stage"] in {"base", "primary_prompt_perturbation"}
            and call["model_id"] == expected_model
        ):
            selected.append(call)
    return selected


def test_frozen_protocol_and_plan_verify_without_execution_authority() -> None:
    summary = verify_cross_family_freeze(ROOT)

    assert summary == CrossFamilyFreezeSummary(
        candidate_model_id=CANDIDATE_MODEL_ID,
        experiment_count=6,
        planned_call_count=624,
        base_call_count=312,
        prompt_perturbation_call_count=312,
        maximum_gpu_seconds=100_000,
        maximum_gpu_cost_usd=69.4,
        hard_incremental_cost_cap_usd=90.0,
        retrospective_only=True,
    )
    protocol = verify_envelope(ROOT / DEFAULT_PROTOCOL_PATH, require_blinded=True)
    assert set(protocol["authority"].values()) == {False}
    with pytest.raises(PermissionError, match="separate execution authorization"):
        require_cross_family_execution_authority(protocol)


def test_model_source_manifest_pins_one_public_ungated_original_checkpoint() -> None:
    manifest = verify_envelope(
        ROOT / DEFAULT_MODEL_SOURCE_MANIFEST_PATH, require_blinded=True
    )

    assert manifest["model_id"] == CANDIDATE_MODEL_ID
    assert manifest["hf_repository"] == (
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    )
    assert manifest["checkpoint_commit"] == (
        "68faf511d618ef198fef186659617cfd2eb8e33a"
    )
    assert manifest["license_id"] == "apache-2.0"
    assert manifest["repository_private"] is False
    assert manifest["repository_gated"] is False
    assert manifest["load_format"] == "mistral_original_consolidated"
    assert manifest["required_gpu"] == "A100-80GB"
    assert manifest["dtype"] == "bfloat16"
    assert manifest["quantization"] == "none"

    files = {entry["path"]: entry for entry in manifest["files"]}
    assert files["consolidated.safetensors"]["content_sha256"] == (
        "d446ca97599fa9d98b2e3744d8b83019837a2fe34a80f4353120b1e9b6249b1e"
    )
    assert files["consolidated.safetensors"]["size_bytes"] == 48_022_792_280
    assert files["tekken.json"]["content_sha256"] == (
        "c604f35d1035f534519622c0ec83fed6184978d4fdee92a5bd2a50bc05438094"
    )
    assert files["tokenizer.json"]["content_sha256"] == (
        "b76085f9923309d873994d444989f7eb6ec074b06f25b58f1e8d7b7741070949"
    )
    assert all(not path.startswith("model-") for path in files)
    assert "model.safetensors.index.json" not in files
    assert manifest["excluded_repository_paths"] == [
        "model-00001-of-00010.safetensors",
        "model-00002-of-00010.safetensors",
        "model-00003-of-00010.safetensors",
        "model-00004-of-00010.safetensors",
        "model-00005-of-00010.safetensors",
        "model-00006-of-00010.safetensors",
        "model-00007-of-00010.safetensors",
        "model-00008-of-00010.safetensors",
        "model-00009-of-00010.safetensors",
        "model-00010-of-00010.safetensors",
        "model.safetensors.index.json",
    ]
    assert manifest["authority"]["model_download_authorized"] is False
    assert manifest["authority"]["modal_resource_creation_authorized"] is False


def test_call_plan_is_exact_one_to_one_replay_of_primary_qwen_grid() -> None:
    plan = verify_envelope(ROOT / DEFAULT_CALL_PLAN_PATH, require_blinded=True)
    source_calls = _source_primary_calls()

    assert len(source_calls) == 624
    assert plan == build_cross_family_call_plan(ROOT)
    assert plan["planned_call_count"] == 624
    assert plan["reserve_call_count"] == 0
    assert Counter(call["stage"] for call in plan["calls"]) == {
        "base": 312,
        "primary_prompt_perturbation": 312,
    }
    assert Counter(call["experiment_id"] for call in plan["calls"]) == (
        EXPECTED_CALL_COUNT_BY_EXPERIMENT
    )
    assert {call["candidate_model_id"] for call in plan["calls"]} == {
        CANDIDATE_MODEL_ID
    }
    assert len({call["candidate_call_id"] for call in plan["calls"]}) == 624
    assert len({call["candidate_artifact_relative_path"] for call in plan["calls"]}) == 624

    by_source = {call["source_call_id"]: call for call in plan["calls"]}
    assert set(by_source) == {call["call_id"] for call in source_calls}
    for source in source_calls:
        candidate = by_source[source["call_id"]]
        for field in (
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
        ):
            assert candidate[field] == source[field]
        assert candidate["source_prompt_sha256"] == source["prompt_sha256"]
        assert len(candidate["candidate_request_spec_sha256"]) == 64


def test_interface_and_failure_rules_are_frozen_before_any_candidate_run() -> None:
    protocol = verify_envelope(ROOT / DEFAULT_PROTOCOL_PATH, require_blinded=True)
    interfaces = protocol["inference_contract"]["interfaces"]

    assert interfaces["forced_choice_next_token"]["preflight_requirement"] == (
        "every answer code is one exact distinct tokenizer token"
    )
    assert interfaces["forced_choice_next_token"]["required_schema_validity"] == 1.0
    assert interfaces["continuous_integer"]["parser"] == (
        "strict_nonnegative_integer_no_repair_no_clamp"
    )
    assert protocol["failure_policy"]["automatic_retries"] == 0
    assert protocol["failure_policy"]["reserve_calls"] == 0
    assert protocol["failure_policy"]["primary_substitution_allowed"] is False
    assert protocol["failure_policy"]["invalid_required_cell"] == (
        "mark_model_task_unavailable_and_preserve_raw_output"
    )
    assert protocol["system_instruction"]["dynamic_date_allowed"] is False
    assert protocol["system_instruction"]["use_repository_system_prompt"] is False


def test_budget_and_regression_thresholds_match_the_existing_machine_gate() -> None:
    protocol = verify_envelope(ROOT / DEFAULT_PROTOCOL_PATH, require_blinded=True)
    budget = protocol["budget"]
    thresholds = protocol["regression_thresholds"]

    assert budget["gpu_type"] == "A100-80GB"
    assert budget["gpu_count"] == 1
    assert budget["maximum_gpu_seconds"] == 100_000
    assert budget["official_gpu_price_usd_per_second"] == pytest.approx(0.000694)
    assert budget["maximum_gpu_cost_usd"] == pytest.approx(69.4)
    assert budget["hard_incremental_cost_cap_usd"] == pytest.approx(90.0)
    assert budget["maximum_gpu_cost_usd"] == pytest.approx(
        budget["maximum_gpu_seconds"]
        * budget["official_gpu_price_usd_per_second"]
    )
    assert thresholds == asdict(ModelVersionRegressionThresholds())


def test_claim_boundary_is_retrospective_and_does_not_inflate_experiment_n() -> None:
    protocol = verify_envelope(ROOT / DEFAULT_PROTOCOL_PATH, require_blinded=True)
    boundary = protocol["claim_boundary"]

    assert protocol["study_role"] == "retrospective_cross_family_robustness"
    assert boundary["prospective_confirmation"] is False
    assert boundary["adds_independent_experiments"] is False
    assert boundary["changes_prospective_experiment_count"] is False
    assert boundary["permitted_claim"] == (
        "architecture-family robustness on the already revealed six-task panel"
    )
    assert boundary["forbidden_claim"] == (
        "new prospective validation or evidence for universal simulator reliability"
    )


def test_all_freeze_artifacts_are_blinded_and_hash_bound() -> None:
    protocol = verify_envelope(ROOT / DEFAULT_PROTOCOL_PATH, require_blinded=True)
    model = verify_envelope(
        ROOT / DEFAULT_MODEL_SOURCE_MANIFEST_PATH, require_blinded=True
    )
    plan = verify_envelope(ROOT / DEFAULT_CALL_PLAN_PATH, require_blinded=True)

    assert_blinded_payload(protocol)
    assert_blinded_payload(model)
    assert_blinded_payload(plan)
    assert plan["protocol_payload_sha256"] == payload_hash(protocol)
    assert plan["model_source_manifest_payload_sha256"] == payload_hash(model)
    source = json.loads(SOURCE_PLAN_PATH.read_text(encoding="utf-8"))
    assert plan["source_confirmation_call_plan_payload_sha256"] == source["sha256"]
    assert plan["authority"] == protocol["authority"]


def test_request_specs_are_deterministic_and_bound_to_source_prompt_hashes() -> None:
    first = build_cross_family_call_plan(ROOT)
    second = build_cross_family_call_plan(ROOT)
    assert payload_hash(first) == payload_hash(second)

    calls = first["calls"]
    assert calls[0]["candidate_request_spec_sha256"] != calls[1][
        "candidate_request_spec_sha256"
    ]
    assert all(
        call["source_prompt_sha256"]
        and call["bundle_payload_sha256"]
        and call["system_instruction_sha256"]
        for call in calls
    )
