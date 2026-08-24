"""Immutable zero-authority freeze for prospective multimodal development."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .balanced_forced_choice import read_json_object
from .multimodal_prospective import (
    EXPERIMENT_IDS,
    MODEL_IDS,
    MODEL_SPECS,
    build_multimodal_prospective_plan,
    multimodal_forced_choice_prompt,
)
from .protocol import assert_blinded_payload, payload_hash, verify_envelope


PLAN_PATH = Path("data/manifests/simulators/prospective_multimodal_plan_v1.json")
MODEL_MANIFEST_PATH = Path(
    "data/manifests/simulators/multimodal_model_file_manifests_v1.json"
)
LOCK_INPUT_PATH = Path("infra/modal/multimodal-requirements.in")
LOCK_PATH = Path("infra/modal/multimodal-requirements.lock")
DISCOVERY_SCORE_PATH = Path(
    "artifacts/balanced_full_action/balanced_full_action_20260813_v1/"
    "retrospective_discovery_score.json"
)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_model_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != "pinned_huggingface_file_manifests.v1":
        raise ValueError("unsupported multimodal model manifest")
    expected = {spec["model_id"]: spec for spec in MODEL_SPECS}
    models = manifest.get("models")
    if not isinstance(models, list) or {
        model.get("model_id") for model in models if isinstance(model, Mapping)
    } != set(expected):
        raise ValueError("multimodal model manifest allowlist drifted")
    for model in models:
        spec = expected[model["model_id"]]
        if (model.get("repository"), model.get("commit")) != (
            spec["hf_repository"],
            spec["checkpoint_commit"],
        ):
            raise ValueError("multimodal model source drifted")
        files = model.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("multimodal model file manifest is empty")
        names: set[str] = set()
        for entry in files:
            if not isinstance(entry, list) or len(entry) != 4:
                raise ValueError("malformed multimodal model file entry")
            name, size, algorithm, digest = entry
            relative = PurePosixPath(name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or name in names
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
                or algorithm not in {"sha256", "git_blob_sha1"}
                or not isinstance(digest, str)
                or len(digest) not in {40, 64}
            ):
                raise ValueError("invalid multimodal model file declaration")
            if len(digest) != (64 if algorithm == "sha256" else 40):
                raise ValueError("multimodal model file digest length mismatch")
            names.add(name)
        if "config.json" not in names or not any(
            name.endswith(".safetensors") for name in names
        ):
            raise ValueError("multimodal model manifest lacks runtime files")


def build_prospective_multimodal_freeze(root: Path) -> dict[str, Any]:
    """Build the immutable response-free freeze from repository inputs."""

    plan = read_json_object(root / PLAN_PATH)
    if plan != build_multimodal_prospective_plan(root):
        raise ValueError("prospective multimodal plan does not replay exactly")
    manifest = read_json_object(root / MODEL_MANIFEST_PATH)
    _validate_model_manifest(manifest)
    discovery_score = verify_envelope(
        root / DISCOVERY_SCORE_PATH, require_blinded=False
    )
    if discovery_score["selected_primary_model_id_for_future_freeze"] != (
        "qwen3_8b_generic"
    ):
        raise ValueError("discovery-selected text model drifted")
    assets = sorted(
        {
            (call["asset_path"], call["asset_sha256"])
            for call in plan["calls"]
            if call["asset_path"] is not None
        }
    )
    for relative, expected_hash in assets:
        if sha256_file(root / relative) != expected_hash:
            raise ValueError("prospective multimodal asset hash mismatch")
    implementation_paths = (
        "src/intervenebench/simulators.py",
        "src/intervenebench/multimodal_prospective.py",
        "src/intervenebench/multimodal_freeze.py",
        "scripts/build_multimodal_prospective_plan.py",
        "scripts/build_multimodal_prospective_freeze.py",
        "infra/modal/prospective_multimodal_app.py",
        "scripts/authorize_prospective_multimodal.py",
        "scripts/run_prospective_multimodal.py",
    )
    payload = {
        "schema_version": "prospective_multimodal_freeze.v4",
        "freeze_id": "intervenebench-prospective-multimodal-20260813-v4",
        "freeze_date": "2026-08-13",
        "status": "frozen_nonexecuting_zero_authority",
        "evidence_tier": "prospective_development_outcome_sealed",
        "experiment_ids": list(EXPERIMENT_IDS),
        "plan": {
            "path": PLAN_PATH.as_posix(),
            "file_sha256": sha256_file(root / PLAN_PATH),
            "payload_sha256": payload_hash(plan),
            "logical_call_count": 54,
            "vision_call_count": 36,
            "text_ablation_call_count": 18,
        },
        "models": list(MODEL_SPECS),
        "model_manifest": {
            "path": MODEL_MANIFEST_PATH.as_posix(),
            "file_sha256": sha256_file(root / MODEL_MANIFEST_PATH),
            "payload_sha256": payload_hash(manifest),
        },
        "discovery_selection_basis": {
            "path": DISCOVERY_SCORE_PATH.as_posix(),
            "envelope_payload_sha256": payload_hash(discovery_score),
            "selected_text_model_id": "qwen3_8b_generic",
            "transfer_boundary": (
                "The revealed text-only result selects the Qwen3 family baseline, "
                "not a vision model's prospective accuracy. The target VLM choice "
                "is frozen before all three target outcomes."
            ),
        },
        "method": {
            "method_id": "forced_choice_next_token_softmax.v1",
            "generation_calls": 0,
            "temperature": 1.0,
            "orders": ["source", "reverse"],
            "aggregation_formula": (
                "p_balanced(y|arm)=0.5*p_source(y|arm)+0.5*p_reverse(y|arm)"
            ),
            "recommendation_rule": (
                "maximize balanced expected normalized utility over all source arms; "
                "tie by source arm order"
            ),
            "primary_model_id": "qwen3_vl_8b_primary",
            "comparator_model_id": "qwen2_5_vl_7b_comparator",
            "text_ablation_model_id": "qwen3_8b_text_ablation",
            "arbitrary_permutation_invariance_claimed": False,
            "repair_allowed": False,
            "semantic_retry_allowed": False,
            "transport_retry_allowed": False,
        },
        "diagnostics": {
            "frozen_before_target_outcomes": True,
            "primary": [
                "primary_model_balanced_winner_margin",
                "primary_model_source_reverse_choice_stability",
                "primary_model_mean_arm_source_reverse_total_variation",
                "two_vlm_complete_action_choice_agreement",
                "vision_vs_accessible_text_choice_agreement",
            ],
            "secondary": [
                "primary_model_chosen_arm_normalized_response_entropy",
                "per_arm_two_vlm_expected_utility_dispersion",
            ],
            "directions": {
                "primary_model_balanced_winner_margin": "larger_hypothesized_more_reliable",
                "primary_model_source_reverse_choice_stability": "stable_hypothesized_more_reliable",
                "primary_model_mean_arm_source_reverse_total_variation": "smaller_hypothesized_more_reliable",
                "two_vlm_complete_action_choice_agreement": "agreement_hypothesized_more_reliable",
                "vision_vs_accessible_text_choice_agreement": "agreement_hypothesized_more_reliable",
                "primary_model_chosen_arm_normalized_response_entropy": "smaller_hypothesized_more_reliable",
                "per_arm_two_vlm_expected_utility_dispersion": "smaller_hypothesized_more_reliable",
            },
            "evaluation": (
                "continuous regret ranking and fixed risk-coverage only; no classifier, "
                "calibration, or threshold claim from three experiments"
            ),
            "model_rows_do_not_inflate_experiment_count": True,
        },
        "future_score_fields": [
            "human_arm_mean_normalized_utility",
            "human_treatment_effect_vs_frozen_reference",
            "synthetic_arm_mean_normalized_utility",
            "synthetic_treatment_effect_vs_frozen_reference",
            "treatment_effect_absolute_error",
            "human_best_arm",
            "synthetic_selected_arm",
            "correct_intervention_choice",
            "decision_regret",
            "practically_reliable_at_contract_tolerance",
            "outcome_free_diagnostics",
        ],
        "runtime": {
            "python": "3.11",
            "torch": "2.9.1",
            "torchvision": "0.24.1",
            "transformers": "4.57.6",
            "pillow": "11.3.0",
            "gpu": "NVIDIA L40S:1",
            "trust_remote_code": False,
            "network_during_inference": False,
            "dependency_input_path": LOCK_INPUT_PATH.as_posix(),
            "dependency_input_sha256": sha256_file(root / LOCK_INPUT_PATH),
            "dependency_lock_path": LOCK_PATH.as_posix(),
            "dependency_lock_sha256": sha256_file(root / LOCK_PATH),
            "image_recipe": {
                "base": "modal.Image.debian_slim",
                "python_version": "3.11",
                "installer": "uv_pip_install",
                "uv_version": "0.12.4",
                "require_hashes": True,
                "lock_path": LOCK_PATH.as_posix(),
                "embedded_files": [
                    MODEL_MANIFEST_PATH.as_posix(),
                    "configs/simulators/prospective_multimodal_v4.json",
                    LOCK_PATH.as_posix(),
                    "infra/modal/prospective_multimodal_app.py",
                ],
                "embedded_stimuli_root": "data/derived/stimuli",
                "modal_volume": "intervenebench-model-cache-v1",
            },
        },
        "assets": [
            {"path": relative, "file_sha256": digest}
            for relative, digest in assets
        ],
        "implementation_hashes": [
            {"path": path, "file_sha256": sha256_file(root / path)}
            for path in implementation_paths
        ],
        "limits": {
            "maximum_planned_calls": 54,
            "maximum_model_attempts": 54,
            "maximum_model_loads": 3,
            "maximum_aggregate_gpu_seconds": 3600,
            "l40s_price_per_second_usd": 0.000542,
            "maximum_gpu_cost_usd": 1.9512,
            "hard_incremental_execution_cap_usd": 5.0,
            "automatic_retries": 0,
        },
        "authority": {
            "model_download_authorized": False,
            "image_materialization_authorized": False,
            "modal_execution_authorized": False,
            "paid_inference_authorized": False,
            "human_outcome_access_authorized": False,
            "outcome_reveal_authorized": False,
            "trust_threshold_selection_authorized": False,
            "automatic_next_stage_authorized": False,
        },
        "claim_boundary": (
            "The three target experiments are prospective development evidence, not "
            "a canonical held-out test. Vision-model choice, prompts, orders, action "
            "sets, and diagnostics are frozen before target human outcomes."
        ),
    }
    assert_blinded_payload(payload)
    return payload


def verify_prospective_multimodal_freeze(
    root: Path, freeze: Mapping[str, Any]
) -> dict[str, Any]:
    expected = build_prospective_multimodal_freeze(root)
    if dict(freeze) != expected:
        raise ValueError("prospective multimodal freeze does not replay exactly")
    if freeze["status"] != "frozen_nonexecuting_zero_authority" or any(
        freeze["authority"].values()
    ):
        raise PermissionError("prospective multimodal freeze expands authority")
    return {
        "experiment_count": len(freeze["experiment_ids"]),
        "model_count": len(freeze["models"]),
        "call_count": freeze["plan"]["logical_call_count"],
        "maximum_incremental_execution_usd": freeze["limits"][
            "hard_incremental_execution_cap_usd"
        ],
        "freeze_payload_sha256": payload_hash(freeze),
    }


def prepare_multimodal_requests(root: Path) -> tuple[dict[str, Any], ...]:
    """Reconstruct the exact prompt and asset for each frozen logical call."""

    plan = read_json_object(root / PLAN_PATH)
    if plan != build_multimodal_prospective_plan(root):
        raise ValueError("prospective multimodal plan does not replay exactly")
    requests: list[dict[str, Any]] = []
    for call in plan["calls"]:
        bundle = read_json_object(
            root
            / f"data/manifests/contracts/{call['experiment_id']}_blinded_bundle.json"
        )
        prompt = multimodal_forced_choice_prompt(
            bundle,
            arm_id=call["arm_id"],
            repository_root=root,
            option_order=call["option_order"],
            include_exact_image=call["modality"] == "exact_png_vision",
        )
        if sha256(prompt.text.encode("utf-8")).hexdigest() != call["prompt_sha256"]:
            raise ValueError("prospective multimodal prompt hash drifted")
        asset_path = (
            str(Path(prompt.asset_path).resolve().relative_to(root.resolve()))
            if prompt.asset_path is not None
            else None
        )
        if (asset_path, prompt.asset_sha256) != (
            call["asset_path"],
            call["asset_sha256"],
        ):
            raise ValueError("prospective multimodal asset binding drifted")
        requests.append({**call, "prompt": prompt.text})
    assert_blinded_payload(requests)
    return tuple(requests)


def build_materialization_authorization(
    *, freeze: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": "prospective_multimodal_materialization_authorization.v1",
        "authorization_id": "intervenebench-prospective-mm-materialize-20260813-v1",
        "scope": "materialize_exact_pinned_multimodal_image_zero_inference",
        "freeze_payload_sha256": payload_hash(freeze),
        "plan_payload_sha256": payload_hash(plan),
        "modal_profile": "yanav",
        "image_materialization_authorized": True,
        "model_download_authorized": False,
        "paid_inference_authorized": False,
        "human_outcome_access_authorized": False,
        "outcome_reveal_authorized": False,
        "automatic_next_stage_authorized": False,
        "maximum_incremental_execution_usd": 5.0,
        "status": "multimodal_image_only_authorized",
    }
    assert_blinded_payload(payload)
    return payload


def validate_materialization_authorization(
    authorization: Mapping[str, Any],
    *,
    freeze: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    if dict(authorization) != build_materialization_authorization(
        freeze=freeze, plan=plan
    ):
        raise PermissionError("multimodal materialization authorization mismatch")


def build_cache_authorization(
    *,
    freeze: Mapping[str, Any],
    plan: Mapping[str, Any],
    modal_image_id: str,
) -> dict[str, Any]:
    if not modal_image_id:
        raise ValueError("multimodal cache authorization requires an image ID")
    payload = {
        "schema_version": "prospective_multimodal_cache_authorization.v1",
        "authorization_id": "intervenebench-prospective-mm-cache-20260813-v1",
        "scope": "download_two_exact_public_vlm_revisions_cpu_only",
        "freeze_payload_sha256": payload_hash(freeze),
        "plan_payload_sha256": payload_hash(plan),
        "modal_profile": "yanav",
        "modal_image_id": modal_image_id,
        "download_model_ids": [
            "qwen3_vl_8b_primary",
            "qwen2_5_vl_7b_comparator",
        ],
        "reuse_cached_model_ids": ["qwen3_8b_text_ablation"],
        "model_download_authorized": True,
        "image_materialization_authorized": False,
        "paid_gpu_inference_authorized": False,
        "human_outcome_access_authorized": False,
        "outcome_reveal_authorized": False,
        "automatic_next_stage_authorized": False,
        "maximum_incremental_execution_usd": 5.0,
        "status": "two_multimodal_checkpoint_downloads_authorized",
    }
    assert_blinded_payload(payload)
    return payload


def validate_cache_authorization(
    authorization: Mapping[str, Any],
    *,
    freeze: Mapping[str, Any],
    plan: Mapping[str, Any],
    modal_image_id: str,
) -> None:
    if dict(authorization) != build_cache_authorization(
        freeze=freeze, plan=plan, modal_image_id=modal_image_id
    ):
        raise PermissionError("multimodal cache authorization mismatch")


def build_execution_authorization(
    *,
    freeze: Mapping[str, Any],
    plan: Mapping[str, Any],
    modal_image_id: str,
    cache_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if not modal_image_id or set(cache_hashes) != set(MODEL_IDS):
        raise ValueError("multimodal execution cache binding is incomplete")
    payload = {
        "schema_version": "prospective_multimodal_execution_authorization.v1",
        "authorization_id": "intervenebench-prospective-mm-execute-20260813-v1",
        "scope": "exact_54_call_outcome_sealed_prospective_development_run",
        "freeze_payload_sha256": payload_hash(freeze),
        "plan_payload_sha256": payload_hash(plan),
        "modal_profile": "yanav",
        "modal_image_id": modal_image_id,
        "cache_attestation_sha256_by_model": dict(sorted(cache_hashes.items())),
        "model_download_authorized": False,
        "modal_execution_authorized": True,
        "paid_inference_authorized": True,
        "human_outcome_access_authorized": False,
        "outcome_reveal_authorized": False,
        "automatic_next_stage_authorized": False,
        "maximum_planned_calls": 54,
        "maximum_model_attempts": 54,
        "maximum_model_loads": 3,
        "maximum_aggregate_gpu_seconds": 3600,
        "maximum_incremental_execution_usd": 5.0,
        "status": "single_prospective_multimodal_run_authorized",
    }
    assert_blinded_payload(payload)
    return payload


def validate_execution_authorization(
    authorization: Mapping[str, Any],
    *,
    freeze: Mapping[str, Any],
    plan: Mapping[str, Any],
    modal_image_id: str,
    cache_hashes: Mapping[str, str],
) -> None:
    if dict(authorization) != build_execution_authorization(
        freeze=freeze,
        plan=plan,
        modal_image_id=modal_image_id,
        cache_hashes=cache_hashes,
    ):
        raise PermissionError("multimodal execution authorization mismatch")


def validate_runtime_attestation(
    request: Mapping[str, Any],
    attestation: Mapping[str, Any],
    *,
    freeze: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> None:
    model = {item["model_id"]: item for item in freeze["models"]}[
        request["model_id"]
    ]
    expected = {
        "modal_sdk_version": "1.5.4",
        "modal_image_id": authorization["modal_image_id"],
        "image_recipe_sha256": payload_hash(freeze["runtime"]["image_recipe"]),
        "dependency_lock_sha256": freeze["runtime"]["dependency_lock_sha256"],
        "transformers_version": "4.57.6",
        "torchvision_version": "0.24.1",
        "pillow_version": "11.3.0",
        "cuda_runtime_version": "12.8",
        "checkpoint_commit": model["checkpoint_commit"],
        "model_file_manifest_payload_sha256": freeze["model_manifest"][
            "payload_sha256"
        ],
        "cache_attestation_sha256": authorization[
            "cache_attestation_sha256_by_model"
        ][request["model_id"]],
        "call_id": request["call_id"],
        "prompt_sha256": request["prompt_sha256"],
        "asset_sha256": request["asset_sha256"],
        "method_id": "forced_choice_next_token_softmax.v1",
    }
    for field, value in expected.items():
        if attestation.get(field) != value:
            raise ValueError(f"multimodal runtime mismatch: {field}")
    if str(attestation.get("torch_version", "")).split("+")[0] != "2.9.1":
        raise ValueError("multimodal runtime mismatch: torch_version")
    if str(attestation.get("python_version", "")).split(".")[:2] != ["3", "11"]:
        raise ValueError("multimodal runtime mismatch: python_version")
    if "L40S" not in str(attestation.get("gpu_name", "")):
        raise ValueError("multimodal runtime mismatch: gpu_name")


def verify_multimodal_raw_result(
    request: Mapping[str, Any], raw: Mapping[str, Any]
) -> dict[str, Any]:
    """Map display-code probabilities back to frozen source response values."""

    if raw.get("call_id") != request["call_id"] or raw.get("model_id") != request[
        "model_id"
    ]:
        raise ValueError("multimodal result identity mismatch")
    probabilities_by_code = raw.get("probabilities_by_code")
    codes = request["answer_codes"]
    if not isinstance(probabilities_by_code, Mapping) or set(
        probabilities_by_code
    ) != set(codes):
        raise ValueError("multimodal result answer support mismatch")
    probabilities = {
        int(value): float(probabilities_by_code[code])
        for code, value in zip(codes, request["display_option_values"])
    }
    if any(value < 0.0 or value > 1.0 for value in probabilities.values()) or abs(
        sum(probabilities.values()) - 1.0
    ) > 1e-6:
        raise ValueError("multimodal probabilities are not normalized")
    payload = {
        "schema_version": "prospective_multimodal_call_output.v1",
        "call_id": request["call_id"],
        "model_id": request["model_id"],
        "experiment_id": request["experiment_id"],
        "arm_id": request["arm_id"],
        "option_order": request["option_order"],
        "modality": request["modality"],
        "prompt_sha256": request["prompt_sha256"],
        "asset_sha256": request["asset_sha256"],
        "probabilities": probabilities,
        "candidate_token_ids": raw["candidate_token_ids"],
        "candidate_token_strings": raw["candidate_token_strings"],
        "runtime_attestation": raw["runtime_attestation"],
        "outcome_access": "not_accessed",
        "status": "strict_forced_choice_output",
    }
    assert_blinded_payload(payload)
    return payload
