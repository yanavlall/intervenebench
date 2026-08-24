"""Offline validation for the non-executing Modal discovery preflight freeze."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .protocol import assert_blinded_payload, canonical_json_bytes, payload_hash
from .simulators import ordinal_probability_prompt, ordinal_variant_contract


EXPERIMENT_IDS = ("5vm8g", "xc4yq", "de5hx", "turagaS11", "wallaceS12")
AUTHORITY_FIELDS = (
    "modal_execution_authorized",
    "model_download_authorized",
    "paid_inference_authorized",
    "sealed_task_inference_authorized",
    "outcome_access_authorized",
    "fine_tuning_authorized",
)
MODEL_REQUIRED_FIELDS = frozenset(
    {
        "model_id",
        "hf_repository",
        "checkpoint_commit",
        "weight_file_manifest_sha256",
        "tokenizer_manifest_sha256",
        "chat_template_sha256",
        "config_sha256",
        "license_id",
        "repository_private",
        "repository_gated",
        "trust_remote_code",
        "dtype",
        "quantization",
        "thinking_mode",
        "maximum_context_tokens",
        "exposure_by_experiment",
    }
)


@dataclass(frozen=True, slots=True)
class ModalPreflightSummary:
    model_count: int
    call_count: int
    calls_per_model: int
    minimum_parse_successes: int
    maximum_gpu_container_seconds: int
    maximum_gpu_cost_usd: float
    hard_total_cost_cap_usd: float


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _require_digest(value: Any, *, field: str, length: int = 64) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{field} must be an immutable {length}-character hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be an immutable hex digest") from error
    return value


def _require_exact_package(package: Any) -> None:
    if not isinstance(package, str) or package.count("==") != 1:
        raise ValueError("runtime packages must use exact name==version pins")
    name, version = package.split("==", 1)
    if not name or not version or any(token in package for token in (">", "<", "~=", "*")):
        raise ValueError("runtime packages must use exact name==version pins")


def _validate_models(freeze: Mapping[str, Any]) -> tuple[str, ...]:
    models = freeze.get("models")
    if not isinstance(models, list) or len(models) != 4:
        raise ValueError("freeze must contain exactly four models")
    model_ids: list[str] = []
    for model in models:
        if not isinstance(model, Mapping) or set(model) != MODEL_REQUIRED_FIELDS:
            raise ValueError("model freeze fields do not match the required schema")
        model_id = model.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id must be non-empty")
        model_ids.append(model_id)
        _require_digest(
            model.get("checkpoint_commit"), field="checkpoint_commit", length=40
        )
        for field in (
            "weight_file_manifest_sha256",
            "tokenizer_manifest_sha256",
            "chat_template_sha256",
            "config_sha256",
        ):
            _require_digest(model.get(field), field=field)
        if model.get("license_id") != "apache-2.0":
            raise ValueError("the frozen model license must be Apache-2.0")
        if model.get("repository_private") is not False or model.get(
            "repository_gated"
        ) is not False:
            raise ValueError("the preflight models must be public and ungated")
        if model.get("trust_remote_code") is not False:
            raise ValueError("trust_remote_code must remain false")
        if model.get("dtype") != "bfloat16" or model.get("quantization") != "none":
            raise ValueError("model precision must remain frozen to unquantized BF16")
        exposure = model.get("exposure_by_experiment")
        if not isinstance(exposure, Mapping) or tuple(exposure) != EXPERIMENT_IDS:
            raise ValueError("model exposure must be frozen for every discovery task")
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("model IDs must be unique")
    return tuple(model_ids)


def _validate_runtime(freeze: Mapping[str, Any]) -> None:
    runtime = freeze.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("runtime must be an object")
    if runtime.get("provider") != "modal" or runtime.get("gpu_type") != "L40S":
        raise ValueError("runtime must freeze Modal on one L40S")
    if runtime.get("gpu_count") != 1 or runtime.get("gpu_fallback_allowed") is not False:
        raise ValueError("GPU count/fallback must remain one L40S with no fallback")
    if (
        runtime.get("worker_shape") != "one_model_group_per_function_input"
        or
        runtime.get("maximum_containers_per_parameterized_model") != 1
        or runtime.get("maximum_total_model_containers") != 4
        or runtime.get("input_concurrency") != 1
    ):
        raise ValueError("runtime concurrency differs from the frozen preflight")
    if runtime.get("python_version") != "3.11":
        raise ValueError("runtime Python must remain 3.11")
    if runtime.get("expected_cuda_runtime_version") != "12.8":
        raise ValueError("runtime CUDA version must remain 12.8")
    for package in runtime.get("packages", ()):
        _require_exact_package(package)
    if not runtime.get("packages"):
        raise ValueError("runtime package pins are required")
    recipe = runtime.get("image_recipe")
    if not isinstance(recipe, Mapping) or recipe.get("base") != "debian_slim":
        raise ValueError("runtime image recipe must be frozen")
    if runtime.get("modal_image_identity_status") != (
        "materialize_modal_image_id_and_attest_before_dispatch"
    ):
        raise ValueError("runtime must require a Modal image attestation before dispatch")


def _bundle_cells(root: Path, freeze: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    task_scope = freeze.get("task_scope")
    if not isinstance(task_scope, Mapping):
        raise ValueError("task_scope must be an object")
    if tuple(task_scope.get("experiment_ids", ())) != EXPERIMENT_IDS:
        raise ValueError("freeze must use the exact task allowlist")
    if task_scope.get("all_unlisted_experiments_denied") is not True:
        raise ValueError("all unlisted experiments must be denied")
    files = task_scope.get("packaged_files")
    if not isinstance(files, list) or len(files) != len(EXPERIMENT_IDS):
        raise ValueError("packaged_files must contain one blinded bundle per task")
    by_id: dict[str, dict[str, Any]] = {}
    for entry in files:
        if not isinstance(entry, Mapping) or set(entry) != {
            "experiment_id",
            "path",
            "file_sha256",
            "payload_sha256",
        }:
            raise ValueError("packaged file entry is malformed")
        experiment_id = entry["experiment_id"]
        if experiment_id not in EXPERIMENT_IDS:
            raise ValueError("packaged file is outside the exact task allowlist")
        relative = PurePosixPath(str(entry["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("packaged file path escapes the repository")
        path = root / relative
        if _sha256_file(path) != entry["file_sha256"]:
            raise ValueError("packaged blinded-bundle file hash mismatch")
        bundle = _read_object(path)
        assert_blinded_payload(bundle)
        if payload_hash(bundle) != entry["payload_sha256"]:
            raise ValueError("packaged blinded-bundle payload hash mismatch")
        if bundle.get("experiment_id") != experiment_id:
            raise ValueError("packaged bundle experiment mismatch")
        if bundle.get("access_regime") != "DESIGN_ONLY":
            raise ValueError("packaged bundle is not DESIGN_ONLY")
        if bundle.get("outcome_access") != "sealed" or bundle.get(
            "reveal_authorized"
        ) is not False:
            raise ValueError("packaged bundle is not outcome sealed")
        by_id[experiment_id] = bundle
    if tuple(by_id) != EXPERIMENT_IDS:
        raise ValueError("packaged bundle order differs from the exact task allowlist")
    return by_id


def _validate_parent_and_implementation_hashes(
    root: Path, freeze: Mapping[str, Any]
) -> None:
    parents = freeze.get("parent_hashes")
    expected_parents = {
        "development_config_payload_sha256": (
            "configs/simulators/development_v1.json"
        ),
        "development_scope_payload_sha256": (
            "data/manifests/benchmark/simulator_development_scope.json"
        ),
    }
    if not isinstance(parents, Mapping) or set(parents) != set(expected_parents):
        raise ValueError("parent hashes do not match the required schema")
    for field, relative in expected_parents.items():
        payload = _read_object(root / relative)
        if payload_hash(payload) != parents[field]:
            raise ValueError(f"parent payload hash mismatch: {field}")

    implementations = freeze.get("implementation_hashes")
    if not isinstance(implementations, list) or not implementations:
        raise ValueError("implementation hashes are required")
    for entry in implementations:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "file_sha256"}:
            raise ValueError("implementation hash entry is malformed")
        relative = PurePosixPath(str(entry["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("implementation path escapes the repository")
        if _sha256_file(root / relative) != entry["file_sha256"]:
            raise ValueError(f"implementation file hash mismatch: {relative}")


def _validate_model_file_manifests(root: Path, freeze: Mapping[str, Any]) -> None:
    reference = freeze.get("model_file_manifest")
    if not isinstance(reference, Mapping) or set(reference) != {
        "path",
        "payload_sha256",
    }:
        raise ValueError("model file manifest reference is malformed")
    relative = PurePosixPath(str(reference["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("model file manifest path escapes the repository")
    manifest = _read_object(root / relative)
    if payload_hash(manifest) != reference["payload_sha256"]:
        raise ValueError("model file manifest payload hash mismatch")
    if manifest.get("schema_version") != "pinned_huggingface_file_manifests.v1":
        raise ValueError("unsupported model file manifest schema")
    entries = manifest.get("models")
    if not isinstance(entries, list) or len(entries) != 4:
        raise ValueError("model file manifest must contain exactly four models")
    frozen = {model["model_id"]: model for model in freeze["models"]}
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {
            "model_id",
            "repository",
            "commit",
            "files",
        }:
            raise ValueError("model file manifest entry is malformed")
        model = frozen.get(entry["model_id"])
        if model is None:
            raise ValueError("model file manifest contains an unknown model")
        if entry["repository"] != model["hf_repository"] or entry["commit"] != model[
            "checkpoint_commit"
        ]:
            raise ValueError("model file manifest identity mismatch")
        files = entry["files"]
        if not isinstance(files, list) or not files:
            raise ValueError("model file manifest has no files")
        seen: set[str] = set()
        for file_entry in files:
            if not isinstance(file_entry, list) or len(file_entry) != 4:
                raise ValueError("model file hash entry is malformed")
            path, size, algorithm, digest = file_entry
            if (
                not isinstance(path, str)
                or not path
                or path in seen
                or PurePosixPath(path).is_absolute()
                or ".." in PurePosixPath(path).parts
            ):
                raise ValueError("model file path is invalid or duplicated")
            seen.add(path)
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError("model file size must be a positive integer")
            if algorithm not in {"sha256", "git_blob_sha1"}:
                raise ValueError("model file hash algorithm is unsupported")
            _require_digest(
                digest,
                field="model file digest",
                length=64 if algorithm == "sha256" else 40,
            )
        if not any(str(path).endswith(".safetensors") for path, *_ in files):
            raise ValueError("model file manifest has no safetensors weights")


def _validate_dependency_lock(root: Path, freeze: Mapping[str, Any]) -> None:
    lock = freeze.get("dependency_lock")
    required = {
        "input_path",
        "input_file_sha256",
        "lock_path",
        "lock_file_sha256",
        "resolver",
        "target",
    }
    if not isinstance(lock, Mapping) or set(lock) != required:
        raise ValueError("dependency lock declaration is malformed")
    for path_field, hash_field in (
        ("input_path", "input_file_sha256"),
        ("lock_path", "lock_file_sha256"),
    ):
        relative = PurePosixPath(str(lock[path_field]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("dependency lock path escapes the repository")
        if _sha256_file(root / relative) != lock[hash_field]:
            raise ValueError("dependency lock file hash mismatch")
    if lock["resolver"] != "uv==0.12.4" or lock["target"] != (
        "CPython 3.11 x86_64 manylinux_2_28"
    ):
        raise ValueError("dependency lock resolver or target drifted")


def _expected_call_fields() -> set[str]:
    return {
        "call_id",
        "model_id",
        "experiment_id",
        "bundle_payload_sha256",
        "arm_id",
        "variant_id",
        "prompt_sha256",
        "json_schema_sha256",
        "parser_id",
        "seed",
        "temperature",
        "top_p",
        "maximum_output_tokens",
        "artifact_relative_path",
    }


def _validate_calls(
    *,
    freeze: Mapping[str, Any],
    call_plan: Mapping[str, Any],
    bundles: Mapping[str, Mapping[str, Any]],
    model_ids: tuple[str, ...],
) -> tuple[int, int]:
    preflight = freeze.get("preflight")
    generation = freeze.get("generation")
    if not isinstance(preflight, Mapping) or not isinstance(generation, Mapping):
        raise ValueError("preflight and generation must be objects")
    if call_plan.get("schema_version") != "modal_preflight_call_plan.v1":
        raise ValueError("unsupported preflight call-plan schema")
    if tuple(call_plan.get("experiment_ids", ())) != EXPERIMENT_IDS:
        raise ValueError("call plan violates the exact task allowlist")
    if tuple(call_plan.get("model_ids", ())) != model_ids:
        raise ValueError("call-plan model order differs from the freeze")
    calls = call_plan.get("calls")
    expected_calls = int(preflight.get("calls_per_model", 0)) * len(model_ids)
    if not isinstance(calls, list) or len(calls) != expected_calls:
        raise ValueError("call plan has the wrong number of calls")
    seen_ids: set[str] = set()
    counts = {model_id: 0 for model_id in model_ids}
    model_task_arms: dict[tuple[str, str], set[str]] = {}
    for call in calls:
        if not isinstance(call, Mapping) or set(call) != _expected_call_fields():
            raise ValueError("preflight call fields do not match the required schema")
        call_id = str(call["call_id"])
        if call_id in seen_ids:
            raise ValueError("duplicate preflight call ID")
        seen_ids.add(call_id)
        model_id = str(call["model_id"])
        experiment_id = str(call["experiment_id"])
        if model_id not in counts or experiment_id not in EXPERIMENT_IDS:
            raise ValueError("preflight call violates the exact task allowlist")
        bundle = bundles[experiment_id]
        if call["bundle_payload_sha256"] != payload_hash(bundle):
            raise ValueError("preflight call bundle hash mismatch")
        arm_id = str(call["arm_id"])
        variant_id = str(call["variant_id"])
        prompt = ordinal_probability_prompt(
            bundle, arm_id=arm_id, variant_id=variant_id
        )
        if call["prompt_sha256"] != sha256(prompt.encode("utf-8")).hexdigest():
            raise ValueError("preflight prompt hash mismatch")
        if call["json_schema_sha256"] != generation.get("json_schema_sha256"):
            raise ValueError("preflight JSON schema hash mismatch")
        if call["parser_id"] != generation.get("parser_id"):
            raise ValueError("preflight parser mismatch")
        for field in ("temperature", "top_p", "maximum_output_tokens"):
            if call[field] != generation.get(field):
                raise ValueError(f"preflight {field} differs from the freeze")
        artifact = PurePosixPath(str(call["artifact_relative_path"]))
        if artifact.is_absolute() or ".." in artifact.parts:
            raise ValueError("preflight artifact path escapes its root")
        counts[model_id] += 1
        model_task_arms.setdefault((model_id, experiment_id), set()).add(arm_id)
    calls_per_model = int(preflight.get("calls_per_model", 0))
    if any(count != calls_per_model for count in counts.values()):
        raise ValueError("preflight calls are not balanced across models")
    if any(
        len(model_task_arms.get((model_id, experiment_id), set())) != 2
        for model_id in model_ids
        for experiment_id in EXPERIMENT_IDS
    ):
        raise ValueError("preflight must cover two arms from every task per model")
    return len(calls), calls_per_model


def verify_modal_preflight_freeze(
    root: Path,
    *,
    freeze_path: Path,
    call_plan_path: Path,
    freeze_override: Mapping[str, Any] | None = None,
    call_plan_override: Mapping[str, Any] | None = None,
) -> ModalPreflightSummary:
    """Verify a hash-bound, zero-authority discovery preflight without importing Modal."""

    freeze = dict(freeze_override) if freeze_override is not None else _read_object(freeze_path)
    call_plan = (
        dict(call_plan_override)
        if call_plan_override is not None
        else _read_object(call_plan_path)
    )
    assert_blinded_payload(freeze)
    assert_blinded_payload(call_plan)
    if freeze.get("schema_version") not in {
        "modal_discovery_preflight_freeze.v1",
        "modal_discovery_preflight_freeze.v2",
    }:
        raise ValueError("unsupported Modal preflight freeze schema")
    if freeze.get("status") != "frozen_nonexecuting_zero_authority":
        raise ValueError("Modal preflight must remain a non-executing freeze")
    authority = freeze.get("authority")
    if not isinstance(authority, Mapping) or set(authority) != set(AUTHORITY_FIELDS):
        raise ValueError("Modal preflight authority fields are incomplete")
    if any(authority[field] is not False for field in AUTHORITY_FIELDS):
        raise ValueError("Modal preflight authority must remain entirely false")
    model_ids = _validate_models(freeze)
    _validate_runtime(freeze)
    _validate_parent_and_implementation_hashes(root, freeze)
    _validate_model_file_manifests(root, freeze)
    _validate_dependency_lock(root, freeze)
    bundles = _bundle_cells(root, freeze)

    generation = freeze.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("generation must be an object")
    if generation.get("malformed_output_retry_count") != 0:
        raise ValueError("malformed outputs must never be semantically retried")
    if generation.get("identical_transport_retry_limit") != 1:
        raise ValueError("transport retry limit must remain exactly one identical retry")
    schema = generation.get("json_schema")
    if not isinstance(schema, Mapping) or payload_hash(schema) != generation.get(
        "json_schema_sha256"
    ):
        raise ValueError("generation JSON schema hash mismatch")

    expected_relative = str(call_plan_path.relative_to(root))
    preflight = freeze.get("preflight")
    if not isinstance(preflight, Mapping) or preflight.get("call_plan_path") != expected_relative:
        raise ValueError("freeze references a different preflight call plan")
    if (
        call_plan_override is None
        and payload_hash(call_plan) != preflight.get("call_plan_payload_sha256")
    ):
        raise ValueError("preflight call plan payload hash mismatch")
    call_count, calls_per_model = _validate_calls(
        freeze=freeze, call_plan=call_plan, bundles=bundles, model_ids=model_ids
    )

    limits = freeze.get("limits")
    if not isinstance(limits, Mapping):
        raise ValueError("limits must be an object")
    seconds = limits.get("maximum_gpu_container_seconds")
    rate = limits.get("frozen_gpu_usd_per_second")
    reserve = limits.get("startup_and_ancillary_reserve_usd")
    prior_reserve = limits.get("prior_closed_v1_reserve_usd", 0.0)
    cap = limits.get("hard_total_cost_cap_usd")
    for field, value in (
        ("maximum_gpu_container_seconds", seconds),
        ("frozen_gpu_usd_per_second", rate),
        ("startup_and_ancillary_reserve_usd", reserve),
        ("hard_total_cost_cap_usd", cap),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(f"{field} must be finite and positive")
    if (
        isinstance(prior_reserve, bool)
        or not isinstance(prior_reserve, (int, float))
        or not isfinite(float(prior_reserve))
        or float(prior_reserve) < 0
    ):
        raise ValueError("prior_closed_v1_reserve_usd must be finite and non-negative")
    gpu_cost = float(seconds) * float(rate)
    if gpu_cost + float(reserve) + float(prior_reserve) > float(cap) + 1e-9:
        raise ValueError("preflight limits exceed the hard total cost cap")
    if limits.get("maximum_model_attempts") != call_count + 4:
        raise ValueError("attempt ceiling must include at most one transport retry per model")
    if limits.get("abort_before_dispatch_when_next_call_exceeds_budget") is not True:
        raise ValueError("budget must be checked before every dispatch")
    required = int(preflight.get("required_parse_successes_per_model", 0))
    if required != calls_per_model:
        raise ValueError("preflight parse gate must require every planned call to parse")
    return ModalPreflightSummary(
        model_count=len(model_ids),
        call_count=call_count,
        calls_per_model=calls_per_model,
        minimum_parse_successes=required * len(model_ids),
        maximum_gpu_container_seconds=int(seconds),
        maximum_gpu_cost_usd=gpu_cost,
        hard_total_cost_cap_usd=float(cap),
    )


def assert_modal_execution_ready(freeze: Mapping[str, Any]) -> None:
    """Fail closed: an authorization must be a separate future artifact."""

    authority = freeze.get("authority")
    blocked = (
        list(AUTHORITY_FIELDS)
        if not isinstance(authority, Mapping)
        else [field for field in AUTHORITY_FIELDS if authority.get(field) is False]
    )
    raise PermissionError(
        "Modal preflight is a frozen non-executing package; a separate hash-bound "
        f"authorization is required (blocked={blocked})"
    )
