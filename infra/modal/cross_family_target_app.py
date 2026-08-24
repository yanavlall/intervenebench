"""Pinned Modal image and future workers for the Mistral target replay.

The image contains only outcome-blind plans, public stimuli, pinned inference
code, and dependency metadata.  It exposes no model-download function.  Its
one target-free JSON canary and 624-call target worker remain unusable until a
pure-local wrapper validates later, separately frozen authorizations.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import subprocess
import time
from typing import Any

import modal


if modal.is_local():
    ROOT = Path(__file__).resolve().parents[2]
    CONFIG_PATH = ROOT / "configs/simulators/cross_family_execution_v1.json"
    MODEL_MANIFEST_PATH = (
        ROOT
        / "data/manifests/simulators/mistral_small_3_1_24b_source_manifest_v1.json"
    )
    LOCK_PATH = ROOT / "infra/modal/cross-family-requirements.lock"
    STIMULI_PATH = ROOT / "data/derived/stimuli/pb2rr"
    SOURCE_PATH = Path(__file__)
else:
    CONFIG_PATH = Path("/opt/intervenebench/cross_family_execution_v1.json")
    MODEL_MANIFEST_PATH = Path(
        "/opt/intervenebench/mistral_small_3_1_24b_source_manifest_v1.json"
    )
    LOCK_PATH = Path("/opt/intervenebench/cross-family-requirements.lock")
    STIMULI_PATH = Path("/opt/intervenebench/data/derived/stimuli/pb2rr")
    SOURCE_PATH = Path("/root/infra/modal/cross_family_target_app.py")

REMOTE_SOURCE_PATH = Path("/root/infra/modal/cross_family_target_app.py")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _payload_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
MODEL_ENVELOPE = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
if set(MODEL_ENVELOPE) != {"payload", "sha256"}:
    raise RuntimeError("embedded Mistral model manifest is malformed")
MODEL = MODEL_ENVELOPE["payload"]
if _payload_hash(MODEL) != MODEL_ENVELOPE["sha256"]:
    raise RuntimeError("embedded Mistral model manifest hash is invalid")
if MODEL_ENVELOPE["sha256"] != CONFIG["model"]["source_manifest_payload_sha256"]:
    raise RuntimeError("target image/model source binding mismatch")
if CONFIG["authority"]["target_inference_authorized"] is not False:
    raise RuntimeError("target image config must remain zero authority")

MODEL_ROOT = Path("/models")
MODEL_PATH = (
    MODEL_ROOT
    / CONFIG["runtime"]["model_cache_subdirectory"]
    / CONFIG["model"]["checkpoint_commit"]
)
REQUEST_HASHES = CONFIG["call_plan"]["request_payload_sha256_by_call_id"]
IMAGE_RECIPE_SHA256 = CONFIG["runtime"]["image_recipe_sha256"]
DEPENDENCY_LOCK_SHA256 = CONFIG["runtime"]["dependency_lock_sha256"]

app = modal.App(CONFIG["runtime"]["app_name"], include_source=False)
model_volume = modal.Volume.from_name(
    CONFIG["runtime"]["model_volume_name"], create_if_missing=False
)
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(
        requirements=[str(LOCK_PATH)],
        extra_options="--require-hashes",
        uv_version="0.12.4",
    )
    .add_local_file(
        CONFIG_PATH,
        "/opt/intervenebench/cross_family_execution_v1.json",
        copy=True,
    )
    .add_local_file(
        MODEL_MANIFEST_PATH,
        "/opt/intervenebench/mistral_small_3_1_24b_source_manifest_v1.json",
        copy=True,
    )
    .add_local_file(
        LOCK_PATH,
        "/opt/intervenebench/cross-family-requirements.lock",
        copy=True,
    )
    .add_local_file(SOURCE_PATH, str(REMOTE_SOURCE_PATH), copy=True)
)
target_image = base_image.add_local_dir(
    STIMULI_PATH,
    "/opt/intervenebench/data/derived/stimuli/pb2rr",
    copy=True,
)


def _hash_file(path: Path, algorithm: str) -> str:
    if algorithm == "sha256":
        digest = hashlib.sha256()
        prefix = b""
    elif algorithm == "git_blob_sha1":
        digest = hashlib.sha1()
        prefix = f"blob {path.stat().st_size}\0".encode("ascii")
    else:
        raise ValueError("unsupported checkpoint digest algorithm")
    digest.update(prefix)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_checkpoint_files() -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for item in MODEL["files"]:
        path = MODEL_PATH / item["path"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"]:
            raise RuntimeError(f"cached checkpoint file mismatch: {item['path']}")
        content_sha256 = _hash_file(path, "sha256")
        if item["content_sha256"] is not None:
            if content_sha256 != item["content_sha256"]:
                raise RuntimeError(
                    f"cached checkpoint content digest mismatch: {item['path']}"
                )
            verification = "source_content_sha256"
        else:
            if _hash_file(path, "git_blob_sha1") != item["git_oid"]:
                raise RuntimeError(
                    f"cached checkpoint Git-blob digest mismatch: {item['path']}"
                )
            verification = "source_git_blob_sha1"
        verified.append(
            {
                "path": item["path"],
                "size_bytes": item["size_bytes"],
                "content_sha256": content_sha256,
                "source_verification": verification,
            }
        )
    return verified


def _load_cache_attestation(expected_sha256: str) -> dict[str, Any]:
    path = MODEL_PATH / "cache_attestation.json"
    if not path.is_file():
        raise RuntimeError("verified Mistral checkpoint cache is absent")
    attestation = json.loads(path.read_text(encoding="utf-8"))
    if attestation.get("verified_files") != _verify_checkpoint_files():
        raise RuntimeError("Mistral cache attestation/file replay failed")
    if _payload_hash(attestation) != expected_sha256:
        raise RuntimeError("Mistral cache authorization binding failed")
    if expected_sha256 != CONFIG["model"]["cache_attestation_payload_sha256"]:
        raise RuntimeError("Mistral cache differs from target freeze")
    return attestation


def _runtime_attestation(
    *, expected_image_id: str, cache_attestation: dict[str, Any]
) -> dict[str, Any]:
    import torch
    import transformers
    import vllm

    image_id = os.environ.get("MODAL_IMAGE_ID", "")
    if not image_id or image_id != expected_image_id:
        raise RuntimeError("target Modal image authorization mismatch")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    if "A100" not in gpu_name:
        raise RuntimeError(f"frozen target GPU mismatch: {gpu_name!r}")
    versions = {
        "vllm": importlib.metadata.version("vllm"),
        "mistral_common": importlib.metadata.version("mistral-common"),
        "torch": str(torch.__version__).split("+")[0],
        "transformers": transformers.__version__,
    }
    expected = {
        "vllm": CONFIG["runtime"]["vllm_version"],
        "mistral_common": CONFIG["runtime"]["mistral_common_version"],
        "torch": CONFIG["runtime"]["torch_version"],
        "transformers": CONFIG["runtime"]["transformers_version"],
    }
    if versions != expected:
        raise RuntimeError(f"target dependency mismatch: {versions!r}")
    if platform.python_version().split(".")[:2] != ["3", "11"]:
        raise RuntimeError("target Python version mismatch")
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,name,uuid,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "modal_sdk_version": "1.5.4",
        "modal_image_id": image_id,
        "image_recipe_sha256": IMAGE_RECIPE_SHA256,
        "dependency_lock_sha256": DEPENDENCY_LOCK_SHA256,
        "python_version": platform.python_version(),
        "package_versions": versions,
        "cuda_runtime_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "nvidia_smi": smi,
        "checkpoint_commit": MODEL["checkpoint_commit"],
        "cache_attestation_sha256": _payload_hash(cache_attestation),
        "modal_function_call_id": str(modal.current_function_call_id()),
        "modal_input_id": str(modal.current_input_id()),
        "vllm_module_version": getattr(vllm, "__version__", ""),
    }


_LLM: Any | None = None


def _load_llm() -> Any:
    global _LLM
    if _LLM is None:
        from vllm import LLM

        _LLM = LLM(
            model=str(MODEL_PATH),
            tokenizer_mode=CONFIG["runtime"]["tokenizer_mode"],
            config_format=CONFIG["runtime"]["config_format"],
            load_format=CONFIG["runtime"]["load_format"],
            dtype=CONFIG["runtime"]["dtype"],
            tensor_parallel_size=1,
            trust_remote_code=False,
            max_model_len=CONFIG["runtime"]["maximum_model_length"],
            limit_mm_per_prompt={"image": 1},
            gpu_memory_utilization=0.9,
            enforce_eager=True,
        )
    return _LLM


def _asset_bytes(request: dict[str, Any]) -> bytes | None:
    relative = request.get("asset_path")
    if relative is None:
        if request.get("asset_sha256") is not None:
            raise RuntimeError("text target request has an asset digest")
        return None
    prefix = "data/derived/stimuli/pb2rr/"
    if not isinstance(relative, str) or not relative.startswith(prefix):
        raise RuntimeError("target asset path is outside the embedded public assets")
    name = relative.removeprefix(prefix)
    if not name or "/" in name or ".." in name:
        raise RuntimeError("target asset filename is invalid")
    path = STIMULI_PATH / name
    data = path.read_bytes()
    if (
        data[:8] != b"\x89PNG\r\n\x1a\n"
        or hashlib.sha256(data).hexdigest() != request["asset_sha256"]
    ):
        raise RuntimeError("embedded target PNG does not verify")
    return data


def _messages(request: dict[str, Any]) -> list[dict[str, Any]]:
    data = _asset_bytes(request)
    if data is None:
        user_content: Any = request["prompt"]
    else:
        user_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(data).decode("ascii")
                },
            },
            {"type": "text", "text": request["prompt"]},
        ]
    return [
        {"role": "system", "content": CONFIG["system_instruction"]["text"]},
        {"role": "user", "content": user_content},
    ]


def _exact_answer_token_ids(tokenizer: Any, codes: list[str]) -> list[int]:
    token_ids: list[int] = []
    for code in codes:
        encoded = tokenizer.encode(code, add_special_tokens=False)
        if not isinstance(encoded, list) or len(encoded) != 1:
            raise RuntimeError(f"answer code is not exactly one token: {code}")
        token_id = encoded[0]
        if tokenizer.decode([token_id]) != code:
            raise RuntimeError(f"answer token does not decode exactly: {code}")
        token_ids.append(token_id)
    if len(set(token_ids)) != len(token_ids):
        raise RuntimeError("answer codes do not map to distinct tokens")
    return token_ids


def _forced_choice(llm: Any, request: dict[str, Any]) -> dict[str, Any]:
    from vllm import SamplingParams

    codes = request["answer_codes"]
    token_ids = _exact_answer_token_ids(llm.get_tokenizer(), codes)
    params = SamplingParams(
        temperature=1.0,
        max_tokens=1,
        min_tokens=1,
        seed=int(request["generation_seed"]),
        allowed_token_ids=token_ids,
        logprobs=len(token_ids),
        detokenize=False,
    )
    outputs = llm.chat(_messages(request), sampling_params=params, use_tqdm=False)
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError("forced-choice target call returned an unexpected output count")
    generated = outputs[0].outputs[0]
    if len(generated.token_ids) != 1 or len(generated.logprobs or []) != 1:
        raise RuntimeError("forced-choice target call did not return one token")
    entries = generated.logprobs[0]
    log_values: list[float] = []
    for token_id in token_ids:
        entry = entries.get(token_id)
        if entry is None or not math.isfinite(float(entry.logprob)):
            raise RuntimeError("forced-choice target call omitted an allowed-token logprob")
        log_values.append(float(entry.logprob))
    maximum = max(log_values)
    weights = [math.exp(value - maximum) for value in log_values]
    denominator = sum(weights)
    return {
        "probabilities_by_code": {
            code: weight / denominator
            for code, weight in zip(codes, weights, strict=True)
        },
        "candidate_token_ids": token_ids,
        "free_generation_used": False,
        "engine_probe_tokens": 1,
    }


def _continuous_raw(llm: Any, request: dict[str, Any]) -> dict[str, Any]:
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=float(request["temperature"]),
        top_p=float(request["top_p"]),
        max_tokens=int(request["max_new_tokens"]),
        seed=int(request["generation_seed"]),
    )
    outputs = llm.chat(_messages(request), sampling_params=params, use_tqdm=False)
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError("continuous target call returned an unexpected output count")
    return {
        "raw_text": outputs[0].outputs[0].text,
        "generation_seed": int(request["generation_seed"]),
        "semantic_repair_used": False,
    }


def _parse_exact_json_integer(raw_text: Any) -> int:
    try:
        value = json.loads(raw_text)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("JSON canary output is not valid JSON") from error
    if (
        not isinstance(value, dict)
        or set(value) != {"predicted_value"}
        or not isinstance(value["predicted_value"], int)
        or isinstance(value["predicted_value"], bool)
        or value["predicted_value"] < 0
    ):
        raise ValueError("JSON canary output violates the integer schema")
    return value["predicted_value"]


@app.function(
    image=target_image,
    cpu=0.25,
    memory=256,
    timeout=120,
    retries=0,
    max_containers=1,
    name="cross_family_target_startup_smoke",
    include_source=False,
)
def cross_family_target_startup_smoke(
    expected_freeze_payload_sha256: str,
    expected_modal_image_id: str,
) -> dict[str, Any]:
    if _payload_hash(CONFIG) != expected_freeze_payload_sha256:
        raise RuntimeError("startup smoke is bound to another target freeze")
    image_id = os.environ.get("MODAL_IMAGE_ID", "")
    if image_id != expected_modal_image_id:
        raise RuntimeError("startup smoke image authorization mismatch")
    if Path(__file__).resolve() != REMOTE_SOURCE_PATH:
        raise RuntimeError("target app remote module path mismatch")
    return {
        "schema_version": "intervenebench.cross_family_target_startup_smoke.v1",
        "status": "passed",
        "modal_image_id": image_id,
        "module_path": str(Path(__file__).resolve()),
        "model_downloaded": False,
        "inference_calls_made": 0,
        "human_outcomes_accessed": False,
    }


@app.function(
    image=target_image,
    gpu="A100-80GB:1",
    volumes={MODEL_ROOT: model_volume.with_mount_options(read_only=True)},
    timeout=3600,
    startup_timeout=1800,
    scaledown_window=60,
    retries=0,
    max_containers=1,
    block_network=True,
    restrict_modal_access=True,
    name="run_cross_family_json_canary",
    include_source=False,
)
def run_cross_family_json_canary(
    expected_freeze_payload_sha256: str,
    expected_modal_image_id: str,
    expected_cache_attestation_sha256: str,
    expected_canary_prompt_sha256: str,
) -> dict[str, Any]:
    """Future separately-authorized target-free one-call JSON schema canary."""

    if _payload_hash(CONFIG) != expected_freeze_payload_sha256:
        raise RuntimeError("JSON canary is bound to another target freeze")
    spec = CONFIG["required_json_canary"]
    if (
        spec["planned_call_count"] != 1
        or spec["prompt_sha256"] != expected_canary_prompt_sha256
        or hashlib.sha256(spec["prompt"].encode("utf-8")).hexdigest()
        != expected_canary_prompt_sha256
    ):
        raise RuntimeError("JSON canary spec authorization mismatch")
    cache = _load_cache_attestation(expected_cache_attestation_sha256)
    runtime = _runtime_attestation(
        expected_image_id=expected_modal_image_id,
        cache_attestation=cache,
    )
    request = {
        "prompt": spec["prompt"],
        "modality": "text",
        "asset_path": None,
        "asset_sha256": None,
        "temperature": spec["temperature"],
        "top_p": spec["top_p"],
        "max_new_tokens": spec["max_new_tokens"],
        "generation_seed": spec["seed"],
    }
    raw = _continuous_raw(_load_llm(), request)["raw_text"]
    try:
        parsed_value = _parse_exact_json_integer(raw)
    except ValueError:
        return {
            "schema_version": "intervenebench.cross_family_json_canary_result.v1",
            "status": "failed_target_free_json_schema_stop",
            "canary_id": spec["canary_id"],
            "prompt_sha256": spec["prompt_sha256"],
            "raw_text": raw,
            "semantic_repair_used": False,
            "runtime_attestation": runtime,
            "target_calls_made": 0,
            "human_outcomes_accessed": False,
            "participant_rows_read": 0,
            "automatic_next_stage": False,
        }
    return {
        "schema_version": "intervenebench.cross_family_json_canary_result.v1",
        "status": "passed_target_free_json_schema",
        "canary_id": spec["canary_id"],
        "prompt_sha256": spec["prompt_sha256"],
        "raw_text": raw,
        "parsed_value": parsed_value,
        "semantic_repair_used": False,
        "modal_image_id": expected_modal_image_id,
        "runtime_attestation": runtime,
        "target_calls_made": 0,
        "human_outcomes_accessed": False,
        "participant_rows_read": 0,
        "automatic_next_stage": False,
    }


@app.function(
    image=target_image,
    gpu="A100-80GB:1",
    volumes={MODEL_ROOT: model_volume.with_mount_options(read_only=True)},
    timeout=10800,
    startup_timeout=1800,
    scaledown_window=60,
    retries=0,
    max_containers=1,
    block_network=True,
    restrict_modal_access=True,
    name="run_cross_family_target_group",
    include_source=False,
)
def run_cross_family_target_group(
    requests: list[dict[str, Any]],
    expected_freeze_payload_sha256: str,
    expected_modal_image_id: str,
    expected_cache_attestation_sha256: str,
) -> dict[str, Any]:
    """Future separately-authorized worker for a complete target call group."""

    if _payload_hash(CONFIG) != expected_freeze_payload_sha256:
        raise RuntimeError("target group is bound to another target freeze")
    if not requests:
        raise RuntimeError("target group must be non-empty")
    seen: set[str] = set()
    for request in requests:
        call_id = request.get("call_id")
        if call_id in seen or call_id not in REQUEST_HASHES:
            raise RuntimeError("target group contains duplicate or unauthorized call ID")
        seen.add(call_id)
        if _payload_hash(request) != REQUEST_HASHES[call_id]:
            raise RuntimeError("target request payload hash drifted")
        if hashlib.sha256(request["prompt"].encode("utf-8")).hexdigest() != request[
            "source_prompt_sha256"
        ]:
            raise RuntimeError("target request prompt hash drifted")
    cache = _load_cache_attestation(expected_cache_attestation_sha256)
    runtime = _runtime_attestation(
        expected_image_id=expected_modal_image_id,
        cache_attestation=cache,
    )
    llm = _load_llm()
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    for request in requests:
        if request["method_id"] == "forced_choice_next_token_softmax.v1":
            raw = _forced_choice(llm, request)
        elif request["method_id"] == "continuous_constrained_integer_generation.v1":
            raw = _continuous_raw(llm, request)
        else:
            raise RuntimeError("target request method is not allowlisted")
        results.append(
            {
                "call_id": request["call_id"],
                "model_id": request["model_id"],
                "request_payload_sha256": REQUEST_HASHES[request["call_id"]],
                "result": raw,
                "runtime_attestation": {
                    **runtime,
                    "call_id": request["call_id"],
                    "prompt_sha256": request["source_prompt_sha256"],
                    "asset_sha256": request["asset_sha256"],
                    "method_id": request["method_id"],
                },
            }
        )
    return {
        "schema_version": "intervenebench.cross_family_target_group.v1",
        "elapsed_seconds": time.monotonic() - started,
        "attempt_count": len(requests),
        "results": results,
        "model_downloaded": False,
        "human_outcomes_accessed": False,
        "participant_rows_read": 0,
        "automatic_next_stage": False,
    }


def materialized_target_image_id() -> str:
    image_id = target_image.object_id
    if not image_id:
        raise RuntimeError("Modal did not hydrate the cross-family target image")
    return image_id
