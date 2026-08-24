"""Modal workers for the target-free Mistral cross-family preflight.

Only three remote entrypoints exist: a target-free startup smoke test, an exact
public-checkpoint cache, and three synthetic canaries.  This image intentionally
embeds no target call plan, target prompt, target asset, participant data, or
human outcome data.
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
import re
import shutil
import subprocess
from typing import Any

import modal


if modal.is_local():
    ROOT = Path(__file__).resolve().parents[2]
    CONFIG_PATH = ROOT / "configs/simulators/cross_family_modal_preflight_v1.json"
    MODEL_MANIFEST_PATH = (
        ROOT
        / "data/manifests/simulators/mistral_small_3_1_24b_source_manifest_v1.json"
    )
    LOCK_PATH = ROOT / "infra/modal/cross-family-requirements.lock"
    SOURCE_PATH = Path(__file__)
else:
    CONFIG_PATH = Path("/opt/intervenebench/cross_family_modal_preflight_v1.json")
    MODEL_MANIFEST_PATH = Path(
        "/opt/intervenebench/mistral_small_3_1_24b_source_manifest_v1.json"
    )
    LOCK_PATH = Path("/opt/intervenebench/cross-family-requirements.lock")
    SOURCE_PATH = Path("/root/infra/modal/cross_family_app.py")

REMOTE_SOURCE_PATH = Path("/root/infra/modal/cross_family_app.py")


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
    raise RuntimeError("embedded model source manifest is malformed")
MODEL = MODEL_ENVELOPE["payload"]
if _payload_hash(MODEL) != MODEL_ENVELOPE["sha256"]:
    raise RuntimeError("embedded model source manifest hash is invalid")
if MODEL_ENVELOPE["sha256"] != CONFIG["model"]["source_manifest_payload_sha256"]:
    raise RuntimeError("preflight/model source binding mismatch")

MODEL_ROOT = Path("/models")
MODEL_PATH = (
    MODEL_ROOT
    / CONFIG["runtime"]["model_cache_subdirectory"]
    / CONFIG["model"]["checkpoint_commit"]
)
APP_NAME = CONFIG["runtime"]["app_name"]
IMAGE_RECIPE_SHA256 = CONFIG["runtime"]["image_recipe_sha256"]
DEPENDENCY_LOCK_SHA256 = CONFIG["runtime"]["dependency_lock_sha256"]

app = modal.App(APP_NAME, include_source=False)
model_volume = modal.Volume.from_name(
    CONFIG["runtime"]["model_volume_name"], create_if_missing=False
)
runtime_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(
        requirements=[str(LOCK_PATH)],
        extra_options="--require-hashes",
        uv_version="0.12.4",
    )
    .add_local_file(
        CONFIG_PATH,
        "/opt/intervenebench/cross_family_modal_preflight_v1.json",
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
    # Modal imports a function using its defining Python module name.  Keep the
    # source at that exact namespace path so remote workers can import
    # ``infra.modal.cross_family_app`` before executing an entrypoint.
    .add_local_file(SOURCE_PATH, str(REMOTE_SOURCE_PATH), copy=True)
)


@app.function(
    image=runtime_image,
    cpu=0.25,
    memory=256,
    timeout=120,
    retries=0,
    max_containers=1,
    name="cross_family_startup_smoke",
    include_source=False,
)
def cross_family_startup_smoke(
    expected_freeze_payload_sha256: str,
    expected_modal_image_id: str,
) -> dict[str, Any]:
    """Prove the packaged module imports before a long cache call is submitted."""

    print("STARTUP_SMOKE imported_remote_module", flush=True)
    if _payload_hash(CONFIG) != expected_freeze_payload_sha256:
        raise RuntimeError("startup smoke is bound to another preflight freeze")
    image_id = os.environ.get("MODAL_IMAGE_ID", "")
    if not image_id or image_id != expected_modal_image_id:
        raise RuntimeError("startup-smoke Modal image authorization mismatch")
    module_path = Path(__file__).resolve()
    if module_path != REMOTE_SOURCE_PATH:
        raise RuntimeError(f"remote module path mismatch: {module_path}")
    print("STARTUP_SMOKE passed", flush=True)
    return {
        "schema_version": "intervenebench.cross_family_startup_smoke.v1",
        "status": "passed",
        "modal_image_id": image_id,
        "module_path": str(module_path),
        "model_downloaded": False,
        "inference_calls_made": 0,
        "target_prompts_or_assets_accessed": False,
    }


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


def _verify_checkpoint_files(model_path: Path) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for item in MODEL["files"]:
        path = model_path / item["path"]
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
            git_blob_sha1 = _hash_file(path, "git_blob_sha1")
            if git_blob_sha1 != item["git_oid"]:
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


@app.function(
    image=runtime_image,
    cpu=2.0,
    memory=16384,
    volumes={MODEL_ROOT: model_volume},
    timeout=7200,
    retries=0,
    max_containers=1,
    name="cache_cross_family_checkpoint",
    include_source=False,
)
def cache_cross_family_checkpoint(
    expected_freeze_payload_sha256: str,
    expected_modal_image_id: str,
) -> dict[str, Any]:
    """Download exactly one allowlisted public checkpoint revision."""

    from huggingface_hub import snapshot_download

    print("CACHE_STAGE worker_started", flush=True)
    if _payload_hash(CONFIG) != expected_freeze_payload_sha256:
        raise RuntimeError("cache request is bound to another preflight freeze")
    image_id = os.environ.get("MODAL_IMAGE_ID", "")
    if not image_id or image_id != expected_modal_image_id:
        raise RuntimeError("cache Modal image authorization mismatch")
    attestation_path = MODEL_PATH / "cache_attestation.json"
    if attestation_path.is_file():
        print("CACHE_STAGE existing_cache_verification_started", flush=True)
        verified_files = _verify_checkpoint_files(MODEL_PATH)
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
        if attestation.get("verified_files") != verified_files:
            raise RuntimeError("existing cache attestation does not replay")
        print("CACHE_STAGE existing_cache_verified", flush=True)
        return attestation
    if MODEL_PATH.exists():
        raise RuntimeError("unattested model cache path already exists")
    input_id = str(modal.current_input_id())
    if not input_id or "/" in input_id or ".." in input_id:
        raise RuntimeError("invalid Modal input identifier")
    staging = MODEL_ROOT / ".staging" / f"cross-family-{input_id}"
    if staging.exists():
        raise RuntimeError("create-only checkpoint staging path already exists")
    staging.mkdir(parents=True)
    try:
        print(
            "CACHE_STAGE download_started "
            f"files={len(MODEL['files'])} bytes={sum(row['size_bytes'] for row in MODEL['files'])}",
            flush=True,
        )
        snapshot_download(
            repo_id=MODEL["hf_repository"],
            revision=MODEL["checkpoint_commit"],
            allow_patterns=[item["path"] for item in MODEL["files"]],
            local_dir=staging,
            token=False,
            max_workers=8,
        )
        print("CACHE_STAGE download_completed verification_started", flush=True)
        verified_files = _verify_checkpoint_files(staging)
        print("CACHE_STAGE verification_completed", flush=True)
        attestation = {
            "schema_version": "intervenebench.cross_family_cache_attestation.v1",
            "model_id": MODEL["model_id"],
            "hf_repository": MODEL["hf_repository"],
            "checkpoint_commit": MODEL["checkpoint_commit"],
            "model_source_manifest_payload_sha256": MODEL_ENVELOPE["sha256"],
            "modal_image_id": image_id,
            "dependency_lock_sha256": DEPENDENCY_LOCK_SHA256,
            "verified_files": verified_files,
            "unexpected_repository_files_downloaded": False,
            "target_prompts_or_assets_present": False,
        }
        (staging / "cache_attestation.json").write_text(
            json.dumps(attestation, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(MODEL_PATH)
        model_volume.commit()
        print("CACHE_STAGE volume_committed", flush=True)
        return attestation
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _runtime_attestation(cache_attestation: dict[str, Any]) -> dict[str, Any]:
    import torch
    import transformers
    import vllm

    image_id = os.environ.get("MODAL_IMAGE_ID", "")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    if "A100" not in gpu_name:
        raise RuntimeError(f"frozen GPU mismatch: {gpu_name!r}")
    if platform.python_version().split(".")[:2] != ["3", "11"]:
        raise RuntimeError("remote Python version mismatch")
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
        raise RuntimeError(f"remote dependency mismatch: {versions!r}")
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


def _messages(request: dict[str, Any]) -> list[dict[str, Any]]:
    system = CONFIG["system_instruction"]["text"]
    if request["modality"] == "text":
        user_content: Any = request["prompt"]
    else:
        asset = request["asset"]
        data = base64.b64decode(asset["base64"], validate=True)
        if (
            data[:8] != b"\x89PNG\r\n\x1a\n"
            or hashlib.sha256(data).hexdigest() != asset["sha256"]
        ):
            raise RuntimeError("synthetic canary PNG does not verify")
        user_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{asset['mime_type']};base64,{asset['base64']}"
                },
            },
            {"type": "text", "text": request["prompt"]},
        ]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def _forced_choice_probe(llm: Any, request: dict[str, Any]) -> dict[str, Any]:
    from vllm import SamplingParams

    codes = request["answer_codes"]
    token_ids = _exact_answer_token_ids(llm.get_tokenizer(), codes)
    params = SamplingParams(
        temperature=1.0,
        max_tokens=1,
        min_tokens=1,
        seed=request["seed"],
        allowed_token_ids=token_ids,
        logprobs=len(token_ids),
        detokenize=False,
    )
    outputs = llm.chat(_messages(request), sampling_params=params, use_tqdm=False)
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError("forced-choice canary returned an unexpected output count")
    generated = outputs[0].outputs[0]
    if len(generated.token_ids) != 1 or len(generated.logprobs or []) != 1:
        raise RuntimeError("forced-choice canary did not return one probe token")
    sampled_token_id = generated.token_ids[0]
    if sampled_token_id not in token_ids:
        raise RuntimeError("forced-choice sampled token escaped its allowlist")
    entries = generated.logprobs[0]
    log_values: list[float] = []
    for token_id in token_ids:
        entry = entries.get(token_id)
        if entry is None or not math.isfinite(float(entry.logprob)):
            raise RuntimeError("forced-choice canary omitted an allowed-token logprob")
        log_values.append(float(entry.logprob))
    maximum = max(log_values)
    weights = [math.exp(value - maximum) for value in log_values]
    denominator = sum(weights)
    probabilities = {
        code: weight / denominator for code, weight in zip(codes, weights, strict=True)
    }
    return {
        "schema_version": "intervenebench.masked_next_token_probe.v1",
        "answer_codes": codes,
        "token_ids": token_ids,
        "probabilities": probabilities,
        "sampled_code": codes[token_ids.index(sampled_token_id)],
        "free_generation_used": False,
        "engine_probe_tokens": 1,
    }


_STRICT_INTEGER = re.compile(r"(?:0|[1-9][0-9]*)\Z")


def _continuous_probe(llm: Any, request: dict[str, Any]) -> dict[str, Any]:
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=request["temperature"],
        top_p=request["top_p"],
        max_tokens=request["max_new_tokens"],
        seed=request["seed"],
    )
    outputs = llm.chat(_messages(request), sampling_params=params, use_tqdm=False)
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError("continuous canary returned an unexpected output count")
    text = outputs[0].outputs[0].text
    if not isinstance(text, str) or _STRICT_INTEGER.fullmatch(text) is None:
        raise RuntimeError("continuous canary failed the strict integer schema")
    return {
        "schema_version": "intervenebench.strict_nonnegative_integer_probe.v1",
        "raw_text": text,
        "parsed_value": int(text),
        "semantic_repair_used": False,
    }


@app.function(
    image=runtime_image,
    gpu="A100-80GB:1",
    volumes={MODEL_ROOT: model_volume.with_mount_options(read_only=True)},
    timeout=3600,
    startup_timeout=1800,
    scaledown_window=2,
    retries=0,
    max_containers=1,
    block_network=True,
    restrict_modal_access=True,
    name="run_cross_family_canary",
    include_source=False,
)
def run_cross_family_canary(
    expected_freeze_payload_sha256: str,
    expected_modal_image_id: str,
    expected_cache_attestation_sha256: str,
    expected_canary_manifest_payload_sha256: str,
) -> dict[str, Any]:
    """Run exactly three synthetic target-free canaries on one A100-80GB."""

    from vllm import LLM

    if _payload_hash(CONFIG) != expected_freeze_payload_sha256:
        raise RuntimeError("canary request is bound to another preflight freeze")
    canary = CONFIG["canary"]["manifest"]
    if _payload_hash(canary) != expected_canary_manifest_payload_sha256:
        raise RuntimeError("canary manifest authorization mismatch")
    if canary["planned_call_count"] != 3 or len(canary["requests"]) != 3:
        raise RuntimeError("canary request count drifted")
    image_id = os.environ.get("MODAL_IMAGE_ID", "")
    if not image_id or image_id != expected_modal_image_id:
        raise RuntimeError("canary Modal image authorization mismatch")
    cache_path = MODEL_PATH / "cache_attestation.json"
    if not cache_path.is_file():
        raise RuntimeError("verified checkpoint cache is absent")
    cache_attestation = json.loads(cache_path.read_text(encoding="utf-8"))
    verified_files = _verify_checkpoint_files(MODEL_PATH)
    if cache_attestation.get("verified_files") != verified_files:
        raise RuntimeError("cache attestation/file replay failed")
    if _payload_hash(cache_attestation) != expected_cache_attestation_sha256:
        raise RuntimeError("cache attestation authorization mismatch")
    runtime = _runtime_attestation(cache_attestation)
    if runtime["modal_image_id"] != expected_modal_image_id:
        raise RuntimeError("runtime Modal image authorization mismatch")

    llm = LLM(
        model=str(MODEL_PATH),
        tokenizer_mode=CONFIG["runtime"]["tokenizer_mode"],
        config_format=CONFIG["runtime"]["config_format"],
        load_format=CONFIG["runtime"]["load_format"],
        dtype=CONFIG["runtime"]["dtype"],
        tensor_parallel_size=CONFIG["runtime"]["tensor_parallel_size"],
        trust_remote_code=CONFIG["runtime"]["trust_remote_code"],
        max_model_len=CONFIG["runtime"]["maximum_model_length"],
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.9,
        enforce_eager=True,
    )
    canary_results: list[dict[str, Any]] = []
    for request in canary["requests"]:
        if hashlib.sha256(request["prompt"].encode("utf-8")).hexdigest() != request[
            "prompt_sha256"
        ]:
            raise RuntimeError("synthetic canary prompt hash mismatch")
        if request["adapter"] == "forced_choice_next_token_softmax.v1":
            result = _forced_choice_probe(llm, request)
        elif request["adapter"] == "continuous_constrained_integer_generation.v1":
            result = _continuous_probe(llm, request)
        else:
            raise RuntimeError("synthetic canary adapter is not allowlisted")
        canary_results.append(
            {
                "canary_id": request["canary_id"],
                "prompt_sha256": request["prompt_sha256"],
                "result": result,
            }
        )
    return {
        "schema_version": "intervenebench.cross_family_canary_run.v1",
        "status": "completed_requires_local_adjudication",
        "runtime_attestation": runtime,
        "canary_manifest_payload_sha256": expected_canary_manifest_payload_sha256,
        "canary_results": canary_results,
        "target_calls_made": 0,
        "target_prompts_or_assets_accessed": False,
        "human_data_accessed": False,
        "automatic_next_stage": False,
    }
