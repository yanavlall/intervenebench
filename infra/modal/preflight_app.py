"""Modal definitions for the hash-bound InterveneBench discovery preflight.

This module is intentionally not a CLI entrypoint. The local wrapper verifies a
separate authorization before importing it, so importing Modal definitions
cannot precede the project's authority checks.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import modal


if modal.is_local():
    ROOT = Path(__file__).resolve().parents[2]
    FREEZE_PATH = ROOT / "configs/simulators/modal_discovery_preflight_v2.json"
    MODEL_MANIFEST_PATH = (
        ROOT / "data/manifests/simulators/model_file_manifests_v1.json"
    )
    LOCK_PATH = ROOT / "infra/modal/preflight-requirements.lock"
    SOURCE_PATH = Path(__file__)
else:
    FREEZE_PATH = Path("/opt/intervenebench/modal_discovery_preflight_v2.json")
    MODEL_MANIFEST_PATH = Path(
        "/opt/intervenebench/model_file_manifests_v1.json"
    )
    LOCK_PATH = Path("/opt/intervenebench/preflight-requirements.lock")
    SOURCE_PATH = Path("/root/preflight_app.py")

FREEZE = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
MODEL_FILE_MANIFEST = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
MODELS = {model["model_id"]: model for model in FREEZE["models"]}
FILE_MANIFESTS = {
    model["model_id"]: model for model in MODEL_FILE_MANIFEST["models"]
}

APP_NAME = FREEZE["runtime"]["app_name"]
VOLUME_NAME = "intervenebench-model-cache-v1"
REMOTE_MANIFEST_PATH = "/opt/intervenebench/model_file_manifests_v1.json"
REMOTE_FREEZE_PATH = "/opt/intervenebench/modal_discovery_preflight_v2.json"
MODEL_ROOT = Path("/models")


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


IMAGE_RECIPE_SHA256 = _payload_hash(FREEZE["runtime"]["image_recipe"])
DEPENDENCY_LOCK_SHA256 = FREEZE["dependency_lock"]["lock_file_sha256"]

app = modal.App(APP_NAME, include_source=False)
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

cache_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "huggingface-hub==0.36.2",
        "hf-xet==1.6.0",
        uv_version="0.12.4",
    )
    .add_local_file(
        MODEL_MANIFEST_PATH,
        REMOTE_MANIFEST_PATH,
        copy=True,
    )
    .add_local_file(FREEZE_PATH, REMOTE_FREEZE_PATH, copy=True)
    .add_local_file(
        LOCK_PATH,
        "/opt/intervenebench/preflight-requirements.lock",
        copy=True,
    )
    .add_local_file(SOURCE_PATH, "/root/preflight_app.py", copy=True)
)

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        requirements=[str(LOCK_PATH)],
        extra_options="--require-hashes",
        uv_version="0.12.4",
    )
    .add_local_file(
        MODEL_MANIFEST_PATH,
        REMOTE_MANIFEST_PATH,
        copy=True,
    )
    .add_local_file(FREEZE_PATH, REMOTE_FREEZE_PATH, copy=True)
    .add_local_file(
        LOCK_PATH,
        "/opt/intervenebench/preflight-requirements.lock",
        copy=True,
    )
    .add_local_file(SOURCE_PATH, "/root/preflight_app.py", copy=True)
)

import_smoke_image = modal.Image.debian_slim(python_version="3.11").add_local_file(
    SOURCE_PATH, "/root/preflight_app.py", copy=True
).add_local_file(
    MODEL_MANIFEST_PATH,
    REMOTE_MANIFEST_PATH,
    copy=True,
).add_local_file(
    FREEZE_PATH,
    REMOTE_FREEZE_PATH,
    copy=True,
).add_local_file(
    LOCK_PATH,
    "/opt/intervenebench/preflight-requirements.lock",
    copy=True,
)


def _hash_file(path: Path, algorithm: str) -> str:
    if algorithm == "sha256":
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    if algorithm == "git_blob_sha1":
        digest = hashlib.sha1()
        digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    raise ValueError(f"unsupported file hash algorithm: {algorithm}")


def _verify_model_files(model_id: str, model_path: Path) -> list[dict[str, Any]]:
    expected = FILE_MANIFESTS[model_id]["files"]
    verified: list[dict[str, Any]] = []
    for relative, expected_size, algorithm, expected_digest in expected:
        path = model_path / relative
        if not path.is_file():
            raise RuntimeError(f"missing pinned model file: {relative}")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(f"model file size mismatch: {relative}")
        actual_digest = _hash_file(path, algorithm)
        if actual_digest != expected_digest:
            raise RuntimeError(f"model file digest mismatch: {relative}")
        verified.append(
            {
                "path": relative,
                "size": actual_size,
                "algorithm": algorithm,
                "digest": actual_digest,
            }
        )
    return verified


@app.function(
    image=import_smoke_image,
    cpu=0.125,
    memory=128,
    timeout=60,
    retries=0,
    max_containers=1,
    name="container_import_smoke",
    include_source=False,
)
def container_import_smoke() -> dict[str, Any]:
    """Prove the shipped module and frozen inputs import before any cache call."""

    return {
        "schema_version": "modal_container_import_smoke.v1",
        "module_path": __file__,
        "freeze_payload_sha256": _payload_hash(FREEZE),
        "model_manifest_payload_sha256": _payload_hash(MODEL_FILE_MANIFEST),
        "dependency_lock_sha256": DEPENDENCY_LOCK_SHA256,
        "status": "import_ok",
    }


@app.function(
    image=cache_image,
    cpu=2.0,
    memory=4096,
    volumes={MODEL_ROOT: model_volume},
    timeout=1200,
    retries=0,
    max_containers=1,
    name="cache_checkpoint",
    include_source=False,
)
def cache_checkpoint(model_id: str) -> dict[str, Any]:
    """Download one allowlisted revision and publish it only after verification."""

    from huggingface_hub import snapshot_download

    manifest = json.loads(Path(REMOTE_MANIFEST_PATH).read_text(encoding="utf-8"))
    by_id = {entry["model_id"]: entry for entry in manifest["models"]}
    if model_id not in by_id:
        raise ValueError("model ID is outside the frozen cache allowlist")
    entry = by_id[model_id]
    final_path = MODEL_ROOT / model_id / entry["commit"]
    attestation_path = final_path / "cache_attestation.json"
    if attestation_path.is_file():
        existing = json.loads(attestation_path.read_text(encoding="utf-8"))
        verified = _verify_model_files(model_id, final_path)
        if existing.get("verified_files") != verified:
            raise RuntimeError("cached attestation differs from verified model files")
        return existing

    staging_root = MODEL_ROOT / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = staging_root / f"{model_id}-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        snapshot_download(
            repo_id=entry["repository"],
            revision=entry["commit"],
            local_dir=staging,
            allow_patterns=[file_entry[0] for file_entry in entry["files"]],
            token=False,
            max_workers=8,
        )
        verified = _verify_model_files(model_id, staging)
        attestation = {
            "schema_version": "modal_model_cache_attestation.v1",
            "model_id": model_id,
            "repository": entry["repository"],
            "checkpoint_commit": entry["commit"],
            "source_manifest_sha256": _payload_hash(manifest),
            "verified_files": verified,
            "cache_elapsed_seconds": time.monotonic() - started,
            "modal_function_call_id": modal.current_function_call_id(),
            "status": "verified",
        }
        (staging / "cache_attestation.json").write_text(
            json.dumps(attestation, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if final_path.exists():
            raise RuntimeError("cache target appeared during publish")
        staging.rename(final_path)
        model_volume.commit()
        return attestation
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _runtime_attestation(
    *, model_id: str, model_path: Path, cache_attestation: dict[str, Any]
) -> dict[str, Any]:
    import torch
    import transformers

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    if "L40S" not in gpu_name:
        raise RuntimeError(f"frozen GPU mismatch: {gpu_name!r}")
    expected = FREEZE["runtime"]
    if platform.python_version().split(".")[:2] != ["3", "11"]:
        raise RuntimeError("remote Python version differs from the freeze")
    if torch.__version__.split("+")[0] != "2.9.1":
        raise RuntimeError("remote torch version differs from the freeze")
    if transformers.__version__ != "4.57.6":
        raise RuntimeError("remote transformers version differs from the freeze")
    if torch.version.cuda != expected["expected_cuda_runtime_version"]:
        raise RuntimeError("remote CUDA runtime differs from the freeze")
    modal_image_id = os.environ.get("MODAL_IMAGE_ID", "")
    if not modal_image_id:
        raise RuntimeError("MODAL_IMAGE_ID is missing")
    modal_sdk_version = expected["modal_sdk_version"]
    nvidia_smi = subprocess.run(
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
        "modal_sdk_version": modal_sdk_version,
        "modal_image_id": modal_image_id,
        "image_recipe_sha256": IMAGE_RECIPE_SHA256,
        "dependency_lock_sha256": DEPENDENCY_LOCK_SHA256,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_name": gpu_name,
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        "nvidia_smi": nvidia_smi,
        "checkpoint_commit": MODELS[model_id]["checkpoint_commit"],
        "weight_manifest_sha256": MODELS[model_id][
            "weight_file_manifest_sha256"
        ],
        "tokenizer_manifest_sha256": MODELS[model_id][
            "tokenizer_manifest_sha256"
        ],
        "cache_attestation_sha256": _payload_hash(cache_attestation),
        "modal_function_call_id": modal.current_function_call_id(),
        "modal_input_id": modal.current_input_id(),
        "model_path": str(model_path),
        "inference_backend": expected["inference_backend"],
    }


@app.function(
    image=inference_image,
    gpu="L40S:1",
    volumes={MODEL_ROOT: model_volume.with_mount_options(read_only=True)},
    max_containers=4,
    single_use_containers=True,
    timeout=1800,
    startup_timeout=1800,
    scaledown_window=60,
    retries=0,
    block_network=True,
    restrict_modal_access=True,
    name="run_model_group",
    include_source=False,
)
def run_model_group(
    model_id: str,
    requests: list[dict[str, Any]],
    expected_modal_image_id: str,
    expected_cache_attestation_sha256: str,
) -> dict[str, Any]:
    """Load one exact checkpoint once and answer its ten frozen calls."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_id not in MODELS or model_id not in FILE_MANIFESTS:
        raise ValueError("model ID is outside the frozen inference allowlist")
    if len(requests) != 10 or any(request.get("model_id") != model_id for request in requests):
        raise ValueError("remote group must contain exactly ten calls for one model")
    if len({request.get("call_id") for request in requests}) != 10:
        raise ValueError("remote group contains duplicate call IDs")
    model = MODELS[model_id]
    model_path = MODEL_ROOT / model_id / model["checkpoint_commit"]
    cache_path = model_path / "cache_attestation.json"
    if not cache_path.is_file():
        raise RuntimeError("verified checkpoint cache is absent")
    cache_attestation = json.loads(cache_path.read_text(encoding="utf-8"))
    _verify_model_files(model_id, model_path)
    if _payload_hash(cache_attestation) != expected_cache_attestation_sha256:
        raise RuntimeError("cache attestation authorization mismatch")
    runtime = _runtime_attestation(
        model_id=model_id,
        model_path=model_path,
        cache_attestation=cache_attestation,
    )
    if runtime["modal_image_id"] != expected_modal_image_id:
        raise RuntimeError("Modal image authorization mismatch")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    loaded_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    loaded_model.eval()
    results: list[dict[str, Any]] = []
    started = time.monotonic()
    for request in requests:
        if time.monotonic() - started > 1500:
            raise TimeoutError("frozen per-model inference ledger expired")
        prompt = request["prompt"]
        prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if prompt_digest != request["prompt_sha256"]:
            raise ValueError("remote prompt hash mismatch")
        if request["temperature"] != 0.2 or request["top_p"] != 0.95:
            raise ValueError("remote generation settings differ from the freeze")
        if request["maximum_output_tokens"] != 256:
            raise ValueError("remote output limit differs from the freeze")
        template_kwargs: dict[str, Any] = {}
        if model_id.startswith("qwen3_"):
            template_kwargs["enable_thinking"] = False
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to("cuda")
        if inputs["input_ids"].shape[-1] + request["maximum_output_tokens"] > model[
            "maximum_context_tokens"
        ]:
            raise ValueError("frozen model context limit would be exceeded")
        torch.manual_seed(request["seed"])
        torch.cuda.manual_seed_all(request["seed"])
        with torch.inference_mode():
            generated = loaded_model.generate(
                **inputs,
                do_sample=True,
                temperature=request["temperature"],
                top_p=request["top_p"],
                max_new_tokens=request["maximum_output_tokens"],
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion = generated[0, inputs["input_ids"].shape[-1] :]
        raw_text = tokenizer.decode(completion, skip_special_tokens=True).strip()
        if not raw_text:
            raise RuntimeError("model returned an empty response")
        call_attestation = dict(runtime)
        call_attestation["call_id"] = request["call_id"]
        call_attestation["prompt_sha256"] = request["prompt_sha256"]
        results.append(
            {
                "call_id": request["call_id"],
                "model_id": model_id,
                "raw_text": raw_text,
                "runtime_attestation": call_attestation,
            }
        )
    response = {
        "model_id": model_id,
        "elapsed_seconds": time.monotonic() - started,
        "results": results,
    }
    return json.loads(json.dumps(response, allow_nan=False))


def materialized_inference_image_id() -> str:
    """Return the hydrated Modal image object ID while ``app.run`` is active."""

    image_id = inference_image.object_id
    if not image_id:
        raise RuntimeError("Modal did not hydrate the inference image")
    return image_id
