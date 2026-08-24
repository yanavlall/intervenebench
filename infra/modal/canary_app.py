"""Modal definitions for the separately frozen four-call constrained canary."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import modal


if modal.is_local():
    ROOT = Path(__file__).resolve().parents[2]
    FREEZE_PATH = ROOT / "configs/simulators/modal_constrained_canary_v1.json"
    MODEL_MANIFEST_PATH = ROOT / "data/manifests/simulators/model_file_manifests_v1.json"
    LOCK_PATH = ROOT / "infra/modal/canary-requirements.lock"
    SOURCE_PATH = Path(__file__)
else:
    FREEZE_PATH = Path("/opt/intervenebench/modal_constrained_canary_v1.json")
    MODEL_MANIFEST_PATH = Path("/opt/intervenebench/model_file_manifests_v1.json")
    LOCK_PATH = Path("/opt/intervenebench/canary-requirements.lock")
    SOURCE_PATH = Path("/root/canary_app.py")

FREEZE = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
MODEL_FILE_MANIFEST = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
MODELS = {model["model_id"]: model for model in FREEZE["models"]}
FILE_MANIFESTS = {model["model_id"]: model for model in MODEL_FILE_MANIFEST["models"]}
MODEL_ROOT = Path("/models")
VOLUME_NAME = "intervenebench-model-cache-v1"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False
    ).encode("utf-8")


def _payload_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


IMAGE_RECIPE_SHA256 = _payload_hash(FREEZE["runtime"]["image_recipe"])
DEPENDENCY_LOCK_SHA256 = FREEZE["dependency_lock"]["lock_file_sha256"]

app = modal.App(FREEZE["runtime"]["app_name"], include_source=False)
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        requirements=[str(LOCK_PATH)],
        extra_options="--require-hashes",
        uv_version="0.12.4",
    )
    .add_local_file(MODEL_MANIFEST_PATH, "/opt/intervenebench/model_file_manifests_v1.json", copy=True)
    .add_local_file(FREEZE_PATH, "/opt/intervenebench/modal_constrained_canary_v1.json", copy=True)
    .add_local_file(LOCK_PATH, "/opt/intervenebench/canary-requirements.lock", copy=True)
    .add_local_file(SOURCE_PATH, "/root/canary_app.py", copy=True)
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
    raise ValueError(f"unsupported model-file digest algorithm: {algorithm}")


def _verify_model_files(model_id: str, model_path: Path) -> None:
    for relative, size, algorithm, digest in FILE_MANIFESTS[model_id]["files"]:
        path = model_path / relative
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"cached model file mismatch: {relative}")
        if _hash_file(path, algorithm) != digest:
            raise RuntimeError(f"cached model digest mismatch: {relative}")


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
        raise RuntimeError("remote Python version differs from canary freeze")
    if torch.__version__.split("+")[0] != "2.9.1":
        raise RuntimeError("remote torch version differs from canary freeze")
    if transformers.__version__ != "4.57.6":
        raise RuntimeError("remote transformers version differs from canary freeze")
    if metadata.version("outlines") != "1.2.11":
        raise RuntimeError("remote Outlines version differs from canary freeze")
    if metadata.version("outlines-core") != "0.2.14":
        raise RuntimeError("remote Outlines Core version differs from canary freeze")
    if torch.version.cuda != expected["expected_cuda_runtime_version"]:
        raise RuntimeError("remote CUDA runtime differs from canary freeze")
    image_id = os.environ.get("MODAL_IMAGE_ID", "")
    if not image_id:
        raise RuntimeError("MODAL_IMAGE_ID is missing")
    nvidia_smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version,name,uuid,memory.total", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True
    ).stdout.strip()
    model = MODELS[model_id]
    return {
        "modal_sdk_version": expected["modal_sdk_version"],
        "modal_image_id": image_id,
        "image_recipe_sha256": IMAGE_RECIPE_SHA256,
        "dependency_lock_sha256": DEPENDENCY_LOCK_SHA256,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "outlines_version": metadata.version("outlines"),
        "outlines_core_version": metadata.version("outlines-core"),
        "cuda_runtime_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "nvidia_smi": nvidia_smi,
        "checkpoint_commit": model["checkpoint_commit"],
        "weight_manifest_sha256": model["weight_file_manifest_sha256"],
        "tokenizer_manifest_sha256": model["tokenizer_manifest_sha256"],
        "cache_attestation_sha256": _payload_hash(cache_attestation),
        "modal_function_call_id": modal.current_function_call_id(),
        "modal_input_id": modal.current_input_id(),
        "model_path": str(model_path),
        "constraint_backend": "outlines_core_json_schema",
    }


@app.function(
    image=inference_image,
    gpu="L40S:1",
    volumes={MODEL_ROOT: model_volume.with_mount_options(read_only=True)},
    max_containers=4,
    single_use_containers=True,
    timeout=420,
    startup_timeout=900,
    scaledown_window=2,
    retries=0,
    block_network=True,
    restrict_modal_access=True,
    name="run_constrained_canary",
    include_source=False,
)
def run_constrained_canary(
    model_id: str, request: dict[str, Any], expected_modal_image_id: str,
    expected_cache_attestation_sha256: str
) -> dict[str, Any]:
    """Load one cached model and answer exactly one schema-constrained call."""

    import outlines
    import torch
    from outlines.types import JsonSchema
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

    if model_id not in MODELS or model_id not in FILE_MANIFESTS:
        raise ValueError("model is outside canary allowlist")
    if request.get("model_id") != model_id or request.get("experiment_id") != "5vm8g":
        raise ValueError("canary request scope mismatch")
    if request.get("constraint_backend") != "outlines_core_json_schema":
        raise ValueError("canary constraint backend mismatch")
    if request.get("json_schema") != FREEZE["generation"]["json_schema"]:
        raise ValueError("canary JSON schema mismatch")
    if _payload_hash(request["json_schema"]) != request.get("json_schema_sha256"):
        raise ValueError("canary JSON schema hash mismatch")
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or hashlib.sha256(prompt.encode()).hexdigest() != request.get("prompt_sha256"):
        raise ValueError("canary prompt hash mismatch")
    if (request.get("temperature"), request.get("top_p"), request.get("maximum_output_tokens")) != (0.2, 0.95, 128):
        raise ValueError("canary generation settings mismatch")

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
        model_id=model_id, model_path=model_path, cache_attestation=cache_attestation
    )
    if runtime["modal_image_id"] != expected_modal_image_id:
        raise RuntimeError("Modal image authorization mismatch")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    loaded_model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True
    )
    loaded_model.eval()
    template_kwargs: dict[str, Any] = {}
    if model_id.startswith("qwen3_"):
        template_kwargs["enable_thinking"] = False
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, **template_kwargs
    )
    inputs = tokenizer(rendered, return_tensors="pt").to("cuda")
    if inputs["input_ids"].shape[-1] + 128 > model["maximum_context_tokens"]:
        raise ValueError("canary prompt exceeds frozen context limit")

    outlines_model = outlines.from_transformers(loaded_model, tokenizer)
    generator = outlines.Generator(
        outlines_model, output_type=JsonSchema(request["json_schema"]),
        backend="outlines_core"
    )
    processor = generator.logits_processor
    if processor is None:
        raise RuntimeError("Outlines did not create a constrained logits processor")
    processor.reset()
    torch.manual_seed(request["seed"])
    torch.cuda.manual_seed_all(request["seed"])
    started = time.monotonic()
    with torch.inference_mode():
        generated = loaded_model.generate(
            **inputs, do_sample=True, temperature=0.2, top_p=0.95,
            max_new_tokens=128,
            logits_processor=LogitsProcessorList([processor]),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    completion = generated[0, inputs["input_ids"].shape[-1]:]
    raw_text = tokenizer.decode(completion, skip_special_tokens=True).strip()
    decoded = json.loads(raw_text)
    if set(decoded) != {"relative_weights"}:
        raise RuntimeError("constrained output violated top-level schema")
    call_runtime = dict(runtime)
    call_runtime.update(
        {
            "call_id": request["call_id"],
            "prompt_sha256": request["prompt_sha256"],
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    response = {
        "call_id": request["call_id"],
        "model_id": model_id,
        "raw_text": raw_text,
        "runtime_attestation": call_runtime,
    }
    return json.loads(json.dumps(response, allow_nan=False))


def materialized_inference_image_id() -> str:
    image_id = inference_image.object_id
    if not image_id:
        raise RuntimeError("Modal did not hydrate the canary inference image")
    return image_id
