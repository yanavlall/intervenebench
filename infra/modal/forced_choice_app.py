"""Modal definitions for the separately frozen forced-choice logit canary."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import modal


if modal.is_local():
    ROOT = Path(__file__).resolve().parents[2]
    FREEZE_PATH = ROOT / "configs/simulators/modal_forced_choice_v1.json"
    MODEL_MANIFEST_PATH = ROOT / "data/manifests/simulators/model_file_manifests_v1.json"
    LOCK_PATH = ROOT / "infra/modal/preflight-requirements.lock"
    SOURCE_PATH = Path(__file__)
else:
    FREEZE_PATH = Path("/opt/intervenebench/modal_forced_choice_v1.json")
    MODEL_MANIFEST_PATH = Path("/opt/intervenebench/model_file_manifests_v1.json")
    LOCK_PATH = Path("/opt/intervenebench/preflight-requirements.lock")
    SOURCE_PATH = Path("/root/forced_choice_app.py")

FREEZE = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
MODEL_FILE_MANIFEST = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
MODELS = {item["model_id"]: item for item in FREEZE["models"]}
FILE_MANIFESTS = {item["model_id"]: item for item in MODEL_FILE_MANIFEST["models"]}
MODEL_ROOT = Path("/models")


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
model_volume = modal.Volume.from_name(
    "intervenebench-model-cache-v1", create_if_missing=False
)
inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        requirements=[str(LOCK_PATH)], extra_options="--require-hashes",
        uv_version="0.12.4"
    )
    .add_local_file(MODEL_MANIFEST_PATH, "/opt/intervenebench/model_file_manifests_v1.json", copy=True)
    .add_local_file(FREEZE_PATH, "/opt/intervenebench/modal_forced_choice_v1.json", copy=True)
    .add_local_file(LOCK_PATH, "/opt/intervenebench/preflight-requirements.lock", copy=True)
    .add_local_file(SOURCE_PATH, "/root/forced_choice_app.py", copy=True)
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
    raise ValueError(f"unsupported model digest algorithm: {algorithm}")


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
    if platform.python_version().split(".")[:2] != ["3", "11"]:
        raise RuntimeError("remote Python version mismatch")
    if torch.__version__.split("+")[0] != "2.9.1":
        raise RuntimeError("remote torch version mismatch")
    if transformers.__version__ != "4.57.6":
        raise RuntimeError("remote transformers version mismatch")
    if torch.version.cuda != "12.8":
        raise RuntimeError("remote CUDA runtime mismatch")
    image_id = os.environ.get("MODAL_IMAGE_ID", "")
    if not image_id:
        raise RuntimeError("MODAL_IMAGE_ID is missing")
    nvidia_smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version,name,uuid,memory.total", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True
    ).stdout.strip()
    model = MODELS[model_id]
    return {
        "modal_sdk_version": "1.5.4",
        "modal_image_id": image_id,
        "image_recipe_sha256": IMAGE_RECIPE_SHA256,
        "dependency_lock_sha256": DEPENDENCY_LOCK_SHA256,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
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
        "method_id": "forced_choice_next_token_softmax.v1",
    }


@app.function(
    image=inference_image,
    gpu="L40S:1",
    volumes={MODEL_ROOT: model_volume.with_mount_options(read_only=True)},
    max_containers=4,
    single_use_containers=True,
    timeout=300,
    startup_timeout=900,
    scaledown_window=2,
    retries=0,
    block_network=True,
    restrict_modal_access=True,
    name="run_forced_choice",
    include_source=False,
)
def run_forced_choice(
    model_id: str, request: dict[str, Any], expected_modal_image_id: str,
    expected_cache_attestation_sha256: str
) -> dict[str, Any]:
    """Return next-token softmax over five tokenizer-verified answer codes."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_id not in MODELS or request.get("model_id") != model_id:
        raise ValueError("forced-choice model scope mismatch")
    if request.get("experiment_id") != "5vm8g":
        raise ValueError("forced-choice experiment scope mismatch")
    if request.get("method_id") != "forced_choice_next_token_softmax.v1":
        raise ValueError("forced-choice method mismatch")
    if request.get("answer_codes") != ["A", "B", "C", "D", "E"]:
        raise ValueError("forced-choice answer-code contract mismatch")
    if request.get("temperature") != 1.0:
        raise ValueError("forced-choice temperature mismatch")
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or hashlib.sha256(prompt.encode()).hexdigest() != request.get("prompt_sha256"):
        raise ValueError("forced-choice prompt hash mismatch")

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
    template_kwargs: dict[str, Any] = {}
    if model_id.startswith("qwen3_"):
        template_kwargs["enable_thinking"] = False
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, **template_kwargs
    )
    prefix_ids = tokenizer(
        rendered, add_special_tokens=False
    )["input_ids"]
    candidate_ids: list[int] = []
    for code in request["answer_codes"]:
        full_ids = tokenizer(
            rendered + code, add_special_tokens=False
        )["input_ids"]
        if full_ids[:-1] != prefix_ids or len(full_ids) != len(prefix_ids) + 1:
            raise RuntimeError(f"answer code is not one appended token: {code}")
        token_id = full_ids[-1]
        if tokenizer.decode([token_id], clean_up_tokenization_spaces=False) != code:
            raise RuntimeError(f"answer token does not decode exactly: {code}")
        candidate_ids.append(token_id)
    if len(set(candidate_ids)) != 5:
        raise RuntimeError("answer codes do not map to distinct tokens")

    loaded_model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True
    )
    loaded_model.eval()
    inputs = tokenizer(
        rendered, return_tensors="pt", add_special_tokens=False
    ).to("cuda")
    if inputs["input_ids"].shape[-1] > model["maximum_context_tokens"]:
        raise ValueError("forced-choice prompt exceeds frozen context limit")
    started = time.monotonic()
    with torch.inference_mode():
        logits = loaded_model(**inputs, use_cache=False).logits[0, -1]
        choice_logits = logits[
            torch.tensor(candidate_ids, device=logits.device)
        ].float()
        probabilities = torch.softmax(choice_logits, dim=0).cpu().tolist()
    runtime.update(
        {
            "call_id": request["call_id"],
            "prompt_sha256": request["prompt_sha256"],
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    response = {
        "call_id": request["call_id"],
        "model_id": model_id,
        "probabilities_by_code": {
            code: float(probability)
            for code, probability in zip(request["answer_codes"], probabilities)
        },
        "candidate_token_ids": candidate_ids,
        "candidate_token_strings": list(request["answer_codes"]),
        "runtime_attestation": runtime,
    }
    return json.loads(json.dumps(response, allow_nan=False))


def materialized_inference_image_id() -> str:
    image_id = inference_image.object_id
    if not image_id:
        raise RuntimeError("Modal did not hydrate the forced-choice image")
    return image_id
