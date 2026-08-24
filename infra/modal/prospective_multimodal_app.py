"""Modal cache and inference workers for prospective multimodal development."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import modal


if modal.is_local():
    ROOT = Path(__file__).resolve().parents[2]
    FREEZE_PATH = ROOT / "configs/simulators/prospective_multimodal_v4.json"
    MODEL_MANIFEST_PATH = (
        ROOT
        / "data/manifests/simulators/multimodal_model_file_manifests_v1.json"
    )
    LOCK_PATH = ROOT / "infra/modal/multimodal-requirements.lock"
    STIMULI_PATH = ROOT / "data/derived/stimuli"
    SOURCE_PATH = Path(__file__)
else:
    FREEZE_PATH = Path("/opt/intervenebench/prospective_multimodal_v4.json")
    MODEL_MANIFEST_PATH = Path(
        "/opt/intervenebench/multimodal_model_file_manifests_v1.json"
    )
    LOCK_PATH = Path("/opt/intervenebench/multimodal-requirements.lock")
    STIMULI_PATH = Path("/opt/intervenebench/data/derived/stimuli")
    SOURCE_PATH = Path("/root/prospective_multimodal_app.py")

FREEZE = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
MODEL_FILE_MANIFEST = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
MODELS = {item["model_id"]: item for item in FREEZE["models"]}
FILE_MANIFESTS = {
    item["model_id"]: item for item in MODEL_FILE_MANIFEST["models"]
}
MODEL_ROOT = Path("/models")
CACHE_MODEL_ID = {"qwen3_8b_text_ablation": "qwen3_8b_generic"}


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


def _hash_file(path: Path, algorithm: str) -> str:
    if algorithm == "sha256":
        digest = hashlib.sha256()
        prefix = b""
    elif algorithm == "git_blob_sha1":
        digest = hashlib.sha1()
        prefix = f"blob {path.stat().st_size}\0".encode("ascii")
    else:
        raise ValueError("unsupported model digest algorithm")
    digest.update(prefix)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_model_files(model_id: str, model_path: Path) -> None:
    for relative, size, algorithm, digest in FILE_MANIFESTS[model_id]["files"]:
        path = model_path / relative
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"cached model file mismatch: {relative}")
        if _hash_file(path, algorithm) != digest:
            raise RuntimeError(f"cached model digest mismatch: {relative}")


IMAGE_RECIPE_SHA256 = _payload_hash(FREEZE["runtime"]["image_recipe"])
DEPENDENCY_LOCK_SHA256 = FREEZE["runtime"]["dependency_lock_sha256"]
app = modal.App("intervenebench-prospective-multimodal-v4", include_source=False)
model_volume = modal.Volume.from_name(
    "intervenebench-model-cache-v1", create_if_missing=False
)
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        requirements=[str(LOCK_PATH)],
        extra_options="--require-hashes",
        uv_version="0.12.4",
    )
    .add_local_file(
        MODEL_MANIFEST_PATH,
        "/opt/intervenebench/multimodal_model_file_manifests_v1.json",
        copy=True,
    )
    .add_local_file(
        FREEZE_PATH,
        "/opt/intervenebench/prospective_multimodal_v4.json",
        copy=True,
    )
    .add_local_file(
        LOCK_PATH, "/opt/intervenebench/multimodal-requirements.lock", copy=True
    )
    .add_local_file(SOURCE_PATH, "/root/prospective_multimodal_app.py", copy=True)
)
inference_image = base_image.add_local_dir(
    STIMULI_PATH,
    "/opt/intervenebench/data/derived/stimuli",
    copy=True,
)


@app.function(
    image=base_image,
    cpu=2.0,
    memory=8192,
    volumes={MODEL_ROOT: model_volume},
    timeout=3600,
    retries=0,
    max_containers=2,
    name="cache_multimodal_checkpoint",
    include_source=False,
)
def cache_multimodal_checkpoint(model_id: str) -> dict[str, Any]:
    """Download and verify one allowlisted VLM revision without GPU inference."""

    from huggingface_hub import snapshot_download

    if model_id not in {"qwen3_vl_8b_primary", "qwen2_5_vl_7b_comparator"}:
        raise ValueError("checkpoint is not in the multimodal cache allowlist")
    model = MODELS[model_id]
    target = MODEL_ROOT / model_id / model["checkpoint_commit"]
    attestation_path = target / "cache_attestation.json"
    if attestation_path.is_file():
        _verify_model_files(model_id, target)
        return json.loads(attestation_path.read_text(encoding="utf-8"))
    staging = MODEL_ROOT / ".staging" / f"{model_id}-{modal.current_input_id()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        snapshot_download(
            repo_id=model["hf_repository"],
            revision=model["checkpoint_commit"],
            local_dir=staging,
            token=False,
            max_workers=8,
        )
        _verify_model_files(model_id, staging)
        attestation = {
            "schema_version": "multimodal_checkpoint_cache_attestation.v1",
            "model_id": model_id,
            "hf_repository": model["hf_repository"],
            "checkpoint_commit": model["checkpoint_commit"],
            "model_file_manifest_payload_sha256": _payload_hash(
                MODEL_FILE_MANIFEST
            ),
            "verified_file_count": len(FILE_MANIFESTS[model_id]["files"]),
            "network_disabled_during_inference": True,
        }
        (staging / "cache_attestation.json").write_text(
            json.dumps(attestation, sort_keys=True), encoding="utf-8"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(target)
        model_volume.commit()
        return attestation
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _runtime_attestation(
    *, model_id: str, cache_attestation: dict[str, Any]
) -> dict[str, Any]:
    import PIL
    import torch
    import torchvision
    import transformers

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    if "L40S" not in gpu_name:
        raise RuntimeError(f"frozen GPU mismatch: {gpu_name!r}")
    if platform.python_version().split(".")[:2] != ["3", "11"]:
        raise RuntimeError("remote Python version mismatch")
    if torch.__version__.split("+")[0] != "2.9.1":
        raise RuntimeError("remote torch version mismatch")
    if torchvision.__version__.split("+")[0] != "0.24.1":
        raise RuntimeError("remote torchvision version mismatch")
    if transformers.__version__ != "4.57.6" or PIL.__version__ != "11.3.0":
        raise RuntimeError("remote multimodal dependency mismatch")
    if torch.version.cuda != "12.8":
        raise RuntimeError("remote CUDA runtime mismatch")
    image_id = os.environ.get("MODAL_IMAGE_ID", "")
    if not image_id:
        raise RuntimeError("MODAL_IMAGE_ID is missing")
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
        "modal_sdk_version": "1.5.4",
        "modal_image_id": image_id,
        "image_recipe_sha256": IMAGE_RECIPE_SHA256,
        "dependency_lock_sha256": DEPENDENCY_LOCK_SHA256,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__).split("+")[0],
        "transformers_version": transformers.__version__,
        "pillow_version": PIL.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "nvidia_smi": nvidia_smi,
        "checkpoint_commit": MODELS[model_id]["checkpoint_commit"],
        "model_file_manifest_payload_sha256": _payload_hash(MODEL_FILE_MANIFEST),
        "cache_attestation_sha256": _payload_hash(cache_attestation),
        "modal_function_call_id": modal.current_function_call_id(),
        "modal_input_id": modal.current_input_id(),
        "method_id": "forced_choice_next_token_softmax.v1",
    }


def _candidate_token_ids(tokenizer: Any, rendered: str, codes: list[str]) -> list[int]:
    prefix = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    token_ids: list[int] = []
    for code in codes:
        full = tokenizer(rendered + code, add_special_tokens=False)["input_ids"]
        if full[:-1] != prefix or len(full) != len(prefix) + 1:
            raise RuntimeError(f"answer code is not one appended token: {code}")
        token_id = full[-1]
        if tokenizer.decode([token_id], clean_up_tokenization_spaces=False) != code:
            raise RuntimeError(f"answer token does not decode exactly: {code}")
        token_ids.append(token_id)
    if len(set(token_ids)) != len(token_ids):
        raise RuntimeError("answer codes do not map to distinct tokens")
    return token_ids


@app.function(
    image=inference_image,
    gpu="L40S:1",
    volumes={MODEL_ROOT: model_volume.with_mount_options(read_only=True)},
    max_containers=3,
    single_use_containers=True,
    timeout=900,
    startup_timeout=1200,
    scaledown_window=2,
    retries=0,
    block_network=True,
    restrict_modal_access=True,
    name="run_prospective_multimodal_group",
    include_source=False,
)
def run_prospective_multimodal_group(
    model_id: str,
    requests: list[dict[str, Any]],
    expected_modal_image_id: str,
    expected_cache_attestation_sha256: str,
) -> dict[str, Any]:
    """Load one model and execute its exact eighteen ordered calls."""

    import torch
    from PIL import Image
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
    )

    if model_id not in MODELS or len(requests) != 18:
        raise ValueError("prospective multimodal model scope mismatch")
    if len({request.get("call_id") for request in requests}) != 18:
        raise ValueError("prospective multimodal group has duplicate calls")
    for request in requests:
        if request.get("model_id") != model_id:
            raise ValueError("prospective multimodal request model mismatch")
        if request.get("method_id") != "forced_choice_next_token_softmax.v1":
            raise ValueError("prospective multimodal method mismatch")
        if request.get("temperature") != 1.0:
            raise ValueError("prospective multimodal temperature mismatch")
        prompt = request.get("prompt")
        if (
            not isinstance(prompt, str)
            or hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            != request.get("prompt_sha256")
        ):
            raise ValueError("prospective multimodal prompt hash mismatch")
        if request.get("answer_codes") != list("ABCDEFG"):
            raise ValueError("prospective multimodal answer-code mismatch")

    cache_id = CACHE_MODEL_ID.get(model_id, model_id)
    model = MODELS[model_id]
    model_path = MODEL_ROOT / cache_id / model["checkpoint_commit"]
    cache_path = model_path / "cache_attestation.json"
    if not cache_path.is_file():
        raise RuntimeError("verified checkpoint cache is absent")
    cache_attestation = json.loads(cache_path.read_text(encoding="utf-8"))
    _verify_model_files(model_id, model_path)
    if _payload_hash(cache_attestation) != expected_cache_attestation_sha256:
        raise RuntimeError("cache attestation authorization mismatch")
    runtime = _runtime_attestation(
        model_id=model_id, cache_attestation=cache_attestation
    )
    if runtime["modal_image_id"] != expected_modal_image_id:
        raise RuntimeError("Modal image authorization mismatch")

    is_vision = model["modality"] == "exact_png_vision"
    if is_vision:
        processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        tokenizer = processor.tokenizer
        model_class = (
            Qwen3VLForConditionalGeneration
            if model_id == "qwen3_vl_8b_primary"
            else Qwen2_5_VLForConditionalGeneration
        )
        loaded_model = model_class.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
    else:
        processor = None
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
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    for request in requests:
        if time.monotonic() - started > 780:
            raise TimeoutError("prospective multimodal per-model ledger expired")
        image = None
        if is_vision:
            asset_relative = request.get("asset_path")
            asset_sha256 = request.get("asset_sha256")
            if not isinstance(asset_relative, str) or not isinstance(asset_sha256, str):
                raise ValueError("vision request lacks its frozen asset")
            asset_path = Path("/opt/intervenebench") / asset_relative
            if (
                not asset_path.is_file()
                or hashlib.sha256(asset_path.read_bytes()).hexdigest() != asset_sha256
            ):
                raise RuntimeError("vision asset hash mismatch")
            image = Image.open(asset_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": request["prompt"]},
                    ],
                }
            ]
            rendered = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[rendered], images=[image], padding=True, return_tensors="pt"
            ).to("cuda")
        else:
            if request.get("asset_path") is not None:
                raise ValueError("text ablation unexpectedly received an asset")
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": request["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(
                rendered, return_tensors="pt", add_special_tokens=False
            ).to("cuda")
        candidate_ids = _candidate_token_ids(
            tokenizer, rendered, request["answer_codes"]
        )
        if inputs["input_ids"].shape[-1] > 16384:
            raise ValueError("prospective multimodal prompt exceeds context limit")
        with torch.inference_mode():
            logits = loaded_model(**inputs, use_cache=False).logits[0, -1]
            probabilities = torch.softmax(
                logits[torch.tensor(candidate_ids, device=logits.device)].float(),
                dim=0,
            ).cpu().tolist()
        call_runtime = dict(runtime)
        call_runtime.update(
            {
                "call_id": request["call_id"],
                "prompt_sha256": request["prompt_sha256"],
                "asset_sha256": request.get("asset_sha256"),
            }
        )
        results.append(
            {
                "call_id": request["call_id"],
                "model_id": model_id,
                "probabilities_by_code": {
                    code: float(probability)
                    for code, probability in zip(
                        request["answer_codes"], probabilities
                    )
                },
                "candidate_token_ids": candidate_ids,
                "candidate_token_strings": list(request["answer_codes"]),
                "runtime_attestation": call_runtime,
            }
        )
        if image is not None:
            image.close()
    return json.loads(
        json.dumps(
            {
                "model_id": model_id,
                "elapsed_seconds": time.monotonic() - started,
                "results": results,
            },
            allow_nan=False,
        )
    )


def materialized_inference_image_id() -> str:
    image_id = inference_image.object_id
    if not image_id:
        raise RuntimeError("Modal did not hydrate the multimodal image")
    return image_id
