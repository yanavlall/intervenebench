"""Pinned Modal worker for outcome-blind confirmation inference.

This module intentionally exposes no checkpoint-download function.  It can use
only previously verified public checkpoint caches and a separately authorized,
hash-bound call plan.
"""

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
    CONFIG_PATH = ROOT / "configs/simulators/confirmation_execution_v1.json"
    PLAN_PATH = ROOT / "data/manifests/simulators/confirmation_call_plan_v1.json"
    TEXT_MANIFEST_PATH = ROOT / "data/manifests/simulators/model_file_manifests_v1.json"
    MM_MANIFEST_PATH = ROOT / "data/manifests/simulators/multimodal_model_file_manifests_v1.json"
    LOCK_PATH = ROOT / "infra/modal/multimodal-requirements.lock"
    STIMULI_PATH = ROOT / "data/derived/stimuli/pb2rr"
    SOURCE_PATH = Path(__file__)
else:
    CONFIG_PATH = Path("/opt/intervenebench/confirmation_execution_v1.json")
    PLAN_PATH = Path("/opt/intervenebench/confirmation_call_plan_v1.json")
    TEXT_MANIFEST_PATH = Path("/opt/intervenebench/model_file_manifests_v1.json")
    MM_MANIFEST_PATH = Path("/opt/intervenebench/multimodal_model_file_manifests_v1.json")
    LOCK_PATH = Path("/opt/intervenebench/multimodal-requirements.lock")
    STIMULI_PATH = Path("/opt/intervenebench/data/derived/stimuli/pb2rr")
    SOURCE_PATH = Path("/root/confirmation_app.py")


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
PLAN_ENVELOPE = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
if _payload_hash(PLAN_ENVELOPE["payload"]) != PLAN_ENVELOPE["sha256"]:
    raise RuntimeError("embedded confirmation plan envelope is invalid")
PLAN = PLAN_ENVELOPE["payload"]
if _payload_hash(PLAN) != CONFIG["call_plan"]["payload_sha256"]:
    raise RuntimeError("confirmation config/plan binding mismatch")
MODELS = {item["model_id"]: item for item in CONFIG["models"]}
CALLS = {item["call_id"]: item for item in PLAN["calls"]}
TEXT_MANIFEST = json.loads(TEXT_MANIFEST_PATH.read_text(encoding="utf-8"))
MM_MANIFEST = json.loads(MM_MANIFEST_PATH.read_text(encoding="utf-8"))
FILE_MANIFESTS = {
    item["model_id"]: item
    for manifest in (TEXT_MANIFEST, MM_MANIFEST)
    for item in manifest["models"]
}
MODEL_ROOT = Path("/models")
IMAGE_RECIPE_SHA256 = _payload_hash(CONFIG["runtime"]["image_recipe"])

app = modal.App(CONFIG["runtime"]["app_name"], include_source=False)
model_volume = modal.Volume.from_name(
    CONFIG["runtime"]["model_volume_name"], create_if_missing=False
)
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        requirements=[str(LOCK_PATH)],
        extra_options="--require-hashes",
        uv_version="0.12.4",
    )
    .add_local_file(CONFIG_PATH, "/opt/intervenebench/confirmation_execution_v1.json", copy=True)
    .add_local_file(PLAN_PATH, "/opt/intervenebench/confirmation_call_plan_v1.json", copy=True)
    .add_local_file(TEXT_MANIFEST_PATH, "/opt/intervenebench/model_file_manifests_v1.json", copy=True)
    .add_local_file(MM_MANIFEST_PATH, "/opt/intervenebench/multimodal_model_file_manifests_v1.json", copy=True)
    .add_local_file(LOCK_PATH, "/opt/intervenebench/multimodal-requirements.lock", copy=True)
    .add_local_file(SOURCE_PATH, "/root/confirmation_app.py", copy=True)
)
inference_image = base_image.add_local_dir(
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


def _verify_model_files(manifest_model_id: str, model_path: Path) -> None:
    manifest = FILE_MANIFESTS[manifest_model_id]
    for relative, size, algorithm, digest in manifest["files"]:
        path = model_path / relative
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"cached checkpoint file mismatch: {relative}")
        if _hash_file(path, algorithm) != digest:
            raise RuntimeError(f"cached checkpoint digest mismatch: {relative}")


def _runtime_attestation(
    *, cache_model_id: str, cache_attestation: dict[str, Any]
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
        raise RuntimeError("remote dependency version mismatch")
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
        "dependency_lock_sha256": CONFIG["runtime"]["dependency_lock_sha256"],
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__).split("+")[0],
        "transformers_version": transformers.__version__,
        "pillow_version": PIL.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "nvidia_smi": nvidia_smi,
        "cache_model_id": cache_model_id,
        "checkpoint_commit": MODELS[cache_model_id]["checkpoint_commit"],
        "cache_attestation_sha256": _payload_hash(cache_attestation),
        "modal_function_call_id": modal.current_function_call_id(),
        "modal_input_id": modal.current_input_id(),
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


def _render_text(tokenizer: Any, prompt: str, *, qwen3: bool) -> str:
    kwargs = {"enable_thinking": False} if qwen3 else {}
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


@app.function(
    image=inference_image,
    gpu="L40S:1",
    volumes={MODEL_ROOT: model_volume.with_mount_options(read_only=True)},
    max_containers=6,
    single_use_containers=True,
    timeout=10800,
    startup_timeout=1800,
    scaledown_window=2,
    retries=0,
    block_network=True,
    restrict_modal_access=True,
    name="run_confirmation_checkpoint_group",
    include_source=False,
)
def run_confirmation_checkpoint_group(
    cache_model_id: str,
    requests: list[dict[str, Any]],
    expected_modal_image_id: str,
    expected_cache_attestation_sha256: str,
) -> dict[str, Any]:
    """Load one verified checkpoint and execute its complete planned call group."""

    import torch
    from PIL import Image
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
    )

    expected_count = CONFIG["planned_calls_by_cache_model"].get(cache_model_id)
    if cache_model_id not in MODELS or len(requests) != expected_count:
        raise ValueError("confirmation checkpoint group scope mismatch")
    if len({request.get("call_id") for request in requests}) != len(requests):
        raise ValueError("confirmation group contains duplicate call IDs")
    for request in requests:
        call_id = request.get("call_id")
        frozen = CALLS.get(call_id)
        if frozen is None or frozen["stage"] == "outcome_free_adaptive_reserve":
            raise ValueError("request is outside the planned confirmation call set")
        without_prompt = {key: value for key, value in request.items() if key != "prompt"}
        if without_prompt != frozen:
            raise ValueError("confirmation request metadata differs from frozen plan")
        prompt = request.get("prompt")
        if (
            not isinstance(prompt, str)
            or hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            != request.get("prompt_sha256")
        ):
            raise ValueError("confirmation prompt hash mismatch")
        runtime_cache_id = MODELS[request["model_id"]]["cache_model_id"]
        if runtime_cache_id != cache_model_id:
            raise ValueError("request assigned to the wrong checkpoint group")

    model = MODELS[cache_model_id]
    model_path = MODEL_ROOT / cache_model_id / model["checkpoint_commit"]
    cache_path = model_path / "cache_attestation.json"
    if not cache_path.is_file():
        raise RuntimeError("verified checkpoint cache is absent")
    cache_attestation = json.loads(cache_path.read_text(encoding="utf-8"))
    _verify_model_files(cache_model_id, model_path)
    if _payload_hash(cache_attestation) != expected_cache_attestation_sha256:
        raise RuntimeError("cache attestation authorization mismatch")
    runtime = _runtime_attestation(
        cache_model_id=cache_model_id, cache_attestation=cache_attestation
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
            if cache_model_id == "qwen3_vl_8b_primary"
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
        if time.monotonic() - started > 9000:
            raise TimeoutError("confirmation per-checkpoint GPU ledger expired")
        image = None
        if is_vision:
            asset_relative = request.get("asset_path")
            asset_sha256 = request.get("asset_sha256")
            if not isinstance(asset_relative, str) or not isinstance(asset_sha256, str):
                raise ValueError("vision confirmation request lacks its frozen asset")
            asset_path = Path("/opt/intervenebench") / asset_relative
            if (
                not asset_path.is_file()
                or hashlib.sha256(asset_path.read_bytes()).hexdigest() != asset_sha256
            ):
                raise RuntimeError("confirmation vision asset hash mismatch")
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
                raise ValueError("text confirmation request unexpectedly received an asset")
            rendered = _render_text(
                tokenizer,
                request["prompt"],
                qwen3=cache_model_id.startswith("qwen3_"),
            )
            inputs = tokenizer(
                rendered, return_tensors="pt", add_special_tokens=False
            ).to("cuda")
        if inputs["input_ids"].shape[-1] > model["maximum_context_tokens"]:
            raise ValueError("confirmation prompt exceeds frozen context limit")

        result: dict[str, Any] = {
            "call_id": request["call_id"],
            "model_id": request["model_id"],
        }
        if request["method_id"] == "forced_choice_next_token_softmax.v1":
            candidate_ids = _candidate_token_ids(
                tokenizer, rendered, request["answer_codes"]
            )
            with torch.inference_mode():
                logits = loaded_model(**inputs, use_cache=False).logits[0, -1]
                probabilities = torch.softmax(
                    logits[
                        torch.tensor(candidate_ids, device=logits.device)
                    ].float(),
                    dim=0,
                ).cpu().tolist()
            result.update(
                {
                    "probabilities_by_code": {
                        code: float(probability)
                        for code, probability in zip(
                            request["answer_codes"], probabilities, strict=True
                        )
                    },
                    "candidate_token_ids": candidate_ids,
                    "candidate_token_strings": list(request["answer_codes"]),
                }
            )
        elif request["method_id"] == "continuous_constrained_integer_generation.v1":
            if is_vision:
                raise ValueError("continuous generation cannot use a vision checkpoint")
            torch.manual_seed(int(request["generation_seed"]))
            torch.cuda.manual_seed_all(int(request["generation_seed"]))
            with torch.inference_mode():
                output_ids = loaded_model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=float(request["temperature"]),
                    top_p=float(request["top_p"]),
                    max_new_tokens=int(request["max_new_tokens"]),
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = output_ids[0, inputs["input_ids"].shape[-1] :]
            result["raw_text"] = tokenizer.decode(
                generated, skip_special_tokens=True
            ).strip()
            result["generation_seed"] = int(request["generation_seed"])
        else:
            raise ValueError("unsupported confirmation inference method")
        call_runtime = dict(runtime)
        call_runtime.update(
            {
                "call_id": request["call_id"],
                "prompt_sha256": request["prompt_sha256"],
                "asset_sha256": request.get("asset_sha256"),
                "method_id": request["method_id"],
            }
        )
        result["runtime_attestation"] = call_runtime
        results.append(result)
        if image is not None:
            image.close()
    return json.loads(
        json.dumps(
            {
                "cache_model_id": cache_model_id,
                "elapsed_seconds": time.monotonic() - started,
                "results": results,
            },
            allow_nan=False,
        )
    )


def materialized_inference_image_id() -> str:
    image_id = inference_image.object_id
    if not image_id:
        raise RuntimeError("Modal did not hydrate the confirmation inference image")
    return image_id

