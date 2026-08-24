"""Pinned Modal workers for evidence-grounded report generation.

The images contain a public aggregate evidence call plan and verified inference
code. They expose no checkpoint-download function and cannot access participant
rows, experiment-level score stores, human labels, or automated-judge outputs.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import time
from typing import Any

import modal


if modal.is_local():
    ROOT = Path(__file__).resolve().parents[2]
    CONFIG_PATH = ROOT / "configs/simulators/evidence_report_execution_v1.json"
    PLAN_PATH = ROOT / (
        "data/manifests/qualitative_eval/report_generation_plan_v1.json"
    )
    QWEN_MANIFEST_PATH = ROOT / (
        "data/manifests/simulators/model_file_manifests_v1.json"
    )
    MISTRAL_MANIFEST_PATH = ROOT / (
        "data/manifests/simulators/mistral_small_3_1_24b_source_manifest_v1.json"
    )
    QWEN_LOCK_PATH = ROOT / "infra/modal/multimodal-requirements.lock"
    MISTRAL_LOCK_PATH = ROOT / "infra/modal/cross-family-requirements.lock"
    SOURCE_PATH = Path(__file__)
else:
    CONFIG_PATH = Path("/opt/intervenebench/evidence_report_execution_v1.json")
    PLAN_PATH = Path("/opt/intervenebench/report_generation_plan_v1.json")
    QWEN_MANIFEST_PATH = Path("/opt/intervenebench/model_file_manifests_v1.json")
    MISTRAL_MANIFEST_PATH = Path(
        "/opt/intervenebench/mistral_small_3_1_24b_source_manifest_v1.json"
    )
    QWEN_LOCK_PATH = Path("/opt/intervenebench/multimodal-requirements.lock")
    MISTRAL_LOCK_PATH = Path("/opt/intervenebench/cross-family-requirements.lock")
    # The local authority wrapper imports this file as the top-level module
    # ``evidence_report_app``. Modal workers must receive it at the matching
    # top-level import path.
    SOURCE_PATH = Path("/root/evidence_report_app.py")


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


def _load_envelope(path: Path) -> tuple[dict[str, Any], str]:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if set(envelope) != {"payload", "sha256"}:
        raise RuntimeError(f"malformed embedded envelope: {path.name}")
    payload = envelope["payload"]
    if not isinstance(payload, dict) or _payload_hash(payload) != envelope["sha256"]:
        raise RuntimeError(f"invalid embedded envelope hash: {path.name}")
    return payload, envelope["sha256"]


CONFIG, CONFIG_SHA256 = _load_envelope(CONFIG_PATH)
PLAN, PLAN_SHA256 = _load_envelope(PLAN_PATH)
MISTRAL_MANIFEST, MISTRAL_MANIFEST_SHA256 = _load_envelope(
    MISTRAL_MANIFEST_PATH
)
QWEN_MANIFEST = json.loads(QWEN_MANIFEST_PATH.read_text(encoding="utf-8"))
if PLAN_SHA256 != CONFIG["generation_plan_payload_sha256"]:
    raise RuntimeError("embedded report call plan differs from execution freeze")
if CONFIG["planned_call_count"] != 48 or len(PLAN["calls"]) != 48:
    raise RuntimeError("embedded report panel is not the frozen 48-call grid")
if any(CONFIG["authority"].values()):
    raise RuntimeError("embedded report execution freeze must have zero authority")

CALLS = {call["call_id"]: call for call in PLAN["calls"]}
MODELS = {model["model_role"]: model for model in CONFIG["models"]}
QWEN_FILE_MANIFESTS = {
    model["model_id"]: model for model in QWEN_MANIFEST["models"]
}
MODEL_ROOT = Path("/models")

app = modal.App(CONFIG["runtime"]["app_name"], include_source=False)
model_volume = modal.Volume.from_name(
    CONFIG["runtime"]["model_volume_name"], create_if_missing=False
)

REMOTE_EMBEDDED_PATHS = (
    CONFIG_PATH,
    PLAN_PATH,
    QWEN_MANIFEST_PATH,
    MISTRAL_MANIFEST_PATH,
    QWEN_LOCK_PATH,
    MISTRAL_LOCK_PATH,
    SOURCE_PATH,
)

qwen_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        requirements=[str(QWEN_LOCK_PATH)],
        extra_options="--require-hashes",
        uv_version="0.12.4",
    )
    .add_local_file(
        CONFIG_PATH,
        "/opt/intervenebench/evidence_report_execution_v1.json",
        copy=True,
    )
    .add_local_file(
        PLAN_PATH,
        "/opt/intervenebench/report_generation_plan_v1.json",
        copy=True,
    )
    .add_local_file(
        QWEN_MANIFEST_PATH,
        "/opt/intervenebench/model_file_manifests_v1.json",
        copy=True,
    )
    .add_local_file(
        MISTRAL_MANIFEST_PATH,
        "/opt/intervenebench/mistral_small_3_1_24b_source_manifest_v1.json",
        copy=True,
    )
    .add_local_file(
        QWEN_LOCK_PATH,
        "/opt/intervenebench/multimodal-requirements.lock",
        copy=True,
    )
    .add_local_file(
        MISTRAL_LOCK_PATH,
        "/opt/intervenebench/cross-family-requirements.lock",
        copy=True,
    )
    .add_local_file(
        SOURCE_PATH,
        "/root/evidence_report_app.py",
        copy=True,
    )
)
mistral_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(
        requirements=[str(MISTRAL_LOCK_PATH)],
        extra_options="--require-hashes",
        uv_version="0.12.4",
    )
    .add_local_file(
        CONFIG_PATH,
        "/opt/intervenebench/evidence_report_execution_v1.json",
        copy=True,
    )
    .add_local_file(
        PLAN_PATH,
        "/opt/intervenebench/report_generation_plan_v1.json",
        copy=True,
    )
    .add_local_file(
        QWEN_MANIFEST_PATH,
        "/opt/intervenebench/model_file_manifests_v1.json",
        copy=True,
    )
    .add_local_file(
        MISTRAL_MANIFEST_PATH,
        "/opt/intervenebench/mistral_small_3_1_24b_source_manifest_v1.json",
        copy=True,
    )
    .add_local_file(
        QWEN_LOCK_PATH,
        "/opt/intervenebench/multimodal-requirements.lock",
        copy=True,
    )
    .add_local_file(
        MISTRAL_LOCK_PATH,
        "/opt/intervenebench/cross-family-requirements.lock",
        copy=True,
    )
    .add_local_file(
        SOURCE_PATH,
        "/root/evidence_report_app.py",
        copy=True,
    )
)


def _hash_file(path: Path, algorithm: str) -> str:
    if algorithm == "sha256":
        digest = hashlib.sha256()
        prefix = b""
    elif algorithm == "git_blob_sha1":
        digest = hashlib.sha1()
        prefix = f"blob {path.stat().st_size}\0".encode("ascii")
    else:
        raise RuntimeError("unsupported checkpoint digest algorithm")
    digest.update(prefix)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _group_calls(model_role: str) -> list[dict[str, Any]]:
    if model_role not in MODELS:
        raise RuntimeError("report model role is not frozen")
    calls = sorted(
        (call for call in PLAN["calls"] if call["model_role"] == model_role),
        key=lambda call: call["call_id"],
    )
    if len(calls) != CONFIG["planned_calls_by_model_role"][model_role]:
        raise RuntimeError("report model group has incomplete call coverage")
    for call in calls:
        if _payload_hash(call["prompt"]) != call["prompt_sha256"]:
            raise RuntimeError("embedded report prompt hash drifted")
    return calls


def _runtime_attestation(
    *,
    model_role: str,
    expected_image_id: str,
    expected_freeze_payload_sha256: str,
    cache_attestation_payload_sha256: str,
) -> dict[str, Any]:
    import torch
    import transformers

    if expected_freeze_payload_sha256 != CONFIG_SHA256:
        raise RuntimeError("report worker is bound to another execution freeze")
    recipe_key = (
        "qwen_image_recipe"
        if MODELS[model_role]["runtime_family"] == "qwen_transformers"
        else "mistral_image_recipe"
    )
    if _hash_file(SOURCE_PATH, "sha256") != CONFIG["runtime"][recipe_key][
        "app_source_sha256"
    ]:
        raise RuntimeError("report worker source hash differs from execution freeze")
    image_id = os.environ.get("MODAL_IMAGE_ID", "")
    if not image_id or image_id != expected_image_id:
        raise RuntimeError("report worker Modal image authorization mismatch")
    model = MODELS[model_role]
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    if (model["gpu"].startswith("L40S") and "L40S" not in gpu_name) or (
        model["gpu"].startswith("A100") and "A100" not in gpu_name
    ):
        raise RuntimeError(f"report worker GPU mismatch: {gpu_name!r}")
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
        "execution_freeze_payload_sha256": CONFIG_SHA256,
        "generation_plan_payload_sha256": PLAN_SHA256,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "gpu_name": gpu_name,
        "cuda_runtime_version": torch.version.cuda,
        "nvidia_smi": smi,
        "model_role": model_role,
        "cache_model_id": model["cache_model_id"],
        "checkpoint_commit": model["checkpoint_commit"],
        "cache_attestation_payload_sha256": cache_attestation_payload_sha256,
        "modal_function_call_id": str(modal.current_function_call_id()),
        "modal_input_id": str(modal.current_input_id()),
    }


def _verify_qwen_cache(model_role: str, expected_cache_sha256: str) -> Path:
    model = MODELS[model_role]
    if model["runtime_family"] != "qwen_transformers":
        raise RuntimeError("non-Qwen model requested through Qwen worker")
    model_path = (
        MODEL_ROOT / model["cache_model_id"] / model["checkpoint_commit"]
    )
    attestation_path = model_path / "cache_attestation.json"
    if not attestation_path.is_file():
        raise RuntimeError("verified Qwen cache is absent")
    manifest = QWEN_FILE_MANIFESTS[model["cache_model_id"]]
    for relative, size, algorithm, digest in manifest["files"]:
        path = model_path / relative
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"cached Qwen file mismatch: {relative}")
        if _hash_file(path, algorithm) != digest:
            raise RuntimeError(f"cached Qwen digest mismatch: {relative}")
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    actual = _payload_hash(attestation)
    if actual != expected_cache_sha256 or actual != model[
        "cache_attestation_payload_sha256"
    ]:
        raise RuntimeError("Qwen cache attestation authorization mismatch")
    return model_path


def _verify_mistral_cache(expected_cache_sha256: str) -> Path:
    model_role = "mistral_small_3_1_24b_cross_family"
    model = MODELS[model_role]
    model_path = (
        MODEL_ROOT / model["cache_model_id"] / model["checkpoint_commit"]
    )
    attestation_path = model_path / "cache_attestation.json"
    if not attestation_path.is_file():
        raise RuntimeError("verified Mistral cache is absent")
    verified: list[dict[str, Any]] = []
    for item in MISTRAL_MANIFEST["files"]:
        path = model_path / item["path"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"]:
            raise RuntimeError(f"cached Mistral file mismatch: {item['path']}")
        content_sha256 = _hash_file(path, "sha256")
        if item["content_sha256"] is not None:
            if content_sha256 != item["content_sha256"]:
                raise RuntimeError(f"cached Mistral digest mismatch: {item['path']}")
            source_verification = "source_content_sha256"
        else:
            if _hash_file(path, "git_blob_sha1") != item["git_oid"]:
                raise RuntimeError(f"cached Mistral Git digest mismatch: {item['path']}")
            source_verification = "source_git_blob_sha1"
        verified.append(
            {
                "path": item["path"],
                "size_bytes": item["size_bytes"],
                "content_sha256": content_sha256,
                "source_verification": source_verification,
            }
        )
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    if attestation.get("verified_files") != verified:
        raise RuntimeError("Mistral cache attestation/file replay failed")
    actual = _payload_hash(attestation)
    if actual != expected_cache_sha256 or actual != model[
        "cache_attestation_payload_sha256"
    ]:
        raise RuntimeError("Mistral cache attestation authorization mismatch")
    return model_path


def _remote_import_smoke(
    *,
    image_kind: str,
    expected_freeze_payload_sha256: str,
    expected_modal_image_id: str,
) -> dict[str, Any]:
    if image_kind not in {"qwen", "mistral"}:
        raise RuntimeError("unknown report image kind")
    if expected_freeze_payload_sha256 != CONFIG_SHA256:
        raise RuntimeError("import smoke is bound to another execution freeze")
    image_id = os.environ.get("MODAL_IMAGE_ID", "")
    if not image_id or image_id != expected_modal_image_id:
        raise RuntimeError("import smoke Modal image authorization mismatch")
    recipe_key = f"{image_kind}_image_recipe"
    if _hash_file(SOURCE_PATH, "sha256") != CONFIG["runtime"][recipe_key][
        "app_source_sha256"
    ]:
        raise RuntimeError("import-smoke source hash differs from execution freeze")
    missing = [str(path) for path in REMOTE_EMBEDDED_PATHS if not path.is_file()]
    if missing:
        raise RuntimeError(f"import-smoke embedded files absent: {missing}")
    return {
        "schema_version": "intervenebench.report_eval_remote_import_smoke.v1",
        "image_kind": image_kind,
        "modal_image_id": image_id,
        "execution_freeze_payload_sha256": CONFIG_SHA256,
        "source_sha256": _hash_file(SOURCE_PATH, "sha256"),
        "verified_embedded_paths": sorted(str(path) for path in REMOTE_EMBEDDED_PATHS),
        "model_downloaded": False,
        "inference_performed": False,
        "participant_rows_accessed": 0,
        "experiment_level_human_scores_accessed": False,
    }


@app.function(
    image=qwen_image,
    cpu=0.25,
    memory=256,
    timeout=300,
    retries=0,
    max_containers=1,
    single_use_containers=True,
    block_network=True,
    restrict_modal_access=True,
    name="smoke_qwen_evidence_report_import",
    include_source=False,
)
def smoke_qwen_evidence_report_import(
    expected_freeze_payload_sha256: str,
    expected_modal_image_id: str,
) -> dict[str, Any]:
    return _remote_import_smoke(
        image_kind="qwen",
        expected_freeze_payload_sha256=expected_freeze_payload_sha256,
        expected_modal_image_id=expected_modal_image_id,
    )


@app.function(
    image=mistral_image,
    cpu=0.25,
    memory=256,
    timeout=300,
    retries=0,
    max_containers=1,
    single_use_containers=True,
    block_network=True,
    restrict_modal_access=True,
    name="smoke_mistral_evidence_report_import",
    include_source=False,
)
def smoke_mistral_evidence_report_import(
    expected_freeze_payload_sha256: str,
    expected_modal_image_id: str,
) -> dict[str, Any]:
    return _remote_import_smoke(
        image_kind="mistral",
        expected_freeze_payload_sha256=expected_freeze_payload_sha256,
        expected_modal_image_id=expected_modal_image_id,
    )


@app.function(
    image=qwen_image,
    gpu="L40S:1",
    volumes={MODEL_ROOT: model_volume.with_mount_options(read_only=True)},
    timeout=10_800,
    startup_timeout=1_800,
    scaledown_window=2,
    retries=0,
    max_containers=2,
    single_use_containers=True,
    block_network=True,
    restrict_modal_access=True,
    name="run_qwen_evidence_report_group",
    include_source=False,
)
def run_qwen_evidence_report_group(
    model_role: str,
    expected_freeze_payload_sha256: str,
    expected_modal_image_id: str,
    expected_cache_attestation_sha256: str,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    calls = _group_calls(model_role)
    model_path = _verify_qwen_cache(model_role, expected_cache_attestation_sha256)
    runtime = _runtime_attestation(
        model_role=model_role,
        expected_image_id=expected_modal_image_id,
        expected_freeze_payload_sha256=expected_freeze_payload_sha256,
        cache_attestation_payload_sha256=expected_cache_attestation_sha256,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False, use_fast=True
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
    for index, call in enumerate(calls, start=1):
        if time.monotonic() - started > 9_000:
            raise TimeoutError("Qwen report-group GPU ledger expired")
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": call["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(
            rendered, return_tensors="pt", add_special_tokens=False
        ).to("cuda")
        if inputs["input_ids"].shape[-1] > 16_384:
            raise RuntimeError("Qwen report prompt exceeds frozen context")
        torch.manual_seed(int(call["seed"]))
        torch.cuda.manual_seed_all(int(call["seed"]))
        with torch.inference_mode():
            output_ids = loaded_model.generate(
                **inputs,
                do_sample=True,
                temperature=float(PLAN["generation_config"]["temperature"]),
                top_p=float(PLAN["generation_config"]["top_p"]),
                max_new_tokens=int(
                    PLAN["generation_config"]["maximum_new_tokens"]
                ),
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[0, inputs["input_ids"].shape[-1] :]
        raw_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        results.append(
            {
                "call_id": call["call_id"],
                "model_role": model_role,
                "prompt_sha256": call["prompt_sha256"],
                "raw_text": raw_text,
                "runtime_attestation": {
                    **runtime,
                    "call_id": call["call_id"],
                    "prompt_sha256": call["prompt_sha256"],
                    "seed": call["seed"],
                },
            }
        )
        print(f"REPORT_GROUP {model_role} progress={index}/{len(calls)}", flush=True)
    return {
        "schema_version": "intervenebench.evidence_report_raw_group.v1",
        "model_role": model_role,
        "attempt_count": len(calls),
        "elapsed_seconds": time.monotonic() - started,
        "results": results,
        "model_downloaded": False,
        "participant_rows_accessed": 0,
        "experiment_level_human_scores_accessed": False,
        "human_labels_accessed": False,
        "automatic_next_stage": False,
    }


@app.function(
    image=mistral_image,
    gpu="A100-80GB:1",
    volumes={MODEL_ROOT: model_volume.with_mount_options(read_only=True)},
    timeout=10_800,
    startup_timeout=1_800,
    scaledown_window=2,
    retries=0,
    max_containers=1,
    single_use_containers=True,
    block_network=True,
    restrict_modal_access=True,
    name="run_mistral_evidence_report_group",
    include_source=False,
)
def run_mistral_evidence_report_group(
    expected_freeze_payload_sha256: str,
    expected_modal_image_id: str,
    expected_cache_attestation_sha256: str,
) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    model_role = "mistral_small_3_1_24b_cross_family"
    calls = _group_calls(model_role)
    model_path = _verify_mistral_cache(expected_cache_attestation_sha256)
    runtime = _runtime_attestation(
        model_role=model_role,
        expected_image_id=expected_modal_image_id,
        expected_freeze_payload_sha256=expected_freeze_payload_sha256,
        cache_attestation_payload_sha256=expected_cache_attestation_sha256,
    )
    llm = LLM(
        model=str(model_path),
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        dtype="bfloat16",
        tensor_parallel_size=1,
        trust_remote_code=False,
        max_model_len=16_384,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
    )
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    for index, call in enumerate(calls, start=1):
        if time.monotonic() - started > 9_000:
            raise TimeoutError("Mistral report-group GPU ledger expired")
        params = SamplingParams(
            temperature=float(PLAN["generation_config"]["temperature"]),
            top_p=float(PLAN["generation_config"]["top_p"]),
            max_tokens=int(PLAN["generation_config"]["maximum_new_tokens"]),
            seed=int(call["seed"]),
        )
        outputs = llm.chat(
            [{"role": "user", "content": call["prompt"]}],
            sampling_params=params,
            use_tqdm=False,
        )
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise RuntimeError("Mistral report call returned unexpected output count")
        raw_text = outputs[0].outputs[0].text.strip()
        results.append(
            {
                "call_id": call["call_id"],
                "model_role": model_role,
                "prompt_sha256": call["prompt_sha256"],
                "raw_text": raw_text,
                "runtime_attestation": {
                    **runtime,
                    "call_id": call["call_id"],
                    "prompt_sha256": call["prompt_sha256"],
                    "seed": call["seed"],
                },
            }
        )
        print(f"REPORT_GROUP {model_role} progress={index}/{len(calls)}", flush=True)
    return {
        "schema_version": "intervenebench.evidence_report_raw_group.v1",
        "model_role": model_role,
        "attempt_count": len(calls),
        "elapsed_seconds": time.monotonic() - started,
        "results": results,
        "model_downloaded": False,
        "participant_rows_accessed": 0,
        "experiment_level_human_scores_accessed": False,
        "human_labels_accessed": False,
        "automatic_next_stage": False,
    }


def materialized_report_image_ids() -> dict[str, str]:
    qwen = qwen_image.object_id
    mistral = mistral_image.object_id
    if not qwen or not mistral:
        raise RuntimeError("Modal did not hydrate both report-eval images")
    return {"qwen": qwen, "mistral": mistral}
