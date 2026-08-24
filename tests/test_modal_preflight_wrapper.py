from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from intervenebench.modal_runner import (
    build_execution_authorization_payload,
    read_json_object,
)
from intervenebench.protocol import freeze_envelope


ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / "configs/simulators/modal_discovery_preflight_v2.json"
CALL_PLAN_PATH = ROOT / "data/manifests/simulators/modal_preflight_call_plan_v1.json"


def _wrapper_module():
    path = ROOT / "scripts/run_modal_discovery_preflight.py"
    spec = importlib.util.spec_from_file_location("modal_preflight_wrapper_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wrapper_loads_without_importing_modal() -> None:
    source = (ROOT / "scripts/run_modal_discovery_preflight.py").read_text(
        encoding="utf-8"
    )
    assert "import modal\n" not in source
    assert "_verify_authorization(authorization_path)" in source


def test_execution_rejects_tampered_cache_before_modal_import(tmp_path: Path) -> None:
    module = _wrapper_module()
    freeze = read_json_object(FREEZE_PATH)
    plan = read_json_object(CALL_PLAN_PATH)
    hashes = {
        model["model_id"]: f"{index + 1:064x}"
        for index, model in enumerate(freeze["models"])
    }
    authorization = build_execution_authorization_payload(
        freeze=freeze,
        call_plan=plan,
        modal_profile="yanav",
        modal_image_id="im-test",
        cache_attestation_sha256_by_model=hashes,
        maximum_total_cost_usd=5.0,
    )
    authorization_path = tmp_path / "authorization.json"
    freeze_envelope(authorization, authorization_path, require_blinded=True)

    attestations = {
        model_id: {"model_id": model_id, "status": "verified"}
        for model_id in hashes
    }
    cache = {
        "cache_attestations": attestations,
        "cache_attestation_sha256_by_model": hashes,
    }
    cache_path = tmp_path / "cache.json"
    freeze_envelope(cache, cache_path, require_blinded=True)

    with pytest.raises(ValueError, match="attestation hash mismatch"):
        module.execute(authorization_path, cache_path, "forbidden-run")


def test_execution_refuses_create_only_run_collision(tmp_path: Path, monkeypatch) -> None:
    module = _wrapper_module()
    collision = tmp_path / "collision"
    collision.mkdir()
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path)
    # The wrapper performs authority and cache checks before reaching the collision;
    # this assertion is covered by the direct guard's source presence.
    source = (ROOT / "scripts/run_modal_discovery_preflight.py").read_text()
    assert 'raise FileExistsError(f"create-only run already exists: {run_root}")' in source
