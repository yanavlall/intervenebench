from __future__ import annotations

import ast
from pathlib import Path

from intervenebench.protocol import verify_envelope


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "infra/modal/evidence_report_app.py"
WRAPPER_PATH = ROOT / "scripts/run_evidence_report_eval.py"
FREEZE_PATH = ROOT / "configs/simulators/evidence_report_execution_v1.json"


def test_modal_app_has_two_import_smokes_two_inference_workers_and_no_download_path() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    assert "snapshot_download" not in source
    assert "hf_hub_download" not in source
    assert source.count("@app.function(") == 4
    assert source.count("retries=0") == 4
    assert source.count("block_network=True") == 4
    assert "create_if_missing=False" in source
    assert "with_mount_options(read_only=True)" in source
    assert "smoke_qwen_evidence_report_import" in source
    assert "smoke_mistral_evidence_report_import" in source
    assert "run_qwen_evidence_report_group" in source
    assert "run_mistral_evidence_report_group" in source
    assert 'SOURCE_PATH = Path("/root/evidence_report_app.py")' in source
    assert source.count('"/root/evidence_report_app.py"') == 3


def test_each_image_embeds_every_file_loaded_during_remote_module_import() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    qwen_block = source[
        source.index("qwen_image ="):source.index("mistral_image =")
    ]
    mistral_block = source[
        source.index("mistral_image ="):source.index("def _hash_file")
    ]
    required_remote_paths = {
        "/opt/intervenebench/evidence_report_execution_v1.json",
        "/opt/intervenebench/report_generation_plan_v1.json",
        "/opt/intervenebench/model_file_manifests_v1.json",
        "/opt/intervenebench/mistral_small_3_1_24b_source_manifest_v1.json",
        "/opt/intervenebench/multimodal-requirements.lock",
        "/opt/intervenebench/cross-family-requirements.lock",
        "/root/evidence_report_app.py",
    }
    for image_block in (qwen_block, mistral_block):
        for path in required_remote_paths:
            assert path in image_block


def test_local_wrapper_does_not_import_modal_before_authority_validation() -> None:
    tree = ast.parse(WRAPPER_PATH.read_text(encoding="utf-8"))
    top_level_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        not (
            isinstance(node, ast.Import)
            and any(alias.name == "modal" for alias in node.names)
        )
        and not (isinstance(node, ast.ImportFrom) and node.module == "modal")
        for node in top_level_imports
    )
    source = WRAPPER_PATH.read_text(encoding="utf-8")
    assert 'spec_from_file_location("evidence_report_app", APP_PATH)' in source
    assert source.index("validate_report_materialization_authorization(") < source.index(
        "modal_app = _load_modal_app()"
    )
    assert source.index("validate_report_import_smoke_authorization(") < source.index(
        "modal_app = _load_modal_app()", source.index("def smoke(")
    )


def test_execution_freeze_bars_downloads_retries_scores_and_automatic_judging() -> None:
    freeze = verify_envelope(FREEZE_PATH, require_blinded=True)
    assert freeze["authority"] == {
        "modal_image_materialization_authorized": False,
        "model_download_authorized": False,
        "inference_authorized": False,
        "automatic_retries_authorized": False,
        "reserve_calls_authorized": False,
        "participant_row_access_authorized": False,
        "experiment_level_human_score_access_authorized": False,
        "automatic_judging_authorized": False,
        "automatic_next_stage_authorized": False,
    }
    assert freeze["privacy"]["participant_rows_allowed"] is False
    assert freeze["privacy"]["experiment_level_human_scores_allowed"] is False
    assert freeze["limits"]["automatic_retries"] == 0
    assert freeze["limits"]["reserve_calls"] == 0
