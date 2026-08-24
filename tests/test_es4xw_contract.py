from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from intervenebench.protocol import assert_blinded_payload
from intervenebench.simulators import (
    ordinal_png_multimodal_prompt,
    validate_ordinal_png_multimodal_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = ROOT / "data/manifests/contracts"


def _load(name: str) -> dict:
    return json.loads((CONTRACT_DIR / name).read_text(encoding="utf-8"))


def test_es4xw_contract_is_source_only_sealed_and_runnable() -> None:
    task = _load("es4xw_decision_task_candidate.json")
    bundle = _load("es4xw_blinded_bundle.json")
    assert task["task_id"] == bundle["task_id"]
    assert task["experiment_id"] == bundle["experiment_id"] == "es4xw"
    assert task["socsci210_task_num"] is None
    assert task["socsci210_scoring_status"] == "barred_source_instrument_mismatch"
    assert task["outcome_access"] == bundle["outcome_access"] == "sealed"
    assert task["reveal_authorized"] is bundle["reveal_authorized"] is False
    assert list(task["released_rows_per_arm_before_outcome_missingness"].values()) == [
        636,
        644,
        625,
        607,
    ]
    assert task["source_data_locator"]["authorized_projection"] == [
        "caseid",
        "weight1",
        "XTESS040",
        "Q1",
    ]
    assert set(task["source_variable_mapping"]["assignment_to_arm"].values()) == {
        arm["arm_id"] for arm in task["arms"]
    }
    task_options = [
        {
            "value": option["raw_value"],
            "label": option["label"],
            "normalized_utility": option["normalized_utility"],
        }
        for option in task["response_options"]
    ]
    assert task_options == bundle["response_options"]
    assert_blinded_payload(task)
    assert_blinded_payload(bundle)
    validate_ordinal_png_multimodal_bundle(bundle)
    for arm in bundle["arms"]:
        prompt = ordinal_png_multimodal_prompt(
            bundle, arm_id=arm["arm_id"], repository_root=ROOT
        )
        assert prompt.asset_sha256 == (arm["asset"]["sha256"],)


def test_es4xw_assets_and_source_sav_are_hash_pinned_without_opening_outcomes() -> None:
    task = _load("es4xw_decision_task_candidate.json")
    questionnaire = ROOT / "data/raw/sources/es4xw/Tess2_040_Bauman_final.docx"
    archive = ROOT / task["source_data_locator"]["container_path"]
    assert hashlib.sha256(questionnaire.read_bytes()).hexdigest() == task[
        "source_hashes"
    ]["final_questionnaire_sha256"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == task[
        "source_data_locator"
    ]["container_sha256"]
    with ZipFile(questionnaire) as source:
        for index in range(1, 5):
            member = source.read(f"word/media/image{index}.png")
            derived = (ROOT / f"data/derived/stimuli/es4xw/image{index}.png").read_bytes()
            assert member == derived
            assert hashlib.sha256(member).hexdigest() == task["source_hashes"][
                f"picture_{index}_png_sha256"
            ]
    with ZipFile(archive) as source:
        info = source.getinfo(task["source_data_locator"]["sav_member"])
        assert info.file_size > 0
        # The contract pins the member hash; it is intentionally not read here.
        assert task["source_data_locator"]["sav_member_sha256"] == task[
            "source_hashes"
        ]["source_sav_sha256"]
