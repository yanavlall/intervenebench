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


def test_e2pyb_three_arm_contract_is_sealed_and_runnable() -> None:
    task = _load("e2pyb_decision_task_candidate.json")
    bundle = _load("e2pyb_blinded_bundle.json")
    assert task["task_id"] == bundle["task_id"]
    assert task["experiment_id"] == bundle["experiment_id"] == "e2pyb"
    assert task["socsci210_task_num"] == 0
    assert task["outcome_access"] == bundle["outcome_access"] == "sealed"
    assert task["reveal_authorized"] is bundle["reveal_authorized"] is False
    assert [arm["socsci_condition_num"] for arm in task["arms"]] == [0, 1, 2]
    assert list(task["released_rows_per_arm_before_outcome_missingness"].values()) == [
        540,
        532,
        528,
    ]
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


def test_e2pyb_exact_images_match_source_simple_questionnaire() -> None:
    task = _load("e2pyb_decision_task_candidate.json")
    source = ROOT / "data/raw/sources/e2pyb/9089.112.TESS Brown_vFINALsimple.docx"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == task["source_hashes"][
        "final_simple_questionnaire_sha256"
    ]
    expected = {
        "word/media/image3.png": (
            "data/derived/stimuli/e2pyb/health_disparities.png",
            task["source_hashes"]["health_infographic_sha256"],
        ),
        "word/media/image4.png": (
            "data/derived/stimuli/e2pyb/economic_disparities.png",
            task["source_hashes"]["economic_infographic_sha256"],
        ),
        "word/media/image5.png": (
            "data/derived/stimuli/e2pyb/belonging_disparities.png",
            task["source_hashes"]["belonging_infographic_sha256"],
        ),
    }
    with ZipFile(source) as archive:
        for member, (relative, digest) in expected.items():
            source_bytes = archive.read(member)
            assert source_bytes == (ROOT / relative).read_bytes()
            assert hashlib.sha256(source_bytes).hexdigest() == digest
