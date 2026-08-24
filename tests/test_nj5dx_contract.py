from __future__ import annotations

import hashlib
import json
from pathlib import Path

from intervenebench.protocol import assert_blinded_payload
from intervenebench.simulators import (
    ordinal_png_multimodal_prompt,
    validate_ordinal_png_multimodal_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = ROOT / "data/manifests/contracts"


def _load(name: str) -> dict:
    return json.loads((CONTRACT_DIR / name).read_text(encoding="utf-8"))


def test_nj5dx_contract_is_sealed_source_faithful_and_runnable() -> None:
    task = _load("nj5dx_decision_task_candidate.json")
    bundle = _load("nj5dx_blinded_bundle.json")

    assert task["experiment_id"] == bundle["experiment_id"] == "nj5dx"
    assert task["task_id"] == bundle["task_id"]
    assert task["socsci210_task_num"] == 0
    assert task["outcome_access"] == bundle["outcome_access"] == "sealed"
    assert task["reveal_authorized"] is bundle["reveal_authorized"] is False
    assert task["source_data_mapping_status"].endswith(
        "ordinal_png_multimodal_mapping"
    )
    assert [arm["socsci_condition_num"] for arm in task["arms"]] == [0, 1]
    assert list(task["released_rows_per_arm_before_outcome_missingness"].values()) == [
        907,
        903,
    ]
    assert task["source_variable_mapping"] == {
        "assignment_variable": "INFO",
        "outcome_variable": "Q3",
        "weight_variable": "WEIGHT1",
        "source_missing_codes": [77, 98, 99],
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
        asset = ROOT / arm["asset"]["path"]
        assert asset.is_file()
        assert hashlib.sha256(asset.read_bytes()).hexdigest() == arm["asset"][
            "sha256"
        ]
        prompt = ordinal_png_multimodal_prompt(
            bundle, arm_id=arm["arm_id"], repository_root=ROOT
        )
        assert prompt.asset_sha256 == (arm["asset"]["sha256"],)
        assert prompt.asset_paths == (str(asset.resolve()),)


def test_nj5dx_assets_match_the_exact_source_container_members() -> None:
    import zipfile

    task = _load("nj5dx_decision_task_candidate.json")
    container = ROOT / "data/raw/sources/nj5dx/TESS-0961.R1 description revised.docx"
    assert hashlib.sha256(container.read_bytes()).hexdigest() == task["source_hashes"][
        "source_description_container_sha256"
    ]
    expected = {
        "word/media/image1.png": (
            "data/derived/stimuli/nj5dx/lower_class_disadvantage.png",
            task["source_hashes"]["lower_class_infographic_sha256"],
        ),
        "word/media/image2.png": (
            "data/derived/stimuli/nj5dx/upper_class_privilege.png",
            task["source_hashes"]["upper_class_infographic_sha256"],
        ),
    }
    with zipfile.ZipFile(container) as archive:
        for member, (relative, digest) in expected.items():
            member_bytes = archive.read(member)
            derived_bytes = (ROOT / relative).read_bytes()
            assert member_bytes == derived_bytes
            assert hashlib.sha256(member_bytes).hexdigest() == digest
