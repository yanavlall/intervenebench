from __future__ import annotations

import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from intervenebench.bounded_numeric import (
    bounded_numeric_prompt,
    freeze_bounded_numeric_recommendation_from_outputs,
    freeze_bounded_numeric_reveal_authorization,
    replay_bounded_numeric_score,
    score_frozen_bounded_numeric_recommendation,
    validate_bounded_numeric_blinded_bundle,
)
from intervenebench.protocol import assert_blinded_payload, verify_frozen_recommendation


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = ROOT / "data/manifests/contracts"
TASK_PATH = CONTRACT_DIR / "9tv2b_continuous_task_candidate.json"
BUNDLE_PATH = CONTRACT_DIR / "9tv2b_continuous_blinded_bundle.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _docx_paragraphs(path: Path) -> tuple[str, ...]:
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with ZipFile(path) as archive:
        document = ElementTree.fromstring(archive.read("word/document.xml"))
    return tuple(
        "".join(node.text or "" for node in paragraph.iter(f"{namespace}t"))
        for paragraph in document.iter(f"{namespace}p")
    )


def test_9tv2b_contract_is_source_faithful_sealed_and_runnable() -> None:
    task = _load(TASK_PATH)
    bundle = _load(BUNDLE_PATH)

    assert task["task_id"] == bundle["task_id"]
    assert task["experiment_id"] == bundle["experiment_id"] == "9tv2b"
    assert task["socsci210_task_num"] == 0
    assert task["canonical_split_status"] == "unassigned"
    assert task["outcome_access"] == bundle["outcome_access"] == "sealed"
    assert task["reveal_authorized"] is bundle["reveal_authorized"] is False
    assert task["utility_transform"] == "U=y/100"
    assert task["valid_response"] == {
        "lower_bound": 0,
        "upper_bound": 100,
        "integer_only": True,
        "missing_codes": [],
        "missing_labels": ["source or SocSci null only"],
    }
    assert [arm["condition_num"] for arm in task["arms"]] == [0, 1]
    assert task["released_rows_per_arm_before_outcome_missingness"] == {
        "democratic_party_wording": 943,
        "democrat_party_wording": 966,
    }
    assert task["source_assignment_support_before_outcome_missingness"] == {
        "democratic_party_wording": 994,
        "democrat_party_wording": 1015,
    }
    assert task["source_variable_mapping"] == {
        "assignment_variable": "P_DEM",
        "outcome_variable": "U1",
        "weight_variable": "WEIGHT",
        "assignment_to_arm": {
            "1": "democrat_party_wording",
            "2": "democratic_party_wording",
        },
        "utility_transform": "U1/100",
        "valid_outcome_range": [0, 100],
        "missing_outcome_rule": "null_only",
    }
    assert task["estimator"]["normalized_for_pooled_regret"] is True
    assert task["scoring_binding"]["real_outcome_access_before_frozen_recommendation"] is False
    assert_blinded_payload(task)
    assert_blinded_payload(bundle)
    assert task["scoring_binding"] == {
        "recommendation_schema": "bounded_numeric_recommendation.v1",
        "reveal_authorization_schema": "bounded_numeric_reveal_authorization.v1",
        "score_schema": "bounded_numeric_score.v1",
        "required_hashes": [
            "split_manifest_sha256",
            "decision_task_sha256",
            "blinded_bundle_sha256",
            "simulator_outputs_sha256",
        ],
        "scoring_adapter": "intervenebench.bounded_numeric",
        "real_outcome_access_before_frozen_recommendation": False,
        "explicit_reveal_authorization_after_frozen_recommendation": True,
        "source_sensitivity_requires_separate_projection_authorization": True,
    }
    validate_bounded_numeric_blinded_bundle(bundle)

    prompt = bounded_numeric_prompt(bundle, arm_id="democrat_party_wording")
    assert "between 0 and 100 inclusive" in prompt
    assert bundle["arms"][1]["message"] in prompt
    assert bundle["outcome_question"] in prompt


def test_9tv2b_exact_text_and_source_container_are_hash_pinned() -> None:
    task = _load(TASK_PATH)
    questionnaire = ROOT / "data/raw/sources/9tv2b/8041.082_TESS_Utych_vFINALsimple.docx"
    programmed = ROOT / "data/raw/sources/9tv2b/8041.082_TESS_Utych_vFINAL.docx"
    archive_path = ROOT / task["source_data_locator"]["container_path"]

    assert hashlib.sha256(questionnaire.read_bytes()).hexdigest() == task[
        "source_hashes"
    ]["final_simple_questionnaire_sha256"]
    assert hashlib.sha256(programmed.read_bytes()).hexdigest() == task[
        "source_hashes"
    ]["final_programmed_questionnaire_sha256"]
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == task[
        "source_data_locator"
    ]["container_sha256"]

    paragraphs = _docx_paragraphs(questionnaire)
    bundle = _load(BUNDLE_PATH)
    for arm in bundle["arms"]:
        for source_paragraph in arm["message"].split("\n\n"):
            assert source_paragraph in paragraphs
    visible_paragraphs = tuple(
        paragraph.replace("<u>", "").replace("</u>", "")
        for paragraph in paragraphs
    )
    assert bundle["outcome_question"] in visible_paragraphs

    with ZipFile(archive_path) as archive:
        info = archive.getinfo(task["source_data_locator"]["dta_member"])
        assert info.file_size == 646333
        # The DTA member is deliberately not extracted or opened before reveal.
        assert len(task["source_data_locator"]["dta_member_sha256"]) == 64


def _write_validation_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    task = _load(TASK_PATH)
    task["split"] = "validation"
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    split = {
        "experiment_to_split": {"9tv2b": "validation"},
        "test_outcomes_sealed": True,
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    parquet_path = tmp_path / "outcomes.parquet"
    pq.write_table(
        pa.table(
            {
                "study_id": ["9tv2b"] * 4,
                "sample_id": [0, 1, 2, 3],
                "participant": [10, 11, 12, 13],
                "condition_num": [0, 0, 1, 1],
                "task_num": [0, 0, 0, 0],
                "response": [20, 40, 10, 30],
            }
        ),
        parquet_path,
    )
    return task_path, split_path, parquet_path


def test_9tv2b_bounded_recommendation_is_hash_bound_before_normalized_scoring(
    tmp_path: Path,
) -> None:
    task_path, split_path, parquet_path = _write_validation_fixture(tmp_path)
    raw_path = tmp_path / "raw.json"
    recommendation_path = tmp_path / "recommendation.json"
    authorization_path = tmp_path / "authorization.json"
    score_path = tmp_path / "score.json"
    outputs = (
        {
            "arm_id": "democratic_party_wording",
            "draw_index": 0,
            "raw_response": '{"predicted_value": 40}',
        },
        {
            "arm_id": "democrat_party_wording",
            "draw_index": 0,
            "raw_response": '{"predicted_value": 60}',
        },
    )
    freeze_bounded_numeric_recommendation_from_outputs(
        bundle_path=BUNDLE_PATH,
        split_path=split_path,
        decision_task_path=task_path,
        outputs=outputs,
        raw_output_path=raw_path,
        recommendation_path=recommendation_path,
        simulator_id="fixture-bounded-numeric",
        simulator_revision="1",
        draws=1,
        seed=19,
    )
    recommendation = verify_frozen_recommendation(recommendation_path)
    assert recommendation["selected_arm_id"] == "democrat_party_wording"
    assert recommendation["normalized_for_pooled_regret"] is True
    assert recommendation["synthetic_treatment_effects"] == pytest.approx(
        {"democrat_party_wording": 0.2}
    )

    freeze_bounded_numeric_reveal_authorization(
        recommendation_path=recommendation_path,
        raw_output_path=raw_path,
        split_manifest_path=split_path,
        decision_task_path=task_path,
        blinded_bundle_path=BUNDLE_PATH,
        authorization_path=authorization_path,
    )
    score_frozen_bounded_numeric_recommendation(
        parquet_paths=(parquet_path,),
        decision_task_path=task_path,
        split_manifest_path=split_path,
        blinded_bundle_path=BUNDLE_PATH,
        recommendation_path=recommendation_path,
        raw_output_path=raw_path,
        authorization_path=authorization_path,
        score_path=score_path,
        bootstrap_replicates=20,
        bootstrap_seed=23,
    )
    score = replay_bounded_numeric_score(
        score_path=score_path,
        recommendation_path=recommendation_path,
        raw_output_path=raw_path,
        authorization_path=authorization_path,
    )
    assert score["human_arm_locations_raw"] == {
        "democratic_party_wording": 30.0,
        "democrat_party_wording": 20.0,
    }
    assert score["human_treatment_effects"] == pytest.approx(
        {"democrat_party_wording": -0.1}
    )
    assert score["selected_arm_id"] == "democrat_party_wording"
    assert score["human_best_arm_id"] == "democratic_party_wording"
    assert score["normalized_decision_regret"] == pytest.approx(0.1)
    assert score["normalized_for_pooled_regret"] is True
    assert score["regret_unit"] == "normalized_utility"
    assert "raw_decision_regret" not in score


def test_9tv2b_bounded_adapter_rejects_out_of_range_predictions(
    tmp_path: Path,
) -> None:
    task_path, split_path, _ = _write_validation_fixture(tmp_path)
    with pytest.raises(ValueError, match="outside the frozen response bounds"):
        freeze_bounded_numeric_recommendation_from_outputs(
            bundle_path=BUNDLE_PATH,
            split_path=split_path,
            decision_task_path=task_path,
            outputs=(
                {
                    "arm_id": "democratic_party_wording",
                    "draw_index": 0,
                    "raw_response": '{"predicted_value": 101}',
                },
                {
                    "arm_id": "democrat_party_wording",
                    "draw_index": 0,
                    "raw_response": '{"predicted_value": 50}',
                },
            ),
            raw_output_path=tmp_path / "raw.json",
            recommendation_path=tmp_path / "recommendation.json",
            simulator_id="fixture-bounded-numeric",
            simulator_revision="1",
            draws=1,
            seed=19,
        )
