from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from intervenebench.protocol import assert_blinded_payload
from intervenebench.schemas import (
    Arm,
    DecisionTask,
    DesignType,
    NuisanceStandardizedDecisionTask,
    NuisanceStratum,
    OutcomeDirection,
    OutcomeFamily,
)
from intervenebench.simulators import (
    materialize_sequence_episode,
    sequence_probability_prompt,
    validate_ordinal_blinded_bundle,
    validate_bounded_multimodal_bundle,
    validate_categorical_multimodal_bundle,
    validate_sequence_blinded_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = ROOT / "data" / "manifests" / "contracts"


CASES = {
    "5vm8g": {
        "task_num": 2,
        "source_path": ROOT / "data/raw/sources/5vm8g/Haaland874.zip",
        "archive_hash": (
            "7ce5df7c7066c60664b088ad879a4804a878f417f3ce307030645d2330afb179"
        ),
        "source_member_hash": (
            "45e90954d9e1564abc4eb44c2faaf9a2a1da598f841bad252728550091224ed6"
        ),
        "row_counts": [757, 781],
    },
    "xc4yq": {
        "task_num": 7,
        "source_path": ROOT / "data/raw/sources/xc4yq/Thompson157.zip",
        "archive_hash": (
            "39ff9475a90573039d37f49a48f0a581584af3d87f439cdbb70b39e2e1612f51"
        ),
        "source_member_hash": (
            "b7467979e24199e1d9e19373c6a8d1aae981e524ab62ea767a2b07f72ae8f5d2"
        ),
        "row_counts": [198, 204, 187],
    },
    "de5hx": {
        "task_num": 0,
        "source_path": ROOT / "data/raw/sources/de5hx/Kam006.zip",
        "archive_hash": (
            "a8ed87a14b291cd5638feabecffb70e9efd340d2c8201bc9adfa98a061880bed"
        ),
        "source_member_hash": (
            "3d59c81363d588e2744dedefd20e8dbb56df0525502dd96a3894786acabe53c8"
        ),
        "row_counts": [322, 323, 341],
    },
}

EXTERNAL_CASES = {
    "Blair1131": {
        "source_path": ROOT
        / "data/raw/sources/Blair1131/8041.045_Northwestern_TESS 045 Blair_quex_v3clean.docx",
        "source_hash": (
            "26c97d60874445b54f054460233c569a8a2bd5b7fa7096aca7a54b19c727036e"
        ),
        "row_counts": [104, 88, 99],
        "assignment_variable": "BLAIR",
        "outcome_variable": "Q3",
        "weight_variable": "WEIGHT",
    },
    "turagaS11": {
        "source_path": ROOT / "data/raw/sources/turagaS11/TESS2_030_Turaga_FINAL.doc",
        "source_hash": (
            "8259b893cb29223cdaee42ee2c31f8abe340fc657d209cad228a44b0f7e81fa9"
        ),
        "row_counts": [247, 258, 269],
        "assignment_variable": "XTESS030",
        "outcome_variable": None,
        "weight_variable": "weight",
    },
    "wallaceS12": {
        "source_path": ROOT
        / "data/raw/sources/wallaceS12/K3542_TESS2 097 Wallace_FINAL.docx",
        "source_hash": (
            "b45e023efa0eea3b2274e49bc0a0da49fffad75d8203618b408d18632b964316"
        ),
        "row_counts": [240, 237, 244, 244, 251, 245],
        "assignment_variable": "XTESS097",
        "outcome_variable": "Q1",
        "weight_variable": "weight",
    },
}


def _load(experiment_id: str, suffix: str) -> dict:
    path = CONTRACT_DIR / f"{experiment_id}_{suffix}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _typed_task(candidate: dict) -> DecisionTask:
    options = candidate["response_options"]
    return DecisionTask(
        task_id=candidate["task_id"],
        experiment_id=candidate["experiment_id"],
        source_id=candidate["source_id"],
        paradigm_group=candidate["paradigm_group"],
        design_type=DesignType(candidate["design_type"]),
        randomization_unit=candidate["randomization_unit"],
        arms=tuple(
            Arm(
                arm_id=arm["arm_id"],
                description=arm["description"],
                deployable=arm["deployable"],
            )
            for arm in candidate["arms"]
        ),
        control_arm_id=candidate["control_arm_id"],
        primary_outcome_id=candidate["primary_outcome_id"],
        outcome_family=OutcomeFamily(candidate["outcome_family"]),
        response_options=tuple(float(option["raw_value"]) for option in options),
        scale_lower=float(candidate["scale_lower"]),
        scale_upper=float(candidate["scale_upper"]),
        direction=OutcomeDirection(candidate["direction"]),
        observations_per_arm=tuple(
            candidate["released_rows_per_arm_before_outcome_missingness"].items()
        ),
        weighting_rule=candidate["weighting_rule"],
        missingness_rule=candidate["missingness_rule"],
        practical_regret_tolerance=float(candidate["practical_regret_tolerance"]),
    )


@pytest.mark.parametrize("experiment_id", sorted(CASES))
def test_ordinal_candidate_contract_is_sealed_and_phase1_valid(
    experiment_id: str,
) -> None:
    candidate = _load(experiment_id, "decision_task_candidate")
    bundle = _load(experiment_id, "blinded_bundle")

    assert candidate["socsci210_task_num"] == CASES[experiment_id]["task_num"]
    assert candidate["canonical_split_status"] == "unassigned"
    assert candidate["outcome_access"] == "sealed"
    assert candidate["reveal_authorized"] is False
    assert candidate["decision_maker"].strip()
    assert candidate["beneficiary"].strip()
    assert candidate["utility_rationale"].strip()
    assert candidate["truthfulness_scope"].strip()
    assert list(
        candidate["released_rows_per_arm_before_outcome_missingness"].values()
    ) == CASES[experiment_id]["row_counts"]
    assert CASES[experiment_id]["source_path"].is_file()

    _typed_task(candidate).validate_phase1()
    validate_ordinal_blinded_bundle(bundle)
    assert_blinded_payload(bundle)
    assert bundle["source_material_sha256"] == CASES[experiment_id][
        "source_member_hash"
    ]


def test_candidate_utility_and_bundle_utility_are_identical() -> None:
    for experiment_id in CASES:
        candidate = _load(experiment_id, "decision_task_candidate")
        bundle = _load(experiment_id, "blinded_bundle")
        candidate_options = [
            {
                "value": option["raw_value"],
                "label": option["label"],
                "normalized_utility": option["normalized_utility"],
            }
            for option in candidate["response_options"]
        ]
        assert bundle["response_options"] == candidate_options


def test_declared_source_members_are_present_in_pinned_archives() -> None:
    # The full archives remain pinned; this test intentionally checks bytes and
    # member names only and never extracts or reads participant response files.
    expected_archives = {
        "5vm8g": (
            "Haaland874.zip",
            "[8041.007] TESS Haaland_v2clean.docx",
        ),
        "xc4yq": (
            "Thompson157.zip",
            "TESS2_DHS_02_Thompson_final.doc",
        ),
        "de5hx": (
            "Kam006.zip",
            "tess2_054_kam-simas2_FINAL.doc",
        ),
    }
    import zipfile

    for experiment_id, (archive_name, member_name) in expected_archives.items():
        path = ROOT / "data/raw/sources" / experiment_id / archive_name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == CASES[experiment_id][
            "archive_hash"
        ]
        with zipfile.ZipFile(path) as archive:
            assert member_name in archive.namelist()
            assert hashlib.sha256(archive.read(member_name)).hexdigest() == CASES[
                experiment_id
            ]["source_member_hash"]


@pytest.mark.parametrize("experiment_id", sorted(EXTERNAL_CASES))
def test_external_candidate_contract_is_outcome_blind_and_score_mapped(
    experiment_id: str,
) -> None:
    candidate = _load(experiment_id, "decision_task_candidate")
    bundle = _load(experiment_id, "blinded_bundle")
    source = EXTERNAL_CASES[experiment_id]

    assert candidate["canonical_split_status"] == "unassigned"
    assert candidate["outcome_access"] == "sealed"
    assert candidate["reveal_authorized"] is False
    assert (
        candidate["source_data_mapping_status"]
        == "complete_outcome_blind_schema_and_design_mapping"
    )
    assert list(
        candidate["released_rows_per_arm_before_outcome_missingness"].values()
    ) == source["row_counts"]
    mapping = candidate["source_variable_mapping"]
    assert mapping["assignment_variable"] == source["assignment_variable"]
    assert mapping["weight_variable"] == source["weight_variable"]
    if source["outcome_variable"] is not None:
        assert mapping["outcome_variable"] == source["outcome_variable"]
    assert candidate["decision_maker"].strip()
    assert candidate["beneficiary"].strip()
    assert candidate["utility_rationale"].strip()
    assert candidate["truthfulness_scope"].strip()
    assert source["source_path"].is_file()
    assert hashlib.sha256(source["source_path"].read_bytes()).hexdigest() == source[
        "source_hash"
    ]

    validate_ordinal_blinded_bundle(bundle)
    assert_blinded_payload(bundle)
    assert bundle["source_material_sha256"] == source["source_hash"]


def test_external_schema_mapping_manifest_preserves_the_outcome_boundary() -> None:
    import csv

    path = ROOT / "data/manifests/audits/external_schema_mappings.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["archive_study_id"] for row in rows} == set(EXTERNAL_CASES) | {
        "ShannonS2",
        "KlarS44",
    }
    assert {row["participant_outcomes_opened"] for row in rows} == {"false"}
    assert {row["outcome_summaries_computed"] for row in rows} == {"false"}
    assert {row["temp_participant_file_retained"] for row in rows} == {"false"}
    assert all(len(row["official_archive_sha256"]) == 64 for row in rows)


def test_sequence_contracts_complete_human_and_simulator_mapping() -> None:
    for experiment_id in ("ShannonS2", "KlarS44", "z358z"):
        candidate = _load(experiment_id, "decision_task_candidate")
        bundle = _load(experiment_id, "blinded_bundle")
        assert candidate["outcome_access"] == "sealed"
        assert candidate["reveal_authorized"] is False
        assert candidate["source_data_mapping_status"] == (
            "complete_outcome_blind_schema_design_and_sequence_mapping"
        )
        assert candidate["simulator_sequence_contract"].strip()
        validate_sequence_blinded_bundle(bundle)
        assert_blinded_payload(bundle)
        episode = materialize_sequence_episode(bundle, seed=2102026)
        prompts = [
            sequence_probability_prompt(
                bundle, arm_id=arm["arm_id"], episode=episode
            )
            for arm in bundle["arms"]
        ]
        assert all("{{" not in prompt for prompt in prompts)
        if episode.prior_exposure:
            assert all(episode.prior_exposure in prompt for prompt in prompts)


def test_z358z_freezes_one_context_and_marks_checkpoint_exposure() -> None:
    candidate = _load("z358z", "decision_task_candidate")
    assert candidate["outcome_access"] == "sealed"
    assert candidate["reveal_authorized"] is False
    assert candidate["socsci210_task_num"] == 2
    assert [arm["condition_num"] for arm in candidate["arms"]] == [0, 1]
    assert [cell["condition_num"] for cell in candidate["omitted_source_cells"]] == [
        2,
        3,
    ]
    assert candidate["source_data_mapping_status"] == (
        "complete_outcome_blind_schema_design_and_sequence_mapping"
    )
    assert candidate["checkpoint_compatibility"].startswith(
        "The released Socrates participant_mapping.json lists z358z as seen."
    )
    _typed_task(candidate).validate_phase1()
    assert (CONTRACT_DIR / "z358z_blinded_bundle.json").is_file()


def test_pb2rr_source_primary_uses_equal_name_standardization_and_exact_assets() -> None:
    candidate = _load("pb2rr", "decision_task_candidate")
    bundle = _load("pb2rr", "blinded_bundle")
    levels = candidate["nuisance_factor"]["levels"]
    counts = candidate["source_rows_per_article_name_cell_before_outcome_missingness"]
    task = NuisanceStandardizedDecisionTask(
        task_id=candidate["task_id"],
        experiment_id=candidate["experiment_id"],
        source_id=candidate["source_id"],
        paradigm_group=candidate["paradigm_group"],
        randomization_unit=candidate["randomization_unit"],
        arms=tuple(
            Arm(
                arm_id=arm["arm_id"],
                description=arm["description"],
                deployable=arm["deployable"],
            )
            for arm in candidate["arms"]
        ),
        control_arm_id=candidate["control_arm_id"],
        primary_outcome_id=candidate["primary_outcome_id"],
        outcome_family=OutcomeFamily(candidate["outcome_family"]),
        response_options=tuple(float(value) for value in candidate["response_options"]),
        scale_lower=float(candidate["scale_lower"]),
        scale_upper=float(candidate["scale_upper"]),
        direction=OutcomeDirection(candidate["direction"]),
        nuisance_strata=tuple(
            NuisanceStratum(level["nuisance_id"], float(level["weight"]))
            for level in levels
        ),
        observations_per_cell=tuple(
            (arm_id, level["nuisance_id"], counts[arm_id][level["nuisance_id"]])
            for arm_id in counts
            for level in levels
        ),
        modality="image",
        practical_regret_tolerance=float(candidate["practical_regret_tolerance"]),
    )
    task.validate_factorial_extension()
    validate_bounded_multimodal_bundle(bundle)
    assert candidate["socsci210_task_num"] is None
    assert candidate["socsci210_reconstruction_status"] == (
        "source_primary_absent_from_socsci210"
    )
    assert candidate["source_data_mapping_status"] == (
        "complete_outcome_blind_schema_design_multimodal_and_nuisance_mapping"
    )
    assert sum(level["weight"] for level in levels) == pytest.approx(1.0)
    assert all(counts[arm_id][level["nuisance_id"]] > 0 for arm_id in counts for level in levels)
    for arm in bundle["arms"]:
        path = ROOT / arm["asset"]["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == arm["asset"]["sha256"]


def test_egmxd_development_contract_preserves_categorical_choice_and_images() -> None:
    import zipfile

    candidate = _load("egmxd", "decision_task_candidate")
    bundle = _load("egmxd", "blinded_bundle")
    assert candidate["outcome_access"] == "result_text_exposed_non_test"
    assert candidate["canonical_split_status"] == "unassigned"
    assert candidate["reveal_authorized"] is False
    assert candidate["source_data_mapping_status"] == (
        "complete_outcome_blind_schema_design_categorical_multimodal_mapping"
    )
    assert candidate["socsci210_task_num"] == 0
    assert set(candidate["released_participants_per_arm_before_outcome_missingness"].values()) == {
        1667,
        1671,
        1717,
    }
    assert sum(
        option["normalized_utility"] for option in candidate["choice_options"]
    ) == 9.0
    validate_categorical_multimodal_bundle(bundle)
    assert_blinded_payload(bundle)
    for arm in bundle["arms"]:
        path = ROOT / arm["asset"]["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == arm["asset"]["sha256"]
        source = ROOT / arm["asset"]["source_container_path"]
        with zipfile.ZipFile(source) as questionnaire:
            embedded = questionnaire.read(arm["asset"]["source_member"])
        assert hashlib.sha256(embedded).hexdigest() == arm["asset"]["sha256"]


def test_blair_bundle_marginalizes_randomized_name_equally_within_every_arm() -> None:
    bundle = _load("Blair1131", "blinded_bundle")
    for arm in bundle["arms"]:
        assert [variant["variant_id"] for variant in arm["message_variants"]] == [
            "president_name_eric",
            "president_name_steven",
        ]
        assert [variant["weight"] for variant in arm["message_variants"]] == [
            0.5,
            0.5,
        ]


def test_4w9pz_source_binary_mapping_is_sealed_and_sequence_blocked() -> None:
    candidate = _load("4w9pz", "decision_task_candidate")
    assert candidate["canonical_split_status"] == "unassigned"
    assert candidate["outcome_access"] == "sealed"
    assert candidate["reveal_authorized"] is False
    assert candidate["socsci210_task_num"] is None
    assert candidate["source_question_id"] == "T70_14"
    assert candidate["source_data_mapping_status"] == (
        "complete_outcome_blind_human_mapping_sequence_assets_pending"
    )
    assert candidate["simulator_status"] == (
        "blocked_missing_exact_cofielded_comodule_and_visual_assets"
    )
    assert candidate["released_rows_per_arm_before_outcome_missingness"] == {
        "undermining_teacher_policy": 392,
        "affording_teacher_policy": 411,
    }
    mapping = candidate["source_variable_mapping"]
    assert mapping["assignment_variable"] == "P_COND70"
    assert mapping["outcome_variable"] == "T70_14"
    assert mapping["weight_variable"] == "WEIGHT"
    assert mapping["valid_outcome_values"] == [1, 2]
    assert mapping["missing_outcome_codes"] == [77, 98, 99]
    assert candidate["fielding_sequence_dependency"][
        "adult_campbell_source_may_not_be_substituted"
    ] is True
    assert_blinded_payload(candidate)
    assert not (CONTRACT_DIR / "4w9pz_blinded_bundle.json").exists()

    questionnaire = ROOT / (
        "data/raw/sources/4w9pz/8041.0070_TESS Teen 2_v10clean_FINAL.docx"
    )
    codebook = ROOT / "data/raw/sources/4w9pz/TESS_070_Hecht_Codebook.xlsx"
    archive = ROOT / "data/raw/sources/4w9pz/Hecht1248.zip"
    assert hashlib.sha256(questionnaire.read_bytes()).hexdigest() == candidate[
        "source_hashes"
    ]["final_questionnaire_sha256"]
    assert hashlib.sha256(codebook.read_bytes()).hexdigest() == candidate[
        "source_hashes"
    ]["codebook_sha256"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == candidate[
        "source_hashes"
    ]["official_mixed_archive_sha256"]
