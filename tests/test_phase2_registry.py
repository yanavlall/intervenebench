import csv
import hashlib
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "data" / "manifests" / "audits"
BATCH_PATH = AUDIT_DIR / "phase2_viability_batch.csv"
REGISTRY_PATH = AUDIT_DIR / "phase2_candidate_registry.csv"
SOURCE_BUNDLES_PATH = AUDIT_DIR / "phase2_source_bundles.csv"
PRIMARY_MAPPINGS_PATH = (
    AUDIT_DIR / "phase2_primary_outcome_mappings.csv"
)
EXTERNAL_CROSSWALK_PATH = (
    AUDIT_DIR / "external_archive_crosswalk.csv"
)
EXTERNAL_UNIVERSE_PATH = (
    AUDIT_DIR / "external_candidate_universe_v1.csv"
)
EXTERNAL_AUDIT_REGISTRY_PATH = (
    AUDIT_DIR / "external_source_audit_registry.csv"
)
EXTERNAL_SOURCE_BUNDLES_PATH = (
    AUDIT_DIR / "external_source_bundles.csv"
)
EXTERNAL_AUDIT_BATCHES_PATH = (
    AUDIT_DIR / "external_audit_batches.csv"
)
EXTERNAL_AUDIT_FUNNEL_BACKTEST_PATH = (
    AUDIT_DIR / "external_audit_funnel_backtest.csv"
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_phase2_registry_matches_frozen_batch_and_keeps_outcomes_sealed() -> None:
    batch = _rows(BATCH_PATH)
    registry = _rows(REGISTRY_PATH)

    assert len(batch) == len(registry) == 40
    assert [row["audit_order"] for row in registry] == [
        str(index) for index in range(1, 41)
    ]
    assert [row["experiment_id"] for row in registry] == [
        row["experiment_id"] for row in batch
    ]
    assert {row["outcome_access"] for row in batch} == {"sealed"}
    assert {row["outcome_access"] for row in registry} <= {
        "sealed",
        "result_text_exposed_non_test",
    }
    assert {
        row["experiment_id"]
        for row in registry
        if row["outcome_access"] == "result_text_exposed_non_test"
    } == {"345ms", "mzm26", "egmxd"}


def test_phase2_registry_freezes_viability_classification_counts() -> None:
    registry = _rows(REGISTRY_PATH)

    assert Counter(row["scientific_status"] for row in registry) == {
        "eligible_core": 2,
        "eligible_extension": 26,
        "ineligible": 12,
    }
    assert Counter(row["primary_track"] for row in registry) == {
        "core_simple": 2,
        "extension_continuous": 2,
        "extension_factorial": 5,
        "extension_interactive": 5,
        "extension_multimodal": 11,
        "extension_source_data": 1,
        "extension_utility_sensitivity": 1,
        "extension_composite": 1,
        "ineligible": 12,
    }


def test_phase2_registry_contains_no_outcome_or_result_fields() -> None:
    registry = _rows(REGISTRY_PATH)
    forbidden = {
        "response",
        "arm_mean",
        "treatment_effect",
        "significance",
        "winner",
        "regret",
    }

    assert forbidden.isdisjoint(registry[0])
    assert all(row["outcome_mapping_status"] for row in registry)
    assert all(row["notes"] for row in registry)


def test_phase2_source_bundles_cover_frozen_batch_and_are_sealed() -> None:
    batch = _rows(BATCH_PATH)
    bundles = _rows(SOURCE_BUNDLES_PATH)

    assert [row["experiment_id"] for row in bundles] == [
        row["experiment_id"] for row in batch
    ]
    assert all(int(row["source_file_count"]) >= 2 for row in bundles)
    assert all(len(row["source_bundle_sha256"]) == 64 for row in bundles)
    assert {row["outcome_access"] for row in bundles} <= {
        "sealed",
        "result_text_exposed_non_test",
    }


def test_primary_outcome_mappings_cover_verified_phase2_tasks() -> None:
    registry = {row["experiment_id"]: row for row in _rows(REGISTRY_PATH)}
    mappings = _rows(PRIMARY_MAPPINGS_PATH)

    assert [row["experiment_id"] for row in mappings] == [
        "mzm26",
        "tcg8p",
        "4w9pz",
        "pb2rr",
        "egmxd",
        "de5hx",
        "345ms",
        "gx6hp",
        "hgmu6",
        "z358z",
    ]
    assert {row["outcome_access"] for row in mappings} <= {
        "sealed",
        "result_text_exposed_non_test",
    }
    assert all(len(row["source_questionnaire_sha256"]) == 64 for row in mappings)
    assert all(row["mapping_status"] == registry[row["experiment_id"]]["outcome_mapping_status"] for row in mappings)
    assert Counter(row["primary_track"] for row in mappings) == {
        "core_simple": 2,
        "extension_continuous": 1,
        "extension_source_data": 1,
        "extension_utility_sensitivity": 1,
        "extension_composite": 1,
        "extension_factorial": 3,
        "extension_multimodal": 1,
    }


def test_external_archive_crosswalk_and_universe_are_disjoint_and_sealed() -> None:
    crosswalk = _rows(EXTERNAL_CROSSWALK_PATH)
    universe = _rows(EXTERNAL_UNIVERSE_PATH)

    assert len(crosswalk) == 40
    assert len(universe) == 31
    assert {row["archive_study_id"] for row in crosswalk}.isdisjoint(
        row["archive_study_id"] for row in universe
    )
    assert {row["outcome_access"] for row in crosswalk + universe} == {"sealed"}
    assert Counter(row["match_rule"] for row in crosswalk) == {
        "normalized_exact_title": 39,
        "replication_suffix_match": 1,
    }
    assert Counter(row["deduplication_status"] for row in universe) == {
        "pending_source_audit": 29,
        "pending_duplicate_adjudication": 1,
        "insufficient_metadata": 1,
    }


def test_external_universe_order_and_hashes_are_frozen() -> None:
    universe = _rows(EXTERNAL_UNIVERSE_PATH)

    assert [row["freeze_order"] for row in universe] == [
        str(index) for index in range(1, 32)
    ]
    expected = [
        hashlib.sha256(
            f"external-universe-v1:20260812:{row['archive_study_id']}".encode()
        ).hexdigest()
        for row in universe
    ]
    assert [row["selection_hash"] for row in universe] == expected
    assert expected == sorted(expected)


def test_external_parallel_batches_are_contiguous_frozen_order_blocks() -> None:
    universe = {
        int(row["freeze_order"]): row for row in _rows(EXTERNAL_UNIVERSE_PATH)
    }
    batches = _rows(EXTERNAL_AUDIT_BATCHES_PATH)

    assert batches
    for batch in batches:
        if batch["batch_id"].startswith("external-resolution-"):
            assert batch["candidate_ids"].split("|") == [
                "system_threat",
                "willer845",
            ]
            continue
        start = int(batch["start_freeze_order"])
        end = int(batch["end_freeze_order"])
        expected_ids = [
            universe[order]["archive_study_id"]
            for order in range(start, end + 1)
            if universe[order]["deduplication_status"] == "pending_source_audit"
        ]
        assert batch["candidate_ids"].split("|") == expected_ids
        assert all(
            universe[order]["deduplication_status"] == "pending_source_audit"
            for order in range(start, end + 1)
            if universe[order]["archive_study_id"] in expected_ids
        )
        assert batch["outcome_access"] in {
            "sealed",
            "result_text_exposure_logged",
            "mixed_sealed_and_exposure_logged",
        }


def test_completed_external_batches_are_committed_in_the_audit_registry() -> None:
    audits = {row["archive_study_id"] for row in _rows(EXTERNAL_AUDIT_REGISTRY_PATH)}
    completed_batches = [
        row
        for row in _rows(EXTERNAL_AUDIT_BATCHES_PATH)
        if row["batch_status"] == "complete"
    ]

    assert completed_batches
    assert all(
        set(batch["candidate_ids"].split("|")).issubset(audits)
        for batch in completed_batches
    )


def test_external_audit_funnel_backtest_preserves_completed_adjudications() -> None:
    audits = _rows(EXTERNAL_AUDIT_REGISTRY_PATH)
    backtest = _rows(EXTERNAL_AUDIT_FUNNEL_BACKTEST_PATH)
    audited_prefix = audits[: len(backtest)]

    assert [row["archive_study_id"] for row in backtest] == [
        row["archive_study_id"] for row in audited_prefix
    ]
    assert [row["audit_order"] for row in backtest] == [
        str(index) for index in range(1, len(backtest) + 1)
    ]
    assert all(
        row["final_scientific_status"] == audit["scientific_status"]
        for row, audit in zip(backtest, audited_prefix, strict=True)
    )
    assert all(
        row["outcome_access"] == audit["outcome_access"]
        for row, audit in zip(backtest, audited_prefix, strict=True)
    )
    assert all(
        (row["stage1_disposition"] == "escalate")
        == (row["requires_stage2"] == "true")
        for row in backtest
    )
    assert all(
        row["stage1_disposition"] == "escalate"
        for row in backtest
        if row["final_scientific_status"] != "ineligible"
    )
    assert all(
        row["stage1_disposition"] == "stop"
        for row in backtest
        if row["final_scientific_status"] == "ineligible"
    )


def test_external_source_audits_follow_frozen_order_and_record_exposure() -> None:
    universe = _rows(EXTERNAL_UNIVERSE_PATH)
    audits = _rows(EXTERNAL_AUDIT_REGISTRY_PATH)
    clear_candidates = [
        row for row in universe if row["deduplication_status"] == "pending_source_audit"
    ]

    clear_audits = [
        row
        for row in audits
        if row["archive_study_id"] not in {"system_threat", "willer845"}
    ]
    assert [row["archive_study_id"] for row in clear_audits] == [
        row["archive_study_id"] for row in clear_candidates
    ]
    assert [row["audit_order"] for row in audits] == [
        str(index) for index in range(1, len(audits) + 1)
    ]
    assert {row["outcome_access"] for row in audits} <= {
        "sealed",
        "result_text_exposed_ineligible",
        "result_text_exposed_non_test",
    }
    assert all(
        row["outcome_access"] == "sealed"
        for row in audits
        if row["scientific_status"] != "ineligible"
        and row["outcome_access"] != "result_text_exposed_non_test"
    )
    assert all(
        row["scientific_status"] == "ineligible"
        for row in audits
        if row["outcome_access"] == "result_text_exposed_ineligible"
    )
    assert all(
        row["scientific_status"] == "eligible_extension"
        for row in audits
        if row["outcome_access"] == "result_text_exposed_non_test"
    )


def test_external_source_audit_registry_contains_no_result_fields() -> None:
    audits = _rows(EXTERNAL_AUDIT_REGISTRY_PATH)
    forbidden = {
        "response",
        "arm_mean",
        "treatment_effect",
        "significance",
        "winner",
        "regret",
    }

    assert audits
    assert forbidden.isdisjoint(audits[0])
    assert all(row["scientific_status"] for row in audits)
    assert all(row["notes"] for row in audits)


def test_bougher_external_audit_records_design_based_exclusion() -> None:
    audits = _rows(EXTERNAL_AUDIT_REGISTRY_PATH)
    bougher = next(
        row for row in audits if row["archive_study_id"] == "Bougher893"
    )

    assert bougher["primary_track"] == "ineligible"
    assert bougher["scientific_status"] == "ineligible"
    assert bougher["outcome_mapping_status"] == "not_applicable"
    assert (
        bougher["exclusion_reason"]
        == "target_attributes_and_no_stable_decision_utility"
    )
    assert bougher["outcome_access"] == "result_text_exposed_ineligible"


def test_external_source_bundles_cover_audits_and_have_valid_hashes() -> None:
    audits = _rows(EXTERNAL_AUDIT_REGISTRY_PATH)
    bundles = _rows(EXTERNAL_SOURCE_BUNDLES_PATH)

    assert [row["archive_study_id"] for row in bundles] == [
        row["archive_study_id"] for row in audits
    ]
    assert {row["outcome_access"] for row in bundles} <= {
        "sealed",
        "result_text_exposed_ineligible",
        "result_text_exposed_non_test",
    }
    assert next(
        row["outcome_access"]
        for row in bundles
        if row["archive_study_id"] == "Bougher893"
    ) == "result_text_exposed_ineligible"
    assert {
        row["archive_study_id"]: row["outcome_access"] for row in bundles
    } == {
        row["archive_study_id"]: row["outcome_access"] for row in audits
    }
    assert all(int(row["source_file_count"]) >= 0 for row in bundles)
    assert all(len(row["source_bundle_sha256"]) == 64 for row in bundles)
    for row in bundles:
        source_files = (
            row["source_files"].split("|") if row["source_files"] else []
        )
        assert len(source_files) == int(row["source_file_count"])
        bundle_entries = []
        for source_file in source_files:
            filename, recorded_hash = source_file.rsplit("=", 1)
            if "/" in filename:
                source_path = ROOT / "data" / "raw" / "sources" / filename
            else:
                source_path = (
                    ROOT
                    / "data"
                    / "raw"
                    / "sources"
                    / row["archive_study_id"]
                    / filename
                )
            assert len(recorded_hash) == 64
            assert source_path.is_file()
            actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
            assert recorded_hash == actual_hash
            relative_path = source_path.relative_to(ROOT)
            bundle_entries.append((str(relative_path), actual_hash))
        actual_bundle_hash = hashlib.sha256(
            "".join(
                f"{actual_hash}  {relative_path}\n"
                for relative_path, actual_hash in sorted(bundle_entries)
            ).encode()
        ).hexdigest()
        assert row["source_bundle_sha256"] == actual_bundle_hash


def test_second_parallel_external_batch_records_duplicate_extension_and_block() -> None:
    audits = {row["archive_study_id"]: row for row in _rows(EXTERNAL_AUDIT_REGISTRY_PATH)}

    assert audits["ThorsonS42"]["exclusion_reason"] == "duplicate_of_socsci210_xtvu5"
    assert audits["ThorsonS42"]["outcome_access"] == "sealed"
    assert audits["Harbridge-Yong1032"]["scientific_status"] == "eligible_extension"
    assert audits["Harbridge-Yong1032"]["primary_track"] == "extension_factorial_action_subset"
    assert audits["Harbridge-Yong1032"]["outcome_mapping_status"] == "source_verified"
    assert audits["Harbridge-Yong1032"]["outcome_access"] == "result_text_exposed_non_test"
    assert audits["converseS16"]["scientific_status"] == "ineligible"
    assert (
        audits["converseS16"]["exclusion_reason"]
        == "unresolved_experiment_identity_and_no_verified_action_utility"
    )


def test_third_parallel_external_batch_records_personalized_and_repeated_extensions() -> None:
    audits = {row["archive_study_id"]: row for row in _rows(EXTERNAL_AUDIT_REGISTRY_PATH)}

    assert audits["Krupnikov719"]["scientific_status"] == "ineligible"
    assert (
        audits["Krupnikov719"]["exclusion_reason"]
        == "no_source_verified_stable_utility_and_post_treatment_selection"
    )
    assert audits["Krupnikov719"]["outcome_access"] == "result_text_exposed_ineligible"
    assert audits["ShannonS2"]["scientific_status"] == "eligible_extension"
    assert audits["ShannonS2"]["primary_track"] == "extension_factorial_repeated_message"
    assert audits["ShannonS2"]["outcome_access"] == "sealed"
    assert audits["AnsonBRIEF60"]["scientific_status"] == "eligible_extension"
    assert audits["AnsonBRIEF60"]["primary_track"] == "extension_personalized_policy"
    assert audits["AnsonBRIEF60"]["outcome_access"] == "result_text_exposed_non_test"


def test_fourth_parallel_batch_and_later_resolution_record_decisions() -> None:
    audits = {row["archive_study_id"]: row for row in _rows(EXTERNAL_AUDIT_REGISTRY_PATH)}

    assert (
        audits["system_threat"]["exclusion_reason"]
        == "duplicate_of_socsci210_345ms"
    )
    assert audits["system_threat"]["outcome_access"] == "result_text_exposed_ineligible"
    assert audits["Craig735"]["scientific_status"] == "ineligible"
    assert (
        audits["Craig735"]["exclusion_reason"]
        == "ideologically_contested_policy_outcomes_no_defensible_stable_utility"
    )
    assert audits["Iles1294"]["scientific_status"] == "ineligible"
    assert (
        audits["Iles1294"]["exclusion_reason"]
        == "message_cells_change_factual_evidence_state_not_deployable_actions"
    )
    assert audits["Blair1131"]["scientific_status"] == "eligible_extension"
    assert audits["Blair1131"]["primary_track"] == "extension_factorial_action_subset"
    assert {audits[name]["outcome_access"] for name in ("Craig735", "Iles1294", "Blair1131")} == {"sealed"}


def test_fifth_parallel_batch_records_core_extension_measurement_exclusion_and_duplicate() -> None:
    audits = {row["archive_study_id"]: row for row in _rows(EXTERNAL_AUDIT_REGISTRY_PATH)}

    assert audits["KrupnikovS34"]["scientific_status"] == "ineligible"
    assert (
        audits["KrupnikovS34"]["exclusion_reason"]
        == "measurement_wording_treatment_no_post_intervention_outcome_or_stable_utility"
    )
    assert audits["turagaS11"]["scientific_status"] == "eligible_extension"
    assert audits["turagaS11"]["primary_track"] == "core_simple"
    assert audits["turagaS11"]["outcome_access"] == "sealed"
    assert audits["SchaadS62"]["exclusion_reason"] == "duplicate_of_socsci210_9nphm"
    assert {audits[name]["outcome_access"] for name in ("KrupnikovS34", "turagaS11", "SchaadS62")} == {"sealed"}


def test_first_parallel_external_batch_has_design_based_exclusions() -> None:
    audits = {row["archive_study_id"]: row for row in _rows(EXTERNAL_AUDIT_REGISTRY_PATH)}

    expected_reasons = {
        "brandtS1": "scenario_target_attributes_not_deployable_actions",
        "patriot_act": "no_coherent_full_action_set_or_defensible_stable_utility",
        "Melin1066": "factorial_target_attributes_no_deployable_action_set_or_stable_utility",
    }
    for archive_study_id, reason in expected_reasons.items():
        assert audits[archive_study_id]["scientific_status"] == "ineligible"
        assert audits[archive_study_id]["exclusion_reason"] == reason
        assert (
            audits[archive_study_id]["outcome_access"]
            == "result_text_exposed_ineligible"
        )


def test_completed_external_census_freezes_counts_and_final_resolutions() -> None:
    audits = {
        row["archive_study_id"]: row
        for row in _rows(EXTERNAL_AUDIT_REGISTRY_PATH)
    }

    assert len(audits) == 31
    assert Counter(row["scientific_status"] for row in audits.values()) == {
        "eligible_extension": 7,
        "ineligible": 21,
        "source_blocked": 3,
    }
    assert Counter(row["outcome_access"] for row in audits.values()) == {
        "sealed": 21,
        "result_text_exposed_ineligible": 8,
        "result_text_exposed_non_test": 2,
    }

    assert {
        study_id
        for study_id, row in audits.items()
        if row["scientific_status"] == "eligible_extension"
    } == {
        "Harbridge-Yong1032",
        "ShannonS2",
        "AnsonBRIEF60",
        "Blair1131",
        "turagaS11",
        "wallaceS12",
        "KlarS44",
    }
    assert {
        study_id
        for study_id, row in audits.items()
        if row["scientific_status"] == "source_blocked"
    } == {"death_penalty", "gashS5", "immigration"}

    expected_duplicates = {
        "ThorsonS42": "duplicate_of_socsci210_xtvu5",
        "SchaadS62": "duplicate_of_socsci210_9nphm",
        "McCabeS19": "duplicate_of_socsci210_3pcdm",
        "Howard823": "duplicate_of_socsci210_ztwqy",
        "system_threat": "duplicate_of_socsci210_345ms",
    }
    assert {
        study_id: audits[study_id]["exclusion_reason"]
        for study_id in expected_duplicates
    } == expected_duplicates
    assert audits["willer845"]["scientific_status"] == "ineligible"
    assert (
        audits["willer845"]["exclusion_reason"]
        == "factual_world_or_evidence_change_and_no_coherent_deployable_action_set"
    )
