from __future__ import annotations

import json
from pathlib import Path

import pytest

from intervenebench.simulators import (
    aggregate_categorical_multimodal_predictions,
    aggregate_bounded_multimodal_predictions,
    aggregate_sequence_predictions,
    aggregate_continuous_predictions,
    aggregate_binary_predictions,
    bounded_multimodal_prompt,
    categorical_multimodal_prompt,
    materialize_sequence_episode,
    parse_ordinal_distribution,
    parse_continuous_prediction,
    parse_binary_probability,
    sequence_probability_prompt,
    validate_continuous_blinded_bundle,
    validate_blinded_bundle,
    validate_bounded_multimodal_bundle,
    validate_categorical_multimodal_bundle,
    validate_ordinal_blinded_bundle,
    validate_sequence_blinded_bundle,
)
from intervenebench.sequence_contracts import (
    _powell_paths,
    build_klar_sequence_bundle,
    build_shannon_sequence_bundle,
    build_z358z_sequence_bundle,
)


def valid_bundle() -> dict:
    return {
        "schema_version": "blinded_bundle.v1",
        "task_id": "exp-a:task-0",
        "experiment_id": "exp-a",
        "access_regime": "DESIGN_ONLY",
        "population": {
            "description": "Adults in the United States",
            "roster_id": "aggregate-us-adult-v1",
        },
        "arms": [
            {"arm_id": "a", "message": "Message A"},
            {"arm_id": "b", "message": "Message B"},
        ],
        "common_context": "A fictional scenario.",
        "outcome_question": "Would you do it?",
        "response_options": [
            {"value": 1, "label": "Yes", "normalized_utility": 1.0},
            {"value": 2, "label": "No", "normalized_utility": 0.0},
        ],
        "source_material_sha256": "a" * 64,
    }


def test_bundle_has_exact_allowlisted_shape() -> None:
    validate_blinded_bundle(valid_bundle())
    bundle = valid_bundle()
    bundle["human_winner"] = "a"
    with pytest.raises(ValueError, match="forbidden|unexpected bundle fields"):
        validate_blinded_bundle(bundle)


def test_probability_parser_is_strict() -> None:
    prediction = parse_binary_probability(
        '{"yes_probability":0.65,"no_probability":0.35}'
    )
    assert prediction.yes_probability == pytest.approx(0.65)
    with pytest.raises(ValueError, match="sum to one"):
        parse_binary_probability(
            '{"yes_probability":0.65,"no_probability":0.30}'
        )
    with pytest.raises(ValueError, match="exactly"):
        parse_binary_probability(
            '{"yes_probability":0.65,"no_probability":0.35,"note":"x"}'
        )


def test_aggregation_requires_complete_paired_draws() -> None:
    outputs = [
        {"arm_id": "a", "draw_index": 0, "yes_probability": 0.4},
        {"arm_id": "a", "draw_index": 1, "yes_probability": 0.6},
        {"arm_id": "b", "draw_index": 0, "yes_probability": 0.7},
        {"arm_id": "b", "draw_index": 1, "yes_probability": 0.9},
    ]
    means = aggregate_binary_predictions(outputs, arm_ids=("a", "b"), draws=2)
    assert means == pytest.approx({"a": 0.5, "b": 0.8})
    with pytest.raises(ValueError, match="complete"):
        aggregate_binary_predictions(outputs[:-1], arm_ids=("a", "b"), draws=2)


def test_strict_continuous_parser_accepts_only_one_integer_field() -> None:
    parsed = parse_continuous_prediction('{"predicted_value": 17}', integer_only=True)
    assert parsed.value == 17.0
    with pytest.raises(ValueError, match="exactly"):
        parse_continuous_prediction(
            '{"predicted_value": 17, "explanation": "guess"}', integer_only=True
        )
    with pytest.raises(ValueError, match="integer"):
        parse_continuous_prediction('{"predicted_value": 17.5}', integer_only=True)


def test_continuous_aggregation_is_complete_and_source_aligned() -> None:
    outputs = [
        {"arm_id": "control", "draw_index": 0, "predicted_value": 10},
        {"arm_id": "control", "draw_index": 1, "predicted_value": 30},
        {"arm_id": "notice", "draw_index": 0, "predicted_value": 5},
        {"arm_id": "notice", "draw_index": 1, "predicted_value": 7},
    ]
    assert aggregate_continuous_predictions(
        outputs,
        arm_ids=("control", "notice"),
        draws=2,
        estimator="mean",
    ) == pytest.approx({"control": 20.0, "notice": 6.0})


def test_tcg8p_continuous_bundle_validates() -> None:
    root = Path(__file__).resolve().parents[1]
    bundle = json.loads(
        (
            root
            / "data/manifests/contracts/tcg8p_continuous_blinded_bundle.json"
        ).read_text()
    )
    validate_continuous_blinded_bundle(bundle)


@pytest.mark.parametrize(
    "experiment_id",
    ["5vm8g", "xc4yq", "de5hx", "Blair1131", "turagaS11", "wallaceS12"],
)
def test_sealed_ordinal_candidate_bundle_validates(experiment_id: str) -> None:
    root = Path(__file__).resolve().parents[1]
    bundle = json.loads(
        (
            root
            / "data/manifests/contracts"
            / f"{experiment_id}_blinded_bundle.json"
        ).read_text()
    )

    validate_ordinal_blinded_bundle(bundle)
    assert bundle["outcome_access"] == "sealed"
    assert bundle["reveal_authorized"] is False


def test_ordinal_bundle_rejects_result_fields_and_open_reveal() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "data/manifests/contracts/5vm8g_blinded_bundle.json"
    bundle = json.loads(path.read_text())
    bundle["human_winner"] = "arm_1_callback_discrimination_information"
    with pytest.raises(ValueError, match="forbidden|unexpected"):
        validate_ordinal_blinded_bundle(bundle)

    bundle = json.loads(path.read_text())
    bundle["reveal_authorized"] = True
    with pytest.raises(ValueError, match="sealed"):
        validate_ordinal_blinded_bundle(bundle)


def test_ordinal_bundle_rejects_inconsistent_randomized_nuisance_variants() -> None:
    root = Path(__file__).resolve().parents[1]
    bundle = json.loads(
        (
            root
            / "data/manifests/contracts/Blair1131_blinded_bundle.json"
        ).read_text()
    )
    bundle["arms"][1]["message_variants"][0]["weight"] = 0.6
    with pytest.raises(ValueError, match="sum to one|must match"):
        validate_ordinal_blinded_bundle(bundle)


def test_ordinal_bundle_supports_six_arm_extensions_but_not_seven() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "data/manifests/contracts/wallaceS12_blinded_bundle.json"
    bundle = json.loads(path.read_text())
    validate_ordinal_blinded_bundle(bundle)
    bundle["arms"].append(
        {"arm_id": "inadmissible_seventh_arm", "message": "Not fielded."}
    )
    with pytest.raises(ValueError, match="two to six"):
        validate_ordinal_blinded_bundle(bundle)


def test_pb2rr_multimodal_bundle_hashes_assets_and_pairs_names() -> None:
    root = Path(__file__).resolve().parents[1]
    bundle = json.loads(
        (root / "data/manifests/contracts/pb2rr_blinded_bundle.json").read_text()
    )
    validate_bounded_multimodal_bundle(bundle)
    prompt = bounded_multimodal_prompt(
        bundle,
        arm_id="hispanic_population_growth_article",
        nuisance_id="jamal",
        repository_root=root,
    )
    assert prompt.asset_paths[0].endswith("treatmentprime_new.pdf")
    assert prompt.asset_sha256 == (
        "1b1a8187d7797da5184004a39f5fde0e60f6beac0e7fea60a8b2327394392f65",
    )
    assert "Jamal" in prompt.text
    assert '"0":NUMBER' in prompt.text and '"10":NUMBER' in prompt.text


def test_bounded_multimodal_aggregation_requires_complete_paired_cells() -> None:
    root = Path(__file__).resolve().parents[1]
    bundle = json.loads(
        (root / "data/manifests/contracts/pb2rr_blinded_bundle.json").read_text()
    )
    outputs = []
    for arm in bundle["arms"]:
        for level in bundle["nuisance_contract"]["levels"]:
            value = 4 if arm["arm_id"].startswith("iphone") else 6
            outputs.append(
                {
                    "arm_id": arm["arm_id"],
                    "nuisance_id": level["nuisance_id"],
                    "draw_index": 0,
                    "probabilities": {
                        str(option): 1.0 if option == value else 0.0
                        for option in range(11)
                    },
                }
            )
    means = aggregate_bounded_multimodal_predictions(
        outputs, bundle=bundle, draws=1
    )
    assert means == pytest.approx(
        {
            "iphone_growth_control_article": 0.4,
            "hispanic_population_growth_article": 0.6,
        }
    )
    with pytest.raises(ValueError, match="complete"):
        aggregate_bounded_multimodal_predictions(
            outputs[:-1], bundle=bundle, draws=1
        )


def test_egmxd_categorical_bundle_hashes_assets_and_aggregates_utility() -> None:
    root = Path(__file__).resolve().parents[1]
    bundle = json.loads(
        (root / "data/manifests/contracts/egmxd_blinded_bundle.json").read_text()
    )
    validate_categorical_multimodal_bundle(bundle)
    prompt = categorical_multimodal_prompt(
        bundle,
        arm_id="low_climate_impact_label_menu",
        repository_root=root,
    )
    assert prompt.asset_paths[0].endswith("low_climate_impact_menu.png")
    assert "every option exactly once" in prompt.text
    assert '"14":NUMBER' in prompt.text

    outputs = []
    for arm in bundle["arms"]:
        chosen = "2" if arm["arm_id"] == "low_climate_impact_label_menu" else "1"
        outputs.append(
            {
                "arm_id": arm["arm_id"],
                "draw_index": 0,
                "probabilities": {
                    option["option_id"]: 1.0 if option["option_id"] == chosen else 0.0
                    for option in bundle["response_options"]
                },
            }
        )
    assert aggregate_categorical_multimodal_predictions(
        outputs, bundle=bundle, draws=1
    ) == pytest.approx(
        {
            "qr_code_control_menu": 0.0,
            "low_climate_impact_label_menu": 1.0,
            "high_climate_impact_warning_menu": 0.0,
        }
    )
    with pytest.raises(ValueError, match="complete"):
        aggregate_categorical_multimodal_predictions(
            outputs[:-1], bundle=bundle, draws=1
        )


@pytest.mark.parametrize("experiment_id", ["KlarS44", "ShannonS2", "z358z"])
def test_sequence_bundle_is_sealed_deterministic_and_arm_paired(
    experiment_id: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    bundle = json.loads(
        (
            root
            / "data/manifests/contracts"
            / f"{experiment_id}_blinded_bundle.json"
        ).read_text()
    )
    validate_sequence_blinded_bundle(bundle)
    first = materialize_sequence_episode(bundle, seed=77)
    again = materialize_sequence_episode(bundle, seed=77)
    different = materialize_sequence_episode(bundle, seed=78)
    assert first == again
    assert first.episode_id != different.episode_id
    prompts = [
        sequence_probability_prompt(bundle, arm_id=arm["arm_id"], episode=first)
        for arm in bundle["arms"]
    ]
    if first.prior_exposure:
        assert all(first.prior_exposure in prompt for prompt in prompts)
    assert all("{{" not in prompt for prompt in prompts)


def test_sequence_bundle_artifacts_match_deterministic_builders() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "KlarS44": build_klar_sequence_bundle(),
        "ShannonS2": build_shannon_sequence_bundle(),
        "z358z": build_z358z_sequence_bundle(),
    }
    for experiment_id, built in expected.items():
        path = root / "data/manifests/contracts" / f"{experiment_id}_blinded_bundle.json"
        assert json.loads(path.read_text()) == built


def test_sequence_validator_rejects_unpaired_or_unnormalized_randomization() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "data/manifests/contracts/KlarS44_blinded_bundle.json"
    bundle = json.loads(path.read_text())
    bundle["sequence_contract"]["paired_across_arms"] = False
    with pytest.raises(ValueError, match="paired across arms"):
        validate_sequence_blinded_bundle(bundle)

    bundle = json.loads(path.read_text())
    bundle["sequence_contract"]["randomizations"][0]["levels"][0]["weight"] = 0.6
    with pytest.raises(ValueError, match="sum to one"):
        validate_sequence_blinded_bundle(bundle)


def test_shannon_powell_adapter_preserves_all_programmed_paths_and_labels() -> None:
    paths = _powell_paths()
    assert len(paths) == 48
    assert sum(weight for _, weight, _ in paths) == pytest.approx(1.0)
    assert sum("_adult_" in path_id for path_id, _, _ in paths) == 16
    assert sum("_teen_" in path_id for path_id, _, _ in paths) == 32
    assert any("The men's restroom; The women's restroom" in text for _, _, text in paths)
    assert any("The women's restroom; The men's restroom" in text for _, _, text in paths)
    assert any("The boy's restroom; The girl's restroom" in text for _, _, text in paths)


def test_z358z_paired_profile_adapter_preserves_source_design() -> None:
    bundle = build_z358z_sequence_bundle()
    contract = bundle["sequence_contract"]
    module_order = contract["randomizations"][0]
    profiles = contract["randomizations"][1]
    assert [level["weight"] for level in module_order["levels"]] == [0.25] * 4
    assert profiles["kind"] == "paired_profiles"
    assert profiles["pair_count"] == 3
    assert len(profiles["traits"]) == 6
    assert [len(trait["levels"]) for trait in profiles["traits"]] == [4, 2, 4, 3, 3, 3]
    assert profiles["trait_order_randomized"] is True
    found = set()
    for seed in range(100):
        episode = materialize_sequence_episode(bundle, seed=seed)
        selected_order = dict(episode.selections)[
            "kalla_nayak_saperstein_module_order"
        ]
        found.add(selected_order)
        if selected_order in {
            "kalla_nayak_saperstein",
            "saperstein_kalla_nayak",
        }:
            assert "Pair 1." in episode.prior_exposure
    assert found == {
        "kalla_nayak_saperstein",
        "nayak_kalla_saperstein",
        "saperstein_kalla_nayak",
        "saperstein_nayak_kalla",
    }


def test_paired_profile_validator_fails_closed() -> None:
    bundle = build_z358z_sequence_bundle()
    bundle["sequence_contract"]["randomizations"][1]["pair_count"] = 0
    with pytest.raises(ValueError, match="paired-profile layout"):
        validate_sequence_blinded_bundle(bundle)

def test_ordinal_distribution_and_sequence_aggregation_are_strict() -> None:
    parsed = parse_ordinal_distribution(
        '{"probabilities":{"1":0.2,"2":0.3,"3":0.5}}',
        option_values=(1, 2, 3),
    )
    assert dict(parsed.probabilities) == pytest.approx({1: 0.2, 2: 0.3, 3: 0.5})
    with pytest.raises(ValueError, match="every response"):
        parse_ordinal_distribution(
            '{"probabilities":{"1":0.2,"2":0.8}}', option_values=(1, 2, 3)
        )

    root = Path(__file__).resolve().parents[1]
    bundle = json.loads(
        (root / "data/manifests/contracts/KlarS44_blinded_bundle.json").read_text()
    )
    episodes = tuple(
        materialize_sequence_episode(bundle, seed=seed).episode_id for seed in (1, 2)
    )
    outputs = [
        {
            "arm_id": arm["arm_id"],
            "episode_id": episode_id,
            "probabilities": {
                str(value): 1.0 if value == 1 else 0.0 for value in range(1, 8)
            },
        }
        for arm in bundle["arms"]
        for episode_id in episodes
    ]
    means = aggregate_sequence_predictions(
        outputs, bundle=bundle, episode_ids=episodes
    )
    assert set(means) == {arm["arm_id"] for arm in bundle["arms"]}
    assert set(means.values()) == {1.0}
    with pytest.raises(ValueError, match="complete and paired"):
        aggregate_sequence_predictions(
            outputs[:-1], bundle=bundle, episode_ids=episodes
        )
