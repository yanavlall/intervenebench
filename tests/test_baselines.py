from __future__ import annotations

import pytest

from intervenebench.baselines import (
    LabeledBaselineExample,
    LabeledEffectExample,
    fit_hashed_effect_ridge,
    fit_hashed_ridge,
)


def fixtures() -> list[LabeledBaselineExample]:
    return [
        LabeledBaselineExample(
            "train-a",
            "train",
            {"message_kind": "control", "support_score": 0.0},
            0.1,
        ),
        LabeledBaselineExample(
            "train-a",
            "train",
            {"message_kind": "supportive", "support_score": 1.0},
            0.9,
        ),
        LabeledBaselineExample(
            "train-b",
            "train",
            {"message_kind": "control", "support_score": 0.1},
            0.2,
        ),
        LabeledBaselineExample(
            "train-b",
            "train",
            {"message_kind": "supportive", "support_score": 0.9},
            0.8,
        ),
    ]


SPLIT = {"train-a": "train", "train-b": "train", "held-out": "test"}


def test_hashed_ridge_is_deterministic_bounded_and_learns_fixture_signal() -> None:
    first = fit_hashed_ridge(
        fixtures(), experiment_to_split=SPLIT, feature_dimension=16, l2_penalty=0.1
    )
    second = fit_hashed_ridge(
        fixtures(), experiment_to_split=SPLIT, feature_dimension=16, l2_penalty=0.1
    )
    assert first == second
    control = first.predict({"message_kind": "control", "support_score": 0.0})
    supportive = first.predict(
        {"message_kind": "supportive", "support_score": 1.0}
    )
    assert 0.0 <= control <= 1.0
    assert 0.0 <= supportive <= 1.0
    assert supportive > control
    assert first.training_experiment_ids == ("train-a", "train-b")


@pytest.mark.parametrize("forbidden_split", ["validation", "test"])
def test_classical_baseline_rejects_nontraining_labels(forbidden_split: str) -> None:
    bad = fixtures() + [
        LabeledBaselineExample(
            "held-out",
            forbidden_split,
            {"message_kind": "supportive", "support_score": 1.0},
            0.9,
        )
    ]
    with pytest.raises(ValueError, match="train split labels only"):
        fit_hashed_ridge(bad, experiment_to_split=SPLIT)


def test_classical_baseline_rejects_relabeling_a_held_out_experiment() -> None:
    relabeled = [
        LabeledBaselineExample(
            "held-out",
            "train",
            {"message_kind": "supportive", "support_score": 1.0},
            0.9,
        )
    ]
    with pytest.raises(ValueError, match="supplied split manifest"):
        fit_hashed_ridge(relabeled, experiment_to_split=SPLIT)


def test_classical_baseline_rejects_invalid_features_and_targets() -> None:
    with pytest.raises(ValueError, match="normalized"):
        fit_hashed_ridge(
            [LabeledBaselineExample("train-a", "train", {"x": 1.0}, 2.0)],
            experiment_to_split=SPLIT,
        )
    model = fit_hashed_ridge(
        fixtures(), experiment_to_split=SPLIT, feature_dimension=16
    )
    with pytest.raises(ValueError, match="finite"):
        model.predict({"x": float("nan")})


def test_effect_ridge_supports_signed_targets_and_experiment_weights() -> None:
    examples = [
        LabeledEffectExample(
            "train-a", "train", {"contrast": -1.0}, -0.8, sample_weight=0.5
        ),
        LabeledEffectExample(
            "train-a", "train", {"contrast": -0.5}, -0.4, sample_weight=0.5
        ),
        LabeledEffectExample(
            "train-b", "train", {"contrast": 1.0}, 0.8, sample_weight=1.0
        ),
    ]
    model = fit_hashed_effect_ridge(
        examples,
        experiment_to_split={"train-a": "train", "train-b": "train"},
        feature_dimension=16,
        l2_penalty=0.1,
    )
    assert model.predict_effect({"contrast": -1.0}) < 0.0
    assert model.predict_effect({"contrast": 1.0}) > 0.0
    assert -1.0 <= model.predict_effect({"contrast": 1e9}) <= 1.0
    assert model.training_experiment_weights == (("train-a", 1.0), ("train-b", 1.0))


def test_effect_ridge_rejects_held_out_labels_and_invalid_weights() -> None:
    with pytest.raises(ValueError, match="train split labels only"):
        fit_hashed_effect_ridge(
            [LabeledEffectExample("held", "test", {"x": 1.0}, 0.2)],
            experiment_to_split={"held": "test"},
        )
    with pytest.raises(ValueError, match="sample weights"):
        fit_hashed_effect_ridge(
            [
                LabeledEffectExample(
                    "train", "train", {"x": 1.0}, 0.2, sample_weight=0.0
                )
            ],
            experiment_to_split={"train": "train"},
        )
