from __future__ import annotations

import pytest

from intervenebench.sampling_convergence import choose_sampling_checkpoint


def test_convergence_selects_first_stable_outcome_free_checkpoint() -> None:
    decision = choose_sampling_checkpoint(
        [
            (3, {"control": 0.50, "treatment": 0.54}),
            (5, {"control": 0.505, "treatment": 0.555}),
            (10, {"control": 0.507, "treatment": 0.554}),
            (20, {"control": 0.507, "treatment": 0.553}),
        ]
    )
    assert decision.converged is True
    assert decision.sample_count == 10
    assert decision.winner_arm_id == "treatment"
    assert decision.reason == "winner_and_arm_means_stable"


def test_convergence_retains_cap_and_marks_instability() -> None:
    decision = choose_sampling_checkpoint(
        [
            (3, {"a": 0.51, "b": 0.50}),
            (5, {"a": 0.49, "b": 0.52}),
            (10, {"a": 0.53, "b": 0.50}),
            (20, {"a": 0.49, "b": 0.54}),
        ]
    )
    assert decision.converged is False
    assert decision.sample_count == 20
    assert decision.winner_arm_id == "b"
    assert decision.reason == "maximum_sampling_checkpoint_reached_without_convergence"


def test_small_margin_does_not_pass_despite_small_arm_changes() -> None:
    decision = choose_sampling_checkpoint(
        [
            (3, {"a": 0.500, "b": 0.501}),
            (5, {"a": 0.501, "b": 0.502}),
        ],
        arm_mean_tolerance=0.01,
        margin_multiplier=2.0,
    )
    assert decision.converged is False


def test_convergence_rejects_incomparable_checkpoints() -> None:
    with pytest.raises(ValueError, match="same arm IDs"):
        choose_sampling_checkpoint(
            [(3, {"a": 0.5, "b": 0.6}), (5, {"a": 0.5, "c": 0.6})]
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        choose_sampling_checkpoint(
            [(5, {"a": 0.5, "b": 0.6}), (3, {"a": 0.5, "b": 0.6})]
        )
