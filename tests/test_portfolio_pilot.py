from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from intervenebench.pilot import PILOT_EXPERIMENTS
from intervenebench.portfolio_pilot import (
    validate_portfolio_scope,
    verify_portfolio_run,
    verify_portfolio_scope,
)
from intervenebench.portfolio_development import (
    validate_development_reveal_authorization,
    verify_development_reveal_authorization,
    verify_development_score,
)
from intervenebench.simulators import (
    aggregate_ordinal_predictions,
    ordinal_probability_prompt,
    ordinal_variant_contract,
    parse_ordinal_relative_weights,
)


ROOT = Path(__file__).resolve().parents[1]
SCOPE_PATH = ROOT / "data/manifests/benchmark/portfolio_pilot_scope.json"
REVEAL_PATH = (
    ROOT
    / "data/manifests/benchmark/portfolio_pilot_development_reveal.json"
)


def test_portfolio_scope_authorizes_only_local_response_free_work() -> None:
    scope = verify_portfolio_scope(ROOT)
    assert tuple(scope["experiment_ids"]) == PILOT_EXPERIMENTS
    assert scope["local_zero_cost_inference_authorized"] is True
    assert scope["human_outcome_reveal_authorized"] is False
    assert scope["paid_inference_authorized"] is False
    assert scope["modal_compute_authorized"] is False
    assert scope["fine_tuning_authorized"] is False
    assert scope["trust_model_claim_authorized"] is False


def test_portfolio_scope_rejects_reveal_or_paid_authorization() -> None:
    scope = json.loads(SCOPE_PATH.read_text(encoding="utf-8"))
    for field in ("human_outcome_reveal_authorized", "paid_inference_authorized"):
        changed = deepcopy(scope)
        changed[field] = True
        with pytest.raises(ValueError, match="must keep"):
            validate_portfolio_scope(changed)


def test_ordinal_prompt_and_aggregation_preserve_frozen_variants() -> None:
    bundle = json.loads(
        (ROOT / "data/manifests/contracts/de5hx_blinded_bundle.json").read_text()
    )
    arm_ids = tuple(arm["arm_id"] for arm in bundle["arms"])
    outputs = []
    for arm_index, arm_id in enumerate(arm_ids):
        variants = ordinal_variant_contract(bundle, arm_id=arm_id)
        assert len(variants) == 2
        for variant_id, weight in variants:
            prompt = ordinal_probability_prompt(
                bundle, arm_id=arm_id, variant_id=variant_id
            )
            assert bundle["outcome_question"] in prompt
            assert variant_id in {"response_order_jack_first", "response_order_gary_first"}
            assert weight == 0.5
            for draw_index in range(3):
                winner_value = arm_index + 1
                probabilities = {
                    str(option["value"]): (
                        1.0 if option["value"] == winner_value else 0.0
                    )
                    for option in bundle["response_options"]
                }
                outputs.append(
                    {
                        "arm_id": arm_id,
                        "variant_id": variant_id,
                        "draw_index": draw_index,
                        "probabilities": probabilities,
                    }
                )
    means, by_draw = aggregate_ordinal_predictions(
        outputs, bundle=bundle, draws=3
    )
    assert means == {
        arm_ids[0]: 1.0,
        arm_ids[1]: 2.0 / 3.0,
        arm_ids[2]: 1.0 / 3.0,
    }
    assert set(by_draw) == {0, 1, 2}


def test_ordinal_aggregation_fails_on_missing_draw() -> None:
    bundle = json.loads(
        (ROOT / "data/manifests/contracts/5vm8g_blinded_bundle.json").read_text()
    )
    outputs = []
    for arm in bundle["arms"]:
        outputs.append(
            {
                "arm_id": arm["arm_id"],
                "variant_id": "direct",
                "draw_index": 0,
                "probabilities": {"1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2, "5": 0.2},
            }
        )
    outputs.pop()
    with pytest.raises(ValueError, match="not complete"):
        aggregate_ordinal_predictions(outputs, bundle=bundle, draws=1)


def test_relative_weight_parser_is_explicit_and_fails_closed() -> None:
    distribution, weights = parse_ordinal_relative_weights(
        '{"relative_weights":{"1":14,"2":23,"3":27,"4":12,"5":4}}',
        option_values=(1, 2, 3, 4, 5),
    )
    assert weights == ((1, 14.0), (2, 23.0), (3, 27.0), (4, 12.0), (5, 4.0))
    assert sum(probability for _, probability in distribution.probabilities) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="at least one"):
        parse_ordinal_relative_weights(
            '{"relative_weights":{"1":0,"2":0}}', option_values=(1, 2)
        )


def test_frozen_local_portfolio_run_verifies_without_outcomes() -> None:
    manifest = verify_portfolio_run(
        ROOT,
        ROOT
        / "artifacts/portfolio_pilot/local_llama3_2_3b_20260813_v2/run_manifest.json",
    )
    assert tuple(manifest["experiment_ids"]) == PILOT_EXPERIMENTS
    assert manifest["human_outcomes_opened"] is False
    assert manifest["human_outcome_reveal_authorized"] is False
    assert manifest["paid_cost_usd"] == 0.0
    assert manifest["modal_used"] is False


def test_development_reveal_is_separate_bound_and_noncanonical() -> None:
    authorization = verify_development_reveal_authorization(ROOT)
    assert tuple(authorization["experiment_ids"]) == PILOT_EXPERIMENTS
    assert authorization["permanent_role"] == "development_only_portfolio_reveal"
    assert authorization["canonical_test_eligible"] is False
    assert authorization["canonical_split_status"] == "unassigned"
    assert authorization["paid_inference_authorized"] is False
    assert authorization["modal_compute_authorized"] is False
    assert set(authorization["other_runnable_contracts_must_remain_sealed"]) == {
        "socsci210:tcg8p",
        "socsci210:pb2rr",
        "socsci210:z358z",
        "external_archive_v1:ShannonS2",
        "external_archive_v1:Blair1131",
        "external_archive_v1:KlarS44",
    }


def test_development_reveal_fails_if_fallback_can_adapt_to_outcomes() -> None:
    authorization = json.loads(REVEAL_PATH.read_text(encoding="utf-8"))
    changed = deepcopy(authorization)
    changed["human_fallback_contract"]["no_outcome_adaptive_allocation"] = False
    with pytest.raises(ValueError, match="frozen before outcomes"):
        validate_development_reveal_authorization(changed)


def test_completed_development_score_is_bound_aggregate_only_and_honest() -> None:
    score = verify_development_score(
        ROOT,
        ROOT / "artifacts/portfolio_pilot/development_score_v2.json",
    )
    assert score["development_only"] is True
    assert score["canonical_test_claim"] is False
    assert score["participant_rows_written_to_artifact"] == 0
    assert score["paid_cost_usd"] == 0.0
    assert score["modal_used"] is False
    local = score["portfolio_summary"]["local_llama3_2_3b"]
    baseline = score["portfolio_summary"]["no_effect_control_tie"]
    assert local["correct_intervention_count"] == 3
    assert baseline["correct_intervention_count"] == 0
    assert local["mean_decision_regret"] < baseline["mean_decision_regret"]
    assert local["mean_treatment_effect_mae"] > baseline["mean_treatment_effect_mae"]


def test_public_portfolio_summary_matches_frozen_score() -> None:
    score = verify_development_score(
        ROOT,
        ROOT / "artifacts/portfolio_pilot/development_score_v2.json",
    )
    local = score["portfolio_summary"]["local_llama3_2_3b"]
    baseline = score["portfolio_summary"]["no_effect_control_tie"]
    expected_public_values = {
        f'{local["correct_intervention_count"]}/5',
        f'{baseline["correct_intervention_count"]}/5',
        f'{local["mean_decision_regret"]:.4f}',
        f'{baseline["mean_decision_regret"]:.4f}',
        f'{local["worst_case_decision_regret"]:.4f}',
        f'{baseline["worst_case_decision_regret"]:.4f}',
        f'{local["mean_treatment_effect_mae"]:.4f}',
        f'{baseline["mean_treatment_effect_mae"]:.4f}',
    }
    for relative_path in (
        "README.md",
        "docs/PORTFOLIO_BRIEF.md",
    ):
        public_text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert all(value in public_text for value in expected_public_values)
