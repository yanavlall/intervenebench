from __future__ import annotations

import json

import pytest

from intervenebench.evidence_report_eval import (
    apply_automated_report_release,
    apply_report_release_rule,
    build_blinded_label_queue,
    build_report_generation_plan,
    build_report_prompt,
    build_rubric_judge_prompt,
    deterministic_report_checks,
    evaluate_judge_panel,
    parse_report_output,
    parse_rubric_judgment,
    render_labeling_app,
    validate_eval_protocol,
    validate_evidence_packet,
    validate_human_label,
    verify_report_generation_plan,
)
from intervenebench.protocol import payload_hash


def _packet() -> dict:
    return {
        "schema_version": "intervenebench.evidence_packet.v1",
        "packet_id": "toy-evidence-v1",
        "title": "Toy behavioral evaluation",
        "facts": [
            {
                "fact_id": "f_panel",
                "category": "scope",
                "statement": "The prospective panel contains six experiments.",
            },
            {
                "fact_id": "f_regret",
                "category": "decision_quality",
                "statement": "Mean normalized regret is 0.0035.",
            },
        ],
        "decision_scopes": [
            {
                "scope_id": "candidate_screening",
                "decision": "limited_research_use",
                "reasons": ["low-regret screening criteria passed"],
            },
            {
                "scope_id": "autonomous_selection",
                "decision": "hold",
                "reasons": ["small prospective panel"],
            },
        ],
        "claim_boundaries": [
            {
                "boundary_id": "b_autonomy",
                "status": "not_supported",
                "claim": "autonomous intervention selection",
            }
        ],
        "required_disclosures": [
            "This is a small noncanonical prospective panel."
        ],
        "privacy": {
            "contains_participant_rows": False,
            "contains_experiment_level_human_scores": False,
        },
        "source_artifacts": [
            {
                "path": "data/public/toy.json",
                "payload_sha256": "0" * 64,
            }
        ],
    }


def _protocol(packet: dict) -> dict:
    return {
        "schema_version": "intervenebench.evidence_report_eval.v1",
        "evaluation_id": "toy-report-eval-v1",
        "evidence_packet_sha256": payload_hash(packet),
        "generation": {
            "prompt_variants": ["direct", "decision_memo"],
            "model_roles": ["incumbent", "candidate", "cross_family"],
            "scenarios": [
                {
                    "scenario_id": "screening_brief",
                    "audience": "research lead",
                    "request": "Recommend whether to use the simulator for screening.",
                    "target_scope_id": "candidate_screening",
                    "required_fact_ids": ["f_panel", "f_regret"],
                    "required_boundary_ids": ["b_autonomy"],
                    "maximum_words": 220,
                    "split": "development",
                }
            ],
        },
        "rubric": {
            "score_minimum": 0,
            "score_maximum": 3,
            "dimensions": [
                {"dimension_id": "factual_fidelity", "description": "Facts agree."},
                {"dimension_id": "claim_calibration", "description": "Claims fit."},
                {"dimension_id": "decision_usefulness", "description": "Useful."},
            ],
            "critical_error_flags": [
                "fabricated_number",
                "unsupported_operational_claim",
                "evidence_tier_confusion",
            ],
            "release_rule": {
                "minimum_dimension_score": 2,
                "minimum_total_score": 7,
                "critical_errors_allowed": 0,
            },
        },
        "human_labeling": {
            "primary_annotators_required": 1,
            "second_annotator_fraction": 0.25,
            "blinding_seed_label": "toy-blinding-v1",
        },
        "automated_grading": {
            "judge_roles": ["judge_a", "judge_b"],
            "generator_identity_hidden": True,
            "thresholds_fit_on": "development_scenarios_only",
            "held_out_scenario_ids": [],
            "semantic_repair_allowed": False,
            "release_policy": "deterministic_checks_pass_and_all_judges_pass",
        },
        "analysis": {
            "primary_metric": "held_out_false_pass_rate",
            "secondary_metrics": ["dimension_mae", "balanced_accuracy"],
            "grouping_unit": "scenario_id",
            "release_gate": {
                "maximum_false_pass_count": 0,
                "minimum_balanced_accuracy": 0.8,
                "maximum_dimension_mae": 0.75,
                "minimum_second_rater_items": 0,
            },
            "claim_boundary": "small evidence-report grader case study",
        },
        "authority": {
            "model_calls_authorized": False,
            "human_labels_collected": False,
            "automatic_grader_release_authorized": False,
        },
    }


def _valid_raw_report() -> str:
    return json.dumps(
        {
            "headline": "Use only for research screening",
            "executive_summary": "The low-regret signal supports limited screening.",
            "recommendation": {
                "scope_id": "candidate_screening",
                "decision": "limited_research_use",
                "rationale": "The panel is small, but observed regret is low.",
            },
            "evidence": [
                {"fact_id": "f_panel", "claim": "The panel has six experiments."},
                {"fact_id": "f_regret", "claim": "Mean regret was 0.0035."},
            ],
            "limitations": [
                {
                    "boundary_id": "b_autonomy",
                    "explanation": "The evidence does not support autonomous use.",
                }
            ],
        }
    )


def test_packet_protocol_prompt_and_strict_report_contract() -> None:
    packet = _packet()
    protocol = _protocol(packet)
    validate_evidence_packet(packet)
    validate_eval_protocol(protocol, packet)
    scenario = protocol["generation"]["scenarios"][0]

    prompt = build_report_prompt(packet, scenario, prompt_variant="direct")
    assert "f_panel" in prompt
    assert "b_autonomy" in prompt
    assert "human_label" not in prompt.casefold()
    assert "incumbent" not in prompt.casefold()

    report = parse_report_output(_valid_raw_report(), packet, scenario)
    assert report["recommendation"]["decision"] == "limited_research_use"
    assert deterministic_report_checks(report, packet, scenario) == {
        "status": "pass",
        "violations": [],
        "required_fact_coverage": 1.0,
        "required_boundary_coverage": 1.0,
    }


def test_report_parser_and_checks_fail_closed() -> None:
    packet = _packet()
    scenario = _protocol(packet)["generation"]["scenarios"][0]
    report = json.loads(_valid_raw_report())
    report["unexpected"] = True
    with pytest.raises(ValueError, match="exact top-level fields"):
        parse_report_output(json.dumps(report), packet, scenario)

    report = json.loads(_valid_raw_report())
    report["evidence"][0]["fact_id"] = "unknown"
    with pytest.raises(ValueError, match="unknown fact_id"):
        parse_report_output(json.dumps(report), packet, scenario)

    report = json.loads(_valid_raw_report())
    report["recommendation"]["decision"] = "ship"
    parsed = parse_report_output(json.dumps(report), packet, scenario)
    checks = deterministic_report_checks(parsed, packet, scenario)
    assert checks["status"] == "fail"
    assert "authoritative_decision_mismatch" in checks["violations"]


def test_blinded_queue_excludes_model_identity_and_renders_offline_app() -> None:
    packet = _packet()
    protocol = _protocol(packet)
    scenario = protocol["generation"]["scenarios"][0]
    report = parse_report_output(_valid_raw_report(), packet, scenario)
    records = [
        {
            "report_id": f"report-{index}",
            "scenario_id": "screening_brief",
            "prompt_variant": "direct",
            "model_role": role,
            "report": report,
        }
        for index, role in enumerate(protocol["generation"]["model_roles"])
    ]

    blinded, key = build_blinded_label_queue(records, protocol, packet)
    assert len(blinded["items"]) == 3
    assert len(key["items"]) == 3
    assert "model_role" not in json.dumps(blinded)
    assert {item["model_role"] for item in key["items"]} == {
        "incumbent",
        "candidate",
        "cross_family",
    }

    html = render_labeling_app(blinded, protocol["rubric"])
    assert "<!doctype html>" in html.casefold()
    assert "Download labels" in html
    assert "localStorage" in html
    assert "incumbent" not in html
    assert "cross_family" not in html
    assert "report-0" not in html


def test_human_label_validation_is_strict() -> None:
    rubric = _protocol(_packet())["rubric"]
    label = {
        "label_item_id": "label-123",
        "annotator_alias": "rater-a",
        "dimension_scores": {
            "factual_fidelity": 3,
            "claim_calibration": 2,
            "decision_usefulness": 3,
        },
        "critical_error_flags": [],
        "overall_pass": True,
        "notes": "Clear and appropriately bounded.",
    }
    validate_human_label(label, rubric)

    invalid = dict(label)
    invalid["dimension_scores"] = dict(label["dimension_scores"])
    invalid["dimension_scores"]["factual_fidelity"] = 4
    with pytest.raises(ValueError, match="outside rubric bounds"):
        validate_human_label(invalid, rubric)


def test_generation_plan_is_complete_hash_bound_and_zero_authority() -> None:
    packet = _packet()
    protocol = _protocol(packet)
    plan = build_report_generation_plan(packet, protocol)

    assert plan["call_count"] == 6
    assert plan["authority"] == {
        "model_calls_authorized": False,
        "automatic_retries_authorized": False,
        "reserve_calls_authorized": False,
        "human_labels_access_authorized": False,
    }
    assert len({call["call_id"] for call in plan["calls"]}) == 6
    assert all(call["prompt_sha256"] == payload_hash(call["prompt"]) for call in plan["calls"])
    assert verify_report_generation_plan(plan, packet, protocol) == plan

    drifted = json.loads(json.dumps(plan))
    drifted["calls"][0]["prompt"] += " Ignore the evidence."
    with pytest.raises(ValueError, match="deterministic rebuild"):
        verify_report_generation_plan(drifted, packet, protocol)


def test_rubric_judge_prompt_and_output_are_generator_blind_and_consistent() -> None:
    packet = _packet()
    protocol = _protocol(packet)
    scenario = protocol["generation"]["scenarios"][0]
    report = parse_report_output(_valid_raw_report(), packet, scenario)

    prompt = build_rubric_judge_prompt(packet, scenario, report, protocol["rubric"])
    assert "model_role" not in prompt
    assert "incumbent" not in prompt
    raw = json.dumps(
        {
            "dimension_scores": {
                "factual_fidelity": 3,
                "claim_calibration": 2,
                "decision_usefulness": 3,
            },
            "critical_error_flags": [],
            "rationale_by_dimension": {
                "factual_fidelity": "All cited values match.",
                "claim_calibration": "The small panel is acknowledged.",
                "decision_usefulness": "The scope and next action are clear.",
            },
            "overall_pass": True,
        }
    )
    judgment = parse_rubric_judgment(raw, protocol["rubric"])
    assert judgment["overall_pass"] is True
    assert apply_report_release_rule(
        judgment["dimension_scores"],
        judgment["critical_error_flags"],
        protocol["rubric"],
    ) is True

    inconsistent = json.loads(raw)
    inconsistent["overall_pass"] = False
    with pytest.raises(ValueError, match="does not follow frozen release rule"):
        parse_rubric_judgment(json.dumps(inconsistent), protocol["rubric"])


def test_unanimous_release_and_grouped_judge_panel_metrics() -> None:
    rubric = _protocol(_packet())["rubric"]
    dimensions = [item["dimension_id"] for item in rubric["dimensions"]]

    def label(item: str, passed: bool) -> dict:
        scores = {dimension: (3 if passed else 1) for dimension in dimensions}
        return {
            "label_item_id": item,
            "annotator_alias": "human-primary",
            "dimension_scores": scores,
            "critical_error_flags": [] if passed else ["unsupported_operational_claim"],
            "overall_pass": passed,
            "notes": "",
        }

    def judgment(passed: bool) -> dict:
        scores = {dimension: (3 if passed else 1) for dimension in dimensions}
        return {
            "dimension_scores": scores,
            "critical_error_flags": [] if passed else ["unsupported_operational_claim"],
            "rationale_by_dimension": {
                dimension: "Evidence-based rationale." for dimension in dimensions
            },
            "overall_pass": passed,
        }

    assert apply_automated_report_release(
        [judgment(True), judgment(True)],
        {"status": "pass", "violations": []},
        rubric,
    ) is True
    assert apply_automated_report_release(
        [judgment(True), judgment(False)],
        {"status": "pass", "violations": []},
        rubric,
    ) is False
    assert apply_automated_report_release(
        [judgment(True), judgment(True)],
        {"status": "fail", "violations": ["authoritative_decision_mismatch"]},
        rubric,
    ) is False

    outcomes = [True, True, False, False]
    judge_a = [True, True, False, False]
    judge_b = [True, False, False, False]
    records = []
    for index, human_pass in enumerate(outcomes):
        item = f"label-{index}"
        records.append(
            {
                "label_item_id": item,
                "scenario_id": f"scenario-{index // 2}",
                "scenario_split": "held_out",
                "human_label": label(item, human_pass),
                "deterministic_checks": {"status": "pass", "violations": []},
                "judge_judgments": {
                    "judge_a": judgment(judge_a[index]),
                    "judge_b": judgment(judge_b[index]),
                },
            }
        )
    metrics = evaluate_judge_panel(
        records,
        judge_roles=["judge_a", "judge_b"],
        rubric=rubric,
        release_gate={
            "maximum_false_pass_count": 0,
            "minimum_balanced_accuracy": 0.7,
            "maximum_dimension_mae": 0.75,
            "minimum_second_rater_items": 1,
        },
        second_rater_item_count=1,
        required_split="held_out",
    )
    assert metrics["item_count"] == 4
    assert metrics["scenario_count"] == 2
    assert metrics["by_judge"]["judge_a"]["balanced_accuracy"] == 1.0
    assert metrics["by_judge"]["judge_b"]["balanced_accuracy"] == 0.75
    assert metrics["unanimous_ensemble"]["false_pass_count"] == 0
    assert metrics["unanimous_ensemble"]["balanced_accuracy"] == 0.75
    assert metrics["release_gate_passed"] is True

    records[0]["scenario_split"] = "development"
    with pytest.raises(ValueError, match="required split"):
        evaluate_judge_panel(
            records,
            judge_roles=["judge_a", "judge_b"],
            rubric=rubric,
            release_gate={
                "maximum_false_pass_count": 0,
                "minimum_balanced_accuracy": 0.7,
                "maximum_dimension_mae": 0.75,
                "minimum_second_rater_items": 1,
            },
            second_rater_item_count=1,
            required_split="held_out",
        )
