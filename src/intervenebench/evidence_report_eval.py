"""Reusable evaluation primitives for evidence-grounded qualitative reports.

The module treats report generation, human labeling, automated checks, and
release decisions as separate protocol states.  It intentionally has no model
client and no data-access code: execution requires a separately frozen call
plan, while labels are collected in a blinded offline application.
"""

from __future__ import annotations

import html
import json
from hashlib import sha256
from math import isfinite
from typing import Any, Mapping, Sequence

from .protocol import canonical_json_bytes, payload_hash


_PACKET_KEYS = {
    "schema_version",
    "packet_id",
    "title",
    "facts",
    "decision_scopes",
    "claim_boundaries",
    "required_disclosures",
    "privacy",
    "source_artifacts",
}
_REPORT_KEYS = {
    "headline",
    "executive_summary",
    "recommendation",
    "evidence",
    "limitations",
}
_LABEL_KEYS = {
    "label_item_id",
    "annotator_alias",
    "dimension_scores",
    "critical_error_flags",
    "overall_pass",
    "notes",
}
_JUDGMENT_KEYS = {
    "dimension_scores",
    "critical_error_flags",
    "rationale_by_dimension",
    "overall_pass",
}


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} must contain exact fields {sorted(expected)}")


def _nonempty_string(value: Any, *, label: str, maximum: int = 10_000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} must be a non-empty bounded string")
    return value


def _unique_strings(values: Any, *, label: str, minimum: int = 1) -> list[str]:
    if not isinstance(values, list) or len(values) < minimum:
        raise ValueError(f"{label} must be a non-empty list")
    checked = [_nonempty_string(value, label=label, maximum=200) for value in values]
    if len(set(checked)) != len(checked):
        raise ValueError(f"{label} must be unique")
    return checked


def _sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def validate_evidence_packet(packet: Mapping[str, Any]) -> None:
    _exact_keys(packet, _PACKET_KEYS, label="evidence packet")
    if packet["schema_version"] != "intervenebench.evidence_packet.v1":
        raise ValueError("unsupported evidence packet schema")
    _nonempty_string(packet["packet_id"], label="packet_id", maximum=120)
    _nonempty_string(packet["title"], label="title", maximum=300)

    facts = packet["facts"]
    if not isinstance(facts, list) or not facts:
        raise ValueError("facts must be a non-empty list")
    fact_ids: list[str] = []
    for fact in facts:
        if not isinstance(fact, Mapping):
            raise ValueError("fact records must be objects")
        _exact_keys(
            fact,
            {"fact_id", "category", "statement"},
            label="fact record",
        )
        fact_ids.append(_nonempty_string(fact["fact_id"], label="fact_id", maximum=100))
        _nonempty_string(fact["category"], label="fact category", maximum=100)
        _nonempty_string(fact["statement"], label="fact statement", maximum=1_000)
    if len(set(fact_ids)) != len(fact_ids):
        raise ValueError("fact_id values must be unique")

    scopes = packet["decision_scopes"]
    if not isinstance(scopes, list) or not scopes:
        raise ValueError("decision_scopes must be a non-empty list")
    scope_ids: list[str] = []
    for scope in scopes:
        if not isinstance(scope, Mapping):
            raise ValueError("decision scope records must be objects")
        _exact_keys(
            scope,
            {"scope_id", "decision", "reasons"},
            label="decision scope",
        )
        scope_ids.append(_nonempty_string(scope["scope_id"], label="scope_id", maximum=100))
        _nonempty_string(scope["decision"], label="scope decision", maximum=100)
        _unique_strings(scope["reasons"], label="scope reasons")
    if len(set(scope_ids)) != len(scope_ids):
        raise ValueError("scope_id values must be unique")

    boundaries = packet["claim_boundaries"]
    if not isinstance(boundaries, list) or not boundaries:
        raise ValueError("claim_boundaries must be a non-empty list")
    boundary_ids: list[str] = []
    for boundary in boundaries:
        if not isinstance(boundary, Mapping):
            raise ValueError("claim boundary records must be objects")
        _exact_keys(
            boundary,
            {"boundary_id", "status", "claim"},
            label="claim boundary",
        )
        boundary_ids.append(
            _nonempty_string(boundary["boundary_id"], label="boundary_id", maximum=100)
        )
        if boundary["status"] not in {"supported", "not_supported"}:
            raise ValueError("boundary status must be supported or not_supported")
        _nonempty_string(boundary["claim"], label="boundary claim", maximum=500)
    if len(set(boundary_ids)) != len(boundary_ids):
        raise ValueError("boundary_id values must be unique")

    _unique_strings(packet["required_disclosures"], label="required disclosures")
    if packet["privacy"] != {
        "contains_participant_rows": False,
        "contains_experiment_level_human_scores": False,
    }:
        raise ValueError("evidence packet privacy boundary is invalid")
    sources = packet["source_artifacts"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("source_artifacts must be a non-empty list")
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("source artifact records must be objects")
        _exact_keys(source, {"path", "payload_sha256"}, label="source artifact")
        path = _nonempty_string(source["path"], label="source path", maximum=500)
        if path.startswith("/") or ".." in path.split("/"):
            raise ValueError("source artifact path escapes repository")
        _sha256(source["payload_sha256"], label="source payload_sha256")


def _rubric_metadata(rubric: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    _exact_keys(
        rubric,
        {
            "score_minimum",
            "score_maximum",
            "dimensions",
            "critical_error_flags",
            "release_rule",
        },
        label="rubric",
    )
    minimum = rubric["score_minimum"]
    maximum = rubric["score_maximum"]
    if (
        isinstance(minimum, bool)
        or isinstance(maximum, bool)
        or not isinstance(minimum, int)
        or not isinstance(maximum, int)
        or minimum < 0
        or maximum <= minimum
    ):
        raise ValueError("rubric bounds are invalid")
    dimensions = rubric["dimensions"]
    if not isinstance(dimensions, list) or len(dimensions) < 2:
        raise ValueError("rubric requires at least two dimensions")
    dimension_ids: list[str] = []
    for dimension in dimensions:
        if not isinstance(dimension, Mapping):
            raise ValueError("rubric dimensions must be objects")
        _exact_keys(
            dimension,
            {"dimension_id", "description"},
            label="rubric dimension",
        )
        dimension_ids.append(
            _nonempty_string(
                dimension["dimension_id"], label="dimension_id", maximum=100
            )
        )
        _nonempty_string(
            dimension["description"], label="dimension description", maximum=1_000
        )
    if len(set(dimension_ids)) != len(dimension_ids):
        raise ValueError("rubric dimension IDs must be unique")
    flags = _unique_strings(
        rubric["critical_error_flags"], label="critical error flags"
    )
    rule = rubric["release_rule"]
    if not isinstance(rule, Mapping):
        raise ValueError("release_rule must be an object")
    _exact_keys(
        rule,
        {
            "minimum_dimension_score",
            "minimum_total_score",
            "critical_errors_allowed",
        },
        label="release rule",
    )
    if (
        not isinstance(rule["minimum_dimension_score"], int)
        or not minimum <= rule["minimum_dimension_score"] <= maximum
        or not isinstance(rule["minimum_total_score"], int)
        or not len(dimension_ids) * minimum
        <= rule["minimum_total_score"]
        <= len(dimension_ids) * maximum
        or rule["critical_errors_allowed"] != 0
    ):
        raise ValueError("release rule is incompatible with rubric bounds")
    return dimension_ids, flags


def validate_eval_protocol(
    protocol: Mapping[str, Any], packet: Mapping[str, Any]
) -> None:
    validate_evidence_packet(packet)
    _exact_keys(
        protocol,
        {
            "schema_version",
            "evaluation_id",
            "evidence_packet_sha256",
            "generation",
            "rubric",
            "human_labeling",
            "automated_grading",
            "analysis",
            "authority",
        },
        label="evaluation protocol",
    )
    if protocol["schema_version"] != "intervenebench.evidence_report_eval.v1":
        raise ValueError("unsupported evidence-report evaluation schema")
    _nonempty_string(protocol["evaluation_id"], label="evaluation_id", maximum=120)
    if protocol["evidence_packet_sha256"] != payload_hash(packet):
        raise ValueError("evaluation protocol is not bound to evidence packet")

    generation = protocol["generation"]
    if not isinstance(generation, Mapping):
        raise ValueError("generation must be an object")
    _exact_keys(
        generation,
        {"prompt_variants", "model_roles", "scenarios"},
        label="generation",
    )
    variants = _unique_strings(
        generation["prompt_variants"], label="prompt variants", minimum=2
    )
    roles = _unique_strings(generation["model_roles"], label="model roles", minimum=2)
    if set(variants) & set(roles):
        raise ValueError("prompt variants and model roles must be distinct")
    scenarios = generation["scenarios"]
    if not isinstance(scenarios, list) or len(scenarios) < 1:
        raise ValueError("generation requires scenarios")
    fact_ids = {fact["fact_id"] for fact in packet["facts"]}
    boundary_ids = {
        boundary["boundary_id"] for boundary in packet["claim_boundaries"]
    }
    scope_ids = {scope["scope_id"] for scope in packet["decision_scopes"]}
    scenario_ids: list[str] = []
    for scenario in scenarios:
        if not isinstance(scenario, Mapping):
            raise ValueError("scenario records must be objects")
        _exact_keys(
            scenario,
            {
                "scenario_id",
                "audience",
                "request",
                "target_scope_id",
                "required_fact_ids",
                "required_boundary_ids",
                "maximum_words",
                "split",
            },
            label="scenario",
        )
        scenario_ids.append(
            _nonempty_string(scenario["scenario_id"], label="scenario_id", maximum=100)
        )
        _nonempty_string(scenario["audience"], label="scenario audience", maximum=200)
        _nonempty_string(scenario["request"], label="scenario request", maximum=1_000)
        if scenario["target_scope_id"] not in scope_ids:
            raise ValueError("scenario target_scope_id is unknown")
        if not set(
            _unique_strings(scenario["required_fact_ids"], label="required fact IDs")
        ).issubset(fact_ids):
            raise ValueError("scenario contains an unknown required fact")
        if not set(
            _unique_strings(
                scenario["required_boundary_ids"], label="required boundary IDs"
            )
        ).issubset(boundary_ids):
            raise ValueError("scenario contains an unknown required boundary")
        if (
            not isinstance(scenario["maximum_words"], int)
            or not 100 <= scenario["maximum_words"] <= 1_000
        ):
            raise ValueError("scenario maximum_words must lie in [100, 1000]")
        if scenario["split"] not in {"development", "held_out"}:
            raise ValueError("scenario split must be development or held_out")
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("scenario IDs must be unique")

    rubric = protocol["rubric"]
    _rubric_metadata(rubric)
    labeling = protocol["human_labeling"]
    if not isinstance(labeling, Mapping):
        raise ValueError("human_labeling must be an object")
    _exact_keys(
        labeling,
        {
            "primary_annotators_required",
            "second_annotator_fraction",
            "blinding_seed_label",
        },
        label="human_labeling",
    )
    fraction = labeling["second_annotator_fraction"]
    if (
        labeling["primary_annotators_required"] != 1
        or isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not isfinite(fraction)
        or not 0.0 <= float(fraction) <= 1.0
    ):
        raise ValueError("human labeling requirements are invalid")
    _nonempty_string(
        labeling["blinding_seed_label"], label="blinding seed", maximum=200
    )

    grading = protocol["automated_grading"]
    if not isinstance(grading, Mapping):
        raise ValueError("automated_grading must be an object")
    _exact_keys(
        grading,
        {
            "judge_roles",
            "generator_identity_hidden",
            "thresholds_fit_on",
            "held_out_scenario_ids",
            "semantic_repair_allowed",
            "release_policy",
        },
        label="automated_grading",
    )
    _unique_strings(grading["judge_roles"], label="judge roles", minimum=2)
    if (
        grading["generator_identity_hidden"] is not True
        or grading["thresholds_fit_on"] != "development_scenarios_only"
        or grading["semantic_repair_allowed"] is not False
        or grading["release_policy"]
        != "deterministic_checks_pass_and_all_judges_pass"
    ):
        raise ValueError("automated grader blinding or fitting boundary is invalid")
    held_out = grading["held_out_scenario_ids"]
    if not isinstance(held_out, list) or len(set(held_out)) != len(held_out):
        raise ValueError("held_out_scenario_ids must be a unique list")
    expected_held_out = {
        scenario["scenario_id"]
        for scenario in scenarios
        if scenario["split"] == "held_out"
    }
    if set(held_out) != expected_held_out:
        raise ValueError("grader held-out scenarios do not match scenario split")

    analysis = protocol["analysis"]
    if not isinstance(analysis, Mapping):
        raise ValueError("analysis must be an object")
    _exact_keys(
        analysis,
        {
            "primary_metric",
            "secondary_metrics",
            "grouping_unit",
            "release_gate",
            "claim_boundary",
        },
        label="analysis",
    )
    if analysis["primary_metric"] != "held_out_false_pass_rate":
        raise ValueError("primary qualitative grader metric is not frozen")
    _unique_strings(analysis["secondary_metrics"], label="secondary metrics")
    if analysis["grouping_unit"] != "scenario_id":
        raise ValueError("qualitative analysis must group by scenario_id")
    _nonempty_string(
        analysis["claim_boundary"], label="analysis claim boundary", maximum=500
    )
    release_gate = analysis["release_gate"]
    if not isinstance(release_gate, Mapping):
        raise ValueError("analysis release_gate must be an object")
    _exact_keys(
        release_gate,
        {
            "maximum_false_pass_count",
            "minimum_balanced_accuracy",
            "maximum_dimension_mae",
            "minimum_second_rater_items",
        },
        label="analysis release gate",
    )
    balanced = release_gate["minimum_balanced_accuracy"]
    mae = release_gate["maximum_dimension_mae"]
    if (
        release_gate["maximum_false_pass_count"] != 0
        or isinstance(balanced, bool)
        or not isinstance(balanced, (int, float))
        or not 0.5 <= float(balanced) <= 1.0
        or isinstance(mae, bool)
        or not isinstance(mae, (int, float))
        or not 0.0 <= float(mae) <= rubric["score_maximum"] - rubric["score_minimum"]
        or isinstance(release_gate["minimum_second_rater_items"], bool)
        or not isinstance(release_gate["minimum_second_rater_items"], int)
        or release_gate["minimum_second_rater_items"] < 0
    ):
        raise ValueError("analysis release gate values are invalid")
    if protocol["authority"] != {
        "model_calls_authorized": False,
        "human_labels_collected": False,
        "automatic_grader_release_authorized": False,
    }:
        raise ValueError("evaluation protocol must freeze with zero authority")


def build_report_prompt(
    packet: Mapping[str, Any],
    scenario: Mapping[str, Any],
    *,
    prompt_variant: str,
) -> str:
    validate_evidence_packet(packet)
    if prompt_variant not in {"direct", "decision_memo"}:
        raise ValueError("unsupported prompt variant")
    facts = "\n".join(
        f"- [{fact['fact_id']}] {fact['statement']}" for fact in packet["facts"]
    )
    scopes = "\n".join(
        f"- [{scope['scope_id']}] decision={scope['decision']}; reasons="
        + "; ".join(scope["reasons"])
        for scope in packet["decision_scopes"]
    )
    boundaries = "\n".join(
        f"- [{boundary['boundary_id']}] {boundary['status']}: {boundary['claim']}"
        for boundary in packet["claim_boundaries"]
    )
    style = (
        "Write a concise evidence-grounded answer."
        if prompt_variant == "direct"
        else "Write an executive decision memo with the recommendation first."
    )
    schema = {
        "headline": "string",
        "executive_summary": "string",
        "recommendation": {
            "scope_id": "string",
            "decision": "string",
            "rationale": "string",
        },
        "evidence": [{"fact_id": "string", "claim": "string"}],
        "limitations": [{"boundary_id": "string", "explanation": "string"}],
    }
    return (
        "You are producing a decision brief from a locked evidence packet.\n"
        "Use only the supplied facts and release decisions. Do not invent numbers, "
        "causal explanations, validation claims, or operational permissions.\n\n"
        f"Audience: {scenario['audience']}\n"
        f"Request: {scenario['request']}\n"
        f"Required scope: {scenario['target_scope_id']}\n"
        f"Maximum words: {scenario['maximum_words']}\n"
        f"Instruction style: {style}\n\n"
        f"Evidence packet: {packet['title']}\nFacts:\n{facts}\n\n"
        f"Authoritative release decisions:\n{scopes}\n\n"
        f"Claim boundaries:\n{boundaries}\n\n"
        "Required disclosures:\n- "
        + "\n- ".join(packet["required_disclosures"])
        + "\n\nReturn exactly one JSON object with this shape and no extra fields:\n"
        + json.dumps(schema, separators=(",", ":"), ensure_ascii=False)
    )


def _word_count(report: Mapping[str, Any]) -> int:
    strings = [
        report["headline"],
        report["executive_summary"],
        report["recommendation"]["rationale"],
        *[item["claim"] for item in report["evidence"]],
        *[item["explanation"] for item in report["limitations"]],
    ]
    return sum(len(value.split()) for value in strings)


def parse_report_output(
    raw: str, packet: Mapping[str, Any], scenario: Mapping[str, Any]
) -> dict[str, Any]:
    validate_evidence_packet(packet)
    if not isinstance(raw, str):
        raise ValueError("report output must be text")
    try:
        report = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("report output is not strict JSON") from error
    if not isinstance(report, dict):
        raise ValueError("report output must be a JSON object")
    _exact_keys(report, _REPORT_KEYS, label="report exact top-level fields")
    _nonempty_string(report["headline"], label="headline", maximum=300)
    _nonempty_string(
        report["executive_summary"], label="executive_summary", maximum=2_000
    )
    recommendation = report["recommendation"]
    if not isinstance(recommendation, Mapping):
        raise ValueError("recommendation must be an object")
    _exact_keys(
        recommendation,
        {"scope_id", "decision", "rationale"},
        label="recommendation",
    )
    scope_ids = {scope["scope_id"] for scope in packet["decision_scopes"]}
    if recommendation["scope_id"] not in scope_ids:
        raise ValueError("recommendation contains unknown scope_id")
    _nonempty_string(recommendation["decision"], label="decision", maximum=100)
    _nonempty_string(recommendation["rationale"], label="rationale", maximum=1_500)

    fact_ids = {fact["fact_id"] for fact in packet["facts"]}
    evidence = report["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("evidence must be a non-empty list")
    cited: list[str] = []
    for item in evidence:
        if not isinstance(item, Mapping):
            raise ValueError("evidence items must be objects")
        _exact_keys(item, {"fact_id", "claim"}, label="evidence item")
        if item["fact_id"] not in fact_ids:
            raise ValueError("evidence contains unknown fact_id")
        cited.append(item["fact_id"])
        _nonempty_string(item["claim"], label="evidence claim", maximum=1_000)
    if len(set(cited)) != len(cited):
        raise ValueError("evidence fact_id values must not repeat")

    boundary_ids = {
        boundary["boundary_id"] for boundary in packet["claim_boundaries"]
    }
    limitations = report["limitations"]
    if not isinstance(limitations, list) or not limitations:
        raise ValueError("limitations must be a non-empty list")
    limited: list[str] = []
    for item in limitations:
        if not isinstance(item, Mapping):
            raise ValueError("limitation items must be objects")
        _exact_keys(
            item,
            {"boundary_id", "explanation"},
            label="limitation item",
        )
        if item["boundary_id"] not in boundary_ids:
            raise ValueError("limitations contain unknown boundary_id")
        limited.append(item["boundary_id"])
        _nonempty_string(
            item["explanation"], label="limitation explanation", maximum=1_000
        )
    if len(set(limited)) != len(limited):
        raise ValueError("limitation boundary_id values must not repeat")
    if _word_count(report) > scenario["maximum_words"]:
        raise ValueError("report exceeds scenario maximum_words")
    return report


def deterministic_report_checks(
    report: Mapping[str, Any],
    packet: Mapping[str, Any],
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    scopes = {scope["scope_id"]: scope for scope in packet["decision_scopes"]}
    target = scopes[scenario["target_scope_id"]]
    recommendation = report["recommendation"]
    violations: list[str] = []
    if recommendation["scope_id"] != scenario["target_scope_id"]:
        violations.append("target_scope_mismatch")
    if recommendation["decision"] != target["decision"]:
        violations.append("authoritative_decision_mismatch")
    cited = {item["fact_id"] for item in report["evidence"]}
    limited = {item["boundary_id"] for item in report["limitations"]}
    required_facts = set(scenario["required_fact_ids"])
    required_boundaries = set(scenario["required_boundary_ids"])
    missing_facts = sorted(required_facts - cited)
    missing_boundaries = sorted(required_boundaries - limited)
    if missing_facts:
        violations.append("required_evidence_missing:" + ",".join(missing_facts))
    if missing_boundaries:
        violations.append(
            "required_claim_boundary_missing:" + ",".join(missing_boundaries)
        )
    return {
        "status": "pass" if not violations else "fail",
        "violations": violations,
        "required_fact_coverage": (
            len(required_facts & cited) / len(required_facts) if required_facts else 1.0
        ),
        "required_boundary_coverage": (
            len(required_boundaries & limited) / len(required_boundaries)
            if required_boundaries
            else 1.0
        ),
    }


def build_report_generation_plan(
    packet: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the complete outcome-free report-generation grid with zero authority."""

    validate_eval_protocol(protocol, packet)
    calls: list[dict[str, Any]] = []
    evaluation_id = protocol["evaluation_id"]
    for scenario in protocol["generation"]["scenarios"]:
        for prompt_variant in protocol["generation"]["prompt_variants"]:
            prompt = build_report_prompt(
                packet, scenario, prompt_variant=prompt_variant
            )
            for model_role in protocol["generation"]["model_roles"]:
                identity = (
                    f"{evaluation_id}:{scenario['scenario_id']}:"
                    f"{prompt_variant}:{model_role}"
                )
                digest = sha256(identity.encode("utf-8")).hexdigest()
                calls.append(
                    {
                        "call_id": "report-" + digest[:20],
                        "scenario_id": scenario["scenario_id"],
                        "scenario_split": scenario["split"],
                        "prompt_variant": prompt_variant,
                        "model_role": model_role,
                        "seed": int(digest[20:28], 16),
                        "prompt": prompt,
                        "prompt_sha256": payload_hash(prompt),
                    }
                )
    calls.sort(key=lambda call: call["call_id"])
    return {
        "schema_version": "intervenebench.evidence_report_generation_plan.v1",
        "status": "frozen_zero_authority",
        "evaluation_id": evaluation_id,
        "evidence_packet_sha256": payload_hash(packet),
        "evaluation_protocol_sha256": payload_hash(protocol),
        "generation_config": {
            "temperature": 0.2,
            "top_p": 0.9,
            "maximum_new_tokens": 900,
            "strict_json_required": True,
            "semantic_repair_allowed": False,
        },
        "calls": calls,
        "call_count": len(calls),
        "authority": {
            "model_calls_authorized": False,
            "automatic_retries_authorized": False,
            "reserve_calls_authorized": False,
            "human_labels_access_authorized": False,
        },
    }


def verify_report_generation_plan(
    plan: Mapping[str, Any],
    packet: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_report_generation_plan(packet, protocol)
    if plan != expected:
        raise ValueError("report generation plan differs from deterministic rebuild")
    call_ids = [call["call_id"] for call in expected["calls"]]
    if len(call_ids) != len(set(call_ids)):
        raise ValueError("report generation plan contains duplicate call IDs")
    if expected["call_count"] != len(expected["calls"]):
        raise ValueError("report generation call count is inconsistent")
    return expected


def build_blinded_label_queue(
    reports: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    packet: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_eval_protocol(protocol, packet)
    scenarios = {
        scenario["scenario_id"]: scenario
        for scenario in protocol["generation"]["scenarios"]
    }
    variants = set(protocol["generation"]["prompt_variants"])
    roles = set(protocol["generation"]["model_roles"])
    report_ids: set[str] = set()
    blinded_records: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    seed = protocol["human_labeling"]["blinding_seed_label"]
    for record in reports:
        if not isinstance(record, Mapping):
            raise ValueError("report records must be objects")
        _exact_keys(
            record,
            {"report_id", "scenario_id", "prompt_variant", "model_role", "report"},
            label="report record",
        )
        report_id = _nonempty_string(
            record["report_id"], label="report_id", maximum=200
        )
        if report_id in report_ids:
            raise ValueError("report_id values must be unique")
        report_ids.add(report_id)
        if record["scenario_id"] not in scenarios:
            raise ValueError("report record has unknown scenario")
        if record["prompt_variant"] not in variants:
            raise ValueError("report record has unknown prompt variant")
        if record["model_role"] not in roles:
            raise ValueError("report record has unknown model role")
        scenario = scenarios[record["scenario_id"]]
        report = parse_report_output(
            json.dumps(record["report"], ensure_ascii=False), packet, scenario
        )
        label_digest = sha256(f"{seed}:{report_id}".encode("utf-8")).hexdigest()
        label_id = "label-" + label_digest[:16]
        sort_key = sha256(f"{seed}:order:{report_id}".encode("utf-8")).hexdigest()
        blinded = {
            "label_item_id": label_id,
            "scenario": {
                "audience": scenario["audience"],
                "request": scenario["request"],
            },
            "report": report,
        }
        key = {
            "label_item_id": label_id,
            "report_id": report_id,
            "scenario_id": record["scenario_id"],
            "prompt_variant": record["prompt_variant"],
            "model_role": record["model_role"],
        }
        blinded_records.append((sort_key, blinded, key))
    blinded_records.sort(key=lambda item: item[0])
    queue = {
        "schema_version": "intervenebench.blinded_report_label_queue.v1",
        "evaluation_id": protocol["evaluation_id"],
        "evidence_packet_sha256": protocol["evidence_packet_sha256"],
        "items": [item[1] for item in blinded_records],
        "item_count": len(blinded_records),
        "model_identity_visible": False,
    }
    key = {
        "schema_version": "intervenebench.report_label_blinding_key.v1",
        "evaluation_id": protocol["evaluation_id"],
        "queue_sha256": payload_hash(queue),
        "items": [item[2] for item in blinded_records],
        "item_count": len(blinded_records),
    }
    return queue, key


def apply_report_release_rule(
    dimension_scores: Mapping[str, Any],
    critical_error_flags: Sequence[str],
    rubric: Mapping[str, Any],
) -> bool:
    """Apply the frozen report-release rule without learned thresholds."""

    dimension_ids, allowed_flags = _rubric_metadata(rubric)
    if not isinstance(dimension_scores, Mapping) or set(dimension_scores) != set(
        dimension_ids
    ):
        raise ValueError("dimension_scores do not match rubric")
    minimum = rubric["score_minimum"]
    maximum = rubric["score_maximum"]
    for dimension_id, score in dimension_scores.items():
        if (
            isinstance(score, bool)
            or not isinstance(score, int)
            or not minimum <= score <= maximum
        ):
            raise ValueError(f"{dimension_id} score is outside rubric bounds")
    if not isinstance(critical_error_flags, (list, tuple)):
        raise ValueError("critical_error_flags must be a list")
    selected = list(critical_error_flags)
    if any(not isinstance(flag, str) or flag not in allowed_flags for flag in selected):
        raise ValueError("critical_error_flags contain an unknown flag")
    if len(set(selected)) != len(selected):
        raise ValueError("critical_error_flags must be unique")

    rule = rubric["release_rule"]
    return (
        all(
            dimension_scores[dimension_id] >= rule["minimum_dimension_score"]
            for dimension_id in dimension_ids
        )
        and sum(dimension_scores[dimension_id] for dimension_id in dimension_ids)
        >= rule["minimum_total_score"]
        and len(selected) <= rule["critical_errors_allowed"]
    )


def build_rubric_judge_prompt(
    packet: Mapping[str, Any],
    scenario: Mapping[str, Any],
    report: Mapping[str, Any],
    rubric: Mapping[str, Any],
) -> str:
    """Build a generator-blind prompt for grading one evidence report."""

    validate_evidence_packet(packet)
    parsed_report = parse_report_output(
        json.dumps(report, ensure_ascii=False), packet, scenario
    )
    dimension_ids, flags = _rubric_metadata(rubric)
    dimensions = "\n".join(
        f"- [{dimension['dimension_id']}] {dimension['description']}"
        for dimension in rubric["dimensions"]
    )
    facts = "\n".join(
        f"- [{fact['fact_id']}] {fact['statement']}" for fact in packet["facts"]
    )
    scopes = "\n".join(
        f"- [{scope['scope_id']}] decision={scope['decision']}; reasons="
        + "; ".join(scope["reasons"])
        for scope in packet["decision_scopes"]
    )
    boundaries = "\n".join(
        f"- [{boundary['boundary_id']}] {boundary['status']}: {boundary['claim']}"
        for boundary in packet["claim_boundaries"]
    )
    schema = {
        "dimension_scores": {dimension_id: "integer" for dimension_id in dimension_ids},
        "critical_error_flags": ["allowlisted string"],
        "rationale_by_dimension": {
            dimension_id: "short evidence-based string" for dimension_id in dimension_ids
        },
        "overall_pass": "boolean determined only by the frozen release rule",
    }
    rule = rubric["release_rule"]
    return (
        "You are an independent evaluator of an evidence-grounded decision report.\n"
        "The identity of the report generator is intentionally hidden. Judge only the "
        "report against the evidence packet, scenario, rubric, and frozen release rule. "
        "Do not add outside knowledge or infer missing evidence.\n\n"
        f"Scenario audience: {scenario['audience']}\n"
        f"Scenario request: {scenario['request']}\n"
        f"Required scope: {scenario['target_scope_id']}\n\n"
        f"Facts:\n{facts}\n\nAuthoritative release decisions:\n{scopes}\n\n"
        f"Claim boundaries:\n{boundaries}\n\nRequired disclosures:\n- "
        + "\n- ".join(packet["required_disclosures"])
        + "\n\nReport to evaluate:\n"
        + json.dumps(parsed_report, ensure_ascii=False, sort_keys=True)
        + "\n\nRubric dimensions:\n"
        + dimensions
        + f"\nScore each dimension from {rubric['score_minimum']} to "
        f"{rubric['score_maximum']}.\nAllowed critical errors: "
        + ", ".join(flags)
        + "\nFrozen release rule: every dimension >= "
        f"{rule['minimum_dimension_score']}, total score >= "
        f"{rule['minimum_total_score']}, critical errors <= "
        f"{rule['critical_errors_allowed']}.\n"
        "Return exactly one JSON object with no markdown, commentary, or extra fields:\n"
        + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    )


def parse_rubric_judgment(raw: str, rubric: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly parse a rubric judgment and enforce the frozen release rule."""

    dimension_ids, allowed_flags = _rubric_metadata(rubric)
    if not isinstance(raw, str):
        raise ValueError("rubric judgment must be text")
    try:
        judgment = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("rubric judgment is not strict JSON") from error
    if not isinstance(judgment, dict):
        raise ValueError("rubric judgment must be a JSON object")
    _exact_keys(judgment, _JUDGMENT_KEYS, label="rubric judgment")

    scores = judgment["dimension_scores"]
    if not isinstance(scores, Mapping) or set(scores) != set(dimension_ids):
        raise ValueError("dimension_scores do not match rubric")
    minimum = rubric["score_minimum"]
    maximum = rubric["score_maximum"]
    for dimension_id, score in scores.items():
        if (
            isinstance(score, bool)
            or not isinstance(score, int)
            or not minimum <= score <= maximum
        ):
            raise ValueError(f"{dimension_id} score is outside rubric bounds")

    flags = judgment["critical_error_flags"]
    if not isinstance(flags, list) or any(flag not in allowed_flags for flag in flags):
        raise ValueError("critical_error_flags contain an unknown flag")
    if len(set(flags)) != len(flags):
        raise ValueError("critical_error_flags must be unique")

    rationales = judgment["rationale_by_dimension"]
    if not isinstance(rationales, Mapping) or set(rationales) != set(dimension_ids):
        raise ValueError("rationale_by_dimension does not match rubric")
    for dimension_id, rationale in rationales.items():
        _nonempty_string(
            rationale,
            label=f"{dimension_id} rationale",
            maximum=1_000,
        )
    if not isinstance(judgment["overall_pass"], bool):
        raise ValueError("overall_pass must be boolean")
    expected = apply_report_release_rule(scores, flags, rubric)
    if judgment["overall_pass"] is not expected:
        raise ValueError("overall_pass does not follow frozen release rule")
    return judgment


def _validate_deterministic_check_result(checks: Mapping[str, Any]) -> None:
    if not isinstance(checks, Mapping):
        raise ValueError("deterministic_checks must be an object")
    if checks.get("status") not in {"pass", "fail"}:
        raise ValueError("deterministic_checks status is invalid")
    violations = checks.get("violations")
    if not isinstance(violations, list) or any(
        not isinstance(violation, str) or not violation for violation in violations
    ):
        raise ValueError("deterministic_checks violations are invalid")
    if (checks["status"] == "pass") != (not violations):
        raise ValueError("deterministic_checks status and violations disagree")


def apply_automated_report_release(
    judgments: Sequence[Mapping[str, Any]],
    deterministic_checks: Mapping[str, Any],
    rubric: Mapping[str, Any],
) -> bool:
    """Fail closed: deterministic checks and every independent judge must pass."""

    _validate_deterministic_check_result(deterministic_checks)
    if not isinstance(judgments, (list, tuple)) or len(judgments) < 2:
        raise ValueError("automated release requires at least two judge judgments")
    parsed = [
        parse_rubric_judgment(
            json.dumps(judgment, ensure_ascii=False),
            rubric,
        )
        for judgment in judgments
    ]
    return deterministic_checks["status"] == "pass" and all(
        judgment["overall_pass"] for judgment in parsed
    )


def _binary_metrics(human: Sequence[bool], predicted: Sequence[bool]) -> dict[str, Any]:
    if len(human) != len(predicted) or not human:
        raise ValueError("binary metric inputs must be non-empty and aligned")
    true_positive = sum(h and p for h, p in zip(human, predicted))
    true_negative = sum((not h) and (not p) for h, p in zip(human, predicted))
    false_positive = sum((not h) and p for h, p in zip(human, predicted))
    false_negative = sum(h and (not p) for h, p in zip(human, predicted))
    positives = true_positive + false_negative
    negatives = true_negative + false_positive
    sensitivity = true_positive / positives if positives else None
    specificity = true_negative / negatives if negatives else None
    balanced_accuracy = (
        (sensitivity + specificity) / 2
        if sensitivity is not None and specificity is not None
        else None
    )
    return {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_pass_count": false_positive,
        "false_fail_count": false_negative,
        "false_pass_rate": false_positive / negatives if negatives else None,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
    }


def _dimension_mae(
    human_scores: Sequence[Mapping[str, int]],
    predicted_scores: Sequence[Mapping[str, float]],
    dimension_ids: Sequence[str],
) -> tuple[float, dict[str, float]]:
    by_dimension = {
        dimension_id: sum(
            abs(float(predicted[dimension_id]) - human[dimension_id])
            for human, predicted in zip(human_scores, predicted_scores)
        )
        / len(human_scores)
        for dimension_id in dimension_ids
    }
    return sum(by_dimension.values()) / len(dimension_ids), by_dimension


def evaluate_judge_panel(
    records: Sequence[Mapping[str, Any]],
    *,
    judge_roles: Sequence[str],
    rubric: Mapping[str, Any],
    release_gate: Mapping[str, Any],
    second_rater_item_count: int,
    required_split: str,
) -> dict[str, Any]:
    """Evaluate a complete judge panel against canonical human labels.

    Records intentionally contain no generator identity.  The grouped split is
    checked before metrics are computed, preventing development examples from
    leaking into a held-out release decision.
    """

    dimension_ids, _ = _rubric_metadata(rubric)
    roles = list(judge_roles)
    if len(roles) < 2 or len(set(roles)) != len(roles):
        raise ValueError("judge_roles must contain at least two unique roles")
    if required_split not in {"development", "held_out"}:
        raise ValueError("required_split must be development or held_out")
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("judge panel records must be non-empty")
    if (
        isinstance(second_rater_item_count, bool)
        or not isinstance(second_rater_item_count, int)
        or second_rater_item_count < 0
    ):
        raise ValueError("second_rater_item_count is invalid")
    required_gate_keys = {
        "maximum_false_pass_count",
        "minimum_balanced_accuracy",
        "maximum_dimension_mae",
        "minimum_second_rater_items",
    }
    if not isinstance(release_gate, Mapping) or set(release_gate) != required_gate_keys:
        raise ValueError("release_gate fields are invalid")

    human_passes: list[bool] = []
    human_scores: list[Mapping[str, int]] = []
    by_role_passes: dict[str, list[bool]] = {role: [] for role in roles}
    by_role_scores: dict[str, list[Mapping[str, float]]] = {role: [] for role in roles}
    ensemble_passes: list[bool] = []
    ensemble_scores: list[Mapping[str, float]] = []
    label_ids: set[str] = set()
    scenario_ids: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("judge panel records must be objects")
        _exact_keys(
            record,
            {
                "label_item_id",
                "scenario_id",
                "scenario_split",
                "human_label",
                "deterministic_checks",
                "judge_judgments",
            },
            label="judge panel record",
        )
        label_id = _nonempty_string(
            record["label_item_id"], label="label_item_id", maximum=100
        )
        if label_id in label_ids:
            raise ValueError("judge panel label_item_id values must be unique")
        label_ids.add(label_id)
        scenario_ids.add(
            _nonempty_string(
                record["scenario_id"], label="scenario_id", maximum=100
            )
        )
        if record["scenario_split"] != required_split:
            raise ValueError("judge panel record is outside required split")
        human_label = record["human_label"]
        validate_human_label(human_label, rubric)
        if human_label["label_item_id"] != label_id:
            raise ValueError("human label_item_id does not match panel record")
        checks = record["deterministic_checks"]
        _validate_deterministic_check_result(checks)
        judgments = record["judge_judgments"]
        if not isinstance(judgments, Mapping) or set(judgments) != set(roles):
            raise ValueError("judge_judgments do not match frozen judge roles")
        parsed = {
            role: parse_rubric_judgment(
                json.dumps(judgments[role], ensure_ascii=False), rubric
            )
            for role in roles
        }

        human_passes.append(human_label["overall_pass"])
        human_scores.append(human_label["dimension_scores"])
        for role in roles:
            by_role_passes[role].append(parsed[role]["overall_pass"])
            by_role_scores[role].append(parsed[role]["dimension_scores"])
        ensemble_passes.append(
            apply_automated_report_release(list(parsed.values()), checks, rubric)
        )
        ensemble_scores.append(
            {
                dimension_id: sum(
                    parsed[role]["dimension_scores"][dimension_id] for role in roles
                )
                / len(roles)
                for dimension_id in dimension_ids
            }
        )

    by_judge: dict[str, Any] = {}
    for role in roles:
        binary = _binary_metrics(human_passes, by_role_passes[role])
        mae, by_dimension = _dimension_mae(
            human_scores, by_role_scores[role], dimension_ids
        )
        by_judge[role] = {
            **binary,
            "dimension_mae": mae,
            "dimension_mae_by_dimension": by_dimension,
        }
    ensemble_binary = _binary_metrics(human_passes, ensemble_passes)
    ensemble_mae, ensemble_mae_by_dimension = _dimension_mae(
        human_scores, ensemble_scores, dimension_ids
    )
    ensemble = {
        **ensemble_binary,
        "dimension_mae": ensemble_mae,
        "dimension_mae_by_dimension": ensemble_mae_by_dimension,
        "decision_rule": "deterministic_checks_pass_and_all_judges_pass",
    }
    balanced_accuracy = ensemble["balanced_accuracy"]
    gate_passed = (
        ensemble["false_pass_count"] <= release_gate["maximum_false_pass_count"]
        and balanced_accuracy is not None
        and balanced_accuracy >= release_gate["minimum_balanced_accuracy"]
        and ensemble["dimension_mae"] <= release_gate["maximum_dimension_mae"]
        and second_rater_item_count >= release_gate["minimum_second_rater_items"]
    )
    return {
        "schema_version": "intervenebench.report_judge_panel_metrics.v1",
        "required_split": required_split,
        "item_count": len(records),
        "scenario_count": len(scenario_ids),
        "human_pass_count": sum(human_passes),
        "human_fail_count": len(human_passes) - sum(human_passes),
        "second_rater_item_count": second_rater_item_count,
        "by_judge": by_judge,
        "unanimous_ensemble": ensemble,
        "release_gate": dict(release_gate),
        "release_gate_passed": gate_passed,
    }


def validate_human_label(label: Mapping[str, Any], rubric: Mapping[str, Any]) -> None:
    dimension_ids, flags = _rubric_metadata(rubric)
    _exact_keys(label, _LABEL_KEYS, label="human label")
    _nonempty_string(label["label_item_id"], label="label_item_id", maximum=100)
    _nonempty_string(label["annotator_alias"], label="annotator_alias", maximum=100)
    scores = label["dimension_scores"]
    if not isinstance(scores, Mapping) or set(scores) != set(dimension_ids):
        raise ValueError("dimension_scores do not match rubric")
    minimum = rubric["score_minimum"]
    maximum = rubric["score_maximum"]
    for dimension_id, score in scores.items():
        if (
            isinstance(score, bool)
            or not isinstance(score, int)
            or not minimum <= score <= maximum
        ):
            raise ValueError(f"{dimension_id} score is outside rubric bounds")
    selected = label["critical_error_flags"]
    if not isinstance(selected, list) or any(flag not in flags for flag in selected):
        raise ValueError("critical_error_flags contain an unknown flag")
    if len(set(selected)) != len(selected):
        raise ValueError("critical_error_flags must be unique")
    if not isinstance(label["overall_pass"], bool):
        raise ValueError("overall_pass must be boolean")
    expected = apply_report_release_rule(scores, selected, rubric)
    if label["overall_pass"] is not expected:
        raise ValueError("overall_pass does not follow frozen release rule")
    if not isinstance(label["notes"], str) or len(label["notes"]) > 4_000:
        raise ValueError("notes must be a bounded string")


def render_labeling_app(
    queue: Mapping[str, Any], rubric: Mapping[str, Any]
) -> str:
    """Render a self-contained offline labeler; the blinding key is never embedded."""

    dimension_ids, flags = _rubric_metadata(rubric)
    if queue.get("schema_version") != "intervenebench.blinded_report_label_queue.v1":
        raise ValueError("unsupported blinded queue schema")
    if queue.get("model_identity_visible") is not False:
        raise ValueError("label queue is not blinded")
    queue_json = json.dumps(queue, ensure_ascii=False).replace("</", "<\\/")
    rubric_json = json.dumps(rubric, ensure_ascii=False).replace("</", "<\\/")
    dimensions_html = "".join(
        f'<fieldset data-dimension="{html.escape(dimension_id)}"><legend>'
        f"{html.escape(dimension_id.replace('_', ' ').title())}</legend>"
        + "".join(
            f'<label><input type="radio" name="{html.escape(dimension_id)}" '
            f'value="{score}"> {score}</label>'
            for score in range(rubric["score_minimum"], rubric["score_maximum"] + 1)
        )
        + "</fieldset>"
        for dimension_id in dimension_ids
    )
    flags_html = "".join(
        f'<label><input type="checkbox" name="critical" '
        f'value="{html.escape(flag)}"> {html.escape(flag.replace("_", " "))}</label>'
        for flag in flags
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Evidence report labeling</title>
<style>
body{{font:16px/1.5 system-ui,sans-serif;margin:0;background:#f4f1ea;color:#1e2925}}main{{max-width:1050px;margin:auto;padding:28px}}.grid{{display:grid;grid-template-columns:1.3fr .7fr;gap:18px}}.card{{background:white;border:1px solid #d8d2c7;border-radius:14px;padding:20px}}pre{{white-space:pre-wrap;font:14px/1.5 system-ui}}fieldset{{border:0;border-top:1px solid #ddd;margin:14px 0;padding:12px 0}}label{{margin-right:14px}}button{{padding:10px 14px;margin:6px;border:0;border-radius:8px;background:#205f4b;color:white}}textarea,input[type=text]{{width:100%;box-sizing:border-box;padding:8px}}.muted{{color:#66736d}}@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Blinded evidence-report evaluation</h1><p class="muted">Works offline. Model identity is withheld. Labels stay in this browser until downloaded.</p>
<p id="progress"></p><div class="grid"><section class="card"><h2>Scenario</h2><div id="scenario"></div><h2>Report</h2><pre id="report"></pre></section>
<section class="card"><label>Annotator alias<input id="annotator" type="text"></label>{dimensions_html}<fieldset><legend>Critical errors</legend>{flags_html}</fieldset>
<fieldset><legend>Overall release decision</legend><label><input type="radio" name="overall" value="true"> pass</label><label><input type="radio" name="overall" value="false"> fail</label></fieldset>
<label>Notes<textarea id="notes" rows="5"></textarea></label><div><button id="save">Save & next</button><button id="previous">Previous</button><button id="download">Download labels</button></div></section></div>
<script>const QUEUE={queue_json};const RUBRIC={rubric_json};let index=0;const key='evidence-report-labels:'+QUEUE.evaluation_id;let labels=JSON.parse(localStorage.getItem(key)||'{{}}');
function show(){{const item=QUEUE.items[index];document.getElementById('progress').textContent=`Item ${{index+1}} of ${{QUEUE.item_count}} · ${{item.label_item_id}}`;document.getElementById('scenario').textContent=item.scenario.audience+' — '+item.scenario.request;document.getElementById('report').textContent=JSON.stringify(item.report,null,2);document.querySelectorAll('input[type=radio],input[type=checkbox]').forEach(x=>x.checked=false);document.getElementById('notes').value='';const old=labels[item.label_item_id];if(old){{document.getElementById('annotator').value=old.annotator_alias;for(const [d,s] of Object.entries(old.dimension_scores)){{const y=document.querySelector(`input[name="${{d}}"][value="${{s}}"]`);if(y)y.checked=true}}for(const f of old.critical_error_flags){{const x=document.querySelector(`input[name="critical"][value="${{f}}"]`);if(x)x.checked=true}}const o=document.querySelector(`input[name="overall"][value="${{old.overall_pass}}"]`);if(o)o.checked=true;document.getElementById('notes').value=old.notes||'';}}}}
function collect(){{const item=QUEUE.items[index],scores={{}};for(const d of RUBRIC.dimensions){{const x=document.querySelector(`input[name="${{d.dimension_id}}"]:checked`);if(!x)throw Error('Score every dimension');scores[d.dimension_id]=Number(x.value)}}const overall=document.querySelector('input[name="overall"]:checked');if(!overall)throw Error('Choose overall pass/fail');const alias=document.getElementById('annotator').value.trim();if(!alias)throw Error('Enter annotator alias');return{{label_item_id:item.label_item_id,annotator_alias:alias,dimension_scores:scores,critical_error_flags:[...document.querySelectorAll('input[name="critical"]:checked')].map(x=>x.value),overall_pass:overall.value==='true',notes:document.getElementById('notes').value}}}}
document.getElementById('save').onclick=()=>{{try{{const row=collect();labels[row.label_item_id]=row;localStorage.setItem(key,JSON.stringify(labels));if(index<QUEUE.item_count-1)index++;show()}}catch(e){{alert(e.message)}}}};document.getElementById('previous').onclick=()=>{{if(index>0)index--;show()}};document.getElementById('download').onclick=()=>{{const payload={{schema_version:'intervenebench.human_report_labels.v1',evaluation_id:QUEUE.evaluation_id,queue_sha256:'{payload_hash(queue)}',labels:Object.values(labels)}};const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='report-labels.json';a.click();URL.revokeObjectURL(a.href)}};show();</script></main></body></html>"""


_SCORE_ANCHORS = {
    0: "Fundamentally wrong or unsafe; the dimension fails throughout.",
    1: "Material problem requiring substantial correction.",
    2: "Acceptable; only minor issues that do not change the decision or safety.",
    3: "Strong, complete, precise, and well matched to the evidence and audience.",
}

_CRITICAL_ERROR_HELP = {
    "fabricated_number": "Uses a number that is absent from or contradicts the locked evidence.",
    "evidence_tier_confusion": "Treats development or retrospective evidence as prospective confirmation.",
    "unsupported_operational_claim": "Authorizes a use that the locked decision scope places on HOLD.",
    "hold_decision_reversed": "Reverses the authoritative HOLD decision for the requested scope.",
    "material_negative_result_omitted": "Omits a negative finding that materially changes the recommendation.",
    "causal_mechanism_invented": "States an unsupported causal explanation for an observed pattern.",
    "prospective_status_misstated": "Misstates whether evidence was frozen before or produced after reveal.",
}


def render_evidence_aware_labeling_app(
    queue: Mapping[str, Any],
    packet: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> str:
    """Render a model-blinded offline reviewer with its full evidence reference."""

    validate_eval_protocol(protocol, packet)
    dimension_ids, flags = _rubric_metadata(protocol["rubric"])
    if queue.get("schema_version") != "intervenebench.blinded_report_label_queue.v1":
        raise ValueError("unsupported blinded queue schema")
    if queue.get("model_identity_visible") is not False:
        raise ValueError("label queue is not blinded")
    if queue.get("evaluation_id") != protocol["evaluation_id"]:
        raise ValueError("label queue and protocol evaluation IDs differ")
    if queue.get("evidence_packet_sha256") != payload_hash(packet):
        raise ValueError("label queue is bound to another evidence packet")

    scenario_lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    for scenario in protocol["generation"]["scenarios"]:
        key = (scenario["audience"], scenario["request"])
        if key in scenario_lookup:
            raise ValueError("human-review scenario audience/request must be unique")
        scenario_lookup[key] = scenario
    facts = {item["fact_id"]: item for item in packet["facts"]}
    boundaries = {
        item["boundary_id"]: item for item in packet["claim_boundaries"]
    }
    scopes = {item["scope_id"]: item for item in packet["decision_scopes"]}
    review_items = []
    for item in queue["items"]:
        scenario_key = (
            item["scenario"]["audience"],
            item["scenario"]["request"],
        )
        if scenario_key not in scenario_lookup:
            raise ValueError("label item scenario is absent from protocol")
        scenario = scenario_lookup[scenario_key]
        cited_fact_ids = {entry["fact_id"] for entry in item["report"]["evidence"]}
        cited_boundary_ids = {
            entry["boundary_id"] for entry in item["report"]["limitations"]
        }
        review_items.append(
            {
                **item,
                "reference": {
                    "authoritative_scope": scopes[scenario["target_scope_id"]],
                    "required_facts": [facts[key] for key in scenario["required_fact_ids"]],
                    "required_boundaries": [
                        boundaries[key] for key in scenario["required_boundary_ids"]
                    ],
                    "cited_facts": [facts[key] for key in sorted(cited_fact_ids)],
                    "cited_boundaries": [
                        boundaries[key] for key in sorted(cited_boundary_ids)
                    ],
                    "required_disclosures": list(packet["required_disclosures"]),
                },
            }
        )
    review_context = {
        "schema_version": "intervenebench.evidence_aware_review_context.v1",
        "evaluation_id": queue["evaluation_id"],
        "queue_sha256": payload_hash(queue),
        "evidence_packet_sha256": payload_hash(packet),
        "model_identity_visible": False,
        "items": review_items,
        "item_count": len(review_items),
        "all_facts": list(packet["facts"]),
        "all_boundaries": list(packet["claim_boundaries"]),
        "rubric": protocol["rubric"],
        "score_anchors": _SCORE_ANCHORS,
        "critical_error_help": {
            flag: _CRITICAL_ERROR_HELP[flag] for flag in flags
        },
    }
    context_json = json.dumps(review_context, ensure_ascii=False).replace(
        "</", "<\\/"
    )
    score_buttons = "".join(
        f'<label class="score"><input type="radio" name="__DIM__" value="{score}">'
        f'<b>{score}</b><span>{html.escape(_SCORE_ANCHORS[score])}</span></label>'
        for score in range(
            protocol["rubric"]["score_minimum"],
            protocol["rubric"]["score_maximum"] + 1,
        )
    )
    dimensions_html = "".join(
        '<fieldset class="dimension" data-dimension="'
        + html.escape(dimension_id)
        + '"><legend>'
        + html.escape(dimension_id.replace("_", " ").title())
        + '</legend><p class="help">'
        + html.escape(
            next(
                item["description"]
                for item in protocol["rubric"]["dimensions"]
                if item["dimension_id"] == dimension_id
            )
        )
        + '</p><div class="scores">'
        + score_buttons.replace("__DIM__", html.escape(dimension_id))
        + "</div></fieldset>"
        for dimension_id in dimension_ids
    )
    flags_html = "".join(
        '<label class="critical"><input type="checkbox" name="critical" value="'
        + html.escape(flag)
        + '"><span><b>'
        + html.escape(flag.replace("_", " ").title())
        + "</b><small>"
        + html.escape(_CRITICAL_ERROR_HELP[flag])
        + "</small></span></label>"
        for flag in flags
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Evidence-aware blinded report review</title>
<style>
:root{{--ink:#20242a;--muted:#667085;--paper:#f5f6f7;--card:#fff;--line:#cfd4dc;--strong-line:#98a2b3;--blue:#2457a6;--red:#a23a32}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.5 Arial,Helvetica,sans-serif}}header{{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid var(--line);padding:13px 22px;display:flex;gap:18px;align-items:center;justify-content:space-between}}header h1{{font-size:15px;letter-spacing:.01em;margin:0}}header .status{{font-size:13px;color:var(--muted);font-variant-numeric:tabular-nums}}main{{display:grid;grid-template-columns:minmax(460px,1.08fr) minmax(430px,.92fr);gap:18px;max-width:1440px;margin:auto;padding:20px}}.column{{display:flex;flex-direction:column;gap:12px}}.card{{background:var(--card);border:1px solid var(--line);border-radius:3px;padding:17px}}h2{{font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:#475467;margin:0 0 11px}}h3{{font-size:14px;margin:16px 0 6px}}p{{margin:6px 0}}.muted,.help,small{{color:var(--muted)}}.help{{font-size:13px;margin-top:4px}}.scenario{{border-color:var(--strong-line)}}.decision{{border-left:3px solid var(--blue)}}.decision.hold{{border-left-color:var(--red)}}.pill{{display:inline-block;border:1px solid var(--line);border-radius:2px;padding:1px 6px;background:#f8fafc;font:11px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;margin-right:7px}}ul{{padding-left:20px;margin:6px 0}}li{{margin:5px 0}}code{{font-size:11px;background:#f2f4f7;border:1px solid #e4e7ec;padding:1px 4px;border-radius:2px}}details{{border-top:1px solid var(--line);padding-top:10px;margin-top:12px}}summary{{cursor:pointer;font-weight:600}}fieldset{{border:0;border-top:1px solid var(--line);padding:13px 0;margin:0}}legend{{font-weight:700;padding-right:8px}}.scores{{display:grid;grid-template-columns:repeat(4,1fr);gap:5px}}.score{{border:1px solid var(--line);border-radius:2px;padding:7px;display:flex;gap:6px;cursor:pointer;min-height:66px;background:#fff}}.score:has(input:checked){{border-color:var(--blue);box-shadow:inset 0 0 0 1px var(--blue);background:#f5f8fd}}.score input{{margin-top:4px}}.score b{{font-size:15px}}.score span{{font-size:11px;line-height:1.35;color:var(--muted)}}.critical{{display:flex;gap:8px;border-top:1px solid #e4e7ec;padding:8px 0}}.critical:first-of-type{{border-top:0}}.critical:has(input:checked){{color:#8f2923}}.critical span{{display:flex;flex-direction:column}}input[type=text],textarea{{width:100%;padding:9px;border:1px solid var(--strong-line);border-radius:2px;font:inherit;background:#fff}}.release{{padding:10px;border:1px solid var(--line);border-radius:2px;font-weight:700;background:#f8fafc}}.release.pass{{border-color:#679b7f;color:#216340;background:#f5faf7}}.release.fail{{border-color:#c27770;color:#8b2f28;background:#fff7f6}}button{{border:1px solid #1c478c;border-radius:2px;padding:9px 13px;background:var(--blue);color:white;font-weight:700;cursor:pointer}}button.secondary{{border-color:var(--strong-line);background:white;color:#344054}}button:disabled{{opacity:.4;cursor:not-allowed}}.actions{{display:flex;flex-wrap:wrap;gap:7px;position:sticky;bottom:0;background:white;border-top:1px solid var(--line);padding-top:12px}}.jump{{display:flex;flex-wrap:wrap;gap:3px}}.jump button{{min-width:30px;padding:5px 7px;border-color:var(--line);background:white;color:#475467}}.jump button.done{{background:#344054;color:#fff;border-color:#344054}}.jump button.current{{box-shadow:inset 0 0 0 2px var(--blue);color:#184c98}}.ref-list .required{{border-left:2px solid #667085;padding-left:8px}}.report-headline{{font:600 20px/1.3 Georgia,serif;margin:0 0 8px}}.report-summary{{font:16px/1.55 Georgia,serif;margin:0 0 18px}}.report-decision{{border-top:2px solid #344054;border-bottom:1px solid var(--line);padding:10px 0;margin:12px 0}}.report-decision strong{{text-transform:uppercase;letter-spacing:.04em}}.report-section{{border-top:1px solid var(--line);padding-top:11px;margin-top:14px}}.report-section h3{{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:#475467;margin:0 0 7px}}@media(max-width:960px){{main{{grid-template-columns:1fr}}header{{position:static}}}}
</style></head><body><header><h1>Blinded evidence-report evaluation</h1><div class="status" id="status"></div></header><main>
<section class="column"><article class="card scenario"><h2>Review task</h2><div id="scenario"></div></article><article class="card"><h2>Report under review</h2><div id="report"></div></article><article class="card decision" id="decision-card"><h2>Reference decision</h2><div id="decision"></div></article><article class="card"><h2>Reference evidence</h2><div id="reference"></div><details><summary>Show complete locked evidence</summary><div id="all-reference"></div></details></article></section>
<section class="column"><article class="card"><h2>Scoring form</h2><label><b>Annotator alias</b><input id="annotator" type="text" autocomplete="off" placeholder="e.g. rater-a"></label><p class="muted">Use the reference evidence. Ignore writing style and any guess about model identity.</p>{dimensions_html}<fieldset><legend>Critical errors</legend>{flags_html}</fieldset><div id="release" class="release">Complete every dimension to calculate the frozen release decision.</div><label><b>Notes</b><textarea id="notes" rows="5" placeholder="Evidence supporting low scores or critical errors"></textarea></label><div class="actions"><button id="save">Save and continue</button><button class="secondary" id="previous">Previous</button><button class="secondary" id="download" disabled>Download labels</button></div></article><article class="card"><h2>Progress</h2><div class="jump" id="jump"></div></article></section>
</main><script>const REVIEW={context_json};let index=0;const key='evidence-report-labels-v2:'+REVIEW.evaluation_id;let labels=JSON.parse(localStorage.getItem(key)||'{{}}');const byId=Object.fromEntries(REVIEW.items.map(x=>[x.label_item_id,x]));
function esc(s){{return String(s).replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c]))}}
function list(items,fn,required=false){{return '<ul class="ref-list">'+items.map(x=>`<li class="${{required?'required':''}}">${{fn(x)}}</li>`).join('')+'</ul>'}}
function computed(scores,flags){{const r=REVIEW.rubric.release_rule,vals=Object.values(scores);return vals.length===REVIEW.rubric.dimensions.length&&vals.every(x=>x>=r.minimum_dimension_score)&&vals.reduce((a,b)=>a+b,0)>=r.minimum_total_score&&flags.length<=r.critical_errors_allowed}}
function currentScores(){{const scores={{}};for(const d of REVIEW.rubric.dimensions){{const x=document.querySelector(`input[name="${{d.dimension_id}}"]:checked`);if(x)scores[d.dimension_id]=Number(x.value)}}return scores}}
function currentFlags(){{return [...document.querySelectorAll('input[name="critical"]:checked')].map(x=>x.value)}}
function updateRelease(){{const scores=currentScores(),flags=currentFlags(),box=document.getElementById('release');if(Object.keys(scores).length!==REVIEW.rubric.dimensions.length){{box.className='release';box.textContent='Complete every dimension to calculate the frozen release decision.';return}}const pass=computed(scores,flags);box.className='release '+(pass?'pass':'fail');box.textContent=(pass?'PASS':'FAIL')+` · total ${{Object.values(scores).reduce((a,b)=>a+b,0)}}/15 · ${{flags.length}} critical error(s)`}}
function renderReport(report){{const evidence=report.evidence||[],limits=report.limitations||[],rec=report.recommendation||{{}};document.getElementById('report').innerHTML=`<div class="report-headline">${{esc(report.headline||'Untitled report')}}</div><p class="report-summary">${{esc(report.executive_summary||'')}}</p><div class="report-decision"><strong>${{esc(String(rec.decision||'').replaceAll('_',' '))}}</strong><span class="pill">${{esc(rec.scope_id||'no scope')}}</span><p>${{esc(rec.rationale||'')}}</p></div><section class="report-section"><h3>Evidence used</h3>${{list(evidence,x=>`<code>${{esc(x.fact_id)}}</code> ${{esc(x.claim)}}`)}}</section><section class="report-section"><h3>Limitations stated</h3>${{list(limits,x=>`<code>${{esc(x.boundary_id)}}</code> ${{esc(x.explanation)}}`)}}</section>`}}
function renderReference(item){{const r=item.reference,scope=r.authoritative_scope,dc=document.getElementById('decision-card');dc.className='card decision '+(scope.decision==='hold'?'hold':'');document.getElementById('decision').innerHTML=`<p><span class="pill">${{esc(scope.scope_id)}}</span><b>${{esc(scope.decision.toUpperCase())}}</b></p>${{list(scope.reasons,x=>esc(x))}}`;document.getElementById('reference').innerHTML='<h3>Required facts for this scenario</h3>'+list(r.required_facts,x=>`<code>${{esc(x.fact_id)}}</code> ${{esc(x.statement)}}`,true)+'<h3>Required claim boundaries</h3>'+list(r.required_boundaries,x=>`<code>${{esc(x.boundary_id)}}</code> <b>${{esc(x.status)}}</b>: ${{esc(x.claim)}}`,true)+'<h3>Facts cited by this report</h3>'+list(r.cited_facts,x=>`<code>${{esc(x.fact_id)}}</code> ${{esc(x.statement)}}`)+'<h3>Boundaries cited by this report</h3>'+list(r.cited_boundaries,x=>`<code>${{esc(x.boundary_id)}}</code> <b>${{esc(x.status)}}</b>: ${{esc(x.claim)}}`)+'<h3>Required disclosures</h3>'+list(r.required_disclosures,x=>esc(x));document.getElementById('all-reference').innerHTML='<h3>All facts</h3>'+list(REVIEW.all_facts,x=>`<code>${{esc(x.fact_id)}}</code> ${{esc(x.statement)}}`)+'<h3>All boundaries</h3>'+list(REVIEW.all_boundaries,x=>`<code>${{esc(x.boundary_id)}}</code> <b>${{esc(x.status)}}</b>: ${{esc(x.claim)}}`)}}
function renderJump(){{const el=document.getElementById('jump');el.innerHTML=REVIEW.items.map((x,i)=>`<button data-i="${{i}}" class="${{labels[x.label_item_id]?'done ':''}}${{i===index?'current':''}}">${{i+1}}</button>`).join('');el.querySelectorAll('button').forEach(b=>b.onclick=()=>{{index=Number(b.dataset.i);show()}})}}
function show(){{const item=REVIEW.items[index],done=Object.keys(labels).length;document.getElementById('status').textContent=`${{index+1}} of ${{REVIEW.item_count}} · ${{done}} reviewed · blinded`;document.getElementById('scenario').innerHTML=`<p><b>Audience:</b> ${{esc(item.scenario.audience)}}</p><p><b>Request:</b> ${{esc(item.scenario.request)}}</p>`;renderReport(item.report);renderReference(item);document.querySelectorAll('input[type=radio],input[type=checkbox]').forEach(x=>x.checked=false);document.getElementById('notes').value='';const old=labels[item.label_item_id];if(old){{document.getElementById('annotator').value=old.annotator_alias;for(const [d,s] of Object.entries(old.dimension_scores)){{const x=document.querySelector(`input[name="${{d}}"][value="${{s}}"]`);if(x)x.checked=true}}for(const f of old.critical_error_flags){{const x=document.querySelector(`input[name="critical"][value="${{f}}"]`);if(x)x.checked=true}}document.getElementById('notes').value=old.notes||''}}updateRelease();renderJump();document.getElementById('download').disabled=Object.keys(labels).length!==REVIEW.item_count}}
function collect(){{const item=REVIEW.items[index],scores=currentScores(),flags=currentFlags(),alias=document.getElementById('annotator').value.trim();if(!alias)throw Error('Enter an annotator alias');if(Object.keys(scores).length!==REVIEW.rubric.dimensions.length)throw Error('Score every dimension');return{{label_item_id:item.label_item_id,annotator_alias:alias,dimension_scores:scores,critical_error_flags:flags,overall_pass:computed(scores,flags),notes:document.getElementById('notes').value}}}}
document.querySelectorAll('input[type=radio],input[type=checkbox]').forEach(x=>x.onchange=updateRelease);document.getElementById('save').onclick=()=>{{try{{const row=collect();labels[row.label_item_id]=row;localStorage.setItem(key,JSON.stringify(labels));if(index<REVIEW.item_count-1)index++;show()}}catch(e){{alert(e.message)}}}};document.getElementById('previous').onclick=()=>{{if(index>0)index--;show()}};document.getElementById('download').onclick=()=>{{if(Object.keys(labels).length!==REVIEW.item_count)return;const payload={{schema_version:'intervenebench.human_report_labels.v1',evaluation_id:REVIEW.evaluation_id,queue_sha256:REVIEW.queue_sha256,labels:REVIEW.items.map(x=>labels[x.label_item_id])}};const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='report-labels.json';a.click();URL.revokeObjectURL(a.href)}};show();</script></body></html>"""
