#!/usr/bin/env python3
"""Freeze a high-precision, title-only action-oriented TESS audit lane."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path


SCHEMA_VERSION = "intervenebench.targeted_tess_intervention_universe.v1"
BATCH_SIZE = 12
INCLUDE_PATTERNS = (
    r"\bmessages?\b",
    r"\bmessaging\b",
    r"\bcommunicat(?:e|ed|es|ing|ion|ions)\b",
    r"\badvertis(?:e|ed|es|ing|ement|ements)\b",
    r"\bdisclos(?:e|ed|es|ing|ure|ures)\b",
    r"\btransparen(?:cy|t)\b",
    r"\bfact[ -]?checks?\b",
    r"\brhetori(?:c|cal)\b",
    r"\bnarratives?\b",
    r"\bportrayals?\b",
    r"\bappeals\b",
    r"\b(?:frame|framed|frames|framing|reframe|reframed|reframes|reframing)\b",
    r"\bself[ -]?affirmation\b",
    r"\bimplementation intentions?\b",
    r"\bmobiliz(?:e|ed|es|ing|ation)\b",
    r"\binvok(?:e|ed|es|ing|ation)\b",
    r"\breminders?\b",
    r"\bwarnings?\b",
    r"\bfeedback\b",
    r"\bnudges?\b",
    r"\brequests?\b",
    r"\bdonations?\b",
    r"\bdeliberat(?:e|ed|ion|ive)\b",
    r"\bpolicy justifications?\b",
    r"\binterventions? (?:to|for)\b",
    r"\ban intervention\b",
)
EXCLUDE_PATTERNS = (
    r"\b(?:measure|measures|measuring|measurement)\b",
    r"\bquestion (?:wording|order|context)\b",
    r"\border effects?\b",
    r"\blist experiments?\b",
    r"\bitem count\b",
    r"\banchoring vignettes?\b",
    r"\bsurvey (?:instrument|estimation|mode)\b",
    r"\bphone modules?\b",
    r"\bdata collection procedures?\b",
    r"\bpanel conditioning\b",
    r"\bover[ -]?reporting\b",
    r"\bself[ -]?reported\b",
    r"\btest performance\b",
    r"\bresponse (?:quality|scale)\b",
    r"\bscale (?:direction|order)\b",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized(value: str) -> str:
    decoded = html.unescape(value)
    normalized = unicodedata.normalize("NFKC", decoded).casefold()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", normalized).split())


def _matches(patterns: tuple[str, ...], value: str) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, value)]


def build(
    *,
    source_universe: Path,
    source_registry: Path,
    exposure_log: Path,
) -> dict:
    source = json.loads(source_universe.read_text(encoding="utf-8"))
    with source_registry.open(newline="", encoding="utf-8") as handle:
        audited_ids = {row["osf_node_id"] for row in csv.DictReader(handle)}
    exposure_ids: set[str] = set()
    for incident in json.loads(exposure_log.read_text(encoding="utf-8"))["incidents"]:
        exposure_ids.add(incident["candidate_id"])
        alias = incident.get("source_lookup_alias")
        if alias:
            exposure_ids.add(alias)

    selected_by_title: dict[str, dict] = {}
    title_match_count = 0
    audited_match_count = 0
    exposed_match_count = 0
    duplicate_title_count = 0
    for source_row in source["candidate_universe"]:
        normalized_title = _normalized(source_row["title"])
        include_matches = _matches(INCLUDE_PATTERNS, normalized_title)
        exclude_matches = _matches(EXCLUDE_PATTERNS, normalized_title)
        if not include_matches or exclude_matches:
            continue
        title_match_count += 1
        identifier = source_row["osf_node_id"]
        if identifier in audited_ids:
            audited_match_count += 1
            continue
        if identifier in exposure_ids:
            exposed_match_count += 1
            continue
        candidate = {
            "osf_node_id": identifier,
            "title": source_row["title"],
            "source_url": source_row["source_url"],
            "source_freeze_order": source_row["freeze_order"],
            "selection_sha256": source_row["selection_sha256"],
            "title_include_matches": include_matches,
            "outcome_access": "sealed",
            "result_text_exposed": False,
            "source_audit_status": "pending_ordered_source_audit",
        }
        prior = selected_by_title.get(normalized_title)
        if prior is None:
            selected_by_title[normalized_title] = candidate
        else:
            duplicate_title_count += 1
            if candidate["selection_sha256"] < prior["selection_sha256"]:
                selected_by_title[normalized_title] = candidate

    candidates = sorted(
        selected_by_title.values(), key=lambda row: row["selection_sha256"]
    )
    for index, row in enumerate(candidates, start=1):
        row["freeze_order"] = index

    batch = candidates[:BATCH_SIZE]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "outcome_blind_targeted_universe_frozen",
        "freeze_date": "2026-08-14",
        "protocol_revision": "1.0_high_precision_before_source_access",
        "selection_order": "preserve source universe selection_sha256 order",
        "inputs": {
            "source_universe_path": str(source_universe),
            "source_universe_sha256": _sha256(source_universe),
            "source_registry_path": str(source_registry),
            "source_registry_sha256": _sha256(source_registry),
            "exposure_log_path": str(exposure_log),
            "exposure_log_sha256": _sha256(exposure_log),
        },
        "selection_rule": {
            "allowed_metadata": ["osf_node_id", "title", "public source URL"],
            "include_patterns": list(INCLUDE_PATTERNS),
            "exclude_patterns": list(EXCLUDE_PATTERNS),
            "normalization": (
                "HTML-unescape; Unicode NFKC casefold; non-alphanumeric "
                "collapsed to spaces"
            ),
            "exclude_all_previously_audited_ids": True,
            "exclude_all_global_exposure_log_ids": True,
            "deduplicate_normalized_titles": (
                "retain minimum original source-universe selection_sha256"
            ),
            "manual_additions_or_deletions_permitted": False,
            "human_outcomes_or_simulator_results_used": False,
        },
        "counts": {
            "source_candidate_roots": len(source["candidate_universe"]),
            "title_matches_before_prior_audit_exposure_and_duplicate_exclusion": (
                title_match_count
            ),
            "previously_audited_ids_excluded": audited_match_count,
            "exposure_log_ids_excluded": exposed_match_count,
            "duplicate_normalized_titles_excluded": duplicate_title_count,
            "eligible_targeted_candidates": len(candidates),
            "ordered_audit_batch": len(batch),
        },
        "ordered_audit_protocol": {
            "protocol_revision": "1.0",
            "batch_candidate_ids": [row["osf_node_id"] for row in batch],
            "batch_size": BATCH_SIZE,
            "block_size": 3,
            "audit_contiguously_in_freeze_order": True,
            "no_selective_skipping": True,
            "scientific_survivor_definition": (
                "pristine source-faithful task that passes randomization, "
                "deployable fixed-world actions, bounded utility, exact stimuli, "
                "and fielding independence; only mechanical mapping or adapter "
                "implementation may remain"
            ),
            "checkpoint_after_order_6": {
                "minimum_scientific_survivors": 1,
                "action_below_minimum": "close_lane_low_yield",
            },
            "checkpoint_after_order_9": {
                "minimum_scientific_survivors": 2,
                "action_below_minimum": "close_lane_low_yield",
            },
            "continue_after_order_9_only_if_checkpoint_passes": True,
            "finish_current_block_after_scientific_survivor_count": 4,
            "hard_stop_after_order_12": True,
            "checkpoint_basis": (
                "source-audit structure and reconstruction only; no human outcomes "
                "or simulator performance"
            ),
            "result_exposure_is_automatic_prospective_failure": True,
        },
        "candidate_universe": candidates,
        "authority": {
            "authorized_spend_usd": 0,
            "model_calls_authorized": False,
            "human_outcome_reveal_authorized": False,
            "participant_row_access_authorized": False,
            "public_design_source_acquisition_authorized": True,
            "mixed_archive_member_listing_authorized": True,
            "participant_member_extraction_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-universe", required=True, type=Path)
    parser.add_argument("--source-registry", required=True, type=Path)
    parser.add_argument("--exposure-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build(
        source_universe=args.source_universe,
        source_registry=args.source_registry,
        exposure_log=args.exposure_log,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
