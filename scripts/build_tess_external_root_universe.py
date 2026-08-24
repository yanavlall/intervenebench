#!/usr/bin/env python3
"""Build a result-free external TESS root-project universe from OSF API pages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path


LABEL = "external-tess-root-v1:20260814"
SCHEMA_VERSION = "intervenebench.external_tess_root_universe.v1"
BATCH_SIZE = 30


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalized_title(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    return re.sub(r"[^a-z0-9]+", "", decomposed)


def _read_external_titles(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            _normalized_title(row["study_title"])
            for row in csv.DictReader(handle)
            if row.get("study_title")
        }


def build(
    *,
    page_paths: list[Path],
    socsci_nodes_dir: Path,
    prior_external_universe: Path,
) -> dict:
    page_records: list[dict] = []
    all_nodes: list[dict] = []
    advertised_total: int | None = None
    for page_number, path in enumerate(page_paths, start=1):
        raw = path.read_bytes()
        payload = json.loads(raw)
        total = payload["links"]["meta"]["total"]
        if advertised_total is None:
            advertised_total = total
        elif advertised_total != total:
            raise ValueError("OSF page totals disagree")
        all_nodes.extend(payload["data"])
        page_records.append(
            {
                "page": page_number,
                "source_url": (
                    "https://api.osf.io/v2/users/4547c/nodes/"
                    f"?page={page_number}&page%5Bsize%5D=100"
                ),
                "response_sha256": _sha256_bytes(raw),
                "row_count": len(payload["data"]),
            }
        )
    if advertised_total != len(all_nodes):
        raise ValueError("OSF page snapshot is incomplete")
    if len({row["id"] for row in all_nodes}) != len(all_nodes):
        raise ValueError("OSF node IDs are not unique")

    socsci_by_id: dict[str, str] = {}
    for path in sorted(socsci_nodes_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        identifier = payload["data"]["id"]
        title = payload["data"]["attributes"]["title"]
        socsci_by_id[identifier] = title
    socsci_titles = {_normalized_title(value) for value in socsci_by_id.values()}
    prior_external_titles = _read_external_titles(prior_external_universe)

    public_roots: list[dict] = []
    for row in all_nodes:
        attributes = row["attributes"]
        parent = row["relationships"].get("parent", {}).get("data")
        if attributes.get("public") is True and parent is None:
            public_roots.append(
                {
                    "osf_node_id": row["id"],
                    "title": attributes["title"],
                    "date_created": attributes["date_created"],
                    "date_modified": attributes["date_modified"],
                    "category": attributes["category"],
                    "source_url": row["links"]["html"],
                }
            )

    exclusions: list[dict] = []
    candidates: list[dict] = []
    for row in public_roots:
        identifier = row["osf_node_id"]
        normalized = _normalized_title(row["title"])
        if identifier in socsci_by_id:
            exclusions.append({**row, "deduplication_rule": "socsci_node_id"})
            continue
        if normalized in socsci_titles:
            exclusions.append({**row, "deduplication_rule": "socsci_normalized_title"})
            continue
        if normalized in prior_external_titles:
            exclusions.append(
                {**row, "deduplication_rule": "prior_external_normalized_title"}
            )
            continue
        selection_sha256 = hashlib.sha256(
            f"{LABEL}:{identifier}".encode("utf-8")
        ).hexdigest()
        candidates.append(
            {
                **row,
                "selection_sha256": selection_sha256,
                "outcome_access": "sealed",
                "result_text_exposed": False,
                "source_audit_status": "pending_ordered_source_audit",
            }
        )
    candidates.sort(key=lambda row: row["selection_sha256"])
    for index, row in enumerate(candidates, start=1):
        row["freeze_order"] = index

    batch = [row["osf_node_id"] for row in candidates[:BATCH_SIZE]]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "outcome_blind_universe_frozen",
        "freeze_date": "2026-08-14",
        "selection_label": LABEL,
        "source_account": {
            "osf_user_id": "4547c",
            "display_name": "TESS-Experiments",
            "api_collection": "https://api.osf.io/v2/users/4547c/nodes/",
            "pages": page_records,
            "advertised_node_count": advertised_total,
        },
        "deduplication_inputs": {
            "socsci_node_count": len(socsci_by_id),
            "prior_external_title_count": len(prior_external_titles),
            "rules": [
                "exact OSF node ID against pinned SocSci210",
                "normalized title against pinned SocSci210",
                "normalized title against external candidate universe v1",
                "fielding-level deduplication remains mandatory during source audit",
            ],
        },
        "counts": {
            "api_nodes": len(all_nodes),
            "public_root_projects": len(public_roots),
            "first_pass_dedup_exclusions": len(exclusions),
            "candidate_root_projects": len(candidates),
            "ordered_audit_batch": len(batch),
        },
        "ordered_audit_protocol": {
            "protocol_revision": "1.1_no_progress_stop",
            "batch_candidate_ids": batch,
            "batch_size": BATCH_SIZE,
            "audit_contiguously_in_freeze_order": True,
            "interim_stop_after_order_20_if_clean_passes_below": 2,
            "interim_stop_basis": (
                "source-audit yield only; no human outcomes or simulator results"
            ),
            "stop_after": (
                "at order 20 if fewer than two clean tasks pass; otherwise finish "
                "the current six-candidate block after four pristine runnable "
                "tasks pass, or after all 30 rows are adjudicated"
            ),
            "no_selective_skipping": True,
            "result_exposure_is_automatic_prospective_failure": True,
        },
        "public_root_exclusions": exclusions,
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
    parser.add_argument("--page", action="append", required=True, type=Path)
    parser.add_argument("--socsci-nodes-dir", required=True, type=Path)
    parser.add_argument("--prior-external-universe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build(
        page_paths=args.page,
        socsci_nodes_dir=args.socsci_nodes_dir,
        prior_external_universe=args.prior_external_universe,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
