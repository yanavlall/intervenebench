"""Validate the frozen, low-cost Benchmark v1 engineering-pilot budget."""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from .protocol import assert_blinded_payload, payload_hash


def read_budget(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_budget(payload)
    return payload


def validate_budget(payload: Mapping[str, Any]) -> None:
    assert_blinded_payload(payload)
    if payload.get("schema_version") != "engineering_pilot_compute_budget.v1":
        raise ValueError("unsupported engineering-pilot budget schema")
    if payload.get("status") != "frozen_not_authorized_to_spend":
        raise ValueError("budget must remain frozen and unspent before execution")
    if payload.get("reveal_authorized") is not False:
        raise ValueError("compute budget must not authorize human-outcome reveal")
    if payload.get("paid_inference_authorized") is not False:
        raise ValueError("freezing a budget must not authorize paid inference")
    if payload.get("modal_compute_authorized") is not False:
        raise ValueError("engineering pilot must not authorize Modal compute")
    cap = payload.get("hard_total_cap_usd")
    if (
        isinstance(cap, bool)
        or not isinstance(cap, (int, float))
        or not isfinite(cap)
        or cap <= 0
        or cap > 30
    ):
        raise ValueError("engineering-pilot hard cap must be in (0, 30]")
    tiers = payload.get("tiers")
    if not isinstance(tiers, list) or not tiers:
        raise ValueError("at least one compute tier is required")
    allocated = 0.0
    for tier in tiers:
        if not isinstance(tier, Mapping):
            raise ValueError("compute tiers must be objects")
        maximum = tier.get("maximum_usd")
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, (int, float))
            or not isfinite(maximum)
            or maximum < 0
        ):
            raise ValueError("tier maxima must be finite and non-negative")
        allocated += float(maximum)
    if allocated > float(cap) + 1e-9:
        raise ValueError("tier maxima exceed the hard total cap")
    if payload.get("stop_conditions") is None:
        raise ValueError("budget must freeze stop conditions")
    split_path = payload.get("engineering_split_path")
    split_digest = payload.get("engineering_split_sha256")
    if not isinstance(split_path, str) or not split_path.strip():
        raise ValueError("engineering split path is required")
    if not isinstance(split_digest, str) or len(split_digest) != 64:
        raise ValueError("engineering split SHA-256 is required")


def verify_bound_budget(root: Path, budget_path: Path) -> dict[str, Any]:
    payload = read_budget(budget_path)
    split_path = root / payload["engineering_split_path"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if payload_hash(split) != payload["engineering_split_sha256"]:
        raise ValueError("compute budget is not bound to the frozen engineering split")
    if split.get("not_canonical_benchmark_split") is not True:
        raise ValueError("engineering budget may bind only to a noncanonical split")
    if split.get("all_human_outcomes_sealed") is not True:
        raise ValueError("engineering split must keep all human outcomes sealed")
    if split.get("reveal_authorized") is not False:
        raise ValueError("engineering split must not authorize a reveal")
    return payload
