from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from intervenebench.compute_budget import read_budget, validate_budget, verify_bound_budget


ROOT = Path(__file__).resolve().parents[1]
BUDGET_PATH = (
    ROOT
    / "data"
    / "manifests"
    / "benchmark"
    / "supported_ordinal_compute_budget.json"
)


def test_budget_is_bound_capped_unspent_and_reveal_free() -> None:
    payload = verify_bound_budget(ROOT, BUDGET_PATH)
    assert payload == read_budget(BUDGET_PATH)
    assert payload["hard_total_cap_usd"] == 25.0
    assert sum(tier["maximum_usd"] for tier in payload["tiers"]) == 25.0
    assert payload["current_recorded_spend_usd"] == 0.0
    assert payload["paid_inference_authorized"] is False
    assert payload["modal_compute_authorized"] is False
    assert payload["reveal_authorized"] is False
    assert payload["human_outcomes_opened"] is False


def test_budget_rejects_overspend_and_accidental_authorization() -> None:
    payload = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    overspent = deepcopy(payload)
    overspent["tiers"][1]["maximum_usd"] = 30.0
    with pytest.raises(ValueError, match="exceed"):
        validate_budget(overspent)

    authorized = deepcopy(payload)
    authorized["paid_inference_authorized"] = True
    with pytest.raises(ValueError, match="must not authorize"):
        validate_budget(authorized)
