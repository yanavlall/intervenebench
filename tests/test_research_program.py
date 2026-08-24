from __future__ import annotations

import json
from pathlib import Path

import pytest

from intervenebench.research_program import (
    load_and_validate_research_program,
    verify_research_program,
)


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "data/manifests/research/depth_first_v1.json"


def test_active_program_has_fifteen_independent_tasks_and_six_sealed() -> None:
    result = verify_research_program(ROOT)
    assert result["discovery_experiments"] == 6
    assert result["prospective_development_experiments"] == 3
    assert result["prospective_experiments"] == 6
    assert result["total_experiments"] == 15
    assert result["prospective_outcomes_sealed"] is True
    assert result["authorized_spend_usd"] == 0


def test_program_preserves_paradigm_separation() -> None:
    program = load_and_validate_research_program(PROGRAM)
    development = {
        task["paradigm_group"] for task in program["discovery_tasks"]
    } | {
        task["paradigm_group"]
        for task in program["prospective_development_tasks"]
    }
    prospective = {
        task["paradigm_group"]
        for task in program["prospective_evaluation_tasks"]
    }
    assert development.isdisjoint(prospective)


def test_program_fails_if_a_prospective_outcome_is_unsealed(tmp_path: Path) -> None:
    program = json.loads(PROGRAM.read_text(encoding="utf-8"))
    program["prospective_evaluation_tasks"][0]["outcome_access"] = (
        "development_only"
    )
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(program), encoding="utf-8")
    with pytest.raises(ValueError, match="prospective outcome"):
        load_and_validate_research_program(changed)


def test_program_fails_on_result_bearing_keys(tmp_path: Path) -> None:
    program = json.loads(PROGRAM.read_text(encoding="utf-8"))
    program["human_winner"] = "forbidden"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(program), encoding="utf-8")
    with pytest.raises(ValueError, match="result-bearing key"):
        load_and_validate_research_program(changed)
