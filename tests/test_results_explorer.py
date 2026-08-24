from __future__ import annotations

from pathlib import Path

from intervenebench.public_case_study import (
    DEFAULT_PUBLIC_CASE_STUDY_PATH,
    verify_public_case_study,
)
from intervenebench.results_explorer import (
    DEFAULT_RESULTS_EXPLORER_PATH,
    build_results_explorer_html,
)


ROOT = Path(__file__).resolve().parents[1]


def test_explorer_renders_release_scopes_and_regret_comparison() -> None:
    report = verify_public_case_study(ROOT / DEFAULT_PUBLIC_CASE_STUDY_PATH)
    html = build_results_explorer_html(report)

    assert "InterveneBench" in html
    assert "Limited research use" in html
    assert html.count(">Hold<") == 3
    assert "Primary simulator" in html
    assert "Uniform action" in html
    assert "Classical baseline" in html
    assert "0.0035" in html
    assert "0.0410" in html
    assert "1404 / 1464" in html


def test_explorer_is_portable_and_contains_no_detailed_human_data() -> None:
    report = verify_public_case_study(ROOT / DEFAULT_PUBLIC_CASE_STUDY_PATH)
    html = build_results_explorer_html(report)

    assert "https://" not in html
    assert "<script src=" not in html
    assert "participant_id" not in html
    assert "human_arm_means" not in html
    assert "experiment_scores" not in html


def test_checked_in_explorer_is_deterministic() -> None:
    report = verify_public_case_study(ROOT / DEFAULT_PUBLIC_CASE_STUDY_PATH)
    expected = build_results_explorer_html(report)
    assert (ROOT / DEFAULT_RESULTS_EXPLORER_PATH).read_text(encoding="utf-8") == expected
