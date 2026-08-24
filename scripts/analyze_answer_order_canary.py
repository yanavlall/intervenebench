"""Freeze paired outcome-blind answer-order diagnostics."""

from __future__ import annotations

from pathlib import Path

from intervenebench.answer_order_analysis import analyze_answer_order
from intervenebench.protocol import freeze_envelope


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source_root = (
        root / "artifacts/forced_choice_screen/discovery_screen_20260813_v1"
    )
    reverse_root = (
        root / "artifacts/answer_order_canary/answer_order_canary_20260813_v1"
    )
    target = reverse_root / "paired_robustness_diagnostics.json"
    freeze_envelope(
        analyze_answer_order(
            root,
            source_run_root=source_root,
            reverse_run_root=reverse_root,
        ),
        target,
        require_blinded=True,
    )


if __name__ == "__main__":
    main()
