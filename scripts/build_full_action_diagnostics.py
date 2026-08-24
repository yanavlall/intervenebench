"""Freeze the complete outcome-free full-action diagnostic artifact."""

from __future__ import annotations

from pathlib import Path

from intervenebench.full_action_diagnostics import build_full_action_diagnostics
from intervenebench.protocol import freeze_envelope


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    freeze_envelope(
        build_full_action_diagnostics(
            root,
            freeze_path=(
                root / "configs/diagnostics/balanced_full_action_v2.json"
            ),
        ),
        (
            root
            / "artifacts/balanced_full_action/balanced_full_action_20260813_v1/"
            "outcome_free_diagnostics_v2.json"
        ),
        require_blinded=True,
    )


if __name__ == "__main__":
    main()
