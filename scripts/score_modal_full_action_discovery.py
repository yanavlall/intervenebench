"""Freeze retrospective discovery scores for the four full-action models."""

from __future__ import annotations

from pathlib import Path

from intervenebench.modal_discovery_scoring import score_modal_discovery
from intervenebench.protocol import freeze_envelope


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    freeze_envelope(
        score_modal_discovery(root),
        (
            root
            / "artifacts/balanced_full_action/balanced_full_action_20260813_v1/"
            "retrospective_discovery_score.json"
        ),
        require_blinded=False,
    )


if __name__ == "__main__":
    main()
