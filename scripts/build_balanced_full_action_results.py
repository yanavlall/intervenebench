"""Freeze complete outcome-blind recommendations after the 56-call completion."""

from __future__ import annotations

from pathlib import Path

from intervenebench.balanced_forced_choice import build_completed_full_action_artifact
from intervenebench.protocol import freeze_envelope


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run_root = (
        root
        / "artifacts/balanced_full_action/balanced_full_action_20260813_v1"
    )
    freeze_envelope(
        build_completed_full_action_artifact(root, new_run_root=run_root),
        run_root / "full_action_recommendations.json",
        require_blinded=True,
    )


if __name__ == "__main__":
    main()
