"""Freeze balanced predictions from the already collected paired screen."""

from __future__ import annotations

from pathlib import Path

from intervenebench.balanced_forced_choice import build_balanced_discovery_artifact
from intervenebench.protocol import freeze_envelope


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    target = (
        root
        / "artifacts/answer_order_canary/answer_order_canary_20260813_v1/"
        "balanced_discovery_predictions.json"
    )
    freeze_envelope(
        build_balanced_discovery_artifact(
            root,
            source_run_root=(
                root / "artifacts/forced_choice_screen/discovery_screen_20260813_v1"
            ),
            reverse_run_root=(
                root
                / "artifacts/answer_order_canary/answer_order_canary_20260813_v1"
            ),
        ),
        target,
        require_blinded=True,
    )


if __name__ == "__main__":
    main()
