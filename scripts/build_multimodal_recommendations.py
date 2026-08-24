"""Create the outcome-blind aggregate artifact for the 54-call image run."""

from __future__ import annotations

from pathlib import Path

from intervenebench.multimodal_recommendations import (
    build_multimodal_recommendations,
)
from intervenebench.protocol import freeze_envelope


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run_root = (
        root
        / "artifacts/prospective_multimodal/prospective_multimodal_20260813_v4"
    )
    freeze_envelope(
        build_multimodal_recommendations(root, run_root=run_root),
        run_root / "prospective_recommendations.json",
        require_blinded=True,
    )


if __name__ == "__main__":
    main()
