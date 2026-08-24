"""Write the aggregate-only, outcome-blind discovery diagnostics artifact."""

from __future__ import annotations

from pathlib import Path

from intervenebench.forced_choice_screen_analysis import analyze_screen
from intervenebench.protocol import freeze_envelope


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run_root = root / "artifacts/forced_choice_screen/discovery_screen_20260813_v1"
    freeze_envelope(
        analyze_screen(root, run_root=run_root),
        run_root / "outcome_blind_diagnostics.json",
        require_blinded=True,
    )


if __name__ == "__main__":
    main()
