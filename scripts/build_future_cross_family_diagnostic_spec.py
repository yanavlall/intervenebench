#!/usr/bin/env python3
"""Freeze the unvalidated cross-family diagnostic for future untouched tasks."""

from pathlib import Path

from intervenebench.cross_family_retrospective_score import (
    DEFAULT_DIAGNOSTIC_SPEC_PATH,
    DEFAULT_RETROSPECTIVE_SCORE_PATH,
    freeze_future_cross_family_diagnostic_spec,
)
from intervenebench.protocol import verify_envelope


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    score = verify_envelope(ROOT / DEFAULT_RETROSPECTIVE_SCORE_PATH)
    print(
        freeze_future_cross_family_diagnostic_spec(
            score,
            destination=ROOT / DEFAULT_DIAGNOSTIC_SPEC_PATH,
        )
    )


if __name__ == "__main__":
    main()
