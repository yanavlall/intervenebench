#!/usr/bin/env python3
"""Freeze exact aggregate-only authority for retrospective cross-family scoring."""

from pathlib import Path

from intervenebench.cross_family_retrospective_score import (
    DEFAULT_RETROSPECTIVE_SCORE_AUTHORIZATION_PATH,
    build_retrospective_score_authorization,
    validate_retrospective_score_authorization,
)
from intervenebench.protocol import freeze_envelope


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    authorization = build_retrospective_score_authorization(ROOT)
    validate_retrospective_score_authorization(authorization, root=ROOT)
    print(
        freeze_envelope(
            authorization,
            ROOT / DEFAULT_RETROSPECTIVE_SCORE_AUTHORIZATION_PATH,
            require_blinded=False,
        )
    )


if __name__ == "__main__":
    main()
