#!/usr/bin/env python3
"""Materialize the authorized aggregate-only retrospective cross-family score."""

from pathlib import Path

from intervenebench.cross_family_retrospective_score import (
    DEFAULT_RETROSPECTIVE_SCORE_AUTHORIZATION_PATH,
    freeze_retrospective_cross_family_score,
    validate_retrospective_score_authorization,
)
from intervenebench.protocol import verify_envelope


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    authorization = verify_envelope(
        ROOT / DEFAULT_RETROSPECTIVE_SCORE_AUTHORIZATION_PATH
    )
    validate_retrospective_score_authorization(authorization, root=ROOT)
    print(freeze_retrospective_cross_family_score(ROOT, authorization=authorization))


if __name__ == "__main__":
    main()
