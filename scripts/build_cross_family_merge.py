#!/usr/bin/env python3
"""Materialize the authorized zero-call retrospective cross-family merge."""

from __future__ import annotations

from pathlib import Path

from intervenebench.cross_family_merge import (
    DEFAULT_AUTHORIZATION_PATH,
    freeze_cross_family_merge,
    validate_merge_authorization,
)
from intervenebench.protocol import verify_envelope


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    authorization = verify_envelope(
        ROOT / DEFAULT_AUTHORIZATION_PATH,
        require_blinded=True,
    )
    validate_merge_authorization(authorization, root=ROOT)
    digest = freeze_cross_family_merge(ROOT, authorization=authorization)
    print(digest)


if __name__ == "__main__":
    main()
