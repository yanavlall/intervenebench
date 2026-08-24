#!/usr/bin/env python3
"""Freeze exact authority for a zero-call retrospective cross-family merge."""

from __future__ import annotations

from pathlib import Path

from intervenebench.cross_family_merge import (
    DEFAULT_AUTHORIZATION_PATH,
    build_merge_authorization,
    validate_merge_authorization,
)
from intervenebench.protocol import freeze_envelope


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    authorization = build_merge_authorization(ROOT)
    validate_merge_authorization(authorization, root=ROOT)
    digest = freeze_envelope(
        authorization,
        ROOT / DEFAULT_AUTHORIZATION_PATH,
        require_blinded=True,
    )
    print(digest)


if __name__ == "__main__":
    main()
