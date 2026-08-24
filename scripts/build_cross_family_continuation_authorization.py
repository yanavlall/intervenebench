#!/usr/bin/env python3
"""Create or replay the authorized no-rerun 504-call continuation."""

from __future__ import annotations

import argparse
from pathlib import Path

from intervenebench.cross_family_continuation import (
    DEFAULT_CONTINUATION_AUTHORIZATION_PATH,
    build_continuation_authorization,
)
from intervenebench.protocol import freeze_envelope, payload_hash, verify_envelope


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    path = root / DEFAULT_CONTINUATION_AUTHORIZATION_PATH
    expected = build_continuation_authorization(root)
    if args.write:
        digest = freeze_envelope(expected, path, require_blinded=True)
    else:
        actual = verify_envelope(path, require_blinded=True)
        if actual != expected:
            raise ValueError("no-rerun continuation authorization does not replay")
        digest = payload_hash(actual)
    print(digest)


if __name__ == "__main__":
    main()
