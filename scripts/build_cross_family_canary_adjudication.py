#!/usr/bin/env python3
"""Create or replay the target-free Mistral canary adjudication."""

from __future__ import annotations

import argparse
from pathlib import Path

from intervenebench.cross_family_adjudication import (
    DEFAULT_ADJUDICATION_PATH,
    build_cross_family_canary_adjudication,
)
from intervenebench.protocol import freeze_envelope, payload_hash, verify_envelope


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    path = root / DEFAULT_ADJUDICATION_PATH
    expected = build_cross_family_canary_adjudication(root)
    if args.write:
        digest = freeze_envelope(expected, path, require_blinded=True)
    else:
        actual = verify_envelope(path, require_blinded=True)
        if actual != expected:
            raise ValueError("cross-family canary adjudication does not replay")
        digest = payload_hash(actual)
    print(digest)


if __name__ == "__main__":
    main()

