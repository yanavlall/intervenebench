#!/usr/bin/env python3
"""Create or replay the versioned null-seed correction package."""

from __future__ import annotations

import argparse
from pathlib import Path

from intervenebench.cross_family_seedfix import (
    SEEDFIX_FREEZE_PATH,
    SEEDFIX_MATERIALIZATION_AUTHORIZATION_PATH,
    build_seedfix_freeze,
    build_seedfix_materialization_authorization,
)
from intervenebench.protocol import freeze_envelope, payload_hash, verify_envelope


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write-freeze", action="store_true")
    parser.add_argument("--write-materialization-authorization", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    freeze_path = root / SEEDFIX_FREEZE_PATH
    expected = build_seedfix_freeze(root)
    if args.write_freeze:
        freeze_digest = freeze_envelope(expected, freeze_path, require_blinded=True)
    else:
        actual = verify_envelope(freeze_path, require_blinded=True)
        if actual != expected:
            raise ValueError("seed-fix freeze does not replay")
        freeze_digest = payload_hash(actual)
    print(f"seedfix_freeze={freeze_digest}")
    if args.write_materialization_authorization:
        authorization = build_seedfix_materialization_authorization(root)
        digest = freeze_envelope(
            authorization,
            root / SEEDFIX_MATERIALIZATION_AUTHORIZATION_PATH,
            require_blinded=True,
        )
        print(f"materialization_authorization={digest}")


if __name__ == "__main__":
    main()
