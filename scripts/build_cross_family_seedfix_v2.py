#!/usr/bin/env python3
"""Create or replay the v2 logprob-window correction package."""

from __future__ import annotations

import argparse
from pathlib import Path

from intervenebench.cross_family_seedfix import (
    SEEDFIX_V2_FREEZE_PATH,
    SEEDFIX_V2_MATERIALIZATION_AUTHORIZATION_PATH,
    build_seedfix_v2_freeze,
    build_seedfix_v2_materialization_authorization,
)
from intervenebench.protocol import freeze_envelope, payload_hash, verify_envelope


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write-freeze", action="store_true")
    parser.add_argument("--write-materialization-authorization", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    path = root / SEEDFIX_V2_FREEZE_PATH
    expected = build_seedfix_v2_freeze(root)
    if args.write_freeze:
        digest = freeze_envelope(expected, path, require_blinded=True)
    else:
        actual = verify_envelope(path, require_blinded=True)
        if actual != expected:
            raise ValueError("seed-fix v2 freeze does not replay")
        digest = payload_hash(actual)
    print(f"seedfix_v2_freeze={digest}")
    if args.write_materialization_authorization:
        auth = build_seedfix_v2_materialization_authorization(root)
        auth_digest = freeze_envelope(
            auth,
            root / SEEDFIX_V2_MATERIALIZATION_AUTHORIZATION_PATH,
            require_blinded=True,
        )
        print(f"materialization_authorization={auth_digest}")


if __name__ == "__main__":
    main()
