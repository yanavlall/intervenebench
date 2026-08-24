#!/usr/bin/env python3
"""Create or replay the zero-authority Mistral target execution package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intervenebench.cross_family_execution import (
    DEFAULT_EXECUTION_FREEZE_PATH,
    DEFAULT_MATERIALIZATION_AUTHORIZATION_PATH,
    build_cross_family_execution_freeze,
    build_materialization_authorization,
)
from intervenebench.protocol import freeze_envelope, payload_hash, verify_envelope


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--write-materialization-authorization",
        action="store_true",
        help="write the already user-approved image-only authorization",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    freeze_path = root / DEFAULT_EXECUTION_FREEZE_PATH
    expected = build_cross_family_execution_freeze(root)
    if args.write:
        freeze_path.parent.mkdir(parents=True, exist_ok=True)
        with freeze_path.open("x", encoding="utf-8") as stream:
            json.dump(expected, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
        digest = payload_hash(expected)
    else:
        actual = json.loads(freeze_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError("cross-family target execution freeze does not replay")
        digest = payload_hash(actual)
    if args.write_materialization_authorization:
        authorization = build_materialization_authorization(expected)
        auth_path = root / DEFAULT_MATERIALIZATION_AUTHORIZATION_PATH
        auth_digest = freeze_envelope(
            authorization, auth_path, require_blinded=True
        )
        print(f"materialization_authorization={auth_digest}")
    print(f"execution_freeze={digest}")


if __name__ == "__main__":
    main()
