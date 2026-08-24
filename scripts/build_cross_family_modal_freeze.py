#!/usr/bin/env python3
"""Create or verify the zero-authority cross-family Modal preflight freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intervenebench.cross_family_modal import (
    DEFAULT_CROSS_FAMILY_MODAL_FREEZE_PATH,
    build_cross_family_modal_freeze,
    verify_cross_family_modal_freeze,
)
from intervenebench.protocol import payload_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    path = root / DEFAULT_CROSS_FAMILY_MODAL_FREEZE_PATH
    if args.write:
        value = build_cross_family_modal_freeze(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
    value = verify_cross_family_modal_freeze(root, path)
    print(payload_hash(value))


if __name__ == "__main__":
    main()
