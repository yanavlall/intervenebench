#!/usr/bin/env python3
"""Verify or create the zero-authority cross-family call-plan freeze."""

from __future__ import annotations

import argparse
from pathlib import Path

from intervenebench.cross_family_regression import (
    DEFAULT_CALL_PLAN_PATH,
    freeze_cross_family_call_plan,
    verify_cross_family_freeze,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--write",
        action="store_true",
        help="create the default call-plan path; refuses to overwrite",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if args.write:
        digest = freeze_cross_family_call_plan(root)
        print(f"created {root / DEFAULT_CALL_PLAN_PATH} ({digest})")
        return
    print(verify_cross_family_freeze(root))


if __name__ == "__main__":
    main()
