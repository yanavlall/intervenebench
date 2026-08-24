#!/usr/bin/env python3
"""Freeze the explicitly post-reveal confirmation value-audit specification."""

from __future__ import annotations

from pathlib import Path

from intervenebench.confirmation_value_audit import (
    SCORE_PATH,
    SPEC_PATH,
    build_confirmation_value_audit_spec,
)
from intervenebench.protocol import freeze_envelope, verify_envelope


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    score = verify_envelope(ROOT / SCORE_PATH, require_blinded=False)
    spec = build_confirmation_value_audit_spec(ROOT, score=score)
    digest = freeze_envelope(spec, ROOT / SPEC_PATH, require_blinded=False)
    print(f"frozen {SPEC_PATH.as_posix()} payload_sha256={digest}")


if __name__ == "__main__":
    main()
