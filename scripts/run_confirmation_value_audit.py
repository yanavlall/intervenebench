#!/usr/bin/env python3
"""Run and freeze the aggregate-only confirmation value audit."""

from __future__ import annotations

from pathlib import Path

from intervenebench.confirmation_value_audit import (
    AUDIT_PATH,
    SCORE_PATH,
    SPEC_PATH,
    build_confirmation_value_audit_payload,
)
from intervenebench.protocol import freeze_envelope, verify_envelope


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    score = verify_envelope(ROOT / SCORE_PATH, require_blinded=False)
    spec = verify_envelope(ROOT / SPEC_PATH, require_blinded=False)
    audit = build_confirmation_value_audit_payload(ROOT, score=score, spec=spec)
    digest = freeze_envelope(audit, ROOT / AUDIT_PATH, require_blinded=False)
    print(f"frozen {AUDIT_PATH.as_posix()} payload_sha256={digest}")


if __name__ == "__main__":
    main()
