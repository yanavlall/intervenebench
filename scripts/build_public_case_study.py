"""Materialize the aggregate-only public confirmation case study."""

from __future__ import annotations

from pathlib import Path

from intervenebench.protocol import freeze_envelope
from intervenebench.public_case_study import (
    DEFAULT_PUBLIC_CASE_STUDY_PATH,
    build_public_case_study_payload,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output = ROOT / DEFAULT_PUBLIC_CASE_STUDY_PATH
    digest = freeze_envelope(build_public_case_study_payload(ROOT), output)
    print({"path": output.relative_to(ROOT).as_posix(), "payload_sha256": digest})


if __name__ == "__main__":
    main()
