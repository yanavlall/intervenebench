from __future__ import annotations

from pathlib import Path

from intervenebench.fallback_failure_mechanism import (
    DEFAULT_MECHANISM_AUTHORIZATION_PATH,
    freeze_mechanism_audit,
)
from intervenebench.protocol import verify_envelope


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    authorization = verify_envelope(root / DEFAULT_MECHANISM_AUTHORIZATION_PATH)
    digest = freeze_mechanism_audit(root, authorization=authorization)
    print(digest)


if __name__ == "__main__":
    main()
