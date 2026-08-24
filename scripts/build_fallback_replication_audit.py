from __future__ import annotations

from pathlib import Path

from intervenebench.fallback_replication_audit import (
    DEFAULT_FALLBACK_REPLICATION_AUTHORIZATION_PATH,
    freeze_fallback_replication_audit,
)
from intervenebench.protocol import verify_envelope


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    authorization = verify_envelope(
        root / DEFAULT_FALLBACK_REPLICATION_AUTHORIZATION_PATH
    )
    digest = freeze_fallback_replication_audit(root, authorization=authorization)
    print(digest)


if __name__ == "__main__":
    main()
