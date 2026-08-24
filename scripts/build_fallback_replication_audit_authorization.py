from __future__ import annotations

from pathlib import Path

from intervenebench.fallback_replication_audit import (
    DEFAULT_FALLBACK_REPLICATION_AUTHORIZATION_PATH,
    build_fallback_replication_audit_authorization,
)
from intervenebench.protocol import freeze_envelope


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = build_fallback_replication_audit_authorization(root)
    digest = freeze_envelope(
        payload, root / DEFAULT_FALLBACK_REPLICATION_AUTHORIZATION_PATH
    )
    print(digest)


if __name__ == "__main__":
    main()
