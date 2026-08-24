"""Build the aggregate-only nine-experiment development evidence registry."""

from __future__ import annotations

from pathlib import Path

from intervenebench.development_evidence import (
    DEFAULT_DEVELOPMENT_EVIDENCE_PATH,
    write_development_evidence,
)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    write_development_evidence(root, root / DEFAULT_DEVELOPMENT_EVIDENCE_PATH)
