"""Create the immutable pre-reveal protocol for the image development set."""

from __future__ import annotations

import json
from pathlib import Path

from intervenebench.prospective_development_protocol import (
    PROTOCOL_PATH,
    build_pre_reveal_protocol,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = build_pre_reveal_protocol(root)
    path = root / PROTOCOL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
