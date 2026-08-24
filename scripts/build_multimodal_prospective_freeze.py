"""Create the immutable zero-authority multimodal prospective freeze."""

from __future__ import annotations

import json
from pathlib import Path

from intervenebench.multimodal_freeze import build_prospective_multimodal_freeze


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = root / "configs/simulators/prospective_multimodal_v4.json"
    payload = build_prospective_multimodal_freeze(root)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
