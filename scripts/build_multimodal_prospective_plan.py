"""Create the zero-authority prospective multimodal call plan."""

from __future__ import annotations

import json
from pathlib import Path

from intervenebench.multimodal_prospective import build_multimodal_prospective_plan


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = (
        root / "data/manifests/simulators/prospective_multimodal_plan_v1.json"
    )
    payload = build_multimodal_prospective_plan(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
