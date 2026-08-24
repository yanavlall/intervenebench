"""Materialize the deterministic reverse-answer-order call plan."""

from __future__ import annotations

import json
from pathlib import Path

from intervenebench.answer_order_canary import build_call_plan


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    target = root / "data/manifests/simulators/answer_order_canary_plan_v1.json"
    target.write_text(
        json.dumps(build_call_plan(root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
