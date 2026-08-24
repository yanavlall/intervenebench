"""Materialize the deterministic balanced full-action logical call plan."""

from __future__ import annotations

import json
from pathlib import Path

from intervenebench.balanced_forced_choice import build_full_action_plan


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    target = (
        root / "data/manifests/simulators/balanced_full_action_plan_v1.json"
    )
    target.write_text(
        json.dumps(build_full_action_plan(root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
