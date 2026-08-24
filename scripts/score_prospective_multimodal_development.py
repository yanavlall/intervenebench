"""Run the separately authorized three-task development reveal and scoring."""

from __future__ import annotations

from pathlib import Path

from intervenebench.prospective_development_score import (
    DEFAULT_SCORE_PATH,
    score_prospective_development,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    score_prospective_development(root, output_path=root / DEFAULT_SCORE_PATH)


if __name__ == "__main__":
    main()
