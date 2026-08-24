"""Materialize the portable aggregate-only results explorer."""

from __future__ import annotations

from pathlib import Path

from intervenebench.public_case_study import (
    DEFAULT_PUBLIC_CASE_STUDY_PATH,
    verify_public_case_study,
)
from intervenebench.results_explorer import (
    DEFAULT_RESULTS_EXPLORER_PATH,
    build_results_explorer_html,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output = ROOT / DEFAULT_RESULTS_EXPLORER_PATH
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing explorer: {output}")
    report = verify_public_case_study(ROOT / DEFAULT_PUBLIC_CASE_STUDY_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(build_results_explorer_html(report))
    print({"path": output.relative_to(ROOT).as_posix()})


if __name__ == "__main__":
    main()
