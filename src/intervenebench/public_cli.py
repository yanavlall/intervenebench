"""Dependency-light commands for the aggregate-only public research release."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .public_case_study import (
    DEFAULT_PUBLIC_CASE_STUDY_PATH,
    render_public_case_study,
    verify_public_case_study,
)
from .release_audit import verify_research_release
from .research_findings_release import (
    DEFAULT_RESEARCH_FINDINGS_PATH,
    render_research_findings_markdown,
    verify_research_findings,
)


def _root(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"repository root is not a directory: {path}")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="intervenebench-release",
        description=(
            "Inspect and verify the aggregate-only InterveneBench research release "
            "without model calls or participant-row access."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify", help="verify release integrity")
    verify.add_argument("--root", type=_root, default=Path.cwd())
    verify.add_argument(
        "--deep-replay",
        action="store_true",
        help="also rebuild findings from restricted local aggregate source artifacts",
    )

    case = commands.add_parser("case-study", help="render scoped release decisions")
    case.add_argument("--root", type=_root, default=Path.cwd())

    findings = commands.add_parser("findings", help="render authoritative findings")
    findings.add_argument("--root", type=_root, default=Path.cwd())
    return parser


def _verify(root: Path, *, deep_replay: bool) -> str:
    report = verify_research_release(root, rebuild_findings=deep_replay)
    lines = [
        "InterveneBench portable research release: PASS",
        f"Verified files: {report['verified_file_count']}",
        "Findings artifact: hash-bound and schema-verified",
        "Public case study: hash-bound and schema-verified",
        (
            "Deep provenance rebuild: exact"
            if deep_replay
            else "Deep provenance rebuild: not requested"
        ),
        "Model calls: 0",
        "Participant rows accessed: 0",
        f"Manifest payload SHA-256: {report['manifest_payload_sha256']}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "verify":
        print(_verify(arguments.root, deep_replay=arguments.deep_replay), end="")
        return
    if arguments.command == "case-study":
        report = verify_public_case_study(
            arguments.root / DEFAULT_PUBLIC_CASE_STUDY_PATH
        )
        print(render_public_case_study(report), end="")
        return
    payload = verify_research_findings(
        arguments.root / DEFAULT_RESEARCH_FINDINGS_PATH
    )
    print(render_research_findings_markdown(payload), end="")


if __name__ == "__main__":
    main()
