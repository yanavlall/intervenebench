"""Verify the public research release without models or participant rows."""

from __future__ import annotations

import argparse
from pathlib import Path

from intervenebench.release_audit import verify_research_release


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deep-replay",
        action="store_true",
        help="rebuild findings from restricted local aggregate source artifacts",
    )
    arguments = parser.parse_args()
    report = verify_research_release(
        ROOT, rebuild_findings=arguments.deep_replay
    )
    print("InterveneBench research release: PASS")
    print(f"Verified files: {report['verified_file_count']}")
    print("Findings artifact: hash-bound and schema-verified")
    print(
        "Deep provenance rebuild: "
        + ("exact" if arguments.deep_replay else "not requested")
    )
    print("Model calls: 0")
    print("Participant rows accessed: 0")
    print(f"Manifest payload SHA-256: {report['manifest_payload_sha256']}")


if __name__ == "__main__":
    main()
