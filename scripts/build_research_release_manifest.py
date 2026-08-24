"""Freeze hashes for the public InterveneBench research release."""

from __future__ import annotations

import argparse
from pathlib import Path

from intervenebench.protocol import freeze_envelope
from intervenebench.release_audit import (
    DEFAULT_RELEASE_MANIFEST_PATH,
    build_research_release_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replace",
        action="store_true",
        help="atomically replace the existing release manifest",
    )
    arguments = parser.parse_args()
    output = ROOT / DEFAULT_RELEASE_MANIFEST_PATH
    if output.exists() and not arguments.replace:
        raise FileExistsError(
            f"{output} already exists; pass --replace to publish a new release manifest"
        )
    target = output
    if arguments.replace:
        target = output.with_suffix(output.suffix + ".tmp")
        if target.exists():
            raise FileExistsError(target)
    digest = freeze_envelope(build_research_release_manifest(ROOT), target)
    if arguments.replace:
        target.replace(output)
    print({"path": output.relative_to(ROOT).as_posix(), "payload_sha256": digest})


if __name__ == "__main__":
    main()
