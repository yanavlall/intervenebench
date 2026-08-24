from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from intervenebench.release_audit import (
    DEFAULT_RELEASE_MANIFEST_PATH,
    DEFAULT_TRACKED_PATHS,
    verify_research_release,
)


ROOT = Path(__file__).resolve().parents[1]


def _copy_portable_release(destination: Path) -> None:
    paths = (*DEFAULT_TRACKED_PATHS, DEFAULT_RELEASE_MANIFEST_PATH)
    for relative in paths:
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def test_portable_release_verifies_without_local_artifacts(tmp_path: Path) -> None:
    _copy_portable_release(tmp_path)

    report = verify_research_release(tmp_path, rebuild_findings=False)

    assert not (tmp_path / "artifacts").exists()
    assert report["status"] == "pass"
    assert report["verification_mode"] == "portable_integrity"
    assert report["findings_payload_verified"] is True
    assert report["public_case_study_verified"] is True
    assert report["findings_payload_matches_rebuild"] is None
    assert report["model_calls_made"] == 0
    assert report["participant_rows_accessed"] == 0


def test_public_cli_runs_from_portable_release(tmp_path: Path) -> None:
    _copy_portable_release(tmp_path)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path / "src")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "intervenebench.public_cli",
            "verify",
            "--root",
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "InterveneBench portable research release: PASS" in completed.stdout
    assert "Participant rows accessed: 0" in completed.stdout
    assert "Deep provenance rebuild: not requested" in completed.stdout


def test_public_cli_renders_case_study_without_local_artifacts(tmp_path: Path) -> None:
    _copy_portable_release(tmp_path)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path / "src")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "intervenebench.public_cli",
            "case-study",
            "--root",
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "InterveneBench prospective case study" in completed.stdout
    assert "Exact human-best intervention: 3/6" in completed.stdout
    assert "Candidate screening: LIMITED RESEARCH USE" in completed.stdout
