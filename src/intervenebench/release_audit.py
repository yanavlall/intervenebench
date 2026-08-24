"""One-command, fail-closed verification for the public research release."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from .protocol import payload_hash, verify_envelope
from .public_case_study import verify_public_case_study
from .research_findings_release import (
    DEFAULT_RESEARCH_FINDINGS_PATH,
    build_research_findings_payload,
    verify_research_findings,
)


DEFAULT_RELEASE_MANIFEST_PATH = Path(
    "data/public/research_release_manifest_v1.json"
)
DEFAULT_TRACKED_PATHS = (
    Path("pyproject.toml"),
    Path("README.md"),
    Path("docs/README.md"),
    Path("docs/PORTFOLIO_BRIEF.md"),
    Path("docs/SIMILE_EVALS_CASE_STUDY.md"),
    Path("docs/decisions/role_focused_evaluation_program_v1.md"),
    Path("docs/reports/research_findings_v1.md"),
    Path("docs/reports/figures/fallback_failure_mechanism_v1.svg"),
    Path("docs/results/index.html"),
    Path("data/manifests/research/role_focused_evaluation_program_v1.json"),
    Path("data/public/confirmation_case_study_v1.json"),
    DEFAULT_RESEARCH_FINDINGS_PATH,
    Path("src/intervenebench/__init__.py"),
    Path("src/intervenebench/schemas.py"),
    Path("src/intervenebench/protocol.py"),
    Path("src/intervenebench/release_decision.py"),
    Path("src/intervenebench/public_case_study.py"),
    Path("src/intervenebench/research_findings_release.py"),
    Path("src/intervenebench/release_audit.py"),
    Path("src/intervenebench/public_cli.py"),
    Path("scripts/build_research_findings.py"),
    Path("scripts/build_research_release_manifest.py"),
    Path("scripts/verify_research_release.py"),
    Path("tests/test_research_findings_release.py"),
    Path("tests/test_public_release_portability.py"),
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_path(root: Path, raw: str, *, label: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path escapes repository")
    return root / relative


def _verify_portable_public_chain(
    root: Path, findings: dict[str, Any]
) -> dict[str, Any]:
    """Verify the public aggregate chain without restricted local artifacts."""

    record = findings.get("provenance", {}).get("public_confirmation")
    if not isinstance(record, dict) or set(record) != {
        "path",
        "file_sha256",
        "payload_sha256",
    }:
        raise ValueError("research findings public-confirmation provenance is invalid")
    case_path = _repository_path(
        root, record["path"], label="public confirmation"
    )
    if _file_sha256(case_path) != record["file_sha256"]:
        raise ValueError("public confirmation file hash drifted")
    case_report = verify_public_case_study(case_path)
    case_payload = case_report["payload"]
    if payload_hash(case_payload) != record["payload_sha256"]:
        raise ValueError("public confirmation payload hash drifted")

    case_provenance = case_payload.get("provenance", {})
    role_path = _repository_path(
        root,
        case_provenance.get("role_program_path", ""),
        label="role program",
    )
    if _file_sha256(role_path) != case_provenance.get(
        "role_program_file_sha256"
    ):
        raise ValueError("role-focused program file hash drifted")
    role_program = json.loads(role_path.read_text(encoding="utf-8"))
    authority = role_program.get("authority", {})
    if (
        role_program.get("schema_version")
        != "intervenebench.role_focused_program.v1"
        or role_program.get("status") != "evaluation_product_build_frozen"
        or authority.get("authorized_spend_usd") != 0
        or authority.get("model_calls_authorized") is not False
        or authority.get("human_outcome_reveal_authorized") is not False
        or authority.get("participant_row_access_authorized") is not False
        or authority.get("schema_only_mapping_authorized") is not False
    ):
        raise ValueError("role-focused program identity or authority drifted")
    return {
        "public_case_study_verified": True,
        "public_case_study_payload_sha256": payload_hash(case_payload),
        "role_program_verified": True,
    }


def build_research_release_manifest(
    root: Path, *, tracked_paths: Iterable[Path] = DEFAULT_TRACKED_PATHS
) -> dict[str, Any]:
    paths = tuple(tracked_paths)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("release manifest paths must be unique and non-empty")
    files = []
    for relative in paths:
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("release manifest paths must stay inside the repository")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(
            {"path": relative.as_posix(), "file_sha256": _file_sha256(path)}
        )
    findings_payload_sha256 = None
    if DEFAULT_RESEARCH_FINDINGS_PATH in paths:
        findings_payload_sha256 = payload_hash(
            verify_research_findings(root / DEFAULT_RESEARCH_FINDINGS_PATH)
        )
    return {
        "schema_version": "intervenebench.research_release_manifest.v1",
        "status": "frozen_public_research_release",
        "files": files,
        "file_count": len(files),
        "findings_payload_sha256": findings_payload_sha256,
        "model_calls_authorized": False,
        "participant_row_access_authorized": False,
        "automatic_next_stage_authorized": False,
    }


def verify_research_release(
    root: Path,
    *,
    manifest_path: Path | None = None,
    rebuild_findings: bool = True,
) -> dict[str, Any]:
    path = manifest_path or root / DEFAULT_RELEASE_MANIFEST_PATH
    manifest = verify_envelope(path)
    if (
        manifest.get("schema_version")
        != "intervenebench.research_release_manifest.v1"
        or manifest.get("status") != "frozen_public_research_release"
        or manifest.get("model_calls_authorized") is not False
        or manifest.get("participant_row_access_authorized") is not False
        or manifest.get("automatic_next_stage_authorized") is not False
    ):
        raise ValueError("research release manifest identity or authority drifted")
    files = manifest.get("files")
    if not isinstance(files, list) or manifest.get("file_count") != len(files):
        raise ValueError("research release manifest file count is invalid")
    seen: set[str] = set()
    for record in files:
        if set(record) != {"path", "file_sha256"}:
            raise ValueError("malformed research release file record")
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("research release path escapes repository")
        if record["path"] in seen:
            raise ValueError("duplicate research release path")
        seen.add(record["path"])
        if _file_sha256(root / relative) != record["file_sha256"]:
            raise ValueError(f"file hash drifted: {record['path']}")

    findings_verified = False
    public_chain = {
        "public_case_study_verified": False,
        "public_case_study_payload_sha256": None,
        "role_program_verified": False,
    }
    checked: dict[str, Any] | None = None
    if DEFAULT_RESEARCH_FINDINGS_PATH.as_posix() in seen:
        checked = verify_research_findings(root / DEFAULT_RESEARCH_FINDINGS_PATH)
        if manifest.get("findings_payload_sha256") != payload_hash(checked):
            raise ValueError("release manifest findings payload binding drifted")
        findings_verified = True
        public_chain = _verify_portable_public_chain(root, checked)

    matches: bool | None = None
    if rebuild_findings:
        if checked is None:
            raise ValueError("deep replay requires the tracked research findings")
        rebuilt = build_research_findings_payload(root)
        matches = checked == rebuilt
        if not matches:
            raise ValueError("checked-in findings do not match deterministic rebuild")

    return {
        "status": "pass",
        "verification_mode": (
            "deep_provenance_replay" if rebuild_findings else "portable_integrity"
        ),
        "verified_file_count": len(files),
        "findings_payload_verified": findings_verified,
        "findings_payload_matches_rebuild": matches,
        "manifest_payload_sha256": payload_hash(manifest),
        "model_calls_made": 0,
        "participant_rows_accessed": 0,
        **public_chain,
    }
