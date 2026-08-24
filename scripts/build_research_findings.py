"""Build the public aggregate-only research findings and report."""

from __future__ import annotations

from pathlib import Path

from intervenebench.protocol import freeze_envelope
from intervenebench.research_findings_release import (
    DEFAULT_RESEARCH_FINDINGS_PATH,
    DEFAULT_RESEARCH_FINDINGS_REPORT_PATH,
    build_research_findings_payload,
    render_research_findings_markdown,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    payload = build_research_findings_payload(ROOT)
    output = ROOT / DEFAULT_RESEARCH_FINDINGS_PATH
    digest = freeze_envelope(payload, output)
    report = ROOT / DEFAULT_RESEARCH_FINDINGS_REPORT_PATH
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as stream:
        stream.write(render_research_findings_markdown(payload))
    print(
        {
            "artifact": output.relative_to(ROOT).as_posix(),
            "report": report.relative_to(ROOT).as_posix(),
            "payload_sha256": digest,
        }
    )


if __name__ == "__main__":
    main()
