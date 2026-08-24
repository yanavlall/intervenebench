"""Freeze the common nine-task development fallback method."""

from pathlib import Path

from intervenebench.development_fallback import write_development_fallback_protocol


if __name__ == "__main__":
    write_development_fallback_protocol(Path(__file__).resolve().parents[1])

