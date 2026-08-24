#!/usr/bin/env python3
"""Freeze confirmation scoring and fallback before opening target outcomes."""

from pathlib import Path

from intervenebench.confirmation_scoring import write_confirmation_scoring_protocol


if __name__ == "__main__":
    write_confirmation_scoring_protocol(Path(__file__).resolve().parents[1])
