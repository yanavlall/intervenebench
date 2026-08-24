#!/usr/bin/env python3
"""Freeze all planned and conditional confirmation call definitions."""

from __future__ import annotations

from pathlib import Path

from intervenebench.confirmation_calls import write_confirmation_call_plan


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(write_confirmation_call_plan(root))

