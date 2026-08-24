#!/usr/bin/env python3
"""Freeze the outcome-blind six-task confirmation preparation artifact."""

from __future__ import annotations

from pathlib import Path

from intervenebench.confirmation_preparation import write_confirmation_preparation


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(write_confirmation_preparation(root))

