#!/usr/bin/env python3
"""Freeze the zero-authority Modal confirmation execution environment."""

from __future__ import annotations

from pathlib import Path

from intervenebench.confirmation_execution import write_confirmation_execution_freeze


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(write_confirmation_execution_freeze(root))

