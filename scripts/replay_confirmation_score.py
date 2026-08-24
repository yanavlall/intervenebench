#!/usr/bin/env python3
"""Replay confirmation scoring with versioned post-freeze fallback adapters."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from intervenebench.confirmation_fallback import (
    ConfirmationFallbackObservation,
    evaluate_confirmation_eb_human_fallback,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / "scripts/score_confirmation.py"
    spec = importlib.util.spec_from_file_location("_frozen_confirmation_score", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load frozen confirmation scorer")
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    scorer.FallbackObservation = ConfirmationFallbackObservation
    scorer.evaluate_eb_human_fallback = evaluate_confirmation_eb_human_fallback
    scorer.score(authorization_path=args.authorization, output_path=args.output)


if __name__ == "__main__":
    main()
