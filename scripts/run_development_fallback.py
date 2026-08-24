"""Run the frozen common development-only fallback evaluation."""

from pathlib import Path

from intervenebench.development_fallback import (
    DEFAULT_FALLBACK_PATH,
    run_development_fallback,
)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    run_development_fallback(root, root / DEFAULT_FALLBACK_PATH)
