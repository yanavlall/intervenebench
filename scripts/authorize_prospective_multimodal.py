"""Create staged, hash-bound authorizations for prospective multimodal inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from intervenebench.balanced_forced_choice import read_json_object
from intervenebench.multimodal_freeze import (
    build_cache_authorization,
    build_execution_authorization,
    build_materialization_authorization,
    verify_prospective_multimodal_freeze,
)
from intervenebench.protocol import freeze_envelope, verify_envelope


ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / "configs/simulators/prospective_multimodal_v4.json"
PLAN_PATH = ROOT / "data/manifests/simulators/prospective_multimodal_plan_v1.json"


def _common() -> tuple[dict, dict]:
    freeze = read_json_object(FREEZE_PATH)
    plan = read_json_object(PLAN_PATH)
    verify_prospective_multimodal_freeze(ROOT, freeze)
    return freeze, plan


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    materialize = sub.add_parser("materialize")
    materialize.add_argument("--output", type=Path, required=True)
    cache = sub.add_parser("cache")
    cache.add_argument("--materialization", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    execute = sub.add_parser("execute")
    execute.add_argument("--cache-manifest", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze, plan = _common()
    if args.stage == "materialize":
        payload = build_materialization_authorization(freeze=freeze, plan=plan)
    elif args.stage == "cache":
        materialization = verify_envelope(
            args.materialization, require_blinded=True
        )
        payload = build_cache_authorization(
            freeze=freeze,
            plan=plan,
            modal_image_id=materialization["modal_image_id"],
        )
    else:
        cache_manifest = verify_envelope(args.cache_manifest, require_blinded=True)
        payload = build_execution_authorization(
            freeze=freeze,
            plan=plan,
            modal_image_id=cache_manifest["modal_image_id"],
            cache_hashes=cache_manifest["cache_attestation_sha256_by_model"],
        )
    freeze_envelope(payload, args.output, require_blinded=True)


if __name__ == "__main__":
    main()
