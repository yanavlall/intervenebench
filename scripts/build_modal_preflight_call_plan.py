"""Materialize the frozen, outcome-blind 40-call Modal parser preflight plan."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from intervenebench.protocol import payload_hash
from intervenebench.simulators import ordinal_probability_prompt, ordinal_variant_contract


EXPERIMENT_IDS = ("5vm8g", "xc4yq", "de5hx", "turagaS11", "wallaceS12")
MODEL_IDS = (
    "qwen3_8b_generic",
    "qwen3_14b_generic",
    "qwen2_5_14b_generic",
    "socrates_qwen2_5_14b_sft",
)
BASE_SEED = 21_026_000
JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["probabilities"],
    "properties": {
        "probabilities": {
            "type": "object",
            "additionalProperties": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        }
    },
}


def build(root: Path) -> dict:
    schema_hash = payload_hash(JSON_SCHEMA)
    calls: list[dict] = []
    seed_index = 0
    for model_id in MODEL_IDS:
        for experiment_id in EXPERIMENT_IDS:
            bundle_path = (
                root
                / "data"
                / "manifests"
                / "contracts"
                / f"{experiment_id}_blinded_bundle.json"
            )
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            for arm in (bundle["arms"][0], bundle["arms"][-1]):
                arm_id = arm["arm_id"]
                variant_id = ordinal_variant_contract(bundle, arm_id=arm_id)[0][0]
                prompt = ordinal_probability_prompt(
                    bundle, arm_id=arm_id, variant_id=variant_id
                )
                calls.append(
                    {
                        "call_id": f"{model_id}--{experiment_id}--{arm_id}",
                        "model_id": model_id,
                        "experiment_id": experiment_id,
                        "bundle_payload_sha256": payload_hash(bundle),
                        "arm_id": arm_id,
                        "variant_id": variant_id,
                        "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
                        "json_schema_sha256": schema_hash,
                        "parser_id": "parse_ordinal_distribution.v1",
                        "seed": BASE_SEED + seed_index,
                        "temperature": 0.2,
                        "top_p": 0.95,
                        "maximum_output_tokens": 256,
                        "artifact_relative_path": (
                            f"{model_id}/{experiment_id}/{arm_id}.json"
                        ),
                    }
                )
                seed_index += 1
    return {
        "schema_version": "modal_preflight_call_plan.v1",
        "plan_id": "intervenebench-modal-discovery-parser-preflight-v1",
        "freeze_date": "2026-08-13",
        "status": "frozen_nonexecuting_zero_authority",
        "experiment_ids": list(EXPERIMENT_IDS),
        "model_ids": list(MODEL_IDS),
        "selection_rule": (
            "first_and_last_source_order_arm_per_experiment; "
            "first_declared_variant when nuisance variants exist"
        ),
        "calls": calls,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = (
        root / "data/manifests/simulators/modal_preflight_call_plan_v1.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build(root), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
