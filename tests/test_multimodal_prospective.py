from __future__ import annotations

from pathlib import Path

from intervenebench.multimodal_prospective import (
    EXPERIMENT_IDS,
    MODEL_IDS,
    build_multimodal_prospective_plan,
    multimodal_forced_choice_prompt,
)
from intervenebench.balanced_forced_choice import read_json_object


ROOT = Path(__file__).resolve().parents[1]


def test_prospective_multimodal_plan_is_complete_and_zero_authority() -> None:
    plan = build_multimodal_prospective_plan(ROOT)
    assert plan["experiment_ids"] == list(EXPERIMENT_IDS)
    assert [model["model_id"] for model in plan["model_specs"]] == list(MODEL_IDS)
    assert plan["logical_call_count"] == 54
    assert plan["vision_call_count"] == 36
    assert plan["text_ablation_call_count"] == 18
    assert len({call["call_id"] for call in plan["calls"]}) == 54
    assert not any(plan["authority"].values())
    assert plan["selection_boundary"]["target_human_outcomes_accessed"] is False


def test_every_model_covers_every_arm_in_both_orders() -> None:
    plan = build_multimodal_prospective_plan(ROOT)
    observed = {
        (call["model_id"], call["experiment_id"], call["arm_id"], call["option_order"])
        for call in plan["calls"]
    }
    expected = set()
    for experiment_id in EXPERIMENT_IDS:
        bundle = read_json_object(
            ROOT / f"data/manifests/contracts/{experiment_id}_blinded_bundle.json"
        )
        for model_id in MODEL_IDS:
            for arm in bundle["arms"]:
                for option_order in ("source", "reverse"):
                    expected.add((model_id, experiment_id, arm["arm_id"], option_order))
    assert observed == expected


def test_vision_calls_bind_assets_and_text_ablation_does_not() -> None:
    plan = build_multimodal_prospective_plan(ROOT)
    for call in plan["calls"]:
        if call["modality"] == "exact_png_vision":
            assert call["asset_path"].startswith("data/derived/stimuli/")
            assert len(call["asset_sha256"]) == 64
        else:
            assert call["asset_path"] is None
            assert call["asset_sha256"] is None


def test_source_and_reverse_change_code_mapping_without_changing_values() -> None:
    bundle = read_json_object(
        ROOT / "data/manifests/contracts/nj5dx_blinded_bundle.json"
    )
    source = multimodal_forced_choice_prompt(
        bundle,
        arm_id="lower_class_disadvantage_frame",
        repository_root=ROOT,
        option_order="source",
        include_exact_image=True,
    )
    reverse = multimodal_forced_choice_prompt(
        bundle,
        arm_id="lower_class_disadvantage_frame",
        repository_root=ROOT,
        option_order="reverse",
        include_exact_image=True,
    )
    assert "A: Strongly agree" in source.text
    assert "A: Strongly disagree" in reverse.text
    assert source.asset_sha256 == reverse.asset_sha256


def test_es4xw_text_ablation_does_not_smuggle_visual_identity_labels() -> None:
    bundle = read_json_object(
        ROOT / "data/manifests/contracts/es4xw_blinded_bundle.json"
    )
    prompts = [
        multimodal_forced_choice_prompt(
            bundle,
            arm_id=arm["arm_id"],
            repository_root=ROOT,
            option_order="source",
            include_exact_image=False,
        ).text
        for arm in bundle["arms"]
    ]
    assert len(set(prompts)) == 1
    assert all("protected identity" in prompt for prompt in prompts)
