from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from intervenebench.simulators import (
    aggregate_ordinal_png_multimodal_predictions,
    ordinal_png_multimodal_prompt,
    validate_ordinal_png_multimodal_bundle,
)


def _write_png(root: Path, name: str, suffix: bytes) -> tuple[str, str]:
    relative = f"data/derived/stimuli/{name}.png"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = b"\x89PNG\r\n\x1a\n" + suffix
    path.write_bytes(contents)
    return relative, sha256(contents).hexdigest()


def _bundle(root: Path, *, arm_count: int = 2) -> dict:
    arms = []
    for index in range(arm_count):
        path, digest = _write_png(root, f"arm_{index}", bytes([index]))
        arms.append(
            {
                "arm_id": f"arm_{index}",
                "accessible_text": f"Exact image for arm {index}.",
                "asset": {
                    "path": path,
                    "mime_type": "image/png",
                    "sha256": digest,
                },
            }
        )
    return {
        "schema_version": "ordinal_png_multimodal_bundle.v1",
        "task_id": "fixture:task-0",
        "experiment_id": "fixture",
        "access_regime": "DESIGN_ONLY",
        "population": {
            "description": "Adults in the United States",
            "roster_id": "aggregate-us-adult-v1",
        },
        "arms": arms,
        "common_context": "Participants view exactly one fielded image.",
        "outcome_question": "How strongly do you agree?",
        "response_options": [
            {"value": 1, "label": "Disagree", "normalized_utility": 0.0},
            {"value": 2, "label": "Neutral", "normalized_utility": 0.5},
            {"value": 3, "label": "Agree", "normalized_utility": 1.0},
        ],
        "source_material_sha256": "a" * 64,
        "outcome_access": "sealed",
        "reveal_authorized": False,
    }


def test_ordinal_png_prompt_verifies_exact_asset_and_hash(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    validate_ordinal_png_multimodal_bundle(bundle)
    prompt = ordinal_png_multimodal_prompt(
        bundle, arm_id="arm_0", repository_root=tmp_path
    )
    assert prompt.asset_paths == (
        str((tmp_path / "data/derived/stimuli/arm_0.png").resolve()),
    )
    assert prompt.asset_sha256 == (bundle["arms"][0]["asset"]["sha256"],)
    assert "attached exact fielded image" in prompt.text
    assert '"1":NUMBER' in prompt.text and '"3":NUMBER' in prompt.text


def test_ordinal_png_prompt_reverses_display_order_without_relabeling_values(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    prompt = ordinal_png_multimodal_prompt(
        bundle,
        arm_id="arm_0",
        repository_root=tmp_path,
        option_order="reverse",
    )
    assert prompt.text.index('"3":NUMBER') < prompt.text.index('"1":NUMBER')
    assert prompt.text.index("3=Agree") < prompt.text.index("1=Disagree")
    assert "Output keys remain the original response values" in prompt.text


def test_ordinal_png_aggregation_requires_complete_paired_draws(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    outputs = [
        {
            "arm_id": "arm_0",
            "draw_index": 0,
            "probabilities": {"1": 1.0, "2": 0.0, "3": 0.0},
        },
        {
            "arm_id": "arm_0",
            "draw_index": 1,
            "probabilities": {"1": 0.0, "2": 0.0, "3": 1.0},
        },
        {
            "arm_id": "arm_1",
            "draw_index": 0,
            "probabilities": {"1": 0.0, "2": 1.0, "3": 0.0},
        },
        {
            "arm_id": "arm_1",
            "draw_index": 1,
            "probabilities": {"1": 0.0, "2": 1.0, "3": 0.0},
        },
    ]
    assert aggregate_ordinal_png_multimodal_predictions(
        outputs, bundle=bundle, draws=2
    ) == pytest.approx({"arm_0": 0.5, "arm_1": 0.5})

    with pytest.raises(ValueError, match="complete and paired"):
        aggregate_ordinal_png_multimodal_predictions(
            outputs[:-1], bundle=bundle, draws=2
        )
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_ordinal_png_multimodal_predictions(
            [*outputs, outputs[0]], bundle=bundle, draws=2
        )


def test_ordinal_png_rejects_path_escape_and_asset_mutation(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    escaped = deepcopy(bundle)
    escaped["arms"][0]["asset"]["path"] = "data/derived/stimuli/../../outside.png"
    with pytest.raises(ValueError, match="asset declaration"):
        validate_ordinal_png_multimodal_bundle(escaped)

    path = tmp_path / bundle["arms"][0]["asset"]["path"]
    path.write_bytes(path.read_bytes() + b"mutated")
    with pytest.raises(ValueError, match="hash does not match"):
        ordinal_png_multimodal_prompt(
            bundle, arm_id="arm_0", repository_root=tmp_path
        )


def test_ordinal_png_rejects_symlink_escape(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    outside = tmp_path.parent / "outside-ordinal-image.png"
    outside.write_bytes(b"\x89PNG\r\n\x1a\noutside")
    link = tmp_path / bundle["arms"][0]["asset"]["path"]
    link.unlink()
    link.symlink_to(outside)
    bundle["arms"][0]["asset"]["sha256"] = sha256(outside.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="outside the repository"):
        ordinal_png_multimodal_prompt(
            bundle, arm_id="arm_0", repository_root=tmp_path
        )


def test_ordinal_png_rejects_recursive_result_leakage_and_open_access(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    leaked = deepcopy(bundle)
    leaked["arms"][0]["asset"]["metadata"] = {
        "human_arm_means": {"arm_0": 0.9}
    }
    with pytest.raises(ValueError, match="forbidden"):
        validate_ordinal_png_multimodal_bundle(leaked)

    open_access = deepcopy(bundle)
    open_access["outcome_access"] = "revealed"
    with pytest.raises(ValueError, match="outcome sealed"):
        validate_ordinal_png_multimodal_bundle(open_access)

    with_nuisance = deepcopy(bundle)
    with_nuisance["nuisance_contract"] = {"levels": ["invented"]}
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_ordinal_png_multimodal_bundle(with_nuisance)

    leaked_output = [
        {
            "arm_id": "arm_0",
            "draw_index": 0,
            "probabilities": {"1": 1.0, "2": 0.0, "3": 0.0},
            "metadata": {"human_winner": "arm_0"},
        }
    ]
    with pytest.raises(ValueError, match="forbidden"):
        aggregate_ordinal_png_multimodal_predictions(
            leaked_output, bundle=bundle, draws=1
        )


def test_ordinal_png_supports_two_to_six_arms_only(tmp_path: Path) -> None:
    validate_ordinal_png_multimodal_bundle(_bundle(tmp_path, arm_count=6))
    with pytest.raises(ValueError, match="two to six"):
        validate_ordinal_png_multimodal_bundle(_bundle(tmp_path, arm_count=7))
