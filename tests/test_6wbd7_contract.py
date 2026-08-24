from __future__ import annotations

import hashlib
import json
import struct
import xml.etree.ElementTree as ET
import zlib
from collections import Counter
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

import pytest

from intervenebench.bounded_integer_multimodal import (
    aggregate_bounded_integer_png_predictions,
    bounded_integer_png_multimodal_prompt,
    build_bounded_integer_png_call_plan,
    choose_bounded_integer_png_checkpoint,
    parse_bounded_integer_prediction,
    validate_bounded_integer_png_multimodal_bundle,
)
from intervenebench.protocol import assert_blinded_payload


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = ROOT / "data/manifests/contracts"
PROVENANCE_PATH = ROOT / "data/manifests/stimuli/6wbd7_derived_composite_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _png_rgba(data: bytes) -> tuple[int, int, tuple[bytes, ...]]:
    """Decode the non-interlaced 8-bit RGB/RGBA PNGs used by this fixture."""

    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    position = 8
    compressed = bytearray()
    width = height = color_type = -1
    while position < len(data):
        length = struct.unpack(">I", data[position : position + 4])[0]
        kind = data[position + 4 : position + 8]
        payload = data[position + 8 : position + 8 + length]
        position += length + 12
        if kind == b"IHDR":
            width, height, depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            assert depth == 8 and color_type in {2, 6} and interlace == 0
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break
    channels = {2: 3, 6: 4}[color_type]
    stride = width * channels
    raw = zlib.decompress(bytes(compressed))
    rows: list[bytes] = []
    previous = bytearray(stride)
    cursor = 0

    def paeth(left: int, above: int, upper_left: int) -> int:
        estimate = left + above - upper_left
        distances = (
            abs(estimate - left),
            abs(estimate - above),
            abs(estimate - upper_left),
        )
        return (left, above, upper_left)[distances.index(min(distances))]

    for _ in range(height):
        filter_type = raw[cursor]
        cursor += 1
        scanline = bytearray(raw[cursor : cursor + stride])
        cursor += stride
        for index in range(stride):
            left = scanline[index - channels] if index >= channels else 0
            above = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 1:
                scanline[index] = (scanline[index] + left) & 255
            elif filter_type == 2:
                scanline[index] = (scanline[index] + above) & 255
            elif filter_type == 3:
                scanline[index] = (scanline[index] + ((left + above) // 2)) & 255
            elif filter_type == 4:
                scanline[index] = (
                    scanline[index] + paeth(left, above, upper_left)
                ) & 255
            elif filter_type != 0:
                raise AssertionError("unsupported PNG filter")
        if color_type == 2:
            rgba = bytearray()
            for index in range(0, stride, 3):
                rgba.extend(scanline[index : index + 3])
                rgba.append(255)
            rows.append(bytes(rgba))
        else:
            rows.append(bytes(scanline))
        previous = scanline
    return width, height, tuple(rows)


def _fixture_bundle(tmp_path: Path, *, status: str = "runnable") -> dict:
    arms = []
    provenance_arms = []
    for index in range(3):
        relative = f"data/derived/stimuli/fixture/arm_{index}.png"
        asset = tmp_path / relative
        asset.parent.mkdir(parents=True, exist_ok=True)
        contents = b"\x89PNG\r\n\x1a\n" + bytes([index])
        asset.write_bytes(contents)
        digest = hashlib.sha256(contents).hexdigest()
        arms.append(
            {
                "arm_id": f"arm_{index}",
                "accessible_text": f"Arm {index} exact text.",
                "asset": {"path": relative, "mime_type": "image/png", "sha256": digest},
            }
        )
        provenance_arms.append(
            {"arm_id": f"arm_{index}", "output_path": relative, "output_sha256": digest}
        )
    provenance_relative = "data/manifests/stimuli/fixture.json"
    provenance_path = tmp_path / provenance_relative
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(
            {
                "source_instrument": {"path": "source.docx", "sha256": "a" * 64},
                "arms": provenance_arms,
            }
        ),
        encoding="utf-8",
    )
    blocker = "" if status == "runnable" else "mapping remains blocked"
    return {
        "schema_version": "bounded_integer_png_multimodal_bundle.v1",
        "task_id": "fixture:task-0",
        "experiment_id": "fixture",
        "access_regime": "DESIGN_ONLY",
        "population": {"description": "United States adults", "roster_id": "fixture"},
        "arms": arms,
        "common_context": "View one arm image.",
        "outcome_question": "How strongly do you agree?",
        "response_contract": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "unit": "slider points",
            "utility_direction": "lower_is_better",
        },
        "sampling_contract": {
            "paired_across_arms": True,
            "checkpoints_per_arm": [5, 10, 20],
            "arm_mean_tolerance": 0.03,
            "margin_multiplier": 2.0,
        },
        "representation_status": {
            "status": "source_faithful_derived_composite",
            "provenance_path": provenance_relative,
            "provenance_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
            "limitation": "Typography is derived.",
        },
        "source_material_sha256": "a" * 64,
        "outcome_access": "sealed",
        "reveal_authorized": False,
        "execution_status": status,
        "scoring_blocker": blocker,
    }


def test_6wbd7_contract_is_sealed_provenance_bound_and_blocked() -> None:
    candidate = _load(CONTRACT_DIR / "6wbd7_decision_task_candidate.json")
    bundle = _load(CONTRACT_DIR / "6wbd7_blinded_bundle.json")
    provenance = _load(PROVENANCE_PATH)
    assert candidate["experiment_id"] == bundle["experiment_id"] == "6wbd7"
    assert candidate["task_id"] == bundle["task_id"]
    assert candidate["contract_status"] == bundle["execution_status"]
    assert candidate["simulation_authorized"] is False
    assert candidate["outcome_access"] == bundle["outcome_access"] == "sealed"
    assert candidate["reveal_authorized"] is bundle["reveal_authorized"] is False
    assert "frequency-bearing" in candidate["mapping_blocker"]
    assert "separate outcome-blind schema" in bundle["scoring_blocker"]
    assert bundle["representation_status"]["status"] == "source_faithful_derived_composite"
    assert hashlib.sha256(PROVENANCE_PATH.read_bytes()).hexdigest() == bundle[
        "representation_status"
    ]["provenance_sha256"]
    assert "not exact fielded browser screenshots" in provenance["representation_status"][
        "limitation"
    ]
    assert_blinded_payload(candidate)
    assert_blinded_payload(bundle)
    validate_bounded_integer_png_multimodal_bundle(bundle)
    with pytest.raises(ValueError, match="not runnable.*Primary outcome"):
        bounded_integer_png_multimodal_prompt(
            bundle, arm_id=bundle["arms"][0]["arm_id"], repository_root=ROOT
        )
    with pytest.raises(ValueError, match="not runnable.*Primary outcome"):
        build_bounded_integer_png_call_plan(bundle, model_ids=["unconfigured_vlm"])


def test_6wbd7_verbatim_text_and_chart_pixels_match_source() -> None:
    provenance = _load(PROVENANCE_PATH)
    source = ROOT / provenance["source_instrument"]["path"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == provenance[
        "source_instrument"
    ]["sha256"]
    with ZipFile(source) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
        word_namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        paragraphs = {
            "".join(
                node.text or "" for node in paragraph.iter(f"{word_namespace}t")
            ).rstrip()
            for paragraph in document.iter(f"{word_namespace}p")
        }
        for arm in provenance["arms"]:
            assert arm["heading"] in paragraphs
            for paragraph in arm["paragraphs"]:
                assert paragraph in paragraphs, paragraph
            output = ROOT / arm["output_path"]
            assert hashlib.sha256(output.read_bytes()).hexdigest() == arm["output_sha256"]
            member = arm["source_chart_member"]
            if member is None:
                assert arm["chart_placement"] is None
                continue
            source_bytes = archive.read(member)
            assert hashlib.sha256(source_bytes).hexdigest() == arm["source_chart_sha256"]
            source_width, source_height, source_rows = _png_rgba(source_bytes)
            _, _, output_rows = _png_rgba(output.read_bytes())
            placement = arm["chart_placement"]
            assert (source_width, source_height) == (
                placement["width"],
                placement["height"],
            )
            start = placement["x"] * 4
            end = (placement["x"] + source_width) * 4
            assert tuple(
                row[start:end]
                for row in output_rows[
                    placement["y"] : placement["y"] + source_height
                ]
            ) == source_rows


def test_bounded_integer_parser_prompt_and_provenance_are_strict(tmp_path: Path) -> None:
    bundle = _fixture_bundle(tmp_path)
    validate_bounded_integer_png_multimodal_bundle(bundle)
    parsed = parse_bounded_integer_prediction('{"predicted_value":50}')
    assert parsed.value == 50
    for invalid in (
        '{"predicted_value":50.0}',
        '{"predicted_value":true}',
        '{"predicted_value":101}',
        '{"predicted_value":50,"note":"x"}',
    ):
        with pytest.raises(ValueError):
            parse_bounded_integer_prediction(invalid)
    prompt = bounded_integer_png_multimodal_prompt(
        bundle, arm_id="arm_0", repository_root=tmp_path
    )
    assert prompt.asset_sha256 == (bundle["arms"][0]["asset"]["sha256"],)
    assert '{"predicted_value":50}' in prompt.text

    source_mutated = deepcopy(bundle)
    provenance_path = tmp_path / source_mutated["representation_status"]["provenance_path"]
    provenance = _load(provenance_path)
    provenance["source_instrument"]["sha256"] = "b" * 64
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    source_mutated["representation_status"]["provenance_sha256"] = hashlib.sha256(
        provenance_path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="source is not bound"):
        bounded_integer_png_multimodal_prompt(
            source_mutated, arm_id="arm_0", repository_root=tmp_path
        )

    provenance["source_instrument"]["sha256"] = "a" * 64
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    mutated = deepcopy(bundle)
    mutated["representation_status"]["provenance_sha256"] = hashlib.sha256(
        provenance_path.read_bytes()
    ).hexdigest()

    path = tmp_path / mutated["arms"][0]["asset"]["path"]
    path.write_bytes(path.read_bytes() + b"mutation")
    with pytest.raises(ValueError, match="hash or signature"):
        bounded_integer_png_multimodal_prompt(
            mutated, arm_id="arm_0", repository_root=tmp_path
        )


def test_bounded_integer_aggregation_convergence_and_call_plan(tmp_path: Path) -> None:
    bundle = _fixture_bundle(tmp_path)
    values = {"arm_0": 70, "arm_1": 30, "arm_2": 50}
    outputs = [
        {"arm_id": arm_id, "draw_index": draw, "predicted_value": value}
        for draw in range(20)
        for arm_id, value in values.items()
    ]
    assert aggregate_bounded_integer_png_predictions(
        outputs[:30], bundle=bundle, draws=10
    ) == pytest.approx({"arm_0": 0.3, "arm_1": 0.7, "arm_2": 0.5})
    with pytest.raises(ValueError, match="complete and paired"):
        aggregate_bounded_integer_png_predictions(
            outputs[:29], bundle=bundle, draws=10
        )
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_bounded_integer_png_predictions(
            [*outputs[:30], outputs[0]], bundle=bundle, draws=10
        )
    decision = choose_bounded_integer_png_checkpoint(outputs, bundle=bundle)
    assert decision.converged is True
    assert decision.sample_count == 10
    assert decision.winner_arm_id == "arm_1"

    plan = build_bounded_integer_png_call_plan(
        bundle, model_ids=["vlm_primary", "vlm_robustness"]
    )
    assert plan["planned_call_count"] == 2 * 3 * 20
    assert len({call["call_id"] for call in plan["calls"]}) == 120
    assert all(value is False for value in plan["authority"].values())
    paired = Counter(
        (call["model_id"], call["paired_draw_id"])
        for call in plan["calls"]
    )
    assert set(paired.values()) == {3}


def test_bounded_integer_bundle_rejects_leakage_paths_and_unpaired_sampling(
    tmp_path: Path,
) -> None:
    bundle = _fixture_bundle(tmp_path)
    leaked = deepcopy(bundle)
    leaked["arms"][0]["metadata"] = {"human_winner": "arm_0"}
    with pytest.raises(ValueError, match="forbidden"):
        validate_bounded_integer_png_multimodal_bundle(leaked)

    escaped = deepcopy(bundle)
    escaped["arms"][0]["asset"]["path"] = "data/derived/stimuli/../../outside.png"
    with pytest.raises(ValueError, match="asset declaration"):
        validate_bounded_integer_png_multimodal_bundle(escaped)

    unpaired = deepcopy(bundle)
    unpaired["sampling_contract"]["paired_across_arms"] = False
    with pytest.raises(ValueError, match="pair arms"):
        validate_bounded_integer_png_multimodal_bundle(unpaired)
