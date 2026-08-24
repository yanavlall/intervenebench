"""Outcome-blind preparation for the six-experiment confirmation panel.

This module validates contracts, model exposure, exact inference counts, public
stimulus assets, and the frozen development-only classical model.  It never
loads confirmation response data and it grants no execution or reveal authority.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from .baselines import HashedRidgeEffectModel
from .classical_development import (
    DEFAULT_CLASSICAL_MODEL_PATH,
    build_contrast_features,
    verify_classical_model,
)
from .model_exposure import checkpoint_compatibility, read_study_mapping
from .protocol import (
    assert_blinded_payload,
    freeze_envelope,
    payload_hash,
    verify_envelope,
)
from .sequence_contracts import (
    build_klar_sequence_bundle,
    build_shannon_sequence_bundle,
    build_z358z_sequence_bundle,
)
from .simulators import (
    validate_bounded_multimodal_bundle,
    validate_continuous_blinded_bundle,
    validate_ordinal_blinded_bundle,
    validate_sequence_blinded_bundle,
)


CONFIRMATION_IDS = (
    "tcg8p",
    "pb2rr",
    "z358z",
    "ShannonS2",
    "Blair1131",
    "KlarS44",
)
BOUNDED_CONFIRMATION_IDS = CONFIRMATION_IDS[1:]
BASE_CALLS = 1152
PERTURBATION_CALLS = 312
RESERVE_CALLS = 236
PLANNED_CALLS = BASE_CALLS + PERTURBATION_CALLS
MAXIMUM_ATTEMPTS = PLANNED_CALLS + RESERVE_CALLS

DEFAULT_CONFIRMATION_PROTOCOL_PATH = Path(
    "data/manifests/research/confirmation_inference_protocol_v1.json"
)
DEFAULT_CONFIRMATION_PREPARATION_PATH = Path(
    "artifacts/confirmation/confirmation_preparation_v1.json"
)
CHECKPOINT_MAPPING_PATH = Path(
    "data/raw/socsci210/048481111a4425ed83dc0eacf15f8431f252b21a/"
    "metadata/participant_mapping.json"
)

_AUTHORITY_FIELDS = frozenset(
    {
        "automatic_next_stage_authorized",
        "confirmation_outcome_reveal_authorized",
        "fine_tuning_authorized",
        "modal_compute_authorized",
        "model_download_authorized",
        "paid_inference_authorized",
        "participant_row_access_authorized",
        "participant_row_serialization_authorized",
    }
)
_TEXT_CONFIG_PATH = Path("configs/simulators/balanced_full_action_v1.json")
_MULTIMODAL_CONFIG_PATH = Path("configs/simulators/prospective_multimodal_v4.json")
_EXPECTED_MODELS = {
    "qwen3_8b_generic": (
        "Qwen/Qwen3-8B",
        "b968826d9c46dd6066d109eabc6255188de91218",
    ),
    "qwen3_14b_generic": (
        "Qwen/Qwen3-14B",
        "40c069824f4251a91eefaf281ebe4c544efd3e18",
    ),
    "qwen2_5_14b_generic": (
        "Qwen/Qwen2.5-14B-Instruct",
        "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
    ),
    "socrates_qwen2_5_14b_sft": (
        "socratesft/socrates-qwen2.5-14b-sft",
        "6666d399b373dd37a2691a921550732f2fdddb20",
    ),
    "qwen3_vl_8b_primary": (
        "Qwen/Qwen3-VL-8B-Instruct",
        "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    ),
    "qwen2_5_vl_7b_comparator": (
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    ),
    "qwen3_8b_text_ablation": (
        "Qwen/Qwen3-8B",
        "b968826d9c46dd6066d109eabc6255188de91218",
    ),
}
_EXPECTED_TASK_COUNTS = {
    "tcg8p": (3, 20, 4, 240, 60, 60),
    "pb2rr": (2, 32, 3, 192, 64, 0),
    "z358z": (2, 16, 3, 96, 32, 32),
    "ShannonS2": (6, 16, 4, 384, 96, 96),
    "Blair1131": (3, 4, 4, 48, 12, 0),
    "KlarS44": (3, 16, 4, 192, 48, 48),
}
_PB2RR_PNGS = {
    "iphone_growth_control_article": (
        "data/derived/stimuli/pb2rr/iphone_growth_control_article.png",
        "9479a84d348522ce3c99b340d632a937f7c0874a0bb7eda9b57711b6d0e26a18",
    ),
    "hispanic_population_growth_article": (
        "data/derived/stimuli/pb2rr/hispanic_population_growth_article.png",
        "8d40187401625ca591b0f02e93fd5b14187908cad0ca4210cc38c8aaadd68a3e",
    ),
}


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(root: Path, experiment_id: str) -> tuple[Path, Path]:
    contracts = root / "data/manifests/contracts"
    if experiment_id == "tcg8p":
        return (
            contracts / "tcg8p_continuous_task_candidate.json",
            contracts / "tcg8p_continuous_blinded_bundle.json",
        )
    return (
        contracts / f"{experiment_id}_decision_task_candidate.json",
        contracts / f"{experiment_id}_blinded_bundle.json",
    )


def _model_catalog(root: Path) -> dict[str, dict[str, Any]]:
    text_config = _read_object(root / _TEXT_CONFIG_PATH)
    multimodal_config = _read_object(root / _MULTIMODAL_CONFIG_PATH)
    for config in (text_config, multimodal_config):
        authority = config.get("authority")
        if not isinstance(authority, Mapping) or any(
            value is not False for value in authority.values()
        ):
            raise ValueError("parent model freeze must retain zero authority")
    catalog: dict[str, dict[str, Any]] = {}
    for config in (text_config, multimodal_config):
        models = config.get("models")
        if not isinstance(models, list):
            raise ValueError("parent model freeze is missing models")
        for raw in models:
            if not isinstance(raw, Mapping):
                raise ValueError("model declarations must be objects")
            model_id = str(raw.get("model_id", ""))
            if model_id in catalog and catalog[model_id] != dict(raw):
                raise ValueError(f"conflicting declarations for model {model_id}")
            catalog[model_id] = dict(raw)
    for model_id, (repository, commit) in _EXPECTED_MODELS.items():
        model = catalog.get(model_id)
        if model is None:
            raise ValueError(f"missing frozen model {model_id}")
        if model.get("hf_repository") != repository or model.get(
            "checkpoint_commit"
        ) != commit:
            raise ValueError(f"checkpoint identity drifted for {model_id}")
    return {model_id: catalog[model_id] for model_id in _EXPECTED_MODELS}


def _validate_contracts(root: Path) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    expected_sequences = {
        "z358z": build_z358z_sequence_bundle(),
        "ShannonS2": build_shannon_sequence_bundle(),
        "KlarS44": build_klar_sequence_bundle(),
    }
    for experiment_id in CONFIRMATION_IDS:
        candidate_path, bundle_path = _paths(root, experiment_id)
        candidate = _read_object(candidate_path)
        bundle = _read_object(bundle_path)
        assert_blinded_payload(candidate)
        assert_blinded_payload(bundle)
        if candidate.get("experiment_id") != experiment_id or bundle.get(
            "experiment_id"
        ) != experiment_id:
            raise ValueError("confirmation contract experiment identity drifted")
        if candidate.get("outcome_access") != "sealed" or candidate.get(
            "reveal_authorized"
        ) is not False:
            raise ValueError("confirmation decision task must remain sealed")
        candidate_arm_ids = [str(arm["arm_id"]) for arm in candidate["arms"]]
        bundle_arm_ids = [str(arm["arm_id"]) for arm in bundle["arms"]]
        if candidate_arm_ids != bundle_arm_ids:
            raise ValueError("candidate and bundle arm order must match exactly")
        if experiment_id == "tcg8p":
            validate_continuous_blinded_bundle(bundle)
        elif experiment_id == "pb2rr":
            validate_bounded_multimodal_bundle(bundle)
        elif experiment_id in expected_sequences:
            validate_sequence_blinded_bundle(bundle)
            if bundle != expected_sequences[experiment_id]:
                raise ValueError(f"{experiment_id} sequence bundle does not replay")
        else:
            validate_ordinal_blinded_bundle(bundle)
        loaded[experiment_id] = {
            "candidate": candidate,
            "bundle": bundle,
            "candidate_path": candidate_path,
            "bundle_path": bundle_path,
        }
    return loaded


def validate_confirmation_protocol(
    root: Path, protocol: Mapping[str, Any]
) -> None:
    """Reject any scope, count, model, budget, or authority drift."""

    assert_blinded_payload(protocol)
    if protocol.get("schema_version") != "confirmation_inference_protocol.v1":
        raise ValueError("unsupported confirmation protocol")
    if protocol.get("status") != "frozen_outcome_blind_not_authorized_to_execute":
        raise ValueError("confirmation protocol status drifted")
    if tuple(protocol.get("experiment_ids", ())) != CONFIRMATION_IDS:
        raise ValueError("confirmation experiment scope drifted")
    authority = protocol.get("authority")
    if not isinstance(authority, Mapping) or set(authority) != _AUTHORITY_FIELDS:
        raise ValueError("confirmation authority fields drifted")
    if any(value is not False for value in authority.values()):
        raise ValueError("confirmation protocol must retain zero authority")

    catalog = _model_catalog(root)
    tasks = protocol.get("tasks")
    if not isinstance(tasks, list) or tuple(
        row.get("experiment_id") for row in tasks if isinstance(row, Mapping)
    ) != CONFIRMATION_IDS:
        raise ValueError("confirmation task matrix drifted")
    contracts = _validate_contracts(root)
    totals = {"base": 0, "perturbation": 0, "reserve": 0}
    for row in tasks:
        if not isinstance(row, Mapping):
            raise ValueError("confirmation task rows must be objects")
        experiment_id = str(row["experiment_id"])
        expected = _EXPECTED_TASK_COUNTS[experiment_id]
        actual = (
            row.get("arm_count"),
            row.get("base_cells_per_arm"),
            len(row.get("base_model_ids", ())),
            row.get("base_calls"),
            row.get("primary_prompt_perturbation_calls"),
            row.get("outcome_free_adaptive_reserve_calls"),
        )
        if actual != expected:
            raise ValueError(f"{experiment_id} call counts drifted")
        if row["arm_count"] != len(contracts[experiment_id]["candidate"]["arms"]):
            raise ValueError(f"{experiment_id} call plan arm count drifted")
        models = row["base_model_ids"]
        if len(models) != len(set(models)) or any(model not in catalog for model in models):
            raise ValueError(f"{experiment_id} call plan model set is invalid")
        if row.get("primary_model_id") not in models:
            raise ValueError(f"{experiment_id} primary model is outside its call plan")
        computed_base = row["arm_count"] * row["base_cells_per_arm"] * len(models)
        computed_perturbation = (
            row["arm_count"] * row["primary_prompt_perturbation_cells_per_arm"]
        )
        computed_reserve = row["arm_count"] * row["adaptive_reserve_cells_per_arm"]
        if (
            computed_base != row["base_calls"]
            or computed_perturbation != row["primary_prompt_perturbation_calls"]
            or computed_reserve != row["outcome_free_adaptive_reserve_calls"]
        ):
            raise ValueError(f"{experiment_id} call formula drifted")
        totals["base"] += computed_base
        totals["perturbation"] += computed_perturbation
        totals["reserve"] += computed_reserve
    if totals != {
        "base": BASE_CALLS,
        "perturbation": PERTURBATION_CALLS,
        "reserve": RESERVE_CALLS,
    }:
        raise ValueError("aggregate confirmation call totals drifted")
    expected_call_plan = {
        "base_calls": BASE_CALLS,
        "primary_prompt_perturbation_calls": PERTURBATION_CALLS,
        "outcome_free_adaptive_reserve_calls": RESERVE_CALLS,
        "planned_calls": PLANNED_CALLS,
        "maximum_attempts": MAXIMUM_ATTEMPTS,
    }
    if protocol.get("call_plan") != expected_call_plan:
        raise ValueError("confirmation call plan drifted")

    compute = protocol.get("compute_ceiling")
    if not isinstance(compute, Mapping):
        raise ValueError("confirmation compute ceiling is missing")
    gpu_seconds = float(compute.get("maximum_aggregate_gpu_seconds", -1))
    rate = float(compute.get("l40s_price_per_second_usd", -1))
    gpu_cost = float(compute.get("maximum_gpu_cost_usd", -1))
    reserve = float(compute.get("ancillary_reserve_usd", -1))
    hard_cap = float(compute.get("hard_incremental_cost_cap_usd", -1))
    if any(
        not math.isfinite(value) or value < 0
        for value in (gpu_seconds, rate, gpu_cost, reserve, hard_cap)
    ):
        raise ValueError("confirmation compute ceiling contains invalid values")
    if (
        abs(gpu_seconds * rate - gpu_cost) > 1e-9
        or abs(gpu_cost + reserve - hard_cap) > 1e-9
        or hard_cap != 125.0
    ):
        raise ValueError("confirmation compute cost arithmetic drifted")
    trust = protocol.get("trust_evaluation")
    if not isinstance(trust, Mapping):
        raise ValueError("confirmation trust evaluation is missing")
    if trust.get("learned_threshold") is not None or trust.get(
        "accept_abstain_policy"
    ) != "not_validated_not_deployed":
        raise ValueError("an unvalidated trust threshold cannot be deployed")
    if trust.get("coverage_counts") != {
        "50_percent": 3,
        "75_percent": 5,
        "100_percent": 6,
    }:
        raise ValueError("fixed trust coverage counts drifted")
    fallback = protocol.get("human_fallback")
    if not isinstance(fallback, Mapping) or fallback.get("fusion_tuning") != (
        "stopped_after_negative_development_result"
    ):
        raise ValueError("fallback stopping decision drifted")
    if fallback.get("pooled_normalized_experiment_ids") != list(
        BOUNDED_CONFIRMATION_IDS
    ) or fallback.get("raw_unit_separate_experiment_ids") != ["tcg8p"]:
        raise ValueError("continuous task must stay outside pooled normalized regret")


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a valid PNG")
    if header[12:16] != b"IHDR":
        raise ValueError(f"{path} is missing a PNG IHDR")
    return struct.unpack(">II", header[16:24])


def _pb2rr_assets(root: Path, bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in bundle["arms"]:
        arm_id = str(arm["arm_id"])
        source = arm["asset"]
        source_path = root / str(source["path"])
        if _file_sha256(source_path) != source["sha256"]:
            raise ValueError("pb2rr source PDF hash mismatch")
        png_relative, png_sha256 = _PB2RR_PNGS[arm_id]
        png_path = root / png_relative
        if _file_sha256(png_path) != png_sha256:
            raise ValueError("pb2rr rendered PNG hash mismatch")
        width, height = _png_dimensions(png_path)
        if (width, height) != (1600, 1200):
            raise ValueError("pb2rr rendered PNG dimensions drifted")
        rows.append(
            {
                "arm_id": arm_id,
                "source_pdf_path": source["path"],
                "source_pdf_sha256": source["sha256"],
                "source_page": source["page"],
                "png_path": png_relative,
                "png_sha256": png_sha256,
                "png_width": width,
                "png_height": height,
                "rendering": "pdftoppm_160_dpi_png",
                "visual_qa": "passed_full_page_no_clipping",
            }
        )
    return rows


def _socrates_compatibility(root: Path) -> dict[str, dict[str, Any]]:
    mapping = read_study_mapping(root / CHECKPOINT_MAPPING_PATH)
    specs = {
        "tcg8p": ("socsci210", None),
        "pb2rr": ("socsci210", None),
        "z358z": ("socsci210", None),
        "ShannonS2": ("external", None),
        "Blair1131": ("external", None),
        "KlarS44": ("external", "xtvu5"),
    }
    return {
        experiment_id: asdict(
            checkpoint_compatibility(
                experiment_id=experiment_id,
                source_stratum=source_stratum,
                mapping=mapping,
                equivalent_socsci210_id=equivalent,
            )
        )
        for experiment_id, (source_stratum, equivalent) in specs.items()
    }


def _classical_predictions(
    root: Path, contracts: Mapping[str, Mapping[str, Any]]
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    frozen = verify_classical_model(root, root / DEFAULT_CLASSICAL_MODEL_PATH)
    model = HashedRidgeEffectModel(
        coefficients=tuple(float(value) for value in frozen["coefficients"]),
        feature_dimension=int(frozen["feature_dimension"]),
        l2_penalty=float(frozen["l2_penalty"]),
        training_experiment_ids=tuple(frozen["training_experiment_ids"]),
        training_experiment_weights=tuple(
            (str(item[0]), float(item[1]))
            for item in frozen["training_experiment_weights"]
        ),
    )
    predictions: dict[str, dict[str, Any]] = {}
    for experiment_id in BOUNDED_CONFIRMATION_IDS:
        candidate = contracts[experiment_id]["candidate"]
        bundle = contracts[experiment_id]["bundle"]
        control = str(candidate["control_arm_id"])
        arm_ids = [str(arm["arm_id"]) for arm in candidate["arms"]]
        effects = {control: 0.0}
        for arm_id in arm_ids:
            if arm_id == control:
                continue
            effects[arm_id] = model.predict_effect(
                build_contrast_features(
                    candidate,
                    bundle,
                    arm_id=arm_id,
                    control_arm_id=control,
                )
            )
        selected = max(arm_ids, key=lambda arm_id: effects[arm_id])
        predictions[experiment_id] = {
            "control_arm_id": control,
            "predicted_normalized_utility_effects": effects,
            "selected_arm_id": selected,
            "tie_rule": "source_arm_order",
            "human_outcome_accessed": False,
        }
    return list(model.training_experiment_ids), predictions


def build_confirmation_preparation(root: Path) -> dict[str, Any]:
    protocol_path = root / DEFAULT_CONFIRMATION_PROTOCOL_PATH
    protocol = _read_object(protocol_path)
    validate_confirmation_protocol(root, protocol)
    contracts = _validate_contracts(root)
    models = _model_catalog(root)
    compatibility = _socrates_compatibility(root)
    for task in protocol["tasks"]:
        experiment_id = task["experiment_id"]
        includes_socrates = "socrates_qwen2_5_14b_sft" in task["base_model_ids"]
        if includes_socrates != compatibility[experiment_id]["primary_eligible"]:
            raise ValueError("Socrates call scope disagrees with checkpoint exposure")
    training_ids, classical = _classical_predictions(root, contracts)

    source_hashes: dict[str, str] = {
        "protocol_file_sha256": _file_sha256(protocol_path),
        "text_config_file_sha256": _file_sha256(root / _TEXT_CONFIG_PATH),
        "multimodal_config_file_sha256": _file_sha256(
            root / _MULTIMODAL_CONFIG_PATH
        ),
        "text_model_manifest_file_sha256": _file_sha256(
            root / protocol["model_sources"]["text_model_manifest_path"]
        ),
        "multimodal_model_manifest_file_sha256": _file_sha256(
            root / protocol["model_sources"]["multimodal_model_manifest_path"]
        ),
        "classical_model_file_sha256": _file_sha256(
            root / protocol["model_sources"]["classical_model_path"]
        ),
        "development_evidence_file_sha256": _file_sha256(
            root / "artifacts/development/development_evidence_v1.json"
        ),
        "development_fallback_file_sha256": _file_sha256(
            root / "artifacts/development/development_fallback_v1.json"
        ),
        "lora_gate_file_sha256": _file_sha256(
            root / "data/manifests/research/lora_development_gate_v1.json"
        ),
        "checkpoint_mapping_file_sha256": _file_sha256(
            root / CHECKPOINT_MAPPING_PATH
        ),
        "implementation_file_sha256": _file_sha256(Path(__file__)),
    }
    task_rows: list[dict[str, Any]] = []
    for task in protocol["tasks"]:
        experiment_id = task["experiment_id"]
        candidate_path = contracts[experiment_id]["candidate_path"]
        bundle_path = contracts[experiment_id]["bundle_path"]
        source_hashes[f"{experiment_id}_candidate_file_sha256"] = _file_sha256(
            candidate_path
        )
        source_hashes[f"{experiment_id}_bundle_file_sha256"] = _file_sha256(
            bundle_path
        )
        task_rows.append(
            {
                **task,
                "candidate_path": str(candidate_path.relative_to(root)),
                "candidate_payload_sha256": payload_hash(
                    contracts[experiment_id]["candidate"]
                ),
                "blinded_bundle_path": str(bundle_path.relative_to(root)),
                "blinded_bundle_payload_sha256": payload_hash(
                    contracts[experiment_id]["bundle"]
                ),
                "outcome_access": "sealed",
                "reveal_authorized": False,
            }
        )
    payload = {
        "schema_version": "confirmation_preparation.v1",
        "status": "outcome_blind_ready_for_separate_inference_authorization",
        "evidence_tier": "noncanonical_prospective_confirmation",
        "experiment_ids": list(CONFIRMATION_IDS),
        "experiment_count": len(CONFIRMATION_IDS),
        "protocol_path": str(DEFAULT_CONFIRMATION_PROTOCOL_PATH),
        "protocol_payload_sha256": payload_hash(protocol),
        "protocol_snapshot": protocol,
        "authority": dict(protocol["authority"]),
        "call_plan": dict(protocol["call_plan"]),
        "compute_ceiling": dict(protocol["compute_ceiling"]),
        "tasks": task_rows,
        "model_catalog": models,
        "socrates_checkpoint_compatibility": compatibility,
        "pb2rr_modal_assets": _pb2rr_assets(
            root, contracts["pb2rr"]["bundle"]
        ),
        "classical_training_experiment_ids": training_ids,
        "classical_baseline_predictions": classical,
        "trust_evaluation": {
            **protocol["trust_evaluation"],
            "experiment_is_unit": True,
            "frozen_before_confirmation_outcomes": True,
        },
        "human_fallback": dict(protocol["human_fallback"]),
        "source_hashes": source_hashes,
        "confirmation_outcomes_accessed": False,
        "participant_rows_read": 0,
        "participant_rows_serialized": 0,
        "modal_calls_made": 0,
        "incremental_spend_usd": 0.0,
        "claim_boundary": protocol["claim_boundary"],
    }
    assert_blinded_payload(payload)
    return json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))


def write_confirmation_preparation(root: Path) -> Path:
    path = root / DEFAULT_CONFIRMATION_PREPARATION_PATH
    freeze_envelope(
        build_confirmation_preparation(root), path, require_blinded=True
    )
    return path


def verify_confirmation_preparation(root: Path, path: Path) -> dict[str, Any]:
    payload = verify_envelope(path, require_blinded=True)
    expected = build_confirmation_preparation(root)
    if payload != expected:
        raise ValueError("confirmation preparation artifact does not replay")
    return payload

