"""Experiment-held-out classical treatment-effect baseline for development tasks.

The baseline uses only source/design text and aggregate normalized effects from
revealed development experiments. Every prediction is leave-one-experiment-out;
the target experiment's effect labels never enter its fitted model.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from .baselines import LabeledEffectExample, fit_hashed_effect_ridge
from .development_evidence import (
    DEFAULT_DEVELOPMENT_EVIDENCE_PATH,
    DEVELOPMENT_IDS,
    SEALED_CONFIRMATION_IDS,
    verify_development_evidence,
)
from .experiment_statistics import experiment_cluster_bootstrap
from .protocol import (
    assert_blinded_payload,
    freeze_envelope,
    payload_hash,
    verify_envelope,
)


DEFAULT_CLASSICAL_PROTOCOL_PATH = Path(
    "data/manifests/research/classical_baseline_protocol_v1.json"
)
DEFAULT_CLASSICAL_DEVELOPMENT_PATH = Path(
    "artifacts/development/classical_baseline_development_v1.json"
)
DEFAULT_CLASSICAL_MODEL_PATH = Path(
    "artifacts/development/classical_baseline_final_model_v1.json"
)
FEATURE_DIMENSION = 128
L2_PENALTY = 10.0
FEATURE_SCHEMA_VERSION = "arm_contrast_text_metadata.v1"

_TOKEN = re.compile(r"[a-z]+|[0-9]+", re.IGNORECASE)
_DIRECT_TEXT_FIELDS = frozenset(
    {"description", "message", "accessible_text", "message_template"}
)
_NESTED_TEXT_FIELDS = frozenset({"arm_substitutions", "message_variants"})
_CONTEXT_FIELDS = (
    "common_context",
    "outcome_question",
    "population",
)


@dataclass(frozen=True, slots=True)
class ArmEffectRecord:
    experiment_id: str
    arm_id: str
    features: Mapping[str, str | int | float | bool]
    normalized_effect: float


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


def _all_strings(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, Mapping):
        found: list[str] = []
        for nested in value.values():
            found.extend(_all_strings(nested))
        return found
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        found = []
        for nested in value:
            found.extend(_all_strings(nested))
        return found
    return []


def _arm_strings(arm: Mapping[str, Any]) -> list[str]:
    found: list[str] = []
    for key, value in arm.items():
        if key in _DIRECT_TEXT_FIELDS and isinstance(value, str) and value.strip():
            found.append(value.strip())
        elif key in _NESTED_TEXT_FIELDS:
            found.extend(_all_strings(value))
    # Repeated source descriptions should not receive accidental extra weight.
    return list(dict.fromkeys(found))


def _tokens(texts: Sequence[str]) -> list[str]:
    tokens: list[str] = []
    for text in texts:
        for raw in _TOKEN.findall(text.casefold()):
            tokens.append("__number__" if raw.isdigit() else raw)
    return tokens


def _ngrams(tokens: Sequence[str]) -> Counter[str]:
    counts: Counter[str] = Counter(tokens)
    counts.update(
        f"{left}__{right}" for left, right in zip(tokens, tokens[1:], strict=False)
    )
    return counts


def _term_frequencies(tokens: Sequence[str]) -> dict[str, float]:
    if not tokens:
        return {}
    counts = _ngrams(tokens)
    denominator = float(sum(counts.values()))
    return {token: count / denominator for token, count in counts.items()}


def _arm_by_id(payload: Mapping[str, Any], arm_id: str) -> Mapping[str, Any]:
    arms = payload.get("arms")
    if not isinstance(arms, list):
        raise ValueError("contract arms must be a list")
    matches = [arm for arm in arms if isinstance(arm, Mapping) and arm.get("arm_id") == arm_id]
    if len(matches) != 1:
        raise ValueError(f"arm {arm_id!r} must appear exactly once")
    return matches[0]


def _combined_arm_strings(
    candidate: Mapping[str, Any], bundle: Mapping[str, Any], arm_id: str
) -> list[str]:
    strings = [
        *_arm_strings(_arm_by_id(candidate, arm_id)),
        *_arm_strings(_arm_by_id(bundle, arm_id)),
    ]
    return list(dict.fromkeys(strings))


def build_contrast_features(
    candidate: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    arm_id: str,
    control_arm_id: str,
) -> dict[str, str | int | float | bool]:
    """Build response-blind, ID-free text/metadata features for one contrast."""

    assert_blinded_payload(candidate)
    assert_blinded_payload(bundle)
    if candidate.get("experiment_id") != bundle.get("experiment_id"):
        raise ValueError("candidate and bundle experiment IDs must match")
    if arm_id == control_arm_id:
        raise ValueError("arm contrast must not compare the control with itself")

    candidate_arms = candidate.get("arms")
    if not isinstance(candidate_arms, list) or len(candidate_arms) < 2:
        raise ValueError("candidate must contain at least two arms")
    target_tokens = _tokens(_combined_arm_strings(candidate, bundle, arm_id))
    control_tokens = _tokens(
        _combined_arm_strings(candidate, bundle, control_arm_id)
    )
    context_strings: list[str] = []
    for field in _CONTEXT_FIELDS:
        if field in bundle:
            context_strings.extend(_all_strings(bundle[field]))
    if not context_strings:
        context_strings.extend(_all_strings(candidate.get("outcome_question")))
    context_tokens = _tokens(list(dict.fromkeys(context_strings)))

    target_tf = _term_frequencies(target_tokens)
    control_tf = _term_frequencies(control_tokens)
    context_tf = _term_frequencies(context_tokens)
    features: dict[str, str | int | float | bool] = {
        "outcome_family": str(candidate.get("outcome_family", "unknown")),
        "design_type": str(candidate.get("design_type", "unknown")),
        "direction": str(candidate.get("direction", "unknown")),
        "modality": (
            "image_and_text"
            if "png" in str(bundle.get("schema_version", "")).casefold()
            else "text"
        ),
        "arm_count": len(candidate_arms),
        "response_option_count": len(candidate.get("response_options", ())),
        "target_token_count_scaled": len(target_tokens) / 100.0,
        "control_token_count_scaled": len(control_tokens) / 100.0,
        "token_count_difference_scaled": (
            len(target_tokens) - len(control_tokens)
        )
        / 100.0,
    }
    for token, value in target_tf.items():
        features[f"target::{token}"] = value
    for token, value in control_tf.items():
        features[f"control::{token}"] = value
    for token, value in context_tf.items():
        features[f"context::{token}"] = value
    for token in sorted(set(target_tf) | set(control_tf)):
        difference = target_tf.get(token, 0.0) - control_tf.get(token, 0.0)
        if difference:
            features[f"contrast::{token}"] = difference
    return features


def cross_fit_arm_effect_predictions(
    records_by_experiment: Mapping[str, Sequence[ArmEffectRecord]],
    *,
    feature_dimension: int = FEATURE_DIMENSION,
    l2_penalty: float = L2_PENALTY,
) -> dict[str, dict[str, float]]:
    """Predict each experiment from models fit only on the other experiments."""

    if len(records_by_experiment) < 3:
        raise ValueError("cross-fitting requires at least three experiments")
    identifiers = tuple(records_by_experiment)
    if any(not records_by_experiment[identifier] for identifier in identifiers):
        raise ValueError("every experiment needs at least one arm contrast")
    for experiment_id, records in records_by_experiment.items():
        if any(record.experiment_id != experiment_id for record in records):
            raise ValueError("record experiment IDs must match their group")
        arm_ids = [record.arm_id for record in records]
        if len(arm_ids) != len(set(arm_ids)):
            raise ValueError("arm IDs must be unique within an experiment")

    predictions: dict[str, dict[str, float]] = {}
    for target_experiment in identifiers:
        split = {
            experiment_id: (
                "test" if experiment_id == target_experiment else "train"
            )
            for experiment_id in identifiers
        }
        training: list[LabeledEffectExample] = []
        for experiment_id, records in records_by_experiment.items():
            if experiment_id == target_experiment:
                continue
            weight = 1.0 / len(records)
            training.extend(
                LabeledEffectExample(
                    experiment_id=experiment_id,
                    split="train",
                    features=record.features,
                    normalized_effect=record.normalized_effect,
                    sample_weight=weight,
                )
                for record in records
            )
        model = fit_hashed_effect_ridge(
            training,
            experiment_to_split=split,
            feature_dimension=feature_dimension,
            l2_penalty=l2_penalty,
        )
        predictions[target_experiment] = {
            record.arm_id: model.predict_effect(record.features)
            for record in records_by_experiment[target_experiment]
        }
    return predictions


def _paths(root: Path, experiment_id: str) -> tuple[Path, Path]:
    contracts = root / "data/manifests/contracts"
    candidate_name = (
        f"{experiment_id}_decision_task.json"
        if experiment_id == "jf46x"
        else f"{experiment_id}_decision_task_candidate.json"
    )
    return contracts / candidate_name, contracts / f"{experiment_id}_blinded_bundle.json"


def _load_protocol(root: Path) -> dict[str, Any]:
    protocol = _read_object(root / DEFAULT_CLASSICAL_PROTOCOL_PATH)
    if protocol.get("schema_version") != "classical_baseline_protocol.v1":
        raise ValueError("unsupported classical baseline protocol")
    if protocol.get("status") != "retrospective_development_only_frozen":
        raise ValueError("classical baseline protocol has invalid status")
    method = protocol.get("method")
    if not isinstance(method, Mapping):
        raise ValueError("classical baseline protocol method is missing")
    expected = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dimension": FEATURE_DIMENSION,
        "l2_penalty": L2_PENALTY,
        "cross_fit_unit": "experiment",
        "experiment_total_weight": 1.0,
        "hyperparameter_search": "none",
    }
    if any(method.get(key) != value for key, value in expected.items()):
        raise ValueError("classical baseline protocol method drifted")
    if tuple(protocol.get("development_experiment_ids", ())) != DEVELOPMENT_IDS:
        raise ValueError("classical protocol development support drifted")
    if tuple(protocol.get("excluded_confirmation_experiment_ids", ())) != SEALED_CONFIRMATION_IDS:
        raise ValueError("classical protocol must exclude all confirmation experiments")
    return protocol


def _development_records(
    root: Path,
) -> tuple[
    dict[str, tuple[ArmEffectRecord, ...]],
    dict[str, dict[str, Any]],
    dict[str, str],
]:
    evidence = verify_development_evidence(
        root, root / DEFAULT_DEVELOPMENT_EVIDENCE_PATH
    )
    tasks = {row["experiment_id"]: row for row in evidence["tasks"]}
    records: dict[str, tuple[ArmEffectRecord, ...]] = {}
    contracts: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {
        "development_evidence_payload_sha256": payload_hash(evidence)
    }
    for experiment_id in DEVELOPMENT_IDS:
        candidate_path, bundle_path = _paths(root, experiment_id)
        candidate = _read_object(candidate_path)
        bundle = _read_object(bundle_path)
        assert_blinded_payload(candidate)
        assert_blinded_payload(bundle)
        control = candidate.get("control_arm_id")
        if not isinstance(control, str) or not control:
            raise ValueError(f"{experiment_id} control arm is missing")
        effects = tasks[experiment_id]["human_treatment_effects"]
        if not isinstance(effects, Mapping) or not effects:
            raise ValueError(f"{experiment_id} aggregate effects are missing")
        arm_ids = {
            arm["arm_id"]
            for arm in candidate["arms"]
            if isinstance(arm, Mapping) and isinstance(arm.get("arm_id"), str)
        }
        if set(effects) != arm_ids - {control}:
            raise ValueError(f"{experiment_id} treatment-effect support drifted")
        experiment_records = []
        for arm_id, raw_effect in effects.items():
            effect = float(raw_effect)
            if not math.isfinite(effect) or not -1.0 <= effect <= 1.0:
                raise ValueError("development effects must be normalized and finite")
            experiment_records.append(
                ArmEffectRecord(
                    experiment_id=experiment_id,
                    arm_id=arm_id,
                    features=build_contrast_features(
                        candidate,
                        bundle,
                        arm_id=arm_id,
                        control_arm_id=control,
                    ),
                    normalized_effect=effect,
                )
            )
        records[experiment_id] = tuple(experiment_records)
        contracts[experiment_id] = {
            "candidate": candidate,
            "bundle": bundle,
            "control_arm_id": control,
            "human_best_arm_id": tasks[experiment_id]["human_best_arm_id"],
        }
        hashes[f"{experiment_id}_candidate_file_sha256"] = _file_sha256(candidate_path)
        hashes[f"{experiment_id}_bundle_file_sha256"] = _file_sha256(bundle_path)
    return records, contracts, hashes


def _select_arm(effect_by_arm: Mapping[str, float]) -> str:
    if not effect_by_arm:
        raise ValueError("arm effects must be nonempty")
    return max(sorted(effect_by_arm), key=lambda arm_id: effect_by_arm[arm_id])


def build_classical_development(root: Path) -> dict[str, Any]:
    protocol = _load_protocol(root)
    records, contracts, hashes = _development_records(root)
    predictions = cross_fit_arm_effect_predictions(records)
    task_rows: list[dict[str, Any]] = []
    no_effect_rows: list[dict[str, Any]] = []
    for experiment_id in DEVELOPMENT_IDS:
        control = contracts[experiment_id]["control_arm_id"]
        human_effects = {control: 0.0}
        human_effects.update(
            {
                record.arm_id: record.normalized_effect
                for record in records[experiment_id]
            }
        )
        synthetic_effects = {control: 0.0, **predictions[experiment_id]}
        selected = _select_arm(synthetic_effects)
        no_effect_selected = _select_arm(
            {arm_id: 0.0 for arm_id in synthetic_effects}
        )
        human_best = str(contracts[experiment_id]["human_best_arm_id"])
        regret = human_effects[human_best] - human_effects[selected]
        no_effect_regret = human_effects[human_best] - human_effects[no_effect_selected]
        task_rows.append(
            {
                "experiment_id": experiment_id,
                "training_experiment_ids": [
                    value for value in DEVELOPMENT_IDS if value != experiment_id
                ],
                "control_arm_id": control,
                "predicted_treatment_effects": predictions[experiment_id],
                "selected_arm_id": selected,
                "human_best_arm_id": human_best,
                "correct_intervention_choice": selected == human_best,
                "decision_regret": regret,
                "treatment_effect_mae": fmean(
                    abs(predictions[experiment_id][record.arm_id] - record.normalized_effect)
                    for record in records[experiment_id]
                ),
            }
        )
        no_effect_rows.append(
            {
                "experiment_id": experiment_id,
                "selected_arm_id": no_effect_selected,
                "correct_intervention_choice": no_effect_selected == human_best,
                "decision_regret": no_effect_regret,
                "treatment_effect_mae": fmean(
                    abs(record.normalized_effect)
                    for record in records[experiment_id]
                ),
            }
        )
    regrets = {row["experiment_id"]: row["decision_regret"] for row in task_rows}
    bootstrap = asdict(
        experiment_cluster_bootstrap(
            regrets, replicates=10000, seed=2026081401, confidence_level=0.95
        )
    )
    bootstrap["confidence_interval"] = list(bootstrap["confidence_interval"])
    payload = {
        "schema_version": "classical_baseline_development.v1",
        "status": "complete_retrospective_experiment_cross_fit",
        "development_only": True,
        "canonical_test_claim": False,
        "experiment_count": len(DEVELOPMENT_IDS),
        "experiment_ids": list(DEVELOPMENT_IDS),
        "confirmation_experiment_ids": list(SEALED_CONFIRMATION_IDS),
        "confirmation_outcomes_accessed": False,
        "participant_rows_read": 0,
        "participant_rows_serialized": 0,
        "protocol_payload_sha256": payload_hash(protocol),
        "source_hashes": hashes,
        "method": protocol["method"],
        "tasks": task_rows,
        "summary": {
            "correct_intervention_count": sum(
                row["correct_intervention_choice"] for row in task_rows
            ),
            "exact_choice_accuracy": fmean(
                float(row["correct_intervention_choice"]) for row in task_rows
            ),
            "mean_decision_regret": fmean(row["decision_regret"] for row in task_rows),
            "worst_case_decision_regret": max(row["decision_regret"] for row in task_rows),
            "mean_treatment_effect_mae": fmean(
                row["treatment_effect_mae"] for row in task_rows
            ),
            "decision_regret_experiment_cluster_bootstrap": bootstrap,
        },
        "no_effect_comparator": {
            "tasks": no_effect_rows,
            "exact_choice_accuracy": fmean(
                float(row["correct_intervention_choice"]) for row in no_effect_rows
            ),
            "mean_decision_regret": fmean(
                row["decision_regret"] for row in no_effect_rows
            ),
            "mean_treatment_effect_mae": fmean(
                row["treatment_effect_mae"] for row in no_effect_rows
            ),
        },
        "claim_boundary": (
            "This is retrospective development evidence from leave-one-experiment-out "
            "aggregate-effect fitting. It is not a prospective test, and the single "
            "fixed specification was not selected using confirmation outcomes."
        ),
    }
    return json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))


def build_classical_model(root: Path) -> dict[str, Any]:
    protocol = _load_protocol(root)
    records, _contracts, hashes = _development_records(root)
    split = {experiment_id: "train" for experiment_id in DEVELOPMENT_IDS}
    training: list[LabeledEffectExample] = []
    for experiment_id, experiment_records in records.items():
        weight = 1.0 / len(experiment_records)
        training.extend(
            LabeledEffectExample(
                experiment_id=experiment_id,
                split="train",
                features=record.features,
                normalized_effect=record.normalized_effect,
                sample_weight=weight,
            )
            for record in experiment_records
        )
    model = fit_hashed_effect_ridge(
        training,
        experiment_to_split=split,
        feature_dimension=FEATURE_DIMENSION,
        l2_penalty=L2_PENALTY,
    )
    payload = {
        "schema_version": "classical_baseline_model.v1",
        "status": "frozen_development_fit_not_yet_applied_to_confirmation",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dimension": model.feature_dimension,
        "l2_penalty": model.l2_penalty,
        "coefficients": list(model.coefficients),
        "training_experiment_ids": list(model.training_experiment_ids),
        "training_experiment_weights": [
            list(item) for item in model.training_experiment_weights
        ],
        "training_experiment_count": len(model.training_experiment_ids),
        "training_data": "aggregate_normalized_arm_effects_only",
        "participant_rows_read": 0,
        "participant_rows_serialized": 0,
        "confirmation_experiment_ids": list(SEALED_CONFIRMATION_IDS),
        "confirmation_outcomes_accessed": False,
        "protocol_payload_sha256": payload_hash(protocol),
        "source_hashes": hashes,
    }
    return json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))


def write_classical_artifacts(root: Path) -> tuple[Path, Path]:
    result_path = root / DEFAULT_CLASSICAL_DEVELOPMENT_PATH
    model_path = root / DEFAULT_CLASSICAL_MODEL_PATH
    freeze_envelope(build_classical_development(root), result_path)
    freeze_envelope(build_classical_model(root), model_path)
    return result_path, model_path


def verify_classical_development(root: Path, path: Path) -> dict[str, Any]:
    payload = verify_envelope(path)
    expected = build_classical_development(root)
    if payload != expected:
        raise ValueError("classical development artifact does not replay")
    return payload


def verify_classical_model(root: Path, path: Path) -> dict[str, Any]:
    payload = verify_envelope(path)
    expected = build_classical_model(root)
    if payload != expected:
        raise ValueError("classical model artifact does not replay")
    return payload
