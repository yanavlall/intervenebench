"""Governed development reveal and scoring for the frozen five-task portfolio.

This module is intentionally separate from canonical test scoring.  It accepts
only the five tasks named in the development-reveal authorization, verifies the
pre-reveal recommendation hashes, reads only the declared outcome fields, and
writes aggregate artifacts without participant rows.
"""

from __future__ import annotations

import csv
import io
import json
import math
import random
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from math import fsum
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Mapping, Sequence
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq

from .evaluation import choose_best_arm, decision_regret, treatment_effects
from .pilot import PILOT_EXPERIMENTS
from .portfolio_pilot import verify_portfolio_run, verify_portfolio_scope
from .protocol import freeze_envelope, payload_hash, verify_envelope


AUTHORIZATION_PATH = Path(
    "data/manifests/benchmark/portfolio_pilot_development_reveal.json"
)
DEFAULT_RUN_MANIFEST = Path(
    "artifacts/portfolio_pilot/local_llama3_2_3b_20260813_v2/run_manifest.json"
)
SOCSCI_COLUMNS = (
    "study_id",
    "sample_id",
    "participant",
    "condition_num",
    "task_num",
    "response",
)
EXTERNAL_ARCHIVES = {
    "turagaS11": {
        "path": ".work/development_reveal/Turaga789.zip",
        "outer_sha256": "1da9b8a4d6ec30fd8785741a1e9de3daf789e5974c7a84fc83e50f5ee7e14c66",
        "nested_member": "Archive of OSF Storage.zip",
        "sav_member": "tess2_030_turaga_final_data.sav",
    },
    "wallaceS12": {
        "path": ".work/development_reveal/Wallace187.zip",
        "outer_sha256": "1353334cde493dc599a0e276b4081694176752244c99945ee6eff45f98bc1665",
        "nested_member": "Archive of OSF Storage.zip",
        "sav_member": "TESS2 097 Wallace_Client.sav",
    },
}


@dataclass(frozen=True, slots=True)
class RevealedObservation:
    participant_id: str
    arm_id: str
    raw_value: int
    utility: float
    weight: float


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_development_reveal_authorization(
    authorization: Mapping[str, Any],
) -> None:
    if authorization.get("schema_version") != "portfolio_development_reveal.v1":
        raise ValueError("unsupported portfolio development-reveal schema")
    if authorization.get("status") != "development_reveal_authorized":
        raise ValueError("development reveal is not authorized")
    if tuple(authorization.get("experiment_ids", ())) != PILOT_EXPERIMENTS:
        raise ValueError("development reveal does not cover the frozen five tasks")
    if authorization.get("permanent_role") != "development_only_portfolio_reveal":
        raise ValueError("revealed tasks must be permanently development-only")
    if authorization.get("canonical_test_eligible") is not False:
        raise ValueError("development tasks cannot remain canonical-test eligible")
    if authorization.get("canonical_split_status") != "unassigned":
        raise ValueError("development reveal must not create a canonical split")
    if authorization.get("paid_inference_authorized") is not False:
        raise ValueError("development reveal does not authorize paid inference")
    if authorization.get("modal_compute_authorized") is not False:
        raise ValueError("development reveal does not authorize Modal")
    fallback = authorization.get("human_fallback_contract")
    if not isinstance(fallback, Mapping):
        raise ValueError("human-fallback contract is required before reveal")
    if fallback.get("pilot_evaluation_people_disjoint") is not True:
        raise ValueError("fallback pilot and evaluation people must be disjoint")
    if fallback.get("no_outcome_adaptive_allocation") is not True:
        raise ValueError("fallback allocation must be frozen before outcomes")
    if fallback.get("synthetic_prior_pseudocount_per_arm") != 10:
        raise ValueError("unexpected synthetic prior strength")


def verify_development_reveal_authorization(
    root: Path,
    *,
    authorization_path: Path | None = None,
) -> dict[str, Any]:
    path = authorization_path or root / AUTHORIZATION_PATH
    authorization = _read_object(path)
    validate_development_reveal_authorization(authorization)
    scope = verify_portfolio_scope(root)
    if authorization["portfolio_scope_sha256"] != payload_hash(scope):
        raise ValueError("development reveal is not bound to the frozen scope")
    manifest_path = root / authorization["blind_run_manifest_path"]
    manifest = verify_portfolio_run(root, manifest_path)
    if authorization["blind_run_manifest_sha256"] != payload_hash(manifest):
        raise ValueError("development reveal is not bound to the blind run")
    for experiment_id in PILOT_EXPERIMENTS:
        recommendation = verify_envelope(
            manifest_path.parent / f"{experiment_id}_recommendation.json",
            require_blinded=True,
        )
        if (
            payload_hash(recommendation)
            != authorization["recommendation_sha256"][experiment_id]
        ):
            raise ValueError("development reveal recommendation hash mismatch")
    return authorization


def _normalized_utility(task: Mapping[str, Any], raw_value: int) -> float:
    utilities = {
        int(option["raw_value"]): float(option["normalized_utility"])
        for option in task["response_options"]
    }
    if raw_value not in utilities:
        raise ValueError("revealed response lies outside the frozen outcome scale")
    return utilities[raw_value]


def _parse_integer_response(value: Any) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError("revealed response must be a non-missing integer")
    parsed = float(value)
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise ValueError("revealed response must be a finite integer")
    return int(parsed)


def _read_socsci_observations(
    parquet_paths: Sequence[Path],
    *,
    experiment_id: str,
    task: Mapping[str, Any],
    allowlist: Mapping[str, Any],
) -> list[RevealedObservation]:
    if allowlist.get("source") != "socsci210":
        raise ValueError("SocSci reader requires a SocSci allowlist")
    if tuple(allowlist.get("columns", ())) != SOCSCI_COLUMNS:
        raise ValueError("SocSci reveal column allowlist changed")
    task_num = task.get("socsci210_task_num")
    if task_num != allowlist.get("task_num"):
        raise ValueError("SocSci task number does not match reveal authorization")
    tables = [
        pq.read_table(
            path,
            columns=list(SOCSCI_COLUMNS),
            filters=[
                ("study_id", "=", experiment_id),
                ("task_num", "=", task_num),
            ],
        )
        for path in parquet_paths
    ]
    if not tables:
        raise ValueError("at least one SocSci Parquet file is required")
    table = pa.concat_tables(tables)
    arm_by_condition = {
        int(arm["condition_num"]): arm["arm_id"] for arm in task["arms"]
    }
    observations: list[RevealedObservation] = []
    for row in table.to_pylist():
        if row["response"] is None:
            continue
        raw_value = _parse_integer_response(row["response"])
        condition = _parse_integer_response(row["condition_num"])
        if condition not in arm_by_condition:
            raise ValueError("revealed condition is absent from the frozen task")
        observations.append(
            RevealedObservation(
                participant_id=f"{row['sample_id']}:{row['participant']}",
                arm_id=arm_by_condition[condition],
                raw_value=raw_value,
                utility=_normalized_utility(task, raw_value),
                weight=1.0,
            )
        )
    return observations


def _extract_authorized_sav(
    root: Path, *, experiment_id: str, target: Path
) -> Path:
    source = EXTERNAL_ARCHIVES[experiment_id]
    archive_path = root / source["path"]
    if not archive_path.exists():
        raise ValueError(f"authorized source archive is missing: {archive_path}")
    if _file_sha256(archive_path) != source["outer_sha256"]:
        raise ValueError("official mixed-archive hash mismatch")
    with ZipFile(archive_path) as outer:
        if source["nested_member"] not in outer.namelist():
            raise ValueError("authorized nested archive member is missing")
        nested_bytes = outer.read(source["nested_member"])
    with ZipFile(io.BytesIO(nested_bytes)) as nested:
        if source["sav_member"] not in nested.namelist():
            raise ValueError("authorized SAV member is missing")
        target.write_bytes(nested.read(source["sav_member"]))
    return target


def _r_select_sav_columns(
    sav_path: Path, csv_path: Path, columns: Sequence[str]
) -> None:
    expression = (
        "args<-commandArgs(trailingOnly=TRUE);"
        "cols<-strsplit(args[3],',',fixed=TRUE)[[1]];"
        "d<-haven::read_sav(args[1],col_select=tidyselect::all_of(cols));"
        "d<-haven::zap_labels(d);"
        "out<-data.frame(participant_id=seq_len(nrow(d)),"
        "lapply(d,as.numeric),check.names=FALSE);"
        "write.csv(out,args[2],row.names=FALSE,na='')"
    )
    subprocess.run(
        [
            "Rscript",
            "-e",
            expression,
            str(sav_path),
            str(csv_path),
            ",".join(columns),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _read_external_observations(
    root: Path,
    *,
    experiment_id: str,
    task: Mapping[str, Any],
    allowlist: Mapping[str, Any],
) -> list[RevealedObservation]:
    mapping = task["source_variable_mapping"]
    assignment_column = mapping["assignment_variable"]
    weight_column = mapping["weight_variable"]
    outcome_columns = tuple(allowlist["outcome_columns"])
    if allowlist.get("source") != "official_source_sav":
        raise ValueError("external reader requires an official SAV allowlist")
    if assignment_column != allowlist.get("assignment_column"):
        raise ValueError("external assignment allowlist changed")
    if weight_column != allowlist.get("weight_column"):
        raise ValueError("external weight allowlist changed")
    if experiment_id == "turagaS11":
        if set(outcome_columns) != set(mapping["arm_outcome_variables"].values()):
            raise ValueError("Turaga routed-outcome allowlist changed")
        assignment_by_arm = {
            arm["arm_id"]: int(arm["source_assignment"].split("=")[1])
            for arm in task["arms"]
        }
        outcome_by_arm = mapping["arm_outcome_variables"]
    else:
        if outcome_columns != (mapping["outcome_variable"],):
            raise ValueError("Wallace outcome allowlist changed")
        assignment_by_arm = {
            arm["arm_id"]: int(arm["source_assignment"].split("=")[1])
            for arm in task["arms"]
        }
        outcome_by_arm = {
            arm_id: mapping["outcome_variable"] for arm_id in assignment_by_arm
        }
    arm_by_assignment = {value: arm for arm, value in assignment_by_arm.items()}
    columns = (assignment_column, *outcome_columns, weight_column)
    observations: list[RevealedObservation] = []
    with TemporaryDirectory(prefix=f"intervenebench-{experiment_id}-") as temporary:
        temporary_path = Path(temporary)
        sav_path = _extract_authorized_sav(
            root, experiment_id=experiment_id, target=temporary_path / "source.sav"
        )
        csv_path = temporary_path / "selected.csv"
        _r_select_sav_columns(sav_path, csv_path, columns)
        with csv_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                assignment_text = row[assignment_column]
                if not assignment_text:
                    continue
                assignment = _parse_integer_response(assignment_text)
                if assignment not in arm_by_assignment:
                    continue
                arm_id = arm_by_assignment[assignment]
                outcome_text = row[outcome_by_arm[arm_id]]
                if not outcome_text:
                    continue
                raw_value = _parse_integer_response(outcome_text)
                missing_codes = {
                    int(value) for value in mapping.get("missing_outcome_codes", ())
                }
                if raw_value in missing_codes:
                    continue
                weight = float(row[weight_column])
                if not math.isfinite(weight) or weight <= 0.0:
                    raise ValueError("external source weight must be positive and finite")
                observations.append(
                    RevealedObservation(
                        participant_id=f"source-row:{row['participant_id']}",
                        arm_id=arm_id,
                        raw_value=raw_value,
                        utility=_normalized_utility(task, raw_value),
                        weight=weight,
                    )
                )
    return observations


def _grouped(
    observations: Iterable[RevealedObservation],
) -> dict[str, list[RevealedObservation]]:
    groups: dict[str, list[RevealedObservation]] = defaultdict(list)
    for observation in observations:
        groups[observation.arm_id].append(observation)
    return dict(groups)


def _weighted_means(
    grouped: Mapping[str, Sequence[RevealedObservation]],
) -> dict[str, float]:
    means: dict[str, float] = {}
    for arm_id, rows in grouped.items():
        denominator = fsum(row.weight for row in rows)
        if not rows or denominator <= 0.0:
            raise ValueError("every arm must contain positive-weight observations")
        means[arm_id] = fsum(row.utility * row.weight for row in rows) / denominator
    return means


def _weighted_distribution(
    rows: Sequence[RevealedObservation], option_values: Sequence[int]
) -> dict[str, float]:
    denominator = fsum(row.weight for row in rows)
    return {
        str(value): fsum(row.weight for row in rows if row.raw_value == value)
        / denominator
        for value in option_values
    }


def _synthetic_distributions(
    outputs: Mapping[str, Any], arm_ids: Sequence[str], option_values: Sequence[int]
) -> dict[str, dict[str, float]]:
    totals = {
        arm_id: {str(value): 0.0 for value in option_values} for arm_id in arm_ids
    }
    weights = {arm_id: 0.0 for arm_id in arm_ids}
    draws = int(outputs["draws_per_arm_variant"])
    for row in outputs["outputs"]:
        arm_id = row["arm_id"]
        contribution = float(row["variant_weight"]) / draws
        weights[arm_id] += contribution
        for value in option_values:
            totals[arm_id][str(value)] += contribution * float(
                row["probabilities"][str(value)]
            )
    if any(abs(weight - 1.0) > 1e-9 for weight in weights.values()):
        raise ValueError("synthetic distribution weights do not sum to one by arm")
    return totals


def _js_divergence(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    mixture = {key: (first[key] + second[key]) / 2.0 for key in first}

    def kl(left: Mapping[str, float]) -> float:
        return fsum(
            left[key] * math.log(left[key] / mixture[key], 2)
            for key in left
            if left[key] > 0.0
        )

    return (kl(first) + kl(second)) / 2.0


def _distribution_metrics(
    human: Mapping[str, float], synthetic: Mapping[str, float]
) -> dict[str, float]:
    keys = sorted(human, key=int)
    total_variation = 0.5 * fsum(abs(human[key] - synthetic[key]) for key in keys)
    cumulative_human = 0.0
    cumulative_synthetic = 0.0
    emd = 0.0
    for key in keys[:-1]:
        cumulative_human += human[key]
        cumulative_synthetic += synthetic[key]
        emd += abs(cumulative_human - cumulative_synthetic)
    normalized_emd = emd / max(1, len(keys) - 1)
    return {
        "total_variation": total_variation,
        "ordinal_wasserstein_normalized": normalized_emd,
        "jensen_shannon_divergence_bits": _js_divergence(human, synthetic),
    }


def _sign(value: float, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _simulator_score(
    *,
    human_means: Mapping[str, float],
    synthetic_means: Mapping[str, float],
    selected_arm_id: str,
    control_arm_id: str,
    practical_tolerance: float,
) -> dict[str, Any]:
    human_effects = treatment_effects(human_means, control_arm_id=control_arm_id)
    synthetic_effects = treatment_effects(
        synthetic_means, control_arm_id=control_arm_id
    )
    errors = {
        arm_id: synthetic_effects[arm_id] - human_effect
        for arm_id, human_effect in human_effects.items()
    }
    human_best = choose_best_arm(human_means)
    regret = decision_regret(human_means, selected_arm_id)
    return {
        "selected_arm_id": selected_arm_id,
        "human_best_arm_id": human_best,
        "correct_choice": selected_arm_id == human_best,
        "decision_regret": regret,
        "practically_reliable_at_frozen_tolerance": regret <= practical_tolerance,
        "synthetic_arm_means": dict(synthetic_means),
        "synthetic_treatment_effects": synthetic_effects,
        "treatment_effect_errors": errors,
        "treatment_effect_mae": mean(abs(error) for error in errors.values()),
        "effect_sign_accuracy": mean(
            _sign(synthetic_effects[arm_id]) == _sign(human_effect)
            for arm_id, human_effect in human_effects.items()
        ),
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_uncertainty(
    grouped: Mapping[str, Sequence[RevealedObservation]],
    *,
    control_arm_id: str,
    selected_arms: Mapping[str, str],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    wins = {arm_id: 0 for arm_id in grouped}
    effects = {arm_id: [] for arm_id in grouped if arm_id != control_arm_id}
    regrets = {model_id: [] for model_id in selected_arms}
    for _ in range(replicates):
        sampled = {
            arm_id: [rows[rng.randrange(len(rows))] for _ in rows]
            for arm_id, rows in grouped.items()
        }
        means = _weighted_means(sampled)
        winner = choose_best_arm(means)
        wins[winner] += 1
        task_effects = treatment_effects(means, control_arm_id=control_arm_id)
        for arm_id, effect in task_effects.items():
            effects[arm_id].append(effect)
        for model_id, selected_arm in selected_arms.items():
            regrets[model_id].append(decision_regret(means, selected_arm))
    return {
        "replicates": replicates,
        "seed": seed,
        "human_optimal_arm_probability": {
            arm_id: count / replicates for arm_id, count in wins.items()
        },
        "human_treatment_effect_95pct_intervals": {
            arm_id: [_quantile(values, 0.025), _quantile(values, 0.975)]
            for arm_id, values in effects.items()
        },
        "fixed_policy_regret_95pct_intervals": {
            model_id: [_quantile(values, 0.025), _quantile(values, 0.975)]
            for model_id, values in regrets.items()
        },
    }


def _balanced_allocation(arm_ids: Sequence[str], budget: int) -> dict[str, int]:
    base, remainder = divmod(budget, len(arm_ids))
    return {
        arm_id: base + (index < remainder)
        for index, arm_id in enumerate(sorted(arm_ids))
    }


def _intelligent_allocation(
    arm_ids: Sequence[str], ranking: Sequence[str], budget: int
) -> dict[str, int]:
    if budget < len(arm_ids):
        raise ValueError("intelligent allocation requires at least one person per arm")
    allocation = {arm_id: 1 for arm_id in arm_ids}
    top_two = tuple(ranking[:2])
    for index in range(budget - len(arm_ids)):
        allocation[top_two[index % 2]] += 1
    return allocation


def _partition_evaluation_pool(
    grouped: Mapping[str, Sequence[RevealedObservation]],
    rng: random.Random,
) -> tuple[
    dict[str, list[RevealedObservation]], dict[str, list[RevealedObservation]]
]:
    pool: dict[str, list[RevealedObservation]] = {}
    evaluation: dict[str, list[RevealedObservation]] = {}
    for arm_id, rows in grouped.items():
        order = list(range(len(rows)))
        rng.shuffle(order)
        evaluation_count = len(rows) // 3
        evaluation_indexes = set(order[:evaluation_count])
        pool_order = order[evaluation_count:]
        evaluation[arm_id] = [
            row for index, row in enumerate(rows) if index in evaluation_indexes
        ]
        pool[arm_id] = [rows[index] for index in pool_order]
    return pool, evaluation


def _take_pilot(
    pool: Mapping[str, Sequence[RevealedObservation]],
    allocation: Mapping[str, int],
) -> dict[str, list[RevealedObservation]]:
    pilot: dict[str, list[RevealedObservation]] = {}
    for arm_id, rows in pool.items():
        requested = allocation[arm_id]
        if requested <= 0 or requested > len(rows):
            raise ValueError("fallback allocation is unsupported by the pilot pool")
        pilot[arm_id] = list(rows[:requested])
    return pilot


def _fused_means(
    synthetic_means: Mapping[str, float],
    pilot: Mapping[str, Sequence[RevealedObservation]],
    *,
    pseudocount: int,
) -> dict[str, float]:
    pilot_means = _weighted_means(pilot)
    return {
        arm_id: (
            pseudocount * synthetic_means[arm_id]
            + len(pilot[arm_id]) * pilot_means[arm_id]
        )
        / (pseudocount + len(pilot[arm_id]))
        for arm_id in synthetic_means
    }


def _fallback_curve(
    grouped: Mapping[str, Sequence[RevealedObservation]],
    *,
    synthetic_means: Mapping[str, float],
    synthetic_selected: str,
    synthetic_ranking: Sequence[str],
    control_arm_id: str,
    budgets: Sequence[int],
    replicates: int,
    seed: int,
    pseudocount: int,
) -> dict[str, Any]:
    arm_ids = tuple(sorted(grouped))
    result: dict[str, Any] = {}
    for budget in budgets:
        if budget == 0:
            full_means = _weighted_means(grouped)
            full_best = choose_best_arm(full_means)
            result[str(budget)] = {
                "humans_only": {
                    "mean_regret": decision_regret(full_means, control_arm_id),
                    "correct_choice_probability": float(control_arm_id == full_best),
                },
                "synthetic_only": {
                    "mean_regret": decision_regret(full_means, synthetic_selected),
                    "correct_choice_probability": float(synthetic_selected == full_best),
                },
                "synthetic_plus_random_humans": {
                    "mean_regret": decision_regret(full_means, synthetic_selected),
                    "correct_choice_probability": float(synthetic_selected == full_best),
                },
                "synthetic_plus_intelligent_humans": {
                    "mean_regret": decision_regret(full_means, synthetic_selected),
                    "correct_choice_probability": float(synthetic_selected == full_best),
                },
            }
            continue
        metrics = {
            method: {"regret": [], "correct": []}
            for method in (
                "humans_only",
                "synthetic_only",
                "synthetic_plus_random_humans",
                "synthetic_plus_intelligent_humans",
            )
        }
        balanced = _balanced_allocation(arm_ids, budget)
        intelligent = _intelligent_allocation(arm_ids, synthetic_ranking, budget)
        for replicate in range(replicates):
            rng = random.Random(seed + replicate)
            pool, evaluation = _partition_evaluation_pool(grouped, rng)
            pilot = _take_pilot(pool, balanced)
            evaluation_means = _weighted_means(evaluation)
            evaluation_best = choose_best_arm(evaluation_means)
            human_selected = choose_best_arm(_weighted_means(pilot))
            random_fused_selected = choose_best_arm(
                _fused_means(
                    synthetic_means, pilot, pseudocount=pseudocount
                )
            )
            intelligent_pilot = _take_pilot(pool, intelligent)
            intelligent_selected = choose_best_arm(
                _fused_means(
                    synthetic_means,
                    intelligent_pilot,
                    pseudocount=pseudocount,
                )
            )
            selections = {
                "humans_only": (human_selected, evaluation_means, evaluation_best),
                "synthetic_only": (
                    synthetic_selected,
                    evaluation_means,
                    evaluation_best,
                ),
                "synthetic_plus_random_humans": (
                    random_fused_selected,
                    evaluation_means,
                    evaluation_best,
                ),
                "synthetic_plus_intelligent_humans": (
                    intelligent_selected,
                    evaluation_means,
                    evaluation_best,
                ),
            }
            for method, (selected, means, best) in selections.items():
                metrics[method]["regret"].append(decision_regret(means, selected))
                metrics[method]["correct"].append(float(selected == best))
        result[str(budget)] = {
            method: {
                "mean_regret": mean(values["regret"]),
                "regret_95pct_monte_carlo_interval": [
                    _quantile(values["regret"], 0.025),
                    _quantile(values["regret"], 0.975),
                ],
                "correct_choice_probability": mean(values["correct"]),
            }
            for method, values in metrics.items()
        }
    return result


def _score_one(
    root: Path,
    *,
    experiment_id: str,
    observations: Sequence[RevealedObservation],
    recommendation: Mapping[str, Any],
    outputs: Mapping[str, Any],
    baseline: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    task = _read_object(
        root
        / "data/manifests/contracts"
        / f"{experiment_id}_decision_task_candidate.json"
    )
    grouped = _grouped(observations)
    arm_ids = tuple(arm["arm_id"] for arm in task["arms"])
    if set(grouped) != set(arm_ids):
        raise ValueError("revealed data do not cover every frozen arm")
    human_means = _weighted_means(grouped)
    human_effects = treatment_effects(
        human_means, control_arm_id=task["control_arm_id"]
    )
    local_score = _simulator_score(
        human_means=human_means,
        synthetic_means=recommendation["synthetic_arm_means"],
        selected_arm_id=recommendation["selected_arm_id"],
        control_arm_id=task["control_arm_id"],
        practical_tolerance=task["practical_regret_tolerance"],
    )
    no_effect = baseline["predictions"][experiment_id]
    baseline_score = _simulator_score(
        human_means=human_means,
        synthetic_means=no_effect["synthetic_arm_means"],
        selected_arm_id=no_effect["selected_arm_id"],
        control_arm_id=task["control_arm_id"],
        practical_tolerance=task["practical_regret_tolerance"],
    )
    option_values = tuple(
        int(option["raw_value"]) for option in task["response_options"]
    )
    human_distributions = {
        arm_id: _weighted_distribution(grouped[arm_id], option_values)
        for arm_id in arm_ids
    }
    synthetic_distributions = _synthetic_distributions(
        outputs, arm_ids, option_values
    )
    descriptive = {
        arm_id: _distribution_metrics(
            human_distributions[arm_id], synthetic_distributions[arm_id]
        )
        for arm_id in arm_ids
    }
    uncertainty_contract = authorization["uncertainty_contract"]
    uncertainty = _bootstrap_uncertainty(
        grouped,
        control_arm_id=task["control_arm_id"],
        selected_arms={
            "local_llama3_2_3b": recommendation["selected_arm_id"],
            "no_effect_control_tie": no_effect["selected_arm_id"],
        },
        replicates=int(uncertainty_contract["bootstrap_replicates"]),
        seed=int(uncertainty_contract["bootstrap_seed"]),
    )
    fallback_contract = authorization["human_fallback_contract"]
    fallback = _fallback_curve(
        grouped,
        synthetic_means=recommendation["synthetic_arm_means"],
        synthetic_selected=recommendation["selected_arm_id"],
        synthetic_ranking=recommendation["arm_ranking"],
        control_arm_id=task["control_arm_id"],
        budgets=tuple(fallback_contract["budgets_total_participants_per_experiment"]),
        replicates=int(fallback_contract["monte_carlo_replicates"]),
        seed=int(fallback_contract["seed"]),
        pseudocount=int(fallback_contract["synthetic_prior_pseudocount_per_arm"]),
    )
    score: dict[str, Any] = {
        "experiment_id": experiment_id,
        "development_only": True,
        "participant_count_complete_case": len(observations),
        "complete_case_count_by_arm": {
            arm_id: len(grouped[arm_id]) for arm_id in arm_ids
        },
        "human_arm_means": human_means,
        "human_treatment_effects": human_effects,
        "human_best_arm_id": choose_best_arm(human_means),
        "local_llama3_2_3b": local_score,
        "no_effect_control_tie": baseline_score,
        "descriptive_distribution_fidelity": descriptive,
        "mean_total_variation": mean(
            metrics["total_variation"] for metrics in descriptive.values()
        ),
        "mean_ordinal_wasserstein_normalized": mean(
            metrics["ordinal_wasserstein_normalized"]
            for metrics in descriptive.values()
        ),
        "bootstrap_uncertainty": uncertainty,
        "human_fallback": fallback,
    }
    if experiment_id == "de5hx":
        human_gary = {arm_id: 1.0 - value for arm_id, value in human_means.items()}
        synthetic_gary = {
            arm_id: 1.0 - value
            for arm_id, value in recommendation["synthetic_arm_means"].items()
        }
        baseline_gary = {
            arm_id: 1.0 - value
            for arm_id, value in no_effect["synthetic_arm_means"].items()
        }
        score["prespecified_gary_campaign_sensitivity"] = {
            "human_arm_means": human_gary,
            "human_best_arm_id": choose_best_arm(human_gary),
            "local_llama3_2_3b": _simulator_score(
                human_means=human_gary,
                synthetic_means=synthetic_gary,
                selected_arm_id=choose_best_arm(synthetic_gary),
                control_arm_id=task["control_arm_id"],
                practical_tolerance=task["practical_regret_tolerance"],
            ),
            "no_effect_control_tie": _simulator_score(
                human_means=human_gary,
                synthetic_means=baseline_gary,
                selected_arm_id=no_effect["selected_arm_id"],
                control_arm_id=task["control_arm_id"],
                practical_tolerance=task["practical_regret_tolerance"],
            ),
        }
    return score


def _aggregate_scores(tasks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for model_id in ("local_llama3_2_3b", "no_effect_control_tie"):
        model_scores = [task[model_id] for task in tasks.values()]
        aggregate[model_id] = {
            "correct_intervention_count": sum(
                bool(score["correct_choice"]) for score in model_scores
            ),
            "experiment_count": len(model_scores),
            "correct_intervention_rate": mean(
                float(score["correct_choice"]) for score in model_scores
            ),
            "mean_decision_regret": mean(
                score["decision_regret"] for score in model_scores
            ),
            "worst_case_decision_regret": max(
                score["decision_regret"] for score in model_scores
            ),
            "mean_treatment_effect_mae": mean(
                score["treatment_effect_mae"] for score in model_scores
            ),
            "mean_effect_sign_accuracy": mean(
                score["effect_sign_accuracy"] for score in model_scores
            ),
        }
    aggregate["local_descriptive_fidelity"] = {
        "mean_total_variation": mean(
            task["mean_total_variation"] for task in tasks.values()
        ),
        "mean_ordinal_wasserstein_normalized": mean(
            task["mean_ordinal_wasserstein_normalized"] for task in tasks.values()
        ),
    }
    budgets = next(iter(tasks.values()))["human_fallback"].keys()
    aggregate["human_fallback"] = {
        budget: {
            method: {
                "mean_regret_across_experiments": mean(
                    task["human_fallback"][budget][method]["mean_regret"]
                    for task in tasks.values()
                ),
                "mean_correct_choice_probability_across_experiments": mean(
                    task["human_fallback"][budget][method][
                        "correct_choice_probability"
                    ]
                    for task in tasks.values()
                ),
            }
            for method in next(iter(tasks.values()))["human_fallback"][budget]
        }
        for budget in budgets
    }
    return aggregate


def score_development_portfolio(
    root: Path,
    *,
    output_path: Path,
    authorization_path: Path | None = None,
) -> Path:
    """Reveal only the five authorized outcomes and freeze aggregate scores."""

    authorization = verify_development_reveal_authorization(
        root, authorization_path=authorization_path
    )
    manifest_path = root / authorization["blind_run_manifest_path"]
    artifact_dir = manifest_path.parent
    baseline = _read_object(
        root / "data/manifests/benchmark/supported_ordinal_no_effect.json"
    )
    parquet_paths = tuple(
        sorted(
            (
                root
                / "data/raw/socsci210/048481111a4425ed83dc0eacf15f8431f252b21a/data"
            ).glob("*.parquet")
        )
    )
    task_scores: dict[str, Any] = {}
    for experiment_id in PILOT_EXPERIMENTS:
        task = _read_object(
            root
            / "data/manifests/contracts"
            / f"{experiment_id}_decision_task_candidate.json"
        )
        allowlist = authorization["outcome_column_allowlist"][experiment_id]
        if allowlist["source"] == "socsci210":
            observations = _read_socsci_observations(
                parquet_paths,
                experiment_id=experiment_id,
                task=task,
                allowlist=allowlist,
            )
        else:
            observations = _read_external_observations(
                root,
                experiment_id=experiment_id,
                task=task,
                allowlist=allowlist,
            )
        recommendation = verify_envelope(
            artifact_dir / f"{experiment_id}_recommendation.json",
            require_blinded=True,
        )
        outputs = verify_envelope(
            artifact_dir / f"{experiment_id}_outputs.json", require_blinded=True
        )
        task_scores[experiment_id] = _score_one(
            root,
            experiment_id=experiment_id,
            observations=observations,
            recommendation=recommendation,
            outputs=outputs,
            baseline=baseline,
            authorization=authorization,
        )
    payload = {
        "schema_version": "portfolio_development_score.v1",
        "authorization_sha256": payload_hash(authorization),
        "blind_run_manifest_sha256": authorization["blind_run_manifest_sha256"],
        "development_only": True,
        "canonical_test_claim": False,
        "human_outcomes_opened": True,
        "experiment_ids": list(PILOT_EXPERIMENTS),
        "participant_rows_written_to_artifact": 0,
        "paid_cost_usd": 0.0,
        "modal_used": False,
        "tasks": task_scores,
        "portfolio_summary": _aggregate_scores(task_scores),
        "claim_boundary": authorization["claim_boundary"],
    }
    freeze_envelope(payload, output_path, require_blinded=False)
    return output_path


def verify_development_score(root: Path, score_path: Path) -> dict[str, Any]:
    authorization = verify_development_reveal_authorization(root)
    score = verify_envelope(score_path, require_blinded=False)
    if score.get("schema_version") != "portfolio_development_score.v1":
        raise ValueError("unsupported portfolio development-score schema")
    if score.get("authorization_sha256") != payload_hash(authorization):
        raise ValueError("development score is not bound to its authorization")
    if tuple(score.get("experiment_ids", ())) != PILOT_EXPERIMENTS:
        raise ValueError("development score does not cover the frozen five tasks")
    if score.get("development_only") is not True:
        raise ValueError("portfolio score must remain development-only")
    if score.get("canonical_test_claim") is not False:
        raise ValueError("portfolio score cannot claim canonical test status")
    if score.get("participant_rows_written_to_artifact") != 0:
        raise ValueError("development score must not serialize participant rows")
    return score
