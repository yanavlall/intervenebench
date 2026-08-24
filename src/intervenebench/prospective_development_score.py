"""Governed aggregate scoring for three prospective multimodal experiments."""

from __future__ import annotations

import csv
import io
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import fsum
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq

from .evaluation import choose_best_arm, decision_regret, treatment_effects
from .experiment_statistics import (
    experiment_cluster_bootstrap,
    paired_experiment_cluster_bootstrap,
)
from .human_fallback import (
    FallbackObservation,
    evaluate_human_fallback,
)
from .multimodal_prospective import EXPERIMENT_IDS, MODEL_IDS
from .prospective_development_protocol import (
    RECOMMENDATIONS_PATH,
    verify_pre_reveal_protocol,
    verify_reveal_authorization,
)
from .protocol import freeze_envelope, payload_hash, verify_envelope
from .selective_decision import SelectiveDecisionRecord, selective_decision_summary


DEFAULT_SCORE_PATH = Path(
    "artifacts/prospective_multimodal/prospective_multimodal_development_score_v1.json"
)


@dataclass(frozen=True, slots=True)
class RevealedObservation:
    participant_id: str
    arm_id: str
    raw_value: int
    utility: float
    weight: float


def _utility(task: Mapping[str, Any], raw_value: int) -> float:
    mapping = {
        int(option["raw_value"]): float(option["normalized_utility"])
        for option in task["response_options"]
    }
    if raw_value not in mapping:
        raise ValueError("revealed value lies outside the frozen primary scale")
    return mapping[raw_value]


def _integer(value: Any) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError("revealed values must be non-missing integers")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError("revealed values must be finite integers")
    return int(number)


def _sign(value: float, *, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _read_socsci(
    root: Path,
    *,
    experiment_id: str,
    task: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> list[RevealedObservation]:
    allowlist = protocol["outcome_column_allowlist"][experiment_id]
    if allowlist["source"] != "socsci210" or allowlist["task_num"] != 0:
        raise ValueError("SocSci reveal allowlist is invalid")
    source = protocol["outcome_sources"]["socsci210"]
    tables = [
        pq.read_table(
            root / entry["path"],
            columns=allowlist["columns"],
            filters=[
                ("study_id", "=", experiment_id),
                ("task_num", "=", allowlist["task_num"]),
            ],
        )
        for entry in source["shards"]
    ]
    table = pa.concat_tables(tables)
    arm_by_condition = {
        int(arm["socsci_condition_num"]): arm["arm_id"] for arm in task["arms"]
    }
    valid = {
        int(option["raw_value"]) for option in task["response_options"]
    }
    rows: list[RevealedObservation] = []
    for record in table.to_pylist():
        if record["response"] is None:
            continue
        raw_value = _integer(record["response"])
        if raw_value not in valid:
            continue
        condition = _integer(record["condition_num"])
        if condition not in arm_by_condition:
            raise ValueError("SocSci condition is outside the frozen action set")
        rows.append(
            RevealedObservation(
                participant_id=f"{record['sample_id']}:{record['participant']}",
                arm_id=arm_by_condition[condition],
                raw_value=raw_value,
                utility=_utility(task, raw_value),
                weight=1.0,
            )
        )
    return rows


def _r_project_sav(sav_path: Path, csv_path: Path, columns: Sequence[str]) -> None:
    expression = (
        "args<-commandArgs(trailingOnly=TRUE);"
        "cols<-strsplit(args[3],',',fixed=TRUE)[[1]];"
        "d<-haven::read_sav(args[1],col_select=tidyselect::all_of(cols));"
        "d<-haven::zap_labels(d);"
        "write.csv(as.data.frame(d,check.names=FALSE),args[2],row.names=FALSE,na='')"
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


def _read_es4xw(
    root: Path,
    *,
    task: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> list[RevealedObservation]:
    allowlist = protocol["outcome_column_allowlist"]["es4xw"]
    source = protocol["outcome_sources"]["es4xw"]
    if allowlist != {
        "source": "official_source_sav",
        "columns": ["caseid", "weight1", "XTESS040", "Q1"],
        "assignment_column": "XTESS040",
        "outcome_column": "Q1",
        "weight_column": "weight1",
    }:
        raise ValueError("es4xw source projection allowlist drifted")
    archive_path = root / source["path"]
    with ZipFile(archive_path) as archive:
        member = archive.read(source["sav_member"])
    if sha256(member).hexdigest() != source["sav_member_sha256"]:
        raise ValueError("es4xw SAV member hash mismatch")
    arm_by_assignment = {
        int(value): arm_id
        for value, arm_id in task["source_variable_mapping"][
            "assignment_to_arm"
        ].items()
    }
    valid = {
        int(option["raw_value"]) for option in task["response_options"]
    }
    rows: list[RevealedObservation] = []
    with TemporaryDirectory(prefix="intervenebench-es4xw-reveal-") as temporary:
        temporary_path = Path(temporary)
        sav_path = temporary_path / "source.sav"
        csv_path = temporary_path / "projection.csv"
        sav_path.write_bytes(member)
        _r_project_sav(sav_path, csv_path, allowlist["columns"])
        with csv_path.open(newline="", encoding="utf-8") as stream:
            for record in csv.DictReader(stream):
                if not record["XTESS040"] or not record["Q1"]:
                    continue
                assignment = _integer(record["XTESS040"])
                if assignment not in arm_by_assignment:
                    continue
                raw_value = _integer(record["Q1"])
                if raw_value == -1:
                    continue
                if raw_value not in valid:
                    raise ValueError("es4xw outcome lies outside the frozen scale")
                weight = float(record["weight1"])
                if not math.isfinite(weight) or weight <= 0.0:
                    raise ValueError("es4xw primary weight is invalid")
                rows.append(
                    RevealedObservation(
                        participant_id=f"source:{record['caseid']}",
                        arm_id=arm_by_assignment[assignment],
                        raw_value=raw_value,
                        utility=_utility(task, raw_value),
                        weight=weight,
                    )
                )
    return rows


def _grouped(
    observations: Sequence[RevealedObservation], arm_ids: Sequence[str]
) -> dict[str, list[RevealedObservation]]:
    result: dict[str, list[RevealedObservation]] = defaultdict(list)
    seen: set[str] = set()
    for row in observations:
        if row.participant_id in seen:
            raise ValueError("revealed participant IDs must be unique within task")
        if row.arm_id not in arm_ids:
            raise ValueError("revealed observation uses an unknown arm")
        seen.add(row.participant_id)
        result[row.arm_id].append(row)
    if set(result) != set(arm_ids):
        raise ValueError("revealed observations do not support every arm")
    return result


def _weighted_means(
    grouped: Mapping[str, Sequence[RevealedObservation]]
) -> dict[str, float]:
    return {
        arm_id: fsum(row.weight * row.utility for row in rows)
        / fsum(row.weight for row in rows)
        for arm_id, rows in grouped.items()
    }


def _weighted_distribution(
    rows: Sequence[RevealedObservation], values: Sequence[int]
) -> dict[int, float]:
    total = fsum(row.weight for row in rows)
    return {
        value: fsum(row.weight for row in rows if row.raw_value == value) / total
        for value in values
    }


def distribution_metrics(
    human: Mapping[int, float], synthetic: Mapping[int, float]
) -> dict[str, float]:
    """Compute frozen descriptive distances on shared ordered support."""

    if set(human) != set(synthetic) or len(human) < 2:
        raise ValueError("descriptive distributions need shared ordered support")
    keys = sorted(human)
    if any(value < 0.0 for value in (*human.values(), *synthetic.values())):
        raise ValueError("distribution probabilities cannot be negative")
    if abs(fsum(human.values()) - 1.0) > 1e-6 or abs(
        fsum(synthetic.values()) - 1.0
    ) > 1e-6:
        raise ValueError("descriptive distributions must be normalized")
    mixture = {key: 0.5 * (human[key] + synthetic[key]) for key in keys}

    def kl(left: Mapping[int, float]) -> float:
        return fsum(
            left[key] * math.log(left[key] / mixture[key], 2)
            for key in keys
            if left[key] > 0.0
        )

    cumulative_human = 0.0
    cumulative_synthetic = 0.0
    wasserstein = 0.0
    for key in keys[:-1]:
        cumulative_human += human[key]
        cumulative_synthetic += synthetic[key]
        wasserstein += abs(cumulative_human - cumulative_synthetic)
    return {
        "total_variation": 0.5
        * fsum(abs(human[key] - synthetic[key]) for key in keys),
        "ordinal_wasserstein_normalized": wasserstein / (len(keys) - 1),
        "jensen_shannon_divergence_bits": 0.5 * (kl(human) + kl(synthetic)),
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap(
    grouped: Mapping[str, Sequence[RevealedObservation]],
    *,
    arm_ids: Sequence[str],
    control_arm_id: str,
    model_choices: Mapping[str, str],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    effects = {arm_id: [] for arm_id in arm_ids if arm_id != control_arm_id}
    winners = Counter({arm_id: 0 for arm_id in arm_ids})
    regrets = {model_id: [] for model_id in model_choices}
    optimal = {model_id: 0 for model_id in model_choices}
    for _ in range(replicates):
        sampled = {
            arm_id: [rows[rng.randrange(len(rows))] for _ in rows]
            for arm_id, rows in grouped.items()
        }
        means = _weighted_means(sampled)
        winner = choose_best_arm(means)
        winners[winner] += 1
        for arm_id, effect in treatment_effects(
            means, control_arm_id=control_arm_id
        ).items():
            effects[arm_id].append(effect)
        for model_id, choice in model_choices.items():
            regrets[model_id].append(decision_regret(means, choice))
            optimal[model_id] += int(choice == winner)
    return {
        "replicates": replicates,
        "seed": seed,
        "human_optimal_arm_probability": {
            arm_id: winners[arm_id] / replicates for arm_id in arm_ids
        },
        "human_treatment_effect_95pct_interval": {
            arm_id: [_quantile(values, 0.025), _quantile(values, 0.975)]
            for arm_id, values in effects.items()
        },
        "model_selected_arm_optimal_probability": {
            model_id: optimal[model_id] / replicates for model_id in model_choices
        },
        "model_regret_95pct_interval": {
            model_id: [_quantile(values, 0.025), _quantile(values, 0.975)]
            for model_id, values in regrets.items()
        },
    }


def _score_task(
    *,
    experiment_id: str,
    task: Mapping[str, Any],
    observations: Sequence[RevealedObservation],
    recommendations: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    arm_ids = [arm["arm_id"] for arm in task["arms"]]
    grouped = _grouped(observations, arm_ids)
    human_means = _weighted_means(grouped)
    human_best = choose_best_arm(human_means)
    control = task["control_arm_id"]
    human_effects = treatment_effects(human_means, control_arm_id=control)
    option_values = [int(option["raw_value"]) for option in task["response_options"]]
    human_distributions = {
        arm_id: _weighted_distribution(grouped[arm_id], option_values)
        for arm_id in arm_ids
    }
    arm_rows = {
        (row["model_id"], row["arm_id"]): row
        for row in recommendations["balanced_arm_predictions"]
        if row["experiment_id"] == experiment_id
    }
    decisions = {
        row["model_id"]: row
        for row in recommendations["model_decisions"]
        if row["experiment_id"] == experiment_id
    }
    model_scores: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        synthetic_means = {
            arm_id: float(
                arm_rows[(model_id, arm_id)][
                    "balanced_expected_normalized_utility"
                ]
            )
            for arm_id in arm_ids
        }
        synthetic_effects = treatment_effects(
            synthetic_means, control_arm_id=control
        )
        selected = decisions[model_id]["balanced_chosen_arm_id"]
        regret = decision_regret(human_means, selected)
        effect_errors = {
            arm_id: synthetic_effects[arm_id] - human_effects[arm_id]
            for arm_id in human_effects
        }
        descriptive = {
            arm_id: distribution_metrics(
                human_distributions[arm_id],
                {
                    int(key): float(value)
                    for key, value in arm_rows[(model_id, arm_id)][
                        "balanced_probabilities"
                    ].items()
                },
            )
            for arm_id in arm_ids
        }
        model_scores[model_id] = {
            "selected_arm_id": selected,
            "synthetic_arm_means": synthetic_means,
            "synthetic_treatment_effects": synthetic_effects,
            "treatment_effect_errors": effect_errors,
            "treatment_effect_mae": fmean(abs(value) for value in effect_errors.values()),
            "effect_sign_accuracy": fmean(
                float(
                    _sign(synthetic_effects[arm_id])
                    == _sign(human_effects[arm_id])
                )
                for arm_id in human_effects
            ),
            "human_best_arm_id": human_best,
            "correct_intervention_choice": selected == human_best,
            "decision_regret": regret,
            "practically_reliable_at_task_tolerance": regret
            <= float(task["practical_regret_tolerance"]),
            "descriptive_distribution_metrics_by_arm": descriptive,
            "mean_total_variation": fmean(
                row["total_variation"] for row in descriptive.values()
            ),
            "mean_ordinal_wasserstein_normalized": fmean(
                row["ordinal_wasserstein_normalized"]
                for row in descriptive.values()
            ),
            "mean_jensen_shannon_divergence_bits": fmean(
                row["jensen_shannon_divergence_bits"]
                for row in descriptive.values()
            ),
        }
    bootstrap_contract = protocol["primary_scoring"]["within_experiment_bootstrap"]
    bootstrap = _bootstrap(
        grouped,
        arm_ids=arm_ids,
        control_arm_id=control,
        model_choices={
            model_id: model_scores[model_id]["selected_arm_id"]
            for model_id in MODEL_IDS
        },
        replicates=int(bootstrap_contract["replicates"]),
        seed=int(bootstrap_contract["seed"])
        + list(EXPERIMENT_IDS).index(experiment_id),
    )
    decision_rows = [
        row
        for row in recommendations["model_decisions"]
        if row["experiment_id"] == experiment_id
    ]
    winner_votes = Counter({arm_id: 0 for arm_id in arm_ids})
    for row in decision_rows:
        winner_votes[row["source_order_choice"]] += 1
        winner_votes[row["reverse_order_choice"]] += 1
    fallback_contract = protocol["human_fallback"]
    fallback = evaluate_human_fallback(
        [
            FallbackObservation(
                participant_id=row.participant_id,
                arm_id=row.arm_id,
                utility=row.utility,
                weight=row.weight,
            )
            for row in observations
        ],
        arm_ids=arm_ids,
        synthetic_means=model_scores["qwen3_vl_8b_primary"]["synthetic_arm_means"],
        winner_votes=dict(winner_votes),
        budgets=tuple(fallback_contract["budgets_total_outcome_observations"]),
        partitions=int(fallback_contract["partitions"]),
        fold_count=int(fallback_contract["fold_count"]),
        seed=int(fallback_contract["seed"])
        + list(EXPERIMENT_IDS).index(experiment_id),
        pseudocount=int(
            fallback_contract["fusion"]["synthetic_prior_pseudocount_per_arm"]
        ),
        practical_tolerance=float(task["practical_regret_tolerance"]),
    )
    return {
        "experiment_id": experiment_id,
        "source_stratum": (
            "socsci210_primary" if experiment_id != "es4xw" else "official_source_sav"
        ),
        "participant_count": len(observations),
        "participant_rows_serialized": 0,
        "arm_counts": {arm_id: len(grouped[arm_id]) for arm_id in arm_ids},
        "human_arm_means": human_means,
        "human_treatment_effects": human_effects,
        "human_best_arm_id": human_best,
        "human_response_distributions": human_distributions,
        "models": model_scores,
        "within_experiment_bootstrap": bootstrap,
        "human_fallback": fallback,
    }


def _aggregate(
    tasks: Mapping[str, Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    cluster = protocol["primary_scoring"]["experiment_cluster_bootstrap"]
    models: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        regrets = {
            experiment_id: float(task["models"][model_id]["decision_regret"])
            for experiment_id, task in tasks.items()
        }
        models[model_id] = {
            "correct_intervention_count": sum(
                bool(task["models"][model_id]["correct_intervention_choice"])
                for task in tasks.values()
            ),
            "practically_reliable_count": sum(
                bool(
                    task["models"][model_id][
                        "practically_reliable_at_task_tolerance"
                    ]
                )
                for task in tasks.values()
            ),
            "mean_decision_regret": fmean(regrets.values()),
            "worst_case_decision_regret": max(regrets.values()),
            "mean_treatment_effect_mae": fmean(
                task["models"][model_id]["treatment_effect_mae"]
                for task in tasks.values()
            ),
            "mean_total_variation": fmean(
                task["models"][model_id]["mean_total_variation"]
                for task in tasks.values()
            ),
            "decision_regret_experiment_cluster_bootstrap": asdict(
                experiment_cluster_bootstrap(
                    regrets,
                    replicates=int(cluster["replicates"]),
                    seed=int(cluster["seed"]) + list(MODEL_IDS).index(model_id),
                    confidence_level=float(cluster["confidence_level"]),
                )
            ),
        }
    primary = "qwen3_vl_8b_primary"
    confidence = protocol["selective_decision"]["confidence"][
        "confidence_by_experiment"
    ]
    selective = selective_decision_summary(
        [
            SelectiveDecisionRecord(
                experiment_id=experiment_id,
                confidence=float(confidence[experiment_id]),
                regret=float(tasks[experiment_id]["models"][primary]["decision_regret"]),
                exact_correct=bool(
                    tasks[experiment_id]["models"][primary][
                        "correct_intervention_choice"
                    ]
                ),
                practically_reliable=bool(
                    tasks[experiment_id]["models"][primary][
                        "practically_reliable_at_task_tolerance"
                    ]
                ),
            )
            for experiment_id in EXPERIMENT_IDS
        ],
        minimum_class_count=int(
            protocol["selective_decision"]["minimum_class_count_for_binary_metrics"]
        ),
    )
    fallback: dict[str, Any] = {}
    budgets = protocol["human_fallback"]["budgets_total_outcome_observations"]
    policies = protocol["human_fallback"]["policies"]
    for budget in budgets:
        fallback[str(budget)] = {}
        for policy in policies:
            rows = {
                experiment_id: tasks[experiment_id]["human_fallback"]["by_budget"][
                    str(budget)
                ][policy]
                for experiment_id in EXPERIMENT_IDS
            }
            estimated = {
                experiment_id: row for experiment_id, row in rows.items()
                if row["status"] == "estimated"
            }
            if len(estimated) != len(EXPERIMENT_IDS):
                fallback[str(budget)][policy] = {
                    "status": "not_estimable_for_all_experiments",
                    "experiment_count": len(estimated),
                }
                continue
            fallback[str(budget)][policy] = {
                "status": "estimated",
                "experiment_count": len(estimated),
                "mean_regret": fmean(row["mean_regret"] for row in estimated.values()),
                "mean_exact_choice_rate": fmean(
                    row["exact_choice_rate"] for row in estimated.values()
                ),
                "mean_practical_reliability_rate": fmean(
                    row["practical_reliability_rate"] for row in estimated.values()
                ),
                "mean_paired_regret_change_vs_synthetic": fmean(
                    row["paired_mean_regret_change_vs_synthetic"]
                    for row in estimated.values()
                ),
                "mean_negative_value_rate_vs_synthetic": fmean(
                    row["negative_value_rate_vs_synthetic"]
                    for row in estimated.values()
                ),
            }
    return {
        "models": models,
        "primary_selective_decision": asdict(selective),
        "human_fallback": fallback,
        "source_strata": {
            "socsci210_primary": ["nj5dx", "e2pyb"],
            "official_source_sav": ["es4xw"],
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def score_prospective_development(root: Path, *, output_path: Path) -> Path:
    """Open only authorized outcomes and write one aggregate-only artifact."""

    protocol = verify_pre_reveal_protocol(root)
    authorization = verify_reveal_authorization(root)
    recommendations = verify_envelope(
        root / RECOMMENDATIONS_PATH, require_blinded=True
    )
    if payload_hash(recommendations) != authorization[
        "recommendations_payload_sha256"
    ]:
        raise ValueError("reveal is not bound to the frozen recommendations")
    tasks: dict[str, Any] = {}
    for experiment_id in EXPERIMENT_IDS:
        task = json.loads(
            (
                root
                / f"data/manifests/contracts/{experiment_id}_decision_task_candidate.json"
            ).read_text(encoding="utf-8")
        )
        observations = (
            _read_es4xw(root, task=task, protocol=protocol)
            if experiment_id == "es4xw"
            else _read_socsci(
                root,
                experiment_id=experiment_id,
                task=task,
                protocol=protocol,
            )
        )
        tasks[experiment_id] = _score_task(
            experiment_id=experiment_id,
            task=task,
            observations=observations,
            recommendations=recommendations,
            protocol=protocol,
        )
    payload = {
        "schema_version": "prospective_multimodal_development_score.v1",
        "authorization_payload_sha256": payload_hash(authorization),
        "protocol_payload_sha256": payload_hash(protocol),
        "recommendations_payload_sha256": payload_hash(recommendations),
        "evidence_tier": "prospective_development_noncanonical",
        "canonical_test_claim": False,
        "human_outcomes_opened": True,
        "participant_rows_serialized": 0,
        "experiment_ids": list(EXPERIMENT_IDS),
        "tasks": tasks,
        "summary": _aggregate(tasks, protocol),
        "other_experiments_opened": [],
        "modal_used_for_outcome_scoring": False,
        "paid_cost_usd_for_outcome_scoring": 0.0,
        "claim_boundary": protocol["claim_boundary"],
    }
    freeze_envelope(payload, output_path, require_blinded=False)
    return output_path


def verify_prospective_development_score(root: Path, path: Path) -> dict[str, Any]:
    authorization = verify_reveal_authorization(root)
    score = verify_envelope(path, require_blinded=False)
    if (
        score.get("schema_version")
        != "prospective_multimodal_development_score.v1"
        or score.get("authorization_payload_sha256") != payload_hash(authorization)
        or score.get("evidence_tier")
        != "prospective_development_noncanonical"
        or score.get("canonical_test_claim") is not False
        or score.get("participant_rows_serialized") != 0
        or score.get("other_experiments_opened") != []
    ):
        raise ValueError("prospective-development score contract is invalid")
    return score
