"""Aggregate-only scoring for an explicitly authorized confirmation reveal."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import csv
from hashlib import sha256
import io
import json
from math import fsum, isfinite
from pathlib import Path
import random
from statistics import fmean
import subprocess
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq

from .development_evidence import (
    DEFAULT_DEVELOPMENT_EVIDENCE_PATH,
    DEVELOPMENT_IDS,
)
from .eb_fallback import EffectCalibrationTask, fit_effect_prior
from .human_fallback import balanced_allocation
from .protocol import (
    assert_blinded_payload,
    freeze_envelope,
    payload_hash,
    verify_envelope,
)


CONFIRMATION_IDS = (
    "tcg8p",
    "pb2rr",
    "z358z",
    "ShannonS2",
    "Blair1131",
    "KlarS44",
)
AGGREGATION_PATH = Path(
    "artifacts/confirmation/confirmation_20260814_v1/aggregation_v1.json"
)
SCORING_PROTOCOL_PATH = Path(
    "data/manifests/research/confirmation_scoring_protocol_v1.json"
)
SOURCE_ARCHIVES = {
    "pb2rr": {
        "path": "data/raw/sources/pb2rr/Abascal635.zip",
        "outer_sha256": "010b2b9413214458046f7f152ad5ed7625291e014ddc01d0dbc5e5910c6bc24e",
        "member": "TESS3_187_Abascal_Client.sav",
        "member_sha256": "0ffb58b22babb648d339f2755b37ab3fdf6c12b1432ca15d0b4c4c467d682137",
        "columns": ["XTESS187", "DOV_INSERT_NAME", "Q4", "weight"],
    },
    "z358z": {
        "path": "data/raw/sources/z358z/Nayak609.zip",
        "outer_sha256": "091236a19b521a4ca02f6edf6601e40532a89aee71b1ebfcf951688e402999fc",
        "member": "TESS3_175 Nayak_Client.sav",
        "member_sha256": "0da32a838d3bd06d1244f1bf2a17d84b9646cbd66469be78f8a23d913f2d081f",
        "columns": ["XTESS175", "DOV_OPTION", "Q3a", "weight"],
    },
    "ShannonS2": {
        "path": ".work/confirmation_reveal/ShannonS2.zip",
        "outer_sha256": "755bb9261008e2704c1fe33c24dc2b407a50f8e2ad3b98c78bfa4bdef071bae0",
        "member": "ShannonS2/TESS 001_SHANNON_05March17.sav",
        "member_sha256": "f03e64213e6167a6c99278be4b827f177f4c4626b9e8619f3399f0f179bd6341",
        "columns": ["GROUP", "Q1", "Q4", "Q7", "Q10", "Q13", "Q16", "WEIGHT2"],
    },
    "Blair1131": {
        "path": ".work/confirmation_reveal/Blair1131.zip",
        "outer_sha256": "66ec9b7b879a2858f4445cf60eadc1d3b04e32343e8c0c1955dc22d7dc3762c0",
        "member": "8041.045_TESS045_Blair.csv",
        "member_sha256": "ec153dd3ffc257a2de031da4b7f609dbd294d43a6b21d89733d239b6e9ed86ca",
        "columns": ["WEIGHT", "BLAIR", "RND_02", "PARTYID7", "Q3"],
    },
    "KlarS44": {
        "path": ".work/confirmation_reveal/KlarS44.zip",
        "outer_sha256": "8337694961dffa3f95e16c7b0e6f26ca29b86951a5191260ec7c03fdbd620a6c",
        "member": "PN8041.043_TESS043_Klar_20200109.sav",
        "member_sha256": "dbeb6a5d54e57a34fcd192c17fd92686290319280ac30934db29dca10e502f80",
        "columns": ["P_KLAR", "RND_00", "WEIGHT", "Q2_KLAR"],
    },
}


@dataclass(frozen=True, slots=True)
class HumanArmSummary:
    arm_means: Mapping[str, float]
    complete_case_count_by_arm: Mapping[str, int]
    outcome_unit: str


@dataclass(frozen=True, slots=True)
class RawFallbackObservation:
    participant_id: str
    arm_id: str
    location: float
    weight: float = 1.0
    fold_stratum_id: str = ""


@dataclass(frozen=True, slots=True)
class RevealedOutcomeObservation:
    participant_id: str
    arm_id: str
    raw_value: float
    decision_score: float
    weight: float
    fold_stratum_id: str = ""
    standardization_cell_id: str = ""


def _winner(means: Mapping[str, float], arm_ids: Sequence[str]) -> str:
    arms = tuple(arm_ids)
    return max(arms, key=lambda arm: (float(means[arm]), -arms.index(arm)))


def _sign(value: float, *, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def score_synthetic_recommendation(
    *,
    arm_ids: Sequence[str],
    control_arm_id: str,
    human: HumanArmSummary,
    synthetic_arm_scores: Mapping[str, float],
    selected_arm_id: str,
    practical_tolerance: float,
) -> dict[str, Any]:
    """Score a previously frozen choice against human aggregate utilities."""

    arms = tuple(arm_ids)
    if (
        len(arms) < 2
        or len(set(arms)) != len(arms)
        or control_arm_id not in arms
        or set(human.arm_means) != set(arms)
        or set(human.complete_case_count_by_arm) != set(arms)
        or set(synthetic_arm_scores) != set(arms)
        or selected_arm_id not in arms
    ):
        raise ValueError("scoring action sets do not align")
    if any(
        not isfinite(float(value))
        for value in (*human.arm_means.values(), *synthetic_arm_scores.values())
    ) or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in human.complete_case_count_by_arm.values()
    ):
        raise ValueError("scoring aggregates are invalid")
    if not isfinite(practical_tolerance) or practical_tolerance < 0.0:
        raise ValueError("practical tolerance must be non-negative")
    human_means = {arm: float(human.arm_means[arm]) for arm in arms}
    synthetic_means = {arm: float(synthetic_arm_scores[arm]) for arm in arms}
    human_effects = {
        arm: human_means[arm] - human_means[control_arm_id] for arm in arms
    }
    synthetic_effects = {
        arm: synthetic_means[arm] - synthetic_means[control_arm_id] for arm in arms
    }
    contrasts = [arm for arm in arms if arm != control_arm_id]
    effect_errors = {
        arm: abs(synthetic_effects[arm] - human_effects[arm]) for arm in contrasts
    }
    human_winner = _winner(human_means, arms)
    regret = max(human_means.values()) - human_means[selected_arm_id]
    return {
        "human_arm_means": human_means,
        "complete_case_count_by_arm": dict(human.complete_case_count_by_arm),
        "complete_case_count": sum(human.complete_case_count_by_arm.values()),
        "outcome_unit": human.outcome_unit,
        "human_treatment_effects": human_effects,
        "synthetic_treatment_effects": synthetic_effects,
        "absolute_treatment_effect_errors": effect_errors,
        "mean_absolute_treatment_effect_error": fmean(effect_errors.values()),
        "treatment_effect_sign_accuracy": fmean(
            float(_sign(synthetic_effects[arm]) == _sign(human_effects[arm]))
            for arm in contrasts
        ),
        "human_selected_arm_id": human_winner,
        "synthetic_selected_arm_id": selected_arm_id,
        "exact_choice": selected_arm_id == human_winner,
        "decision_regret": regret,
        "practical_tolerance": practical_tolerance,
        "practically_reliable": regret <= practical_tolerance,
        "tie_rule": "source_arm_order",
    }


def _validate_raw_rows(
    rows: Sequence[RawFallbackObservation], arm_ids: Sequence[str]
) -> tuple[RawFallbackObservation, ...]:
    arms = tuple(arm_ids)
    seen: set[str] = set()
    counts = {arm: 0 for arm in arms}
    validated = []
    for row in rows:
        if (
            not isinstance(row, RawFallbackObservation)
            or not row.participant_id
            or row.participant_id in seen
            or row.arm_id not in counts
            or not isfinite(row.location)
            or not isfinite(row.weight)
            or row.weight <= 0.0
            or not isinstance(row.fold_stratum_id, str)
        ):
            raise ValueError("invalid raw fallback observation")
        seen.add(row.participant_id)
        counts[row.arm_id] += 1
        validated.append(row)
    if any(count == 0 for count in counts.values()):
        raise ValueError("each raw fallback arm requires observations")
    return tuple(validated)


def _raw_fold_assignments(
    rows: Sequence[RawFallbackObservation], *, fold_count: int, seed: int
) -> dict[str, int]:
    grouped: dict[tuple[str, str], list[RawFallbackObservation]] = defaultdict(list)
    for row in rows:
        grouped[(row.arm_id, row.fold_stratum_id)].append(row)
    if any(len(group) < fold_count for group in grouped.values()):
        raise ValueError("every raw fallback stratum must support every fold")
    rng = random.Random(seed)
    assignments: dict[str, int] = {}
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda row: row.participant_id)
        rng.shuffle(group)
        for index, row in enumerate(group):
            assignments[row.participant_id] = index % fold_count
    return assignments


def _raw_means(
    rows: Sequence[RawFallbackObservation], arm_ids: Sequence[str]
) -> dict[str, float]:
    grouped: dict[str, list[RawFallbackObservation]] = defaultdict(list)
    for row in rows:
        grouped[row.arm_id].append(row)
    result = {}
    for arm in arm_ids:
        if not grouped[arm]:
            raise ValueError("raw fallback estimate lacks an arm")
        denominator = fsum(row.weight for row in grouped[arm])
        result[arm] = (
            fsum(row.weight * row.location for row in grouped[arm]) / denominator
        )
    return result


def evaluate_raw_human_fallback(
    observations: Sequence[RawFallbackObservation],
    *,
    arm_ids: Sequence[str],
    synthetic_locations: Mapping[str, float],
    budgets: Sequence[int],
    partitions: int,
    fold_count: int,
    seed: int,
    practical_tolerance: float,
) -> dict[str, Any]:
    """Evaluate primary fallback policies for lower-is-better raw outcomes."""

    rows = _validate_raw_rows(observations, arm_ids)
    arms = tuple(arm_ids)
    if set(synthetic_locations) != set(arms) or any(
        not isfinite(float(value)) for value in synthetic_locations.values()
    ):
        raise ValueError("synthetic raw locations must cover all arms")
    if (
        tuple(budgets) != tuple(sorted(set(budgets)))
        or not budgets
        or budgets[0] != 0
        or partitions <= 0
        or fold_count < 2
        or practical_tolerance < 0
    ):
        raise ValueError("invalid raw fallback protocol")
    synthetic_choice = min(
        arms, key=lambda arm: (float(synthetic_locations[arm]), arms.index(arm))
    )
    records: dict[int, dict[str, list[dict[str, float]]]] = {
        budget: {"synthetic_only": [], "human_only_balanced": []}
        for budget in budgets
    }
    for partition in range(partitions):
        assignments = _raw_fold_assignments(
            rows, fold_count=fold_count, seed=seed + partition * 1009
        )
        for fold in range(fold_count):
            evaluation = [
                row for row in rows if assignments[row.participant_id] == fold
            ]
            pool: dict[str, list[RawFallbackObservation]] = defaultdict(list)
            for row in rows:
                if assignments[row.participant_id] != fold:
                    pool[row.arm_id].append(row)
            rng = random.Random(seed + partition * 1009 + fold * 9173 + 41)
            for arm in arms:
                pool[arm].sort(key=lambda row: row.participant_id)
                rng.shuffle(pool[arm])
            evaluation_means = _raw_means(evaluation, arms)
            human_best = min(
                arms, key=lambda arm: (evaluation_means[arm], arms.index(arm))
            )
            synthetic_regret = (
                evaluation_means[synthetic_choice] - evaluation_means[human_best]
            )
            for budget in budgets:
                records[budget]["synthetic_only"].append(
                    {
                        "regret": synthetic_regret,
                        "exact": float(synthetic_choice == human_best),
                        "practical": float(synthetic_regret <= practical_tolerance),
                    }
                )
                if budget == 0:
                    continue
                allocation = balanced_allocation(arms, budget)
                if any(allocation[arm] > len(pool[arm]) for arm in arms):
                    raise ValueError("raw fallback budget exceeds pool capacity")
                pilot = [
                    row
                    for arm in arms
                    for row in pool[arm][: allocation[arm]]
                ]
                pilot_means = _raw_means(pilot, arms)
                selected = min(
                    arms, key=lambda arm: (pilot_means[arm], arms.index(arm))
                )
                regret = evaluation_means[selected] - evaluation_means[human_best]
                records[budget]["human_only_balanced"].append(
                    {
                        "regret": regret,
                        "exact": float(selected == human_best),
                        "practical": float(regret <= practical_tolerance),
                    }
                )
    result: dict[str, Any] = {}
    for budget in budgets:
        result[str(budget)] = {}
        for policy, policy_rows in records[budget].items():
            if not policy_rows:
                result[str(budget)][policy] = {
                    "status": "not_estimable_at_zero_humans",
                    "human_observations": budget,
                }
                continue
            result[str(budget)][policy] = {
                "status": "estimated",
                "human_observations": budget,
                "acquisition_evaluation_replicates": len(policy_rows),
                "mean_regret": fmean(row["regret"] for row in policy_rows),
                "exact_choice_rate": fmean(row["exact"] for row in policy_rows),
                "practical_reliability_rate": fmean(
                    row["practical"] for row in policy_rows
                ),
            }
    return {
        "budgets": list(budgets),
        "partitions": partitions,
        "fold_count": fold_count,
        "seed": seed,
        "outcome_unit": "usd_per_month",
        "direction": "lower_is_better",
        "pilot_evaluation_people_disjoint": True,
        "sampling_without_replacement": True,
        "nested_arm_prefixes_within_policy": True,
        "participant_rows_serialized": 0,
        "by_budget": result,
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_confirmation_scoring_protocol(root: Path) -> dict[str, Any]:
    """Freeze scoring, source projections, uncertainty, and fallback pre-reveal."""

    aggregation = verify_envelope(root / AGGREGATION_PATH, require_blinded=True)
    if aggregation.get("status") != (
        "complete_frozen_outcome_blind_confirmation_aggregation_stop"
    ) or aggregation.get("confirmation_outcomes_accessed") is not False:
        raise ValueError("outcome-blind aggregation is not scoring-ready")
    evidence = verify_envelope(
        root / DEFAULT_DEVELOPMENT_EVIDENCE_PATH, require_blinded=False
    )
    evidence_by_id = {row["experiment_id"]: row for row in evidence["tasks"]}
    calibration = [
        EffectCalibrationTask(
            experiment_id=experiment_id,
            synthetic_effects=evidence_by_id[experiment_id][
                "synthetic_treatment_effects"
            ],
            human_effects=evidence_by_id[experiment_id]["human_treatment_effects"],
        )
        for experiment_id in DEVELOPMENT_IDS
    ]
    prior = fit_effect_prior(calibration, minimum_variance=1e-6)
    verified_sources: dict[str, Any] = {}
    for experiment_id, raw_source in SOURCE_ARCHIVES.items():
        source = dict(raw_source)
        path = root / source["path"]
        if _file_sha256(path) != source["outer_sha256"]:
            raise ValueError(f"{experiment_id} source archive hash mismatch")
        with ZipFile(path) as archive:
            if source["member"] not in archive.namelist():
                raise ValueError(f"{experiment_id} source member is missing")
            member = archive.read(source["member"])
        if sha256(member).hexdigest() != source["member_sha256"]:
            raise ValueError(f"{experiment_id} source member hash mismatch")
        source["member_size_bytes"] = len(member)
        verified_sources[experiment_id] = source
    protocol = {
        "schema_version": "confirmation_scoring_protocol.v1",
        "status": "frozen_before_confirmation_outcome_access",
        "experiment_ids": list(CONFIRMATION_IDS),
        "aggregation_payload_sha256": payload_hash(aggregation),
        "development_evidence_payload_sha256": payload_hash(evidence),
        "recommendations_may_change": False,
        "diagnostics_may_change": False,
        "threshold_tuning": "forbidden",
        "human_scoring": {
            "primary_unit": "experiment",
            "arm_estimator": "frozen_task_specific_hajek_or_unweighted_mean",
            "treatment_effect_reference": "task_control_arm_id",
            "choice_rule": "maximize_frozen_decision_score_with_source_arm_order_ties",
            "headline_metrics": [
                "mean_absolute_treatment_effect_error",
                "treatment_effect_sign_accuracy",
                "correct_intervention_choice",
                "decision_regret",
            ],
            "participant_within_arm_bootstrap": {
                "replicates": 2000,
                "seed": 2026081405,
                "confidence_level": 0.95,
            },
            "experiment_cluster_bootstrap": {
                "replicates": 10000,
                "seed": 2026081404,
                "confidence_level": 0.95,
            },
            "tcg8p": "raw_usd_per_month_lower_is_better_reported_separately",
            "pooled_normalized_experiment_ids": list(CONFIRMATION_IDS[1:]),
        },
        "trust_evaluation": {
            "ranking_source": "frozen_confirmation_aggregation",
            "exact_error_risk_coverage_experiment_ids": list(CONFIRMATION_IDS),
            "normalized_regret_risk_coverage_experiment_ids": list(
                CONFIRMATION_IDS[1:]
            ),
            "coverage_counts": {"50_percent": 3, "75_percent": 5, "100_percent": 6},
            "learned_threshold": None,
            "accept_abstain_policy": "not_validated_not_deployed",
            "minimum_class_count_for_auc": 3,
        },
        "human_fallback": {
            "budgets": [0, 10, 25, 50, 100, 250],
            "partitions": 20,
            "fold_count": 10,
            "seed": 2026081403,
            "fixed_fusion_pseudocount_per_arm": 10,
            "primary_policies": ["synthetic_only", "human_only_balanced"],
            "negative_ablation_policies": [
                "synthetic_plus_balanced_fixed10",
                "synthetic_plus_hedged_fixed10",
                "synthetic_plus_balanced_eb",
                "synthetic_plus_hedged_eb",
            ],
            "effect_prior_frozen_on_all_development_experiments": asdict(prior),
            "participant_rows_serialized": 0,
            "sampling_without_replacement": True,
            "pilot_evaluation_people_disjoint": True,
            "predeclared_infeasible_task_budgets": {
                "Blair1131": {
                    "250": "smallest retained arm has 88 assigned rows before missingness; a disjoint ten-fold 84-per-arm pilot is impossible"
                }
            },
            "tcg8p_policy_scope": ["synthetic_only", "human_only_balanced"],
            "tcg8p_exclusion_from_eb_reason": "uncapped raw-unit outcome is not commensurate with the normalized development effect prior",
        },
        "source_projection": {
            "tcg8p": {
                "source": "SocSci210",
                "revision": "048481111a4425ed83dc0eacf15f8431f252b21a",
                "filters": {"study_id": "tcg8p", "task_num": 0},
                "columns": [
                    "study_id",
                    "sample_id",
                    "participant",
                    "condition_num",
                    "task_num",
                    "response",
                ],
            },
            **verified_sources,
        },
        "aggregate_only_output": True,
        "participant_rows_may_be_serialized": False,
        "model_calls_authorized": False,
        "modal_compute_authorized": False,
        "claim_boundary": (
            "Six-experiment noncanonical prospective confirmation; no universal "
            "trust calibration or canonical benchmark claim."
        ),
    }
    assert_blinded_payload(protocol)
    return json.loads(json.dumps(protocol, sort_keys=True, allow_nan=False))


def write_confirmation_scoring_protocol(root: Path) -> Path:
    path = root / SCORING_PROTOCOL_PATH
    freeze_envelope(
        build_confirmation_scoring_protocol(root), path, require_blinded=True
    )
    return path


def verify_confirmation_scoring_protocol(root: Path) -> dict[str, Any]:
    protocol = verify_envelope(root / SCORING_PROTOCOL_PATH, require_blinded=True)
    if protocol != build_confirmation_scoring_protocol(root):
        raise ValueError("confirmation scoring protocol does not replay")
    return protocol


def _integer(value: Any, *, field: str) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{field} must be a nonmissing integer")
    number = float(value)
    if not isfinite(number) or not number.is_integer():
        raise ValueError(f"{field} must be a finite integer")
    return int(number)


def _utility_mapping(task: Mapping[str, Any]) -> dict[int, float]:
    options = task["response_options"]
    if options and isinstance(options[0], Mapping):
        return {
            int(option["raw_value"]): float(option["normalized_utility"])
            for option in options
        }
    lower = float(task["scale_lower"])
    upper = float(task["scale_upper"])
    return {
        int(value): (float(value) - lower) / (upper - lower) for value in options
    }


def _r_project_sav(
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


def _project_source_rows(
    root: Path, *, experiment_id: str, source: Mapping[str, Any]
) -> list[dict[str, str]]:
    path = root / str(source["path"])
    if _file_sha256(path) != source["outer_sha256"]:
        raise ValueError("source archive hash drifted at reveal")
    with ZipFile(path) as archive:
        member = archive.read(str(source["member"]))
    if sha256(member).hexdigest() != source["member_sha256"]:
        raise ValueError("source participant member hash drifted at reveal")
    columns = tuple(str(value) for value in source["columns"])
    if experiment_id == "Blair1131":
        # The archived NORC export contains Windows smart-quote bytes. Decode
        # deterministically without altering any field values used for scoring.
        decoded = _decode_public_csv(member)
        reader = csv.DictReader(io.StringIO(decoded))
        if not set(columns).issubset(reader.fieldnames or ()):
            raise ValueError("authorized Blair columns are unavailable")
        return [
            {"participant_id": str(index), **{column: row[column] for column in columns}}
            for index, row in enumerate(reader, start=1)
        ]
    with TemporaryDirectory(prefix=f"intervenebench-{experiment_id}-reveal-") as tmp:
        temporary = Path(tmp)
        sav_path = temporary / "source.sav"
        csv_path = temporary / "projection.csv"
        sav_path.write_bytes(member)
        _r_project_sav(sav_path, csv_path, columns)
        with csv_path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))


def _valid_weight(value: Any) -> float:
    weight = float(value)
    if not isfinite(weight) or weight <= 0.0:
        raise ValueError("source survey weights must be positive and finite")
    return weight


def _decode_public_csv(member: bytes) -> str:
    """Decode hash-pinned public CSV exports using a frozen narrow policy."""

    try:
        return member.decode("utf-8-sig")
    except UnicodeDecodeError:
        return member.decode("cp1252")


def read_confirmation_outcomes(
    root: Path,
    *,
    protocol: Mapping[str, Any],
    tasks: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[RevealedOutcomeObservation]]:
    """Read only the six authorized projections after the reveal gate opens."""

    if tuple(protocol["experiment_ids"]) != CONFIRMATION_IDS:
        raise PermissionError("confirmation scoring scope drifted")
    results: dict[str, list[RevealedOutcomeObservation]] = {}
    source = protocol["source_projection"]["tcg8p"]
    shards = sorted(
        (
            root
            / "data/raw/socsci210"
            / source["revision"]
            / "data"
        ).glob("*.parquet")
    )
    tables = [
        pq.read_table(
            path,
            columns=source["columns"],
            filters=[("study_id", "=", "tcg8p"), ("task_num", "=", 0)],
        )
        for path in shards
    ]
    table = pa.concat_tables(tables)
    tcg_task = tasks["tcg8p"]
    arm_by_condition = {
        int(arm["condition_num"]): str(arm["arm_id"])
        for arm in tcg_task["arms"]
    }
    missing = {int(value) for value in tcg_task["valid_response"]["missing_codes"]}
    tcg_rows = []
    for row in table.to_pylist():
        if row["response"] is None:
            continue
        value = _integer(row["response"], field="tcg8p response")
        if value in missing:
            continue
        if value < int(tcg_task["valid_response"]["lower_bound"]):
            raise ValueError("tcg8p response violates its lower bound")
        condition = _integer(row["condition_num"], field="tcg8p condition")
        if condition not in arm_by_condition:
            raise ValueError("tcg8p condition is outside the action set")
        tcg_rows.append(
            RevealedOutcomeObservation(
                participant_id=f"{row['sample_id']}:{row['participant']}",
                arm_id=arm_by_condition[condition],
                raw_value=float(value),
                decision_score=-float(value),
                weight=1.0,
            )
        )
    results["tcg8p"] = tcg_rows

    for experiment_id in CONFIRMATION_IDS[1:]:
        task = tasks[experiment_id]
        mapping = task["source_variable_mapping"]
        raw_rows = _project_source_rows(
            root,
            experiment_id=experiment_id,
            source=protocol["source_projection"][experiment_id],
        )
        utility = _utility_mapping(task)
        valid = set(utility)
        missing = {int(value) for value in mapping["missing_outcome_codes"]}
        observations: list[RevealedOutcomeObservation] = []
        if experiment_id == "pb2rr":
            arm_by_assignment = {
                1: "hispanic_population_growth_article",
                2: "hispanic_population_growth_article",
                3: "iphone_growth_control_article",
                4: "iphone_growth_control_article",
            }
            name_by_value = {
                int(level["source_value"]): str(level["nuisance_id"])
                for level in task["nuisance_factor"]["levels"]
            }
            for row in raw_rows:
                if not row["XTESS187"] or not row["Q4"]:
                    continue
                assignment = _integer(row["XTESS187"], field="XTESS187")
                value = _integer(row["Q4"], field="Q4")
                if value in missing:
                    continue
                if assignment not in arm_by_assignment or value not in valid:
                    raise ValueError("pb2rr source coding violates its contract")
                nuisance = name_by_value[_integer(row["DOV_INSERT_NAME"], field="DOV_INSERT_NAME")]
                observations.append(
                    RevealedOutcomeObservation(
                        participant_id=f"source-row:{row['participant_id']}",
                        arm_id=arm_by_assignment[assignment],
                        raw_value=float(value),
                        decision_score=utility[value],
                        weight=_valid_weight(row["weight"]),
                        fold_stratum_id=nuisance,
                        standardization_cell_id=nuisance,
                    )
                )
        elif experiment_id == "z358z":
            arm_by_option = {1: "general_notification_policy", 2: "verbal_consent_policy"}
            for row in raw_rows:
                if not row["XTESS175"] or not row["DOV_OPTION"] or not row["Q3a"]:
                    continue
                if _integer(row["XTESS175"], field="XTESS175") != 1:
                    continue
                option = _integer(row["DOV_OPTION"], field="DOV_OPTION")
                value = _integer(row["Q3a"], field="Q3a")
                if value in missing:
                    continue
                if option not in arm_by_option or value not in valid:
                    raise ValueError("z358z source coding violates its contract")
                observations.append(
                    RevealedOutcomeObservation(
                        f"source-row:{row['participant_id']}",
                        arm_by_option[option],
                        float(value),
                        utility[value],
                        _valid_weight(row["weight"]),
                    )
                )
        elif experiment_id == "ShannonS2":
            arms = [str(arm["arm_id"]) for arm in task["arms"]]
            outcome_by_arm = mapping["arm_outcome_variables"]
            for row in raw_rows:
                if not row["GROUP"]:
                    continue
                group = _integer(row["GROUP"], field="GROUP")
                if group not in range(1, 7):
                    raise ValueError("Shannon group is outside the action set")
                arm = arms[group - 1]
                outcome_column = outcome_by_arm[arm]
                if not row[outcome_column]:
                    continue
                value = _integer(row[outcome_column], field=outcome_column)
                if value in missing:
                    continue
                if value not in valid:
                    raise ValueError("Shannon outcome violates its contract")
                observations.append(
                    RevealedOutcomeObservation(
                        f"source-row:{row['participant_id']}",
                        arm,
                        float(value),
                        utility[value],
                        _valid_weight(row["WEIGHT2"]),
                    )
                )
        elif experiment_id == "Blair1131":
            arms = [str(arm["arm_id"]) for arm in task["arms"]]
            for row in raw_rows:
                if not row["BLAIR"] or not row["Q3"]:
                    continue
                assignment = _integer(row["BLAIR"], field="BLAIR")
                if assignment not in range(1, 4):
                    continue
                value = _integer(row["Q3"], field="Q3")
                if value in missing:
                    continue
                if value not in valid:
                    raise ValueError("Blair outcome violates its contract")
                name = f"president_name_{_integer(row['RND_02'], field='RND_02')}"
                observations.append(
                    RevealedOutcomeObservation(
                        f"source-row:{row['participant_id']}",
                        arms[assignment - 1],
                        float(value),
                        utility[value],
                        _valid_weight(row["WEIGHT"]),
                        fold_stratum_id=name,
                    )
                )
        elif experiment_id == "KlarS44":
            arms = [str(arm["arm_id"]) for arm in task["arms"]]
            for row in raw_rows:
                if not row["P_KLAR"] or not row["Q2_KLAR"]:
                    continue
                assignment = _integer(row["P_KLAR"], field="P_KLAR")
                value = _integer(row["Q2_KLAR"], field="Q2_KLAR")
                if value in missing:
                    continue
                if assignment not in range(1, 4) or value not in valid:
                    raise ValueError("Klar source coding violates its contract")
                order = f"comodule_order_{_integer(row['RND_00'], field='RND_00')}"
                observations.append(
                    RevealedOutcomeObservation(
                        f"source-row:{row['participant_id']}",
                        arms[assignment - 1],
                        float(value),
                        utility[value],
                        _valid_weight(row["WEIGHT"]),
                        fold_stratum_id=order,
                    )
                )
        else:
            raise PermissionError("unsupported confirmation outcome source")
        results[experiment_id] = observations

    for experiment_id, rows in results.items():
        participant_ids = [row.participant_id for row in rows]
        if not rows or len(participant_ids) != len(set(participant_ids)):
            raise ValueError(f"{experiment_id} outcome rows are empty or duplicated")
        expected_arms = {str(arm["arm_id"]) for arm in tasks[experiment_id]["arms"]}
        if {row.arm_id for row in rows} != expected_arms:
            raise ValueError(f"{experiment_id} complete cases do not cover every arm")
    return results


def summarize_human_arms(
    observations: Sequence[RevealedOutcomeObservation],
    *,
    arm_ids: Sequence[str],
    outcome_unit: str,
    use_weights: bool = True,
    standardize_cells: bool = True,
) -> HumanArmSummary:
    """Compute task-frozen arm means, including optional equal-cell averaging."""

    grouped: dict[str, list[RevealedOutcomeObservation]] = defaultdict(list)
    for row in observations:
        grouped[row.arm_id].append(row)
    means: dict[str, float] = {}
    counts: dict[str, int] = {}
    for arm in arm_ids:
        rows = grouped[arm]
        if not rows:
            raise ValueError("human arm summary lacks an action")
        counts[arm] = len(rows)
        cell_ids = sorted(
            {row.standardization_cell_id for row in rows if row.standardization_cell_id}
        )
        if not standardize_cells or not cell_ids:
            cell_ids = [""]
        cell_means = []
        for cell_id in cell_ids:
            cell_rows = (
                [row for row in rows if row.standardization_cell_id == cell_id]
                if cell_id
                else rows
            )
            weights = [row.weight if use_weights else 1.0 for row in cell_rows]
            denominator = fsum(weights)
            cell_means.append(
                fsum(
                    weight * row.decision_score
                    for weight, row in zip(weights, cell_rows, strict=True)
                )
                / denominator
            )
        means[arm] = fmean(cell_means)
    return HumanArmSummary(
        arm_means=means,
        complete_case_count_by_arm=counts,
        outcome_unit=outcome_unit,
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def participant_bootstrap_score(
    observations: Sequence[RevealedOutcomeObservation],
    *,
    arm_ids: Sequence[str],
    control_arm_id: str,
    synthetic_arm_scores: Mapping[str, float],
    selected_arm_id: str,
    practical_tolerance: float,
    outcome_unit: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Resample within randomized arm/cell and serialize aggregate intervals only."""

    if replicates <= 0:
        raise ValueError("participant bootstrap requires positive replicates")
    groups: dict[tuple[str, str], list[RevealedOutcomeObservation]] = defaultdict(list)
    for row in observations:
        groups[(row.arm_id, row.standardization_cell_id)].append(row)
    rng = random.Random(seed)
    scores = []
    for _ in range(replicates):
        sampled: list[RevealedOutcomeObservation] = []
        for key in sorted(groups):
            source = groups[key]
            sampled.extend(rng.choice(source) for _ in source)
        human = summarize_human_arms(
            sampled, arm_ids=arm_ids, outcome_unit=outcome_unit
        )
        scores.append(
            score_synthetic_recommendation(
                arm_ids=arm_ids,
                control_arm_id=control_arm_id,
                human=human,
                synthetic_arm_scores=synthetic_arm_scores,
                selected_arm_id=selected_arm_id,
                practical_tolerance=practical_tolerance,
            )
        )
    interval = lambda values: [_quantile(values, 0.025), _quantile(values, 0.975)]
    contrasts = [arm for arm in arm_ids if arm != control_arm_id]
    return {
        "replicates": replicates,
        "seed": seed,
        "confidence_level": 0.95,
        "arm_mean_confidence_intervals": {
            arm: interval([score["human_arm_means"][arm] for score in scores])
            for arm in arm_ids
        },
        "treatment_effect_confidence_intervals": {
            arm: interval(
                [score["human_treatment_effects"][arm] for score in scores]
            )
            for arm in contrasts
        },
        "decision_regret_confidence_interval": interval(
            [score["decision_regret"] for score in scores]
        ),
        "bootstrap_exact_choice_rate": fmean(
            float(score["exact_choice"]) for score in scores
        ),
        "bootstrap_practical_reliability_rate": fmean(
            float(score["practically_reliable"]) for score in scores
        ),
        "participant_rows_serialized": 0,
    }


def validate_confirmation_reveal_authorization(
    authorization: Mapping[str, Any],
    *,
    aggregation_payload_sha256: str,
    scoring_protocol_payload_sha256: str,
    development_evidence_payload_sha256: str,
) -> None:
    """Validate the exact reveal/scoring scope and forbid method changes."""

    if authorization.get("schema_version") != "confirmation_reveal_authorization.v1" or authorization.get("status") != "authorized_frozen_confirmation_outcome_scoring":
        raise PermissionError("invalid confirmation reveal authorization")
    required_true = {
        "outcome_reveal_authorized",
        "aggregate_scoring_authorized",
        "human_fallback_authorized",
    }
    if any(authorization.get(key) is not True for key in required_true):
        raise PermissionError("confirmation outcome scoring is not authorized")
    required_false = {
        "model_calls_authorized",
        "modal_compute_authorized",
        "model_download_authorized",
        "recommendation_changes_authorized",
        "diagnostic_changes_authorized",
        "threshold_tuning_authorized",
        "participant_row_serialization_authorized",
        "automatic_followup_authorized",
    }
    if any(authorization.get(key) is not False for key in required_false):
        raise PermissionError("confirmation reveal authority expanded")
    expected = {
        "aggregation_payload_sha256": aggregation_payload_sha256,
        "scoring_protocol_payload_sha256": scoring_protocol_payload_sha256,
        "development_evidence_payload_sha256": development_evidence_payload_sha256,
        "authorized_experiment_ids": list(CONFIRMATION_IDS),
    }
    if any(authorization.get(key) != value for key, value in expected.items()):
        raise PermissionError("confirmation reveal authorization binding drifted")
