#!/usr/bin/env python3
"""Pure-local authority wrapper for report-eval image materialization and generation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

from intervenebench.evidence_report_eval import (
    build_blinded_label_queue,
    deterministic_report_checks,
    parse_report_output,
    render_labeling_app,
    verify_report_generation_plan,
)
from intervenebench.evidence_report_execution import (
    build_report_execution_freeze,
    validate_report_execution_authorization,
    validate_report_import_smoke_authorization,
    validate_report_import_smoke_result,
    validate_report_materialization_authorization,
)
from intervenebench.protocol import freeze_envelope, payload_hash, verify_envelope


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "infra/modal/evidence_report_app.py"
PACKET_PATH = ROOT / (
    "data/manifests/qualitative_eval/"
    "intervenebench_report_evidence_packet_v1.json"
)
PROTOCOL_PATH = ROOT / "data/manifests/research/evidence_report_eval_v1.json"
PLAN_PATH = ROOT / (
    "data/manifests/qualitative_eval/report_generation_plan_v1.json"
)
FREEZE_PATH = ROOT / "configs/simulators/evidence_report_execution_v1.json"
ARTIFACT_ROOT = ROOT / "artifacts/evidence_report_eval"


class _Reporter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, state: str, **details: Any) -> None:
        row = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "state": state,
            **details,
        }
        line = json.dumps(row, sort_keys=True, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)


def _load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    packet = json.loads(PACKET_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    plan = verify_envelope(PLAN_PATH, require_blinded=True)
    verify_report_generation_plan(plan, packet, protocol)
    freeze = verify_envelope(FREEZE_PATH, require_blinded=True)
    expected = build_report_execution_freeze(ROOT, packet, protocol, plan)
    if freeze != expected:
        raise ValueError("persisted report execution freeze does not replay")
    return packet, protocol, plan, freeze


def _load_modal_app() -> Any:
    spec = importlib.util.spec_from_file_location("evidence_report_app", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load evidence-report Modal app")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def materialize(authorization_path: Path, output_path: Path) -> None:
    _, _, _, freeze = _load_inputs()
    authorization = verify_envelope(authorization_path, require_blinded=True)
    validate_report_materialization_authorization(authorization, freeze)
    if output_path.exists():
        raise FileExistsError(f"create-only materialization exists: {output_path}")
    reporter = _Reporter(output_path.with_suffix(".progress.jsonl"))
    reporter.emit("local_authorization_verified", model_calls_authorized=False)
    modal_app = _load_modal_app()
    reporter.emit("modal_images_hydration_started")
    with modal_app.app.run(
        name=freeze["runtime"]["app_name"],
        environment_name="main",
        detach=False,
        interactive=False,
    ):
        image_ids = modal_app.materialized_report_image_ids()
    digest = freeze_envelope(
        {
            "schema_version": "intervenebench.report_eval_materialization.v1",
            "status": "two_images_materialized_zero_inference_stop",
            "execution_freeze_payload_sha256": payload_hash(freeze),
            "authorization_payload_sha256": payload_hash(authorization),
            "modal_image_ids": image_ids,
            "model_downloads": 0,
            "inference_calls": 0,
            "participant_rows_accessed": 0,
            "experiment_level_human_scores_accessed": False,
            "automatic_next_stage": False,
        },
        output_path,
        require_blinded=True,
    )
    reporter.emit("materialization_frozen_stop", payload_sha256=digest, **image_ids)


def smoke(
    authorization_path: Path,
    materialization_path: Path,
    output_path: Path,
) -> None:
    _, _, _, freeze = _load_inputs()
    authorization = verify_envelope(authorization_path, require_blinded=True)
    materialization = verify_envelope(materialization_path, require_blinded=True)
    image_ids = materialization.get("modal_image_ids")
    if not isinstance(image_ids, Mapping):
        raise ValueError("materialization lacks Modal image IDs")
    if materialization.get("execution_freeze_payload_sha256") != payload_hash(freeze):
        raise ValueError("materialization is bound to another execution freeze")
    validate_report_import_smoke_authorization(
        authorization,
        freeze,
        materialized_image_ids=image_ids,
    )
    if output_path.exists():
        raise FileExistsError(f"create-only import-smoke result exists: {output_path}")
    reporter = _Reporter(output_path.with_suffix(".progress.jsonl"))
    reporter.emit("local_import_smoke_authorization_verified", exact_call_count=2)
    modal_app = _load_modal_app()
    with modal_app.app.run(
        name=freeze["runtime"]["app_name"],
        environment_name="main",
        detach=False,
        interactive=False,
    ):
        hydrated = modal_app.materialized_report_image_ids()
        if hydrated != dict(image_ids):
            raise RuntimeError("import-smoke image IDs differ from materialization")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                "qwen": pool.submit(
                    modal_app.smoke_qwen_evidence_report_import.remote,
                    payload_hash(freeze),
                    hydrated["qwen"],
                ),
                "mistral": pool.submit(
                    modal_app.smoke_mistral_evidence_report_import.remote,
                    payload_hash(freeze),
                    hydrated["mistral"],
                ),
            }
            smokes = {kind: future.result() for kind, future in futures.items()}
    result = {
        "schema_version": "intervenebench.report_eval_import_smoke.v1",
        "status": "two_remote_imports_verified_zero_inference",
        "execution_freeze_payload_sha256": payload_hash(freeze),
        "authorization_payload_sha256": payload_hash(authorization),
        "materialization_payload_sha256": payload_hash(materialization),
        "modal_image_ids": dict(image_ids),
        "import_smoke_call_count": 2,
        "smokes": smokes,
        "model_downloads": 0,
        "inference_calls": 0,
        "participant_rows_accessed": 0,
        "experiment_level_human_scores_accessed": False,
        "automatic_next_stage": False,
    }
    validate_report_import_smoke_result(
        result,
        freeze,
        materialized_image_ids=image_ids,
    )
    digest = freeze_envelope(result, output_path, require_blinded=True)
    reporter.emit("remote_import_smokes_frozen_stop", payload_sha256=digest)


def _validate_raw_group(
    group: Mapping[str, Any],
    *,
    model_role: str,
    expected_calls: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if (
        group.get("schema_version")
        != "intervenebench.evidence_report_raw_group.v1"
        or group.get("model_role") != model_role
        or group.get("attempt_count") != len(expected_calls)
        or group.get("model_downloaded") is not False
        or group.get("participant_rows_accessed") != 0
        or group.get("experiment_level_human_scores_accessed") is not False
        or group.get("human_labels_accessed") is not False
        or group.get("automatic_next_stage") is not False
    ):
        raise RuntimeError("remote evidence-report group attestation drifted")
    results = group.get("results")
    if not isinstance(results, list) or len(results) != len(expected_calls):
        raise RuntimeError("remote evidence-report group has incomplete results")
    returned = {result.get("call_id"): result for result in results}
    if set(returned) != set(expected_calls):
        raise RuntimeError("remote evidence-report call IDs do not match frozen group")
    checked: list[dict[str, Any]] = []
    for call_id, call in expected_calls.items():
        result = returned[call_id]
        if set(result) != {
            "call_id",
            "model_role",
            "prompt_sha256",
            "raw_text",
            "runtime_attestation",
        }:
            raise RuntimeError("raw report result fields drifted")
        if (
            result["model_role"] != model_role
            or result["prompt_sha256"] != call["prompt_sha256"]
            or not isinstance(result["raw_text"], str)
        ):
            raise RuntimeError("raw report result binding drifted")
        runtime = result["runtime_attestation"]
        if not isinstance(runtime, Mapping) or any(
            runtime.get(key) != value
            for key, value in {
                "call_id": call_id,
                "prompt_sha256": call["prompt_sha256"],
                "seed": call["seed"],
            }.items()
        ):
            raise RuntimeError("raw report runtime binding drifted")
        checked.append(dict(result))
    return checked


def execute(
    authorization_path: Path,
    materialization_path: Path,
    import_smoke_path: Path,
    run_id: str,
) -> None:
    packet, protocol, plan, freeze = _load_inputs()
    authorization = verify_envelope(authorization_path, require_blinded=True)
    materialization = verify_envelope(materialization_path, require_blinded=True)
    import_smoke = verify_envelope(import_smoke_path, require_blinded=True)
    image_ids = materialization.get("modal_image_ids")
    if not isinstance(image_ids, Mapping):
        raise ValueError("materialization lacks Modal image IDs")
    validate_report_execution_authorization(
        authorization,
        freeze,
        materialized_image_ids=image_ids,
        import_smoke_result=import_smoke,
    )
    if materialization.get("execution_freeze_payload_sha256") != payload_hash(freeze):
        raise ValueError("materialization is bound to another execution freeze")
    if import_smoke.get("materialization_payload_sha256") != payload_hash(
        materialization
    ):
        raise ValueError("import smoke is bound to another materialization")
    run_root = ARTIFACT_ROOT / run_id
    if run_root.exists():
        raise FileExistsError(f"create-only report run exists: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    reporter = _Reporter(run_root / "progress.jsonl")
    reporter.emit(
        "local_execution_authorization_verified",
        exact_call_count=freeze["planned_call_count"],
        hard_incremental_cost_cap_usd=freeze["limits"][
            "hard_incremental_cost_cap_usd"
        ],
    )
    calls_by_role = {
        role: {
            call["call_id"]: call
            for call in plan["calls"]
            if call["model_role"] == role
        }
        for role in freeze["planned_calls_by_model_role"]
    }
    cache_sha = {
        model["model_role"]: model["cache_attestation_payload_sha256"]
        for model in freeze["models"]
    }
    modal_app = _load_modal_app()
    started = time.monotonic()
    raw_groups: dict[str, dict[str, Any]] = {}
    with modal_app.app.run(
        name=freeze["runtime"]["app_name"],
        environment_name="main",
        detach=False,
        interactive=False,
    ):
        hydrated = modal_app.materialized_report_image_ids()
        validate_report_execution_authorization(
            authorization,
            freeze,
            materialized_image_ids=hydrated,
            import_smoke_result=import_smoke,
        )
        reporter.emit("modal_image_bindings_verified", **hydrated)

        def dispatch(role: str) -> tuple[str, Mapping[str, Any]]:
            reporter.emit("model_group_submitted", model_role=role, call_count=16)
            if role == "mistral_small_3_1_24b_cross_family":
                result = modal_app.run_mistral_evidence_report_group.remote(
                    payload_hash(freeze),
                    hydrated["mistral"],
                    cache_sha[role],
                )
            else:
                result = modal_app.run_qwen_evidence_report_group.remote(
                    role,
                    payload_hash(freeze),
                    hydrated["qwen"],
                    cache_sha[role],
                )
            return role, result

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(dispatch, role) for role in calls_by_role]
            for future in as_completed(futures):
                if time.monotonic() - started > freeze["limits"][
                    "maximum_wall_clock_seconds"
                ]:
                    raise TimeoutError("evidence-report wall-clock ledger expired")
                role, raw_group = future.result()
                checked = _validate_raw_group(
                    raw_group,
                    model_role=role,
                    expected_calls=calls_by_role[role],
                )
                raw_groups[role] = dict(raw_group)
                digest = freeze_envelope(
                    {
                        "schema_version": "intervenebench.evidence_report_raw_group.v1",
                        "model_role": role,
                        "execution_freeze_payload_sha256": payload_hash(freeze),
                        "raw_group": raw_group,
                    },
                    run_root / "raw" / f"{role}.json",
                    require_blinded=True,
                )
                reporter.emit(
                    "model_group_raw_frozen",
                    model_role=role,
                    result_count=len(checked),
                    payload_sha256=digest,
                )

    if set(raw_groups) != set(calls_by_role):
        raise RuntimeError("report generation did not complete all model groups")
    scenarios = {
        scenario["scenario_id"]: scenario
        for scenario in protocol["generation"]["scenarios"]
    }
    plan_by_id = {call["call_id"]: call for call in plan["calls"]}
    parsed_records: list[dict[str, Any]] = []
    parse_failures: list[dict[str, Any]] = []
    deterministic: dict[str, dict[str, Any]] = {}
    for role, raw_group in raw_groups.items():
        for raw in raw_group["results"]:
            call = plan_by_id[raw["call_id"]]
            scenario = scenarios[call["scenario_id"]]
            try:
                report = parse_report_output(raw["raw_text"], packet, scenario)
            except ValueError as error:
                parse_failures.append(
                    {
                        "call_id": call["call_id"],
                        "scenario_id": call["scenario_id"],
                        "scenario_split": call["scenario_split"],
                        "prompt_variant": call["prompt_variant"],
                        "model_role": role,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
                continue
            parsed_records.append(
                {
                    "report_id": call["call_id"],
                    "scenario_id": call["scenario_id"],
                    "prompt_variant": call["prompt_variant"],
                    "model_role": role,
                    "report": report,
                }
            )
            deterministic[call["call_id"]] = deterministic_report_checks(
                report, packet, scenario
            )
    parsed_records.sort(key=lambda record: record["report_id"])
    parse_failures.sort(key=lambda record: record["call_id"])
    parsed_payload = {
        "schema_version": "intervenebench.evidence_report_parsed_panel.v1",
        "status": "strict_parse_complete_no_repairs",
        "evaluation_id": protocol["evaluation_id"],
        "execution_freeze_payload_sha256": payload_hash(freeze),
        "attempt_count": 48,
        "valid_report_count": len(parsed_records),
        "strict_parse_failure_count": len(parse_failures),
        "reports": parsed_records,
        "parse_failures": parse_failures,
        "deterministic_checks_by_report_id": dict(sorted(deterministic.items())),
        "semantic_repairs": 0,
        "automatic_retries": 0,
        "participant_rows_accessed": 0,
        "experiment_level_human_scores_accessed": False,
        "automatic_judging_performed": False,
        "automatic_next_stage": False,
    }
    parsed_sha = freeze_envelope(
        parsed_payload,
        run_root / "parsed_reports.json",
        require_blinded=True,
    )
    queue, key = build_blinded_label_queue(parsed_records, protocol, packet)
    queue_sha = freeze_envelope(
        queue, run_root / "labeling" / "blinded_queue.json", require_blinded=True
    )
    key_sha = freeze_envelope(
        key, run_root / "labeling" / "private_blinding_key.json"
    )
    labeler_path = run_root / "labeling" / "labeler.html"
    labeler_path.parent.mkdir(parents=True, exist_ok=True)
    with labeler_path.open("x", encoding="utf-8") as stream:
        stream.write(render_labeling_app(queue, protocol["rubric"]))
    labeler_sha = payload_hash(labeler_path.read_text(encoding="utf-8"))
    final = {
        "schema_version": "intervenebench.evidence_report_generation_run.v1",
        "status": "generation_and_strict_parse_frozen_stop_before_human_labeling",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "execution_freeze_payload_sha256": payload_hash(freeze),
        "execution_authorization_payload_sha256": payload_hash(authorization),
        "materialization_payload_sha256": payload_hash(materialization),
        "attempt_count": 48,
        "valid_report_count": len(parsed_records),
        "strict_parse_failure_count": len(parse_failures),
        "parsed_panel_payload_sha256": parsed_sha,
        "blinded_label_queue_payload_sha256": queue_sha,
        "private_blinding_key_payload_sha256": key_sha,
        "offline_labeler_sha256": labeler_sha,
        "model_downloads": 0,
        "automatic_retries": 0,
        "reserve_calls": 0,
        "semantic_repairs": 0,
        "participant_rows_accessed": 0,
        "experiment_level_human_scores_accessed": False,
        "human_labels_collected": False,
        "automated_judge_calls": 0,
        "automatic_next_stage": False,
        "wall_clock_seconds": time.monotonic() - started,
    }
    final_sha = freeze_envelope(
        final, run_root / "final_manifest.json", require_blinded=True
    )
    reporter.emit(
        "generation_frozen_stop_before_labeling",
        final_manifest_payload_sha256=final_sha,
        valid_report_count=len(parsed_records),
        strict_parse_failure_count=len(parse_failures),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--authorization", type=Path, required=True)
    materialize_parser.add_argument("--output", type=Path, required=True)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--authorization", type=Path, required=True)
    smoke_parser.add_argument("--materialization", type=Path, required=True)
    smoke_parser.add_argument("--output", type=Path, required=True)
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--authorization", type=Path, required=True)
    execute_parser.add_argument("--materialization", type=Path, required=True)
    execute_parser.add_argument("--import-smoke", type=Path, required=True)
    execute_parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.command == "materialize":
        materialize(args.authorization, args.output)
    elif args.command == "smoke":
        smoke(args.authorization, args.materialization, args.output)
    else:
        execute(
            args.authorization,
            args.materialization,
            args.import_smoke,
            args.run_id,
        )


if __name__ == "__main__":
    main()
