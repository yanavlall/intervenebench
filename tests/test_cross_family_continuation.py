from __future__ import annotations

import importlib.util
from pathlib import Path

from intervenebench.cross_family_continuation import (
    EXPECTED_CHUNK_COUNT,
    EXPECTED_REMAINING_CALL_COUNT,
    UNAVAILABLE_EXPERIMENT_ID,
    build_continuation_authorization,
    continuation_partition,
    unavailable_transport_rows,
    validate_continuation_authorization,
)
from intervenebench.cross_family_target_run import build_recommendations
from intervenebench.protocol import payload_hash


ROOT = Path(__file__).resolve().parents[1]


def _strict_output(request: dict) -> dict:
    value = {
        "schema_version": "intervenebench.cross_family_strict_call.v1",
        "call_id": request["call_id"],
        "model_id": request["model_id"],
        "experiment_id": request["experiment_id"],
        "arm_id": request["arm_id"],
        "nuisance_id": request["nuisance_id"],
        "answer_order": request["answer_order"],
        "stage": request["stage"],
        "prompt_variant": request["prompt_variant"],
        "prompt_sha256": request["source_prompt_sha256"],
        "request_payload_sha256": payload_hash(request),
        "runtime_attestation": {},
        "semantic_repair_used": False,
    }
    if request["method_id"] == "continuous_constrained_integer_generation.v1":
        value.update(
            {
                "raw_text": '{"predicted_value": 50}',
                "predicted_value": 50,
                "generation_seed": request["generation_seed"],
            }
        )
    else:
        values = request["source_option_values"]
        probability = 1.0 / len(values)
        value.update(
            {
                "probabilities_by_source_value": {
                    str(source_value): probability for source_value in values
                },
                "candidate_token_ids": list(range(len(values))),
            }
        )
    return value


def test_partition_is_exact_no_rerun_120_plus_504_in_63_small_chunks() -> None:
    partition = continuation_partition(ROOT)
    unavailable = partition["unavailable_requests"]
    remaining = partition["remaining_requests"]
    chunks = partition["chunks"]
    assert len(unavailable) == 120
    assert {row["experiment_id"] for row in unavailable} == {
        UNAVAILABLE_EXPERIMENT_ID
    }
    assert len(remaining) == EXPECTED_REMAINING_CALL_COUNT
    assert all(row["experiment_id"] != UNAVAILABLE_EXPERIMENT_ID for row in remaining)
    assert len(chunks) == EXPECTED_CHUNK_COUNT
    assert {chunk["call_count"] for chunk in chunks} == {8}
    unavailable_ids = {row["call_id"] for row in unavailable}
    chunk_ids = [call_id for chunk in chunks for call_id in chunk["call_ids"]]
    assert len(chunk_ids) == len(set(chunk_ids)) == 504
    assert unavailable_ids.isdisjoint(chunk_ids)
    assert chunk_ids == [row["call_id"] for row in remaining]


def test_continuation_authority_forbids_tcg8p_reruns_and_human_scoring() -> None:
    authorization = build_continuation_authorization(ROOT)
    validate_continuation_authorization(authorization, root=ROOT)
    assert authorization["remaining_call_count"] == 504
    assert authorization["maximum_attempt_count"] == 504
    assert authorization["chunk_count"] == 63
    assert authorization["chunk_size"] == 8
    assert authorization["tcg8p_rerun_authorized"] is False
    assert authorization["automatic_retry_authorized"] is False
    assert authorization["reserve_call_authorized"] is False
    assert authorization["model_download_authorized"] is False
    assert authorization["human_outcome_access_authorized"] is False
    assert authorization["participant_row_access_authorized"] is False
    assert authorization["regression_scoring_authorized"] is False
    assert authorization["automatic_next_stage_authorized"] is False
    assert authorization["new_chunk_failure_policy"].startswith("fail_stop")


def test_tcg8p_unavailable_ledger_yields_five_recommendations() -> None:
    partition = continuation_partition(ROOT)
    requests = [
        *partition["unavailable_requests"], *partition["remaining_requests"]
    ]
    failures = unavailable_transport_rows(partition["unavailable_requests"])
    strict_outputs = {
        request["call_id"]: _strict_output(request)
        for request in partition["remaining_requests"]
    }
    strict_hashes = {
        call_id: payload_hash(output) for call_id, output in strict_outputs.items()
    }
    recommendations = build_recommendations(
        ROOT,
        requests=requests,
        strict_outputs=strict_outputs,
        strict_output_hashes=strict_hashes,
        parse_failures=failures,
    )
    assert recommendations["planned_call_count"] == 624
    assert recommendations["strict_output_count"] == 504
    assert recommendations["strict_parse_failure_count"] == 120
    assert recommendations["recommendation_count"] == 5
    assert recommendations["unavailable_experiment_count"] == 1
    assert recommendations["unavailable_experiments"] == [
        {
            "experiment_id": "tcg8p",
            "reason": "one_or_more_strict_parse_failures_no_rerun",
            "strict_parse_failure_count": 120,
        }
    ]
    assert recommendations["reruns_made"] == 0
    assert recommendations["human_outcomes_accessed"] is False
    assert recommendations["human_outcome_scoring_performed"] is False


def test_continuation_wrapper_validates_before_modal_and_has_no_scoring() -> None:
    path = ROOT / "scripts/run_cross_family_continuation.py"
    source = path.read_text(encoding="utf-8")
    assert "import modal\n" not in source
    assert source.index("validate_continuation_authorization") < source.index(
        "modal_app = _load_app("
    )
    assert "chunk_active" in source
    assert "call.cancel(terminate_containers=True)" in source
    assert '"automatic_retries": 0' in source
    assert '"tcg8p_reruns": 0' in source
    assert '"reserve_calls": 0' in source
    assert '"semantic_repairs": 0' in source
    assert '"human_outcomes_accessed": False' in source
    assert '"human_outcome_scoring_performed": False' in source
    assert "score_confirmation" not in source
    assert "ThreadPoolExecutor" not in source
    spec = importlib.util.spec_from_file_location("cross_family_continuation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
