from __future__ import annotations

from pathlib import Path

from intervenebench.research_progress import (
    ContractBatchRow,
    evaluate_contract_progress,
    load_contract_batch,
)


ROOT = Path(__file__).resolve().parents[1]
TRACKER = ROOT / "data/manifests/audits/depth_first_contract_batches.csv"


def _row(
    order: int,
    *,
    status: str,
    blocker: str = "",
    minutes: int = 0,
) -> ContractBatchRow:
    runnable = status == "runnable"
    return ContractBatchRow(
        batch_id="batch",
        candidate_id=f"candidate-{order}",
        queue_order=order,
        status=status,
        gate_reached="gate_4_runnable" if runnable else "gate_1_stopped",
        focused_minutes=minutes,
        blocker_code=blocker,
        outcome_access="sealed",
        runnable_contract=runnable,
    )


def test_live_tracker_records_progress_without_counting_gate_one_as_runnable() -> None:
    rows = load_contract_batch(TRACKER)
    progress = evaluate_contract_progress(rows)
    assert progress.candidate_count == len(rows)
    assert progress.candidate_count >= 3
    assert progress.runnable_count == sum(row.runnable_contract for row in rows)
    assert progress.in_progress_count == sum(
        row.status == "in_progress" for row in rows
    )
    assert progress.queued_count == sum(row.status == "queued" for row in rows)
    a42yg = next(row for row in rows if row.candidate_id == "socsci210:a42yg")
    assert a42yg.gate_reached == "gate_2_contract"
    assert a42yg.runnable_contract is False


def test_completed_zero_yield_batch_triggers_lane_pause() -> None:
    rows = (
        _row(1, status="excluded", blocker="one"),
        _row(2, status="blocked", blocker="two"),
        _row(3, status="excluded", blocker="three"),
    )
    progress = evaluate_contract_progress(rows)
    assert progress.pause_current_lane is True
    assert "zero runnable" in progress.reasons[0]


def test_consecutive_shared_blocker_triggers_pivot() -> None:
    rows = (
        _row(1, status="excluded", blocker="missing_asset"),
        _row(2, status="blocked", blocker="missing_asset"),
        _row(3, status="queued"),
    )
    progress = evaluate_contract_progress(rows)
    assert progress.repeated_blocker_pivot is True


def test_shared_blockers_separated_by_runnable_candidate_do_not_trigger_pivot() -> None:
    rows = (
        _row(1, status="excluded", blocker="missing_asset"),
        _row(2, status="runnable"),
        _row(3, status="blocked", blocker="missing_asset"),
    )
    progress = evaluate_contract_progress(rows)
    assert progress.repeated_blocker_pivot is False


def test_runnable_task_prevents_zero_yield_pause_and_time_cap_is_explicit() -> None:
    rows = (
        _row(1, status="runnable", minutes=500),
        _row(2, status="excluded", blocker="utility", minutes=500),
        _row(3, status="blocked", blocker="mapping", minutes=300),
    )
    progress = evaluate_contract_progress(rows, focused_hour_cap=20.0)
    assert progress.runnable_count == 1
    assert progress.pause_current_lane is False
    assert progress.time_cap_reached is True
