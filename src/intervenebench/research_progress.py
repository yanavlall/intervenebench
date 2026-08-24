"""Evaluate depth-first research progress and trigger frozen pivot conditions."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


TERMINAL_STATUSES = {"runnable", "excluded", "blocked"}
VALID_STATUSES = TERMINAL_STATUSES | {"queued", "in_progress"}
VALID_GATES = {
    "not_started",
    "gate_1_survived",
    "gate_1_stopped",
    "gate_2_contract",
    "gate_3_tested",
    "gate_4_runnable",
}


@dataclass(frozen=True, slots=True)
class ContractBatchRow:
    batch_id: str
    candidate_id: str
    queue_order: int
    status: str
    gate_reached: str
    focused_minutes: int
    blocker_code: str
    outcome_access: str
    runnable_contract: bool


@dataclass(frozen=True, slots=True)
class ResearchProgress:
    candidate_count: int
    runnable_count: int
    in_progress_count: int
    queued_count: int
    focused_hours: float
    pause_current_lane: bool
    repeated_blocker_pivot: bool
    time_cap_reached: bool
    reasons: tuple[str, ...]


def load_contract_batch(path: Path) -> tuple[ContractBatchRow, ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError("contract batch must contain candidates")
    rows: list[ContractBatchRow] = []
    seen_ids: set[str] = set()
    seen_orders: set[int] = set()
    for raw in raw_rows:
        candidate_id = raw["candidate_id"]
        queue_order = int(raw["queue_order"])
        status = raw["status"]
        gate = raw["gate_reached"]
        minutes = int(raw["focused_minutes"])
        runnable = raw["runnable_contract"].lower() == "true"
        if candidate_id in seen_ids or queue_order in seen_orders:
            raise ValueError("candidate IDs and queue orders must be unique")
        if status not in VALID_STATUSES or gate not in VALID_GATES:
            raise ValueError("unknown contract status or gate")
        if minutes < 0:
            raise ValueError("focused minutes cannot be negative")
        if raw["outcome_access"] != "sealed":
            raise ValueError("prospective contract candidates must stay sealed")
        if runnable != (status == "runnable" and gate == "gate_4_runnable"):
            raise ValueError("runnable status requires the complete Gate 4 state")
        if status in {"excluded", "blocked"} and not raw["blocker_code"]:
            raise ValueError("stopped candidates require a blocker code")
        seen_ids.add(candidate_id)
        seen_orders.add(queue_order)
        rows.append(
            ContractBatchRow(
                batch_id=raw["batch_id"],
                candidate_id=candidate_id,
                queue_order=queue_order,
                status=status,
                gate_reached=gate,
                focused_minutes=minutes,
                blocker_code=raw["blocker_code"],
                outcome_access=raw["outcome_access"],
                runnable_contract=runnable,
            )
        )
    if seen_orders != set(range(1, len(rows) + 1)):
        raise ValueError("queue orders must form a contiguous sequence")
    if len({row.batch_id for row in rows}) != 1:
        raise ValueError("one tracker file must represent one batch")
    return tuple(sorted(rows, key=lambda row: row.queue_order))


def evaluate_contract_progress(
    rows: Sequence[ContractBatchRow], *, focused_hour_cap: float = 20.0
) -> ResearchProgress:
    if not rows:
        raise ValueError("progress evaluation requires candidates")
    terminal = [row for row in rows if row.status in TERMINAL_STATUSES]
    runnable_count = sum(row.runnable_contract for row in rows)
    focused_hours = sum(row.focused_minutes for row in rows) / 60.0
    pause_lane = len(terminal) == len(rows) and runnable_count == 0

    repeated_blocker = any(
        first.status in {"excluded", "blocked"}
        and second.status in {"excluded", "blocked"}
        and bool(first.blocker_code)
        and first.blocker_code == second.blocker_code
        for first, second in zip(rows, rows[1:])
    )
    time_cap = focused_hours >= focused_hour_cap
    reasons: list[str] = []
    if pause_lane:
        reasons.append("completed batch yielded zero runnable contracts")
    if repeated_blocker:
        reasons.append("two consecutive stopped candidates share one blocker")
    if time_cap:
        reasons.append("focused contract-completion time cap reached")
    return ResearchProgress(
        candidate_count=len(rows),
        runnable_count=runnable_count,
        in_progress_count=sum(row.status == "in_progress" for row in rows),
        queued_count=sum(row.status == "queued" for row in rows),
        focused_hours=focused_hours,
        pause_current_lane=pause_lane,
        repeated_blocker_pivot=repeated_blocker,
        time_cap_reached=time_cap,
        reasons=tuple(reasons),
    )
