"""Column-level access barriers for the pinned SocSci210 snapshot."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from .protocol import (
    verify_bound_continuous_reveal_authorization,
    verify_bound_reveal_authorization,
)


AUDIT_COLUMNS = frozenset(
    {
        "study_id",
        "sample_id",
        "participant",
        "condition_num",
        "task_num",
        "stimuli",
    }
)
FORBIDDEN_BLINDED_COLUMNS = frozenset({"response", "reasoning", "prompt", "demographic"})
REVEALED_OUTCOME_COLUMNS = frozenset(
    {"study_id", "sample_id", "participant", "condition_num", "task_num", "response"}
)


def _validate_requested_columns(columns: Iterable[str], allowed: frozenset[str]) -> tuple[str, ...]:
    requested = tuple(columns)
    forbidden = sorted(set(requested) - allowed)
    if forbidden:
        raise ValueError(f"columns are not allowed in this data view: {forbidden}")
    return requested


def read_audit_view(paths: Iterable[Path], columns: Iterable[str]) -> pa.Table:
    """Read structural fields only; response-derived and persona fields fail closed."""

    requested = _validate_requested_columns(columns, AUDIT_COLUMNS)
    files = tuple(paths)
    if not files:
        raise ValueError("at least one Parquet path is required")
    return pa.concat_tables(
        [pq.read_table(path, columns=list(requested)) for path in files]
    )


def read_revealed_outcomes(
    paths: Iterable[Path],
    *,
    experiment_id: str,
    recommendation_path: Path,
    split_manifest_path: Path,
    decision_task_path: Path,
) -> pa.Table:
    """Reveal only the frozen validation task after verifying its recommendation."""

    if not experiment_id.strip():
        raise ValueError("experiment_id must be non-empty")
    recommendation = verify_bound_reveal_authorization(
        recommendation_path,
        experiment_id=experiment_id,
        split_manifest_path=split_manifest_path,
        decision_task_path=decision_task_path,
    )
    task_num = recommendation["task_num"]
    files = tuple(paths)
    if not files:
        raise ValueError("at least one Parquet path is required")
    return pa.concat_tables(
        [
            pq.read_table(
                path,
                columns=sorted(REVEALED_OUTCOME_COLUMNS),
                filters=[
                    ("study_id", "=", experiment_id),
                    ("task_num", "=", task_num),
                ],
            )
            for path in files
        ]
    )


def read_revealed_continuous_outcomes(
    paths: Iterable[Path],
    *,
    experiment_id: str,
    recommendation_path: Path,
    split_manifest_path: Path,
    decision_task_path: Path,
) -> pa.Table:
    """Reveal one continuous validation outcome after estimator-bound verification."""

    if not experiment_id.strip():
        raise ValueError("experiment_id must be non-empty")
    recommendation = verify_bound_continuous_reveal_authorization(
        recommendation_path,
        experiment_id=experiment_id,
        split_manifest_path=split_manifest_path,
        decision_task_path=decision_task_path,
    )
    task_num = recommendation["task_num"]
    files = tuple(paths)
    if not files:
        raise ValueError("at least one Parquet path is required")
    return pa.concat_tables(
        [
            pq.read_table(
                path,
                columns=sorted(REVEALED_OUTCOME_COLUMNS),
                filters=[
                    ("study_id", "=", experiment_id),
                    ("task_num", "=", task_num),
                ],
            )
            for path in files
        ]
    )
