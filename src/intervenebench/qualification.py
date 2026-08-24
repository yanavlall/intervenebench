"""Outcome-independent selection primitives for source-qualification audits."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable


@dataclass(frozen=True, slots=True)
class AuditBatchEntry:
    audit_order: int
    experiment_id: str
    selection_hash: str


def freeze_audit_batch(
    candidate_ids: Iterable[str],
    *,
    excluded_ids: Iterable[str] = (),
    batch_size: int,
    seed_label: str,
) -> tuple[AuditBatchEntry, ...]:
    """Select a reproducible audit batch using IDs and a declared seed only.

    The caller is responsible for deriving ``candidate_ids`` without outcome or
    simulator information. Hash ordering prevents a reviewer from substituting
    convenient studies after reading their source designs.
    """

    candidates = tuple(candidate_ids)
    excluded = tuple(excluded_ids)
    if not seed_label.strip():
        raise ValueError("seed_label must be non-empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if any(not experiment_id.strip() for experiment_id in candidates + excluded):
        raise ValueError("experiment identifiers must be non-empty")
    if len(candidates) != len(set(candidates)):
        raise ValueError("candidate experiment identifiers must be unique")
    if len(excluded) != len(set(excluded)):
        raise ValueError("excluded experiment identifiers must be unique")

    excluded_set = set(excluded)
    available = sorted(set(candidates) - excluded_set)
    if len(available) < batch_size:
        raise ValueError(
            f"requested {batch_size} experiments but only {len(available)} candidates remain"
        )

    ranked = sorted(
        (
            sha256(f"{seed_label}:{experiment_id}".encode()).hexdigest(),
            experiment_id,
        )
        for experiment_id in available
    )
    return tuple(
        AuditBatchEntry(order, experiment_id, digest)
        for order, (digest, experiment_id) in enumerate(ranked[:batch_size], start=1)
    )
