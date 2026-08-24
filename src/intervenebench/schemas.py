"""Typed, outcome-blind schemas for the Phase 1 benchmark path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite
from typing import Any


class DesignType(StrEnum):
    BETWEEN_SUBJECT = "between_subject"
    WITHIN_SUBJECT = "within_subject"
    FACTORIAL = "factorial"
    REPEATED_MEASURES = "repeated_measures"
    MIXED = "mixed"
    OTHER = "other"


class OutcomeFamily(StrEnum):
    BINARY = "binary"
    ORDINAL = "ordinal"
    BOUNDED_NUMERIC = "bounded_numeric"
    CONTINUOUS = "continuous"
    COUNT = "count"
    CATEGORICAL = "categorical"
    TEXT = "text"


class OutcomeDirection(StrEnum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class SplitName(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class Arm:
    arm_id: str
    description: str
    deployable: bool = True

    def __post_init__(self) -> None:
        if not self.arm_id.strip():
            raise ValueError("arm_id must be non-empty")
        if not self.description.strip():
            raise ValueError("arm description must be non-empty")


@dataclass(frozen=True, slots=True)
class NuisanceStratum:
    """One prespecified level over which a factorial decision is standardized."""

    stratum_id: str
    weight: float

    def __post_init__(self) -> None:
        if not self.stratum_id.strip():
            raise ValueError("nuisance stratum ID must be non-empty")
        if not isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("nuisance stratum weight must be positive and finite")


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    experiment_id: str
    paradigm_group: str
    gold_audit: bool = False

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must be non-empty")
        if not self.paradigm_group.strip():
            raise ValueError("paradigm_group must be non-empty")


@dataclass(frozen=True, slots=True)
class DecisionTask:
    task_id: str
    experiment_id: str
    source_id: str
    paradigm_group: str
    design_type: DesignType
    randomization_unit: str
    arms: tuple[Arm, ...]
    control_arm_id: str
    primary_outcome_id: str
    outcome_family: OutcomeFamily
    response_options: tuple[float, ...]
    scale_lower: float
    scale_upper: float
    direction: OutcomeDirection
    observations_per_arm: tuple[tuple[str, int], ...]
    modality: str = "text"
    weighting_rule: str = "unweighted_released_analytic_sample"
    missingness_rule: str = "complete_case_for_declared_outcome"
    practical_regret_tolerance: float = 0.05
    eligible: bool = True
    exclusion_reason: str | None = None
    gold_audit: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("task_id", self.task_id),
            ("experiment_id", self.experiment_id),
            ("source_id", self.source_id),
            ("paradigm_group", self.paradigm_group),
            ("randomization_unit", self.randomization_unit),
            ("primary_outcome_id", self.primary_outcome_id),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")

        if len(self.arms) < 2:
            raise ValueError("a decision task requires at least two arms")
        arm_ids = [arm.arm_id for arm in self.arms]
        if len(arm_ids) != len(set(arm_ids)):
            raise ValueError("arm IDs must be unique within a task")
        if self.control_arm_id not in arm_ids:
            raise ValueError("control arm must be present in the arm set")
        if not next(arm for arm in self.arms if arm.arm_id == self.control_arm_id).deployable:
            raise ValueError("control arm must be an admissible action")

        if not (isfinite(self.scale_lower) and isfinite(self.scale_upper)):
            raise ValueError("scale bounds must be finite")
        if self.scale_lower >= self.scale_upper:
            raise ValueError("scale_lower must be smaller than scale_upper")
        if len(self.response_options) < 2:
            raise ValueError("at least two response options are required")
        if len(self.response_options) != len(set(self.response_options)):
            raise ValueError("response options must be unique")
        if any(
            not isfinite(option)
            or option < self.scale_lower
            or option > self.scale_upper
            for option in self.response_options
        ):
            raise ValueError("response options must be finite and within questionnaire bounds")
        if self.outcome_family is OutcomeFamily.BINARY and len(self.response_options) != 2:
            raise ValueError("binary outcomes require exactly two response options")

        observation_counts = dict(self.observations_per_arm)
        if len(observation_counts) != len(self.observations_per_arm):
            raise ValueError("observations_per_arm contains duplicate arm IDs")
        if set(observation_counts) != set(arm_ids):
            raise ValueError("observations_per_arm must cover every arm exactly once")
        if any(not isinstance(count, int) or count < 0 for count in observation_counts.values()):
            raise ValueError("arm observation counts must be non-negative integers")

        if not 0.0 <= self.practical_regret_tolerance <= 1.0:
            raise ValueError("practical regret tolerance must be in [0, 1]")
        if self.eligible and self.exclusion_reason is not None:
            raise ValueError("eligible tasks cannot have an exclusion reason")
        if not self.eligible and not (self.exclusion_reason or "").strip():
            raise ValueError("ineligible tasks require an exclusion reason")

    @property
    def admissible_arm_ids(self) -> tuple[str, ...]:
        return tuple(arm.arm_id for arm in self.arms if arm.deployable)

    def validate_phase1(self, minimum_observations_per_arm: int = 100) -> None:
        """Fail closed when a task is outside the locked Phase 1 estimator."""

        if not self.eligible:
            raise ValueError("Phase 1 task must be eligible")
        if self.design_type is not DesignType.BETWEEN_SUBJECT:
            raise ValueError("Phase 1 supports only between-subject tasks")
        if self.randomization_unit.casefold() != "participant":
            raise ValueError("Phase 1 requires participant-level randomization")
        if not 2 <= len(self.admissible_arm_ids) <= 4:
            raise ValueError("Phase 1 requires two to four admissible arms")
        if self.modality.casefold() != "text":
            raise ValueError("Phase 1 supports text-only interventions")
        if self.outcome_family not in {OutcomeFamily.BINARY, OutcomeFamily.ORDINAL}:
            raise ValueError("Phase 1 supports binary or ordinal outcomes")
        observation_counts = dict(self.observations_per_arm)
        if any(
            observation_counts[arm_id] < minimum_observations_per_arm
            for arm_id in self.admissible_arm_ids
        ):
            raise ValueError("Phase 1 requires the declared minimum observations per arm")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NuisanceStandardizedDecisionTask:
    """A randomized factorial task scored after prespecified standardization.

    The deployable arms are the decision alternatives. ``nuisance_strata`` are
    independently randomized implementation factors that are not decision arms.
    Every deployable-arm-by-stratum cell must have source support.
    """

    task_id: str
    experiment_id: str
    source_id: str
    paradigm_group: str
    randomization_unit: str
    arms: tuple[Arm, ...]
    control_arm_id: str
    primary_outcome_id: str
    outcome_family: OutcomeFamily
    response_options: tuple[float, ...]
    scale_lower: float
    scale_upper: float
    direction: OutcomeDirection
    nuisance_strata: tuple[NuisanceStratum, ...]
    observations_per_cell: tuple[tuple[str, str, int], ...]
    modality: str
    practical_regret_tolerance: float = 0.05

    def __post_init__(self) -> None:
        for name, value in (
            ("task_id", self.task_id),
            ("experiment_id", self.experiment_id),
            ("source_id", self.source_id),
            ("paradigm_group", self.paradigm_group),
            ("randomization_unit", self.randomization_unit),
            ("primary_outcome_id", self.primary_outcome_id),
            ("modality", self.modality),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")

        if len(self.arms) < 2:
            raise ValueError("a standardized decision task requires at least two arms")
        arm_ids = tuple(arm.arm_id for arm in self.arms)
        if len(set(arm_ids)) != len(arm_ids):
            raise ValueError("arm IDs must be unique within a task")
        if self.control_arm_id not in arm_ids:
            raise ValueError("control arm must be present in the arm set")
        if any(not arm.deployable for arm in self.arms):
            raise ValueError("every retained standardized arm must be deployable")

        stratum_ids = tuple(stratum.stratum_id for stratum in self.nuisance_strata)
        if len(stratum_ids) < 2 or len(set(stratum_ids)) != len(stratum_ids):
            raise ValueError("nuisance strata must contain at least two unique IDs")
        if abs(sum(stratum.weight for stratum in self.nuisance_strata) - 1.0) > 1e-9:
            raise ValueError("nuisance stratum weights must sum to one")

        cells = {(arm_id, stratum_id): count for arm_id, stratum_id, count in self.observations_per_cell}
        if len(cells) != len(self.observations_per_cell):
            raise ValueError("observations_per_cell contains duplicate cells")
        expected = {(arm_id, stratum_id) for arm_id in arm_ids for stratum_id in stratum_ids}
        if set(cells) != expected:
            raise ValueError("observations_per_cell must cover every arm-by-stratum cell")
        if any(not isinstance(count, int) or count <= 0 for count in cells.values()):
            raise ValueError("standardized cell counts must be positive integers")

        if not (isfinite(self.scale_lower) and isfinite(self.scale_upper)):
            raise ValueError("scale bounds must be finite")
        if self.scale_lower >= self.scale_upper:
            raise ValueError("scale_lower must be smaller than scale_upper")
        if len(self.response_options) < 2 or len(set(self.response_options)) != len(self.response_options):
            raise ValueError("response options must contain unique values")
        if any(
            not isfinite(option)
            or option < self.scale_lower
            or option > self.scale_upper
            for option in self.response_options
        ):
            raise ValueError("response options must lie within the declared bounds")
        if not 0.0 <= self.practical_regret_tolerance <= 1.0:
            raise ValueError("practical regret tolerance must be in [0, 1]")

    def validate_factorial_extension(self, minimum_observations_per_arm: int = 100) -> None:
        if self.randomization_unit.casefold() != "participant":
            raise ValueError("factorial extension requires participant randomization")
        if self.outcome_family not in {
            OutcomeFamily.BINARY,
            OutcomeFamily.ORDINAL,
            OutcomeFamily.BOUNDED_NUMERIC,
        }:
            raise ValueError("factorial extension requires a bounded response family")
        if self.modality.casefold() not in {"image", "mixed"}:
            raise ValueError("this extension requires a declared image-bearing modality")
        arm_totals = {arm.arm_id: 0 for arm in self.arms}
        for arm_id, _, count in self.observations_per_cell:
            arm_totals[arm_id] += count
        if any(count < minimum_observations_per_arm for count in arm_totals.values()):
            raise ValueError("factorial extension lacks the declared arm-level support")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ContinuousDecisionTask:
    """Outcome-blind contract for an uncapped continuous decision task."""

    task_id: str
    experiment_id: str
    source_id: str
    paradigm_group: str
    design_type: DesignType
    randomization_unit: str
    arms: tuple[Arm, ...]
    control_arm_id: str
    primary_outcome_id: str
    outcome_unit: str
    direction: OutcomeDirection
    released_rows_per_arm: tuple[tuple[str, int], ...]
    valid_lower_bound: float | None
    valid_upper_bound: float | None
    missing_codes: tuple[float, ...]
    integer_only: bool
    location_estimand: str
    robustness_estimands: tuple[str, ...]
    practical_regret_tolerance: float
    modality: str = "text"
    weighting_rule: str = "unweighted_released_analytic_sample"
    missingness_rule: str = "complete_case_excluding_source_declared_missing_codes"

    def __post_init__(self) -> None:
        for name, value in (
            ("task_id", self.task_id),
            ("experiment_id", self.experiment_id),
            ("source_id", self.source_id),
            ("paradigm_group", self.paradigm_group),
            ("randomization_unit", self.randomization_unit),
            ("primary_outcome_id", self.primary_outcome_id),
            ("outcome_unit", self.outcome_unit),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if len(self.arms) < 2:
            raise ValueError("a continuous decision task requires at least two arms")
        arm_ids = [arm.arm_id for arm in self.arms]
        if len(arm_ids) != len(set(arm_ids)):
            raise ValueError("arm IDs must be unique within a task")
        if self.control_arm_id not in arm_ids:
            raise ValueError("control arm must be present in the arm set")
        if not next(arm for arm in self.arms if arm.arm_id == self.control_arm_id).deployable:
            raise ValueError("control arm must be an admissible action")

        counts = dict(self.released_rows_per_arm)
        if len(counts) != len(self.released_rows_per_arm):
            raise ValueError("released_rows_per_arm contains duplicate arm IDs")
        if set(counts) != set(arm_ids):
            raise ValueError("released_rows_per_arm must cover every arm exactly once")
        if any(not isinstance(count, int) or count < 0 for count in counts.values()):
            raise ValueError("released row counts must be non-negative integers")

        for bound in (self.valid_lower_bound, self.valid_upper_bound):
            if bound is not None and not isfinite(bound):
                raise ValueError("continuous bounds must be finite when declared")
        if (
            self.valid_lower_bound is not None
            and self.valid_upper_bound is not None
            and self.valid_lower_bound >= self.valid_upper_bound
        ):
            raise ValueError("continuous lower bound must be smaller than upper bound")
        if len(self.missing_codes) != len(set(self.missing_codes)) or any(
            not isfinite(code) for code in self.missing_codes
        ):
            raise ValueError("missing codes must be unique finite numbers")
        if self.location_estimand != "mean":
            raise ValueError("the continuous extension currently requires the source-aligned mean")
        if self.robustness_estimands != ("median",):
            raise ValueError("the continuous extension requires median robustness analysis")
        if (
            not isfinite(self.practical_regret_tolerance)
            or self.practical_regret_tolerance < 0
        ):
            raise ValueError("continuous practical regret tolerance must be non-negative")

    def validate_continuous_extension(
        self, minimum_released_rows_per_arm: int = 100
    ) -> None:
        if self.design_type is not DesignType.BETWEEN_SUBJECT:
            raise ValueError("continuous extension supports only between-subject tasks")
        if self.randomization_unit.casefold() != "participant":
            raise ValueError("continuous extension requires participant randomization")
        if not 2 <= len(tuple(arm for arm in self.arms if arm.deployable)) <= 4:
            raise ValueError("continuous extension requires two to four deployable arms")
        if self.modality.casefold() != "text":
            raise ValueError("continuous extension currently supports text-only interventions")
        counts = dict(self.released_rows_per_arm)
        if any(counts[arm.arm_id] < minimum_released_rows_per_arm for arm in self.arms):
            raise ValueError("continuous extension requires the declared minimum rows per arm")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
