from __future__ import annotations

import json
from pathlib import Path

import pytest

from intervenebench.model_exposure import (
    checkpoint_compatibility,
    read_study_mapping,
)


ROOT = Path(__file__).resolve().parents[1]
MAPPING_PATH = (
    ROOT
    / "data/raw/socsci210/048481111a4425ed83dc0eacf15f8431f252b21a"
    / "metadata/participant_mapping.json"
)


def test_socrates_exposure_tracks_released_study_mapping() -> None:
    mapping = read_study_mapping(MAPPING_PATH)
    unseen = checkpoint_compatibility(
        experiment_id="5vm8g", source_stratum="socsci210", mapping=mapping
    )
    seen = checkpoint_compatibility(
        experiment_id="xc4yq", source_stratum="socsci210", mapping=mapping
    )
    assert unseen.status == "confirmed_checkpoint_unseen"
    assert unseen.primary_eligible is True
    assert seen.status == "training_exposed"
    assert seen.primary_eligible is False


def test_external_tasks_are_absent_unless_they_share_a_socsci_fielding() -> None:
    mapping = read_study_mapping(MAPPING_PATH)
    external = checkpoint_compatibility(
        experiment_id="ShannonS2", source_stratum="external", mapping=mapping
    )
    shared = checkpoint_compatibility(
        experiment_id="KlarS44",
        source_stratum="external",
        mapping=mapping,
        equivalent_socsci210_id="xtvu5",
    )
    assert external.status == "external_absent_from_socsci210_training_universe"
    assert external.checkpoint_experiment_id is None
    assert shared.status == "confirmed_checkpoint_unseen"
    assert shared.checkpoint_experiment_id == "xtvu5"


def test_unknown_socsci_disposition_fails_closed() -> None:
    mapping = read_study_mapping(MAPPING_PATH)
    with pytest.raises(ValueError, match="no disposition"):
        checkpoint_compatibility(
            experiment_id="not-a-released-study",
            source_stratum="socsci210",
            mapping=mapping,
        )


def test_checkpoint_mapping_rejects_overlap(tmp_path: Path) -> None:
    path = tmp_path / "mapping.json"
    path.write_text(
        json.dumps({"seen": ["duplicate"], "unseen": ["duplicate"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlaps"):
        read_study_mapping(path)
