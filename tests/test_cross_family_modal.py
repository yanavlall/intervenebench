from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from intervenebench.cross_family_modal import (
    DEFAULT_CROSS_FAMILY_MODAL_FREEZE_PATH,
    build_cross_family_modal_freeze,
    parse_strict_nonnegative_integer,
    validate_cache_authorization,
    validate_canary_authorization,
    validate_forced_choice_probe,
    validate_materialization_authorization,
    verify_cross_family_modal_freeze,
)
from intervenebench.protocol import payload_hash


ROOT = Path(__file__).resolve().parents[1]


def _materialization_authorization(freeze: dict) -> dict:
    return {
        "schema_version": "intervenebench.cross_family_materialization_authorization.v1",
        "status": "authorized_image_build_zero_download_zero_inference",
        "freeze_payload_sha256": payload_hash(freeze),
        "modal_image_materialization_authorized": True,
        "model_download_authorized": False,
        "modal_compute_authorized": False,
        "paid_inference_authorized": False,
        "candidate_inference_authorized": False,
        "target_call_authorized": False,
        "automatic_retry_authorized": False,
        "reserve_call_authorized": False,
        "human_outcome_access_authorized": False,
        "participant_row_access_authorized": False,
        "participant_row_serialization_authorized": False,
        "regression_scoring_authorized": False,
        "automatic_next_stage_authorized": False,
    }


def _cache_authorization(freeze: dict) -> dict:
    return {
        "schema_version": "intervenebench.cross_family_cache_authorization.v1",
        "status": "authorized_exact_public_checkpoint_cache_only",
        "freeze_payload_sha256": payload_hash(freeze),
        "modal_image_id": "im-test",
        "model_id": freeze["model"]["model_id"],
        "checkpoint_commit": freeze["model"]["checkpoint_commit"],
        "model_source_manifest_payload_sha256": freeze["model"][
            "source_manifest_payload_sha256"
        ],
        "maximum_download_bytes": freeze["cache"]["maximum_download_bytes"],
        "hard_incremental_cost_cap_usd": freeze["cache"][
            "hard_incremental_cost_cap_usd"
        ],
        "modal_image_materialization_authorized": False,
        "model_download_authorized": True,
        "modal_compute_authorized": True,
        "paid_inference_authorized": False,
        "candidate_inference_authorized": False,
        "target_call_authorized": False,
        "automatic_retry_authorized": False,
        "reserve_call_authorized": False,
        "human_outcome_access_authorized": False,
        "participant_row_access_authorized": False,
        "participant_row_serialization_authorized": False,
        "regression_scoring_authorized": False,
        "automatic_next_stage_authorized": False,
    }


def _canary_authorization(freeze: dict) -> dict:
    return {
        "schema_version": "intervenebench.cross_family_canary_authorization.v1",
        "status": "authorized_three_synthetic_canaries_only",
        "freeze_payload_sha256": payload_hash(freeze),
        "modal_image_id": "im-test",
        "cache_attestation_sha256": "a" * 64,
        "canary_manifest_payload_sha256": freeze["canary"]["manifest_payload_sha256"],
        "planned_canary_call_count": 3,
        "maximum_attempt_count": 3,
        "hard_incremental_cost_cap_usd": freeze["canary"][
            "hard_incremental_cost_cap_usd"
        ],
        "modal_image_materialization_authorized": False,
        "model_download_authorized": False,
        "modal_compute_authorized": True,
        "paid_inference_authorized": True,
        "candidate_inference_authorized": True,
        "target_call_authorized": False,
        "automatic_retry_authorized": False,
        "reserve_call_authorized": False,
        "human_outcome_access_authorized": False,
        "participant_row_access_authorized": False,
        "participant_row_serialization_authorized": False,
        "regression_scoring_authorized": False,
        "automatic_next_stage_authorized": False,
    }


def test_modal_freeze_is_zero_authority_and_contains_no_target_prompts() -> None:
    freeze = build_cross_family_modal_freeze(ROOT)

    assert set(freeze["authority"].values()) == {False}
    assert freeze["status"] == "frozen_nonexecuting_zero_authority"
    assert freeze["target_execution"]["authorized"] is False
    assert freeze["target_execution"]["planned_call_count"] == 624
    assert freeze["canary"]["planned_call_count"] == 3
    assert freeze["canary"]["target_assets_included"] is False
    serialized = repr(freeze)
    assert "confirmation_call_plan_v1.json" not in serialized
    assert "cross_family_call_plan_v1.json" not in serialized
    assert freeze["target_execution"]["call_plan_payload_sha256"]


def test_modal_freeze_pins_original_format_runtime_and_replays() -> None:
    freeze = verify_cross_family_modal_freeze(
        ROOT, ROOT / DEFAULT_CROSS_FAMILY_MODAL_FREEZE_PATH
    )

    assert freeze == build_cross_family_modal_freeze(ROOT)
    assert freeze["runtime"]["python_version"] == "3.11"
    assert freeze["runtime"]["vllm_version"] == "0.8.5"
    assert freeze["runtime"]["mistral_common_version"] == "1.5.4"
    assert freeze["runtime"]["gpu"] == "A100-80GB:1"
    assert freeze["runtime"]["config_format"] == "mistral"
    assert freeze["runtime"]["load_format"] == "mistral"
    assert freeze["runtime"]["tokenizer_mode"] == "mistral"
    assert freeze["runtime"]["dependency_lock_sha256"]
    assert set(freeze["implementation_hashes"]) >= {
        "src/intervenebench/cross_family_modal.py",
        "infra/modal/cross_family_app.py",
        "scripts/run_cross_family_preflight.py",
        "scripts/build_cross_family_modal_freeze.py",
    }


def test_three_authority_stages_are_narrow_and_fail_closed() -> None:
    freeze = build_cross_family_modal_freeze(ROOT)
    materialize = _materialization_authorization(freeze)
    cache = _cache_authorization(freeze)
    canary = _canary_authorization(freeze)

    validate_materialization_authorization(materialize, freeze=freeze)
    validate_cache_authorization(cache, freeze=freeze, modal_image_id="im-test")
    validate_canary_authorization(
        canary,
        freeze=freeze,
        modal_image_id="im-test",
        cache_attestation_sha256="a" * 64,
    )

    widened = deepcopy(materialize)
    widened["model_download_authorized"] = True
    with pytest.raises(PermissionError, match="authority"):
        validate_materialization_authorization(widened, freeze=freeze)

    widened = deepcopy(cache)
    widened["candidate_inference_authorized"] = True
    with pytest.raises(PermissionError, match="authority"):
        validate_cache_authorization(widened, freeze=freeze, modal_image_id="im-test")

    widened = deepcopy(canary)
    widened["target_call_authorized"] = True
    with pytest.raises(PermissionError, match="authority"):
        validate_canary_authorization(
            widened,
            freeze=freeze,
            modal_image_id="im-test",
            cache_attestation_sha256="a" * 64,
        )
    widened = deepcopy(canary)
    widened["unreviewed_extra_authority"] = True
    with pytest.raises(PermissionError, match="fields"):
        validate_canary_authorization(
            widened,
            freeze=freeze,
            modal_image_id="im-test",
            cache_attestation_sha256="a" * 64,
        )


def test_forced_choice_probe_requires_complete_normalized_code_distribution() -> None:
    parsed = validate_forced_choice_probe(
        {
            "schema_version": "intervenebench.masked_next_token_probe.v1",
            "answer_codes": ["A", "B", "C"],
            "token_ids": [11, 12, 13],
            "probabilities": {"A": 0.2, "B": 0.3, "C": 0.5},
            "sampled_code": "C",
            "free_generation_used": False,
            "engine_probe_tokens": 1,
        },
        expected_codes=["A", "B", "C"],
    )
    assert parsed["probabilities"]["C"] == pytest.approx(0.5)

    bad = deepcopy(parsed)
    bad["probabilities"]["C"] = 0.4
    with pytest.raises(ValueError, match="sum to one"):
        validate_forced_choice_probe(bad, expected_codes=["A", "B", "C"])
    bad = deepcopy(parsed)
    bad["free_generation_used"] = True
    with pytest.raises(ValueError, match="free generation"):
        validate_forced_choice_probe(bad, expected_codes=["A", "B", "C"])


@pytest.mark.parametrize("text,expected", [("0", 0), ("42", 42), ("9999", 9999)])
def test_continuous_canary_parser_is_strict(text: str, expected: int) -> None:
    assert parse_strict_nonnegative_integer(text) == expected


@pytest.mark.parametrize("text", ["-1", "1.0", " 1", "1\n", "one", "", "01"])
def test_continuous_canary_parser_rejects_repairs(text: str) -> None:
    with pytest.raises(ValueError):
        parse_strict_nonnegative_integer(text)


def test_local_wrapper_validates_authority_before_importing_modal_app() -> None:
    path = ROOT / "scripts/run_cross_family_preflight.py"
    source = path.read_text(encoding="utf-8")
    assert "import modal\n" not in source
    assert source.index("_validate_local_authorization") < source.index(
        "importlib.import_module"
    )
    spec = importlib.util.spec_from_file_location("cross_family_wrapper_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_modal_app_has_only_cache_and_synthetic_canary_entrypoints() -> None:
    source = (ROOT / "infra/modal/cross_family_app.py").read_text(encoding="utf-8")
    assert "cache_cross_family_checkpoint" in source
    assert "cross_family_startup_smoke" in source
    assert "run_cross_family_canary" in source
    assert "run_cross_family_target" not in source
    assert "cross_family_call_plan_v1.json" not in source
    assert "confirmation_call_plan_v1.json" not in source
    assert "block_network=True" in source
    assert "create_if_missing=False" in source
    assert "retries=0" in source


def test_remote_source_is_packaged_at_its_import_namespace_path() -> None:
    source = (ROOT / "infra/modal/cross_family_app.py").read_text(encoding="utf-8")
    assert 'Path("/root/infra/modal/cross_family_app.py")' in source
    assert '.add_local_file(SOURCE_PATH, str(REMOTE_SOURCE_PATH), copy=True)' in source
    assert '.add_local_file(SOURCE_PATH, "/root/cross_family_app.py"' not in source


def test_wrapper_has_live_ledger_smoke_gate_and_deadline_cancellation() -> None:
    source = (ROOT / "scripts/run_cross_family_preflight.py").read_text(
        encoding="utf-8"
    )
    assert "--progress-log" in source
    assert "modal.enable_output" in source
    assert "cross_family_startup_smoke" in source
    assert "deadline_exceeded_cancelled" in source
    assert "call.cancel(terminate_containers=True)" in source


def test_watchdog_records_heartbeat_and_completion(tmp_path: Path) -> None:
    wrapper_path = ROOT / "scripts/run_cross_family_preflight.py"
    spec = importlib.util.spec_from_file_location("cross_family_watchdog_test", wrapper_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeRemoteTimeout(Exception):
        pass

    class FakeCall:
        object_id = "fc-test"

        def __init__(self) -> None:
            self.polls = 0

        def get(self, timeout: float) -> str:
            self.polls += 1
            if self.polls == 1:
                raise FakeRemoteTimeout
            return "completed-result"

        def cancel(self, terminate_containers: bool = False) -> None:
            raise AssertionError("successful calls must not be cancelled")

    call = FakeCall()

    class FakeFunction:
        def spawn(self, *args: object) -> FakeCall:
            assert args == ("bound-input",)
            return call

    ticks = iter([0.0, 1.0, 2.0, 3.0, 4.0])
    progress = tmp_path / "progress.jsonl"
    result = module._run_remote_with_watchdog(
        FakeFunction(),
        ("bound-input",),
        reporter=module._ProgressReporter(progress),
        label="test_stage",
        heartbeat_seconds=10.0,
        maximum_seconds=100.0,
        timeout_error_type=FakeRemoteTimeout,
        clock=lambda: next(ticks),
    )

    assert result == "completed-result"
    rows = [json.loads(line) for line in progress.read_text().splitlines()]
    assert [row["state"] for row in rows] == [
        "submitted",
        "remote_call_active",
        "completed",
    ]


def test_watchdog_treats_builtin_timeout_as_healthy_poll(tmp_path: Path) -> None:
    wrapper_path = ROOT / "scripts/run_cross_family_preflight.py"
    spec = importlib.util.spec_from_file_location(
        "cross_family_builtin_timeout_test", wrapper_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeCall:
        object_id = "fc-builtin-timeout"

        def __init__(self) -> None:
            self.polls = 0

        def get(self, timeout: float) -> str:
            self.polls += 1
            if self.polls == 1:
                raise TimeoutError
            return "ok"

        def cancel(self, terminate_containers: bool = False) -> None:
            raise AssertionError("healthy poll timeouts must not cancel the call")

    call = FakeCall()

    class FakeFunction:
        def spawn(self, *args: object) -> FakeCall:
            return call

    ticks = iter([0.0, 1.0, 2.0, 3.0, 4.0])
    progress = tmp_path / "progress.jsonl"
    result = module._run_remote_with_watchdog(
        FakeFunction(),
        (),
        reporter=module._ProgressReporter(progress),
        label="builtin_timeout",
        heartbeat_seconds=60.0,
        maximum_seconds=7200.0,
        timeout_error_type=RuntimeError,
        clock=lambda: next(ticks),
    )

    assert result == "ok"
    rows = [json.loads(line) for line in progress.read_text().splitlines()]
    assert [row["state"] for row in rows] == [
        "submitted",
        "remote_call_active",
        "completed",
    ]


def test_watchdog_cancels_before_client_deadline(tmp_path: Path) -> None:
    wrapper_path = ROOT / "scripts/run_cross_family_preflight.py"
    spec = importlib.util.spec_from_file_location("cross_family_deadline_test", wrapper_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeCall:
        object_id = "fc-deadline"

        def __init__(self) -> None:
            self.cancelled = False

        def get(self, timeout: float) -> None:
            raise AssertionError("expired calls must be cancelled before polling")

        def cancel(self, terminate_containers: bool = False) -> None:
            self.cancelled = terminate_containers

    call = FakeCall()

    class FakeFunction:
        def spawn(self, *args: object) -> FakeCall:
            return call

    ticks = iter([0.0, 181.0])
    progress = tmp_path / "progress.jsonl"
    with pytest.raises(TimeoutError, match="180-second client deadline"):
        module._run_remote_with_watchdog(
            FakeFunction(),
            (),
            reporter=module._ProgressReporter(progress),
            label="startup_smoke",
            heartbeat_seconds=15.0,
            maximum_seconds=180.0,
            timeout_error_type=RuntimeError,
            clock=lambda: next(ticks),
        )

    assert call.cancelled is True
    rows = [json.loads(line) for line in progress.read_text().splitlines()]
    assert rows[-1]["state"] == "deadline_exceeded_cancelled"
