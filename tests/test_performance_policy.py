"""Pure tests for evidence-calibrated memory estimates and tiling decisions."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from ltx_pipelines_mlx.utils.performance_policy import (
    ESTIMATOR_VERSION,
    Workload,
    decide_tiling,
    estimate_peak_memory,
    static_peak_estimate_gb,
)

_NOW = 1_800_000_000.0
_IDENTITY = {
    "runtime_commit": "abc123",
    "runtime_version": "0.14.20.dev1",
    "device_architecture": "apple-gpu-g16",
    "device_name": "Apple M5 Max",
    "device_recommended_working_set_bytes": 32 * 1024**3,
}
_WORKLOAD = Workload(
    height=512,
    width=768,
    frames=129,
    mode="distilled",
    model="local/ltx-2.3-22b-bf16",
)


def _append_run(
    path: Path,
    *,
    run_id: str,
    workload: Workload = _WORKLOAD,
    ratio: float = 1.0,
    timestamp: float = _NOW,
    status: str = "success",
    runtime_version: str = "0.14.19",
    device_architecture: str = "apple-gpu-g16",
) -> None:
    static = static_peak_estimate_gb(workload)
    metadata = {
        "height": workload.height,
        "width": workload.width,
        "frames": workload.frames,
        "mode": workload.mode,
        "model": workload.model,
        "model_precision": workload.model_precision,
        "model_family": workload.model_family,
        "execution_mode": workload.execution_mode,
        "low_ram": workload.low_ram,
        "tile_frames": workload.tile_frames,
        "tile_spatial": workload.tile_spatial,
        "runtime_version": runtime_version,
        "device_architecture": device_architecture,
    }
    events = [
        {
            "schema_version": 2,
            "event": "run_start",
            "run_id": run_id,
            "timestamp_unix_seconds": timestamp,
            "metadata": metadata,
        },
        {
            "schema_version": 2,
            "event": "run_end" if status == "success" else "run_error",
            "run_id": run_id,
            "timestamp_unix_seconds": timestamp + 10,
            "status": status,
            "observed_peak_process_rss_gb": static * ratio,
        },
    ]
    with path.open("a", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event) + "\n")


def test_estimator_uses_conservative_recent_envelope_and_filters_outlier(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "profile.jsonl"
    for index, ratio in enumerate((1.0, 1.02, 1.03, 1.04, 2.4)):
        _append_run(
            ledger,
            run_id=str(index),
            ratio=ratio,
            timestamp=_NOW - (5 - index) * 60,
        )

    estimate = estimate_peak_memory(_WORKLOAD, _IDENTITY, [ledger], now=_NOW)

    assert estimate.estimator_version == ESTIMATOR_VERSION
    assert estimate.sample_count == 4
    assert estimate.confidence == "low"
    assert estimate.fallback_reason is None
    assert estimate.calibration_multiplier == pytest.approx(1.092, abs=0.001)
    assert estimate.predicted_peak_gb > estimate.static_peak_gb
    assert estimate.runtime_commit == "abc123"


def test_estimator_rejects_unsuccessful_stale_quantized_and_incompatible_runs(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "profile.jsonl"
    _append_run(ledger, run_id="failed", status="error")
    _append_run(
        ledger,
        run_id="stale",
        timestamp=_NOW - 60 * 86400,
    )
    _append_run(
        ledger,
        run_id="runtime",
        runtime_version="0.13.9",
    )
    _append_run(
        ledger,
        run_id="device",
        device_architecture="different-gpu",
    )
    _append_run(
        ledger,
        run_id="q8",
        workload=Workload(
            height=512,
            width=768,
            frames=129,
            mode="distilled",
            model="local/ltx-2.3-22b-q8",
        ),
    )

    estimate = estimate_peak_memory(_WORKLOAD, _IDENTITY, [ledger], now=_NOW)

    assert estimate.confidence == "fallback"
    assert estimate.sample_count == 0
    assert estimate.fallback_reason == "insufficient_compatible_samples"
    assert estimate.predicted_peak_gb == estimate.static_peak_gb


def test_non_bf16_workload_never_uses_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "profile.jsonl"
    for index in range(5):
        _append_run(ledger, run_id=str(index))
    q8 = Workload(
        height=512,
        width=768,
        frames=129,
        mode="distilled",
        model="local/ltx-2.3-22b-q8",
    )

    estimate = estimate_peak_memory(q8, _IDENTITY, [ledger], now=_NOW)

    assert estimate.fallback_reason == "calibration_requires_bf16"
    assert estimate.sample_count == 0


def test_auto_tiling_is_off_by_default_and_explicit_values_win() -> None:
    disabled = decide_tiling(
        _WORKLOAD,
        _IDENTITY,
        [],
        auto_enabled=False,
        explicit_tile_frames=None,
        explicit_tile_spatial=None,
    )
    explicit = decide_tiling(
        _WORKLOAD,
        _IDENTITY,
        [],
        auto_enabled=True,
        explicit_tile_frames=1,
        explicit_tile_spatial=2,
        tile_overlap=3,
    )

    assert disabled.decision == "disabled"
    assert (disabled.tile_frames, disabled.tile_spatial) == (1, 1)
    assert explicit.decision == "explicit"
    assert explicit.explicit_override is True
    assert (explicit.tile_frames, explicit.tile_spatial, explicit.tile_overlap) == (
        1,
        2,
        3,
    )


def test_enabled_policy_tiles_under_pressure_and_relaxes_vae_budget() -> None:
    pressured = decide_tiling(
        _WORKLOAD,
        _IDENTITY,
        [],
        auto_enabled=True,
        explicit_tile_frames=None,
        explicit_tile_spatial=None,
    )
    ample_identity = {
        **_IDENTITY,
        "device_recommended_working_set_bytes": 128 * 1024**3,
    }
    ample = decide_tiling(
        _WORKLOAD,
        ample_identity,
        [],
        auto_enabled=True,
        explicit_tile_frames=None,
        explicit_tile_spatial=None,
    )

    assert pressured.decision == "auto_tiled"
    assert pressured.tile_frames > 1 or pressured.tile_spatial > 1
    assert pressured.vae_decode_budget_gb is not None
    assert ample.decision == "auto_no_tiling"
    assert ample.vae_decode_budget_gb == 12.0


def test_profile_json_can_self_calibrate_as_it_accumulates(tmp_path: Path) -> None:
    ledger = tmp_path / "profile.jsonl"
    current = time.time()
    for index in range(3):
        _append_run(
            ledger,
            run_id=str(index),
            timestamp=current - index,
            ratio=1.1,
        )

    estimate = estimate_peak_memory(_WORKLOAD, _IDENTITY, [ledger])

    assert estimate.sample_count == 3
    assert estimate.fallback_reason is None
