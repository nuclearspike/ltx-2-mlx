"""Evidence-calibrated peak-memory estimates and opt-in tiling decisions."""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_GB = 1024**3
ESTIMATOR_VERSION = "ltx.memory.v1"
MIN_CALIBRATION_SAMPLES = 3
MAX_CALIBRATION_SAMPLES = 64
MAX_SAMPLE_AGE_DAYS = 45
_MIN_VALID_RATIO = 0.4
_MAX_VALID_RATIO = 2.5


def infer_model_precision(model: str) -> str:
    """Infer the weight precision from the model identifier."""
    normalized = model.lower()
    if re.search(r"(?:^|[-_/])(q4|int4|4bit)(?:$|[-_/])", normalized):
        return "q4"
    if re.search(r"(?:^|[-_/])(q8|int8|8bit)(?:$|[-_/])", normalized):
        return "q8"
    if any(token in normalized for token in ("bf16", "bfloat16", "fp16", "float16")):
        return "bf16"
    # Do not admit an ambiguous local path into the BF16 calibration lane.
    return "unknown"


def infer_model_family(model: str) -> str:
    """Normalize paths and quantization suffixes to a stable model family."""
    normalized = model.lower().replace("\\", "/")
    if "ltx-2.3" in normalized:
        return "ltx-2.3-22b"
    if "ltx-2" in normalized:
        return "ltx-2"
    leaf = normalized.rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"[-_.](?:bf16|fp16|q[48]|int[48]|[48]bit)(?:$|[-_.].*)", "", leaf)


def runtime_family(version: object) -> str | None:
    """Return a major.minor family for compatible runtime evidence."""
    match = re.match(r"^\s*(\d+)\.(\d+)", str(version or ""))
    return f"{match.group(1)}.{match.group(2)}" if match else None


def device_family(identity: Mapping[str, object]) -> str | None:
    """Return the most stable available MLX device-family identifier."""
    value = identity.get("device_architecture") or identity.get("device_name")
    return str(value).strip().lower() if value else None


@dataclass(frozen=True)
class Workload:
    height: int
    width: int
    frames: int
    mode: str
    model: str
    low_ram: bool = False
    tile_frames: int = 1
    tile_spatial: int = 1
    precision: str | None = None

    @property
    def model_precision(self) -> str:
        return self.precision or infer_model_precision(self.model)

    @property
    def model_family(self) -> str:
        return infer_model_family(self.model)

    @property
    def execution_mode(self) -> str:
        memory_mode = "low_ram" if self.low_ram else "resident"
        return f"{self.mode}:{memory_mode}"

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> Workload | None:
        try:
            return cls(
                height=int(metadata["height"]),
                width=int(metadata["width"]),
                frames=int(metadata["frames"]),
                mode=str(metadata["mode"]),
                model=str(metadata["model"]),
                low_ram=bool(metadata.get("low_ram", False)),
                tile_frames=max(1, int(metadata.get("tile_frames", 1))),
                tile_spatial=max(1, int(metadata.get("tile_spatial", 1))),
                precision=(
                    str(metadata["model_precision"])
                    if metadata.get("model_precision")
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError):
            return None


def static_peak_estimate_gb(workload: Workload) -> float:
    """Conservative static peak estimate normalized by latent workload size."""
    base_by_precision = {
        "bf16": 37.0,
        "q8": 20.0,
        "q4": 13.0,
    }
    base = base_by_precision.get(workload.model_precision, 30.0)
    if workload.low_ram:
        base *= 0.34

    latent_frames = max(1, math.ceil((workload.frames - 1) / 8) + 1)
    latent_height = max(1, math.ceil(workload.height / 32))
    latent_width = max(1, math.ceil(workload.width / 32))
    latent_tokens = latent_frames * latent_height * latent_width
    activation = max(1.0, 3.25 * latent_tokens / 6528)

    tile_count = max(1, workload.tile_frames * workload.tile_spatial**2)
    activation /= tile_count**0.6
    mode_overhead = {
        "distilled": 0.0,
        "one_stage": 2.0,
        "two_stage": 2.5,
        "two_stages_hq": 3.0,
    }.get(workload.mode, 2.0)
    return round(base + mode_overhead + activation, 3)


@dataclass(frozen=True)
class PeakEstimate:
    estimator_version: str
    predicted_peak_gb: float
    static_peak_gb: float
    sample_count: int
    confidence: str
    fallback_reason: str | None
    calibration_multiplier: float
    runtime_commit: str | None
    runtime_family: str | None
    model_family: str
    execution_mode: str
    model_precision: str

    def to_metadata(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _LedgerSample:
    timestamp: float
    ratio: float


def _observed_peak(record: Mapping[str, object]) -> float | None:
    candidates: list[float] = []
    for key in (
        "observed_peak_process_rss_gb",
        "observed_peak_mlx_gb",
        "process_rss_gb",
        "mlx_peak_gb",
    ):
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
            candidates.append(float(value))
    return max(candidates, default=None)


def _read_successful_runs(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    runs: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        try:
            stream = path.open("r", encoding="utf-8")
        except OSError:
            continue
        with stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(record, dict) or not record.get("run_id"):
                    continue
                key = (str(path.resolve()), str(record["run_id"]))
                run = runs.setdefault(
                    key,
                    {
                        "metadata": None,
                        "timestamp": 0.0,
                        "peak_gb": 0.0,
                        "success": False,
                        "terminal": False,
                    },
                )
                timestamp = record.get("timestamp_unix_seconds")
                if isinstance(timestamp, (int, float)):
                    run["timestamp"] = max(float(timestamp), run["timestamp"])
                peak = _observed_peak(record)
                if peak is not None:
                    run["peak_gb"] = max(peak, run["peak_gb"])
                event = record.get("event")
                if event == "run_start" and isinstance(record.get("metadata"), dict):
                    run["metadata"] = record["metadata"]
                elif event == "run_end":
                    run["terminal"] = True
                    run["success"] = record.get("status") == "success"
                elif event == "run_error":
                    run["terminal"] = True
                    run["success"] = False
    return [
        run
        for run in runs.values()
        if run["terminal"] and run["success"] and isinstance(run["metadata"], dict)
    ]


def _fallback_estimate(
    workload: Workload,
    identity: Mapping[str, object],
    reason: str,
    *,
    sample_count: int = 0,
) -> PeakEstimate:
    static = static_peak_estimate_gb(workload)
    return PeakEstimate(
        estimator_version=ESTIMATOR_VERSION,
        predicted_peak_gb=static,
        static_peak_gb=static,
        sample_count=sample_count,
        confidence="fallback",
        fallback_reason=reason,
        calibration_multiplier=1.0,
        runtime_commit=(
            str(identity["runtime_commit"]) if identity.get("runtime_commit") else None
        ),
        runtime_family=runtime_family(identity.get("runtime_version")),
        model_family=workload.model_family,
        execution_mode=workload.execution_mode,
        model_precision=workload.model_precision,
    )


def estimate_peak_memory(
    workload: Workload,
    identity: Mapping[str, object],
    ledger_paths: Sequence[str | Path],
    *,
    now: float | None = None,
) -> PeakEstimate:
    """Estimate peak memory from compatible ledger samples or a static fallback."""
    if workload.model_precision != "bf16":
        return _fallback_estimate(workload, identity, "calibration_requires_bf16")
    if not ledger_paths:
        return _fallback_estimate(workload, identity, "no_ledger")

    current_runtime_family = runtime_family(identity.get("runtime_version"))
    current_device_family = device_family(identity)
    if current_runtime_family is None or current_device_family is None:
        return _fallback_estimate(workload, identity, "missing_runtime_or_device_identity")

    cutoff = (now if now is not None else time.time()) - MAX_SAMPLE_AGE_DAYS * 86400
    samples: list[_LedgerSample] = []
    for run in _read_successful_runs(ledger_paths):
        metadata = run["metadata"]
        sample_workload = Workload.from_metadata(metadata)
        if sample_workload is None or sample_workload.model_precision != "bf16":
            continue
        sample_runtime_family = (
            str(metadata["runtime_family"])
            if metadata.get("runtime_family")
            else runtime_family(metadata.get("runtime_version"))
        )
        sample_device_family = (
            str(metadata["device_family"]).lower()
            if metadata.get("device_family")
            else device_family(metadata)
        )
        sample_execution_mode = str(
            metadata.get("execution_mode", sample_workload.execution_mode)
        )
        sample_model_family = str(
            metadata.get("model_family", sample_workload.model_family)
        )
        if (
            run["timestamp"] < cutoff
            or sample_runtime_family != current_runtime_family
            or sample_device_family != current_device_family
            or sample_execution_mode != workload.execution_mode
            or sample_model_family != workload.model_family
        ):
            continue
        static = static_peak_estimate_gb(sample_workload)
        ratio = float(run["peak_gb"]) / static if static > 0 else 0.0
        if _MIN_VALID_RATIO <= ratio <= _MAX_VALID_RATIO:
            samples.append(_LedgerSample(timestamp=run["timestamp"], ratio=ratio))

    samples.sort(key=lambda sample: sample.timestamp)
    samples = samples[-MAX_CALIBRATION_SAMPLES:]
    if len(samples) < MIN_CALIBRATION_SAMPLES:
        return _fallback_estimate(
            workload,
            identity,
            "insufficient_compatible_samples",
            sample_count=len(samples),
        )

    median_ratio = statistics.median(sample.ratio for sample in samples)
    deviations = [abs(sample.ratio - median_ratio) for sample in samples]
    mad = statistics.median(deviations)
    envelope = max(0.15 * median_ratio, 3.0 * mad)
    filtered = [
        sample for sample in samples if abs(sample.ratio - median_ratio) <= envelope
    ]
    if len(filtered) < MIN_CALIBRATION_SAMPLES:
        return _fallback_estimate(
            workload,
            identity,
            "insufficient_samples_after_outlier_filter",
            sample_count=len(filtered),
        )

    sorted_ratios = sorted(sample.ratio for sample in filtered)
    p90_index = max(0, math.ceil(0.9 * len(sorted_ratios)) - 1)
    high_quantile = sorted_ratios[p90_index]
    recent_envelope = max(sample.ratio for sample in filtered[-min(5, len(filtered)) :])
    multiplier = min(
        _MAX_VALID_RATIO,
        max(0.75, high_quantile, recent_envelope) * 1.05,
    )
    static = static_peak_estimate_gb(workload)
    sample_count = len(filtered)
    confidence = "high" if sample_count >= 8 else "medium" if sample_count >= 5 else "low"
    return PeakEstimate(
        estimator_version=ESTIMATOR_VERSION,
        predicted_peak_gb=round(static * multiplier, 3),
        static_peak_gb=static,
        sample_count=sample_count,
        confidence=confidence,
        fallback_reason=None,
        calibration_multiplier=round(multiplier, 4),
        runtime_commit=(
            str(identity["runtime_commit"]) if identity.get("runtime_commit") else None
        ),
        runtime_family=current_runtime_family,
        model_family=workload.model_family,
        execution_mode=workload.execution_mode,
        model_precision=workload.model_precision,
    )


@dataclass(frozen=True)
class TilingDecision:
    auto_tiling_enabled: bool
    decision: str
    explicit_override: bool
    tile_frames: int
    tile_spatial: int
    tile_overlap: int
    policy_memory_budget_gb: float | None
    vae_decode_budget_gb: float | None
    estimate: PeakEstimate

    def to_metadata(self) -> dict[str, object]:
        return {
            **self.estimate.to_metadata(),
            "auto_tiling_enabled": self.auto_tiling_enabled,
            "decision": self.decision,
            "explicit_override": self.explicit_override,
            "tile_frames": self.tile_frames,
            "tile_spatial": self.tile_spatial,
            "tile_overlap": self.tile_overlap,
            "policy_memory_budget_gb": self.policy_memory_budget_gb,
            "vae_decode_budget_gb": self.vae_decode_budget_gb,
        }


def decide_tiling(
    workload: Workload,
    identity: Mapping[str, object],
    ledger_paths: Sequence[str | Path],
    *,
    auto_enabled: bool,
    explicit_tile_frames: int | None,
    explicit_tile_spatial: int | None,
    tile_overlap: int = 2,
) -> TilingDecision:
    """Return an explicit-overrides-first, opt-in automatic tiling decision."""
    estimate = estimate_peak_memory(workload, identity, ledger_paths)
    recommended_bytes = identity.get("device_recommended_working_set_bytes")
    policy_budget_gb = (
        round(float(recommended_bytes) / _GB * 0.78, 3)
        if isinstance(recommended_bytes, (int, float)) and recommended_bytes > 0
        else None
    )

    if explicit_tile_frames is not None or explicit_tile_spatial is not None:
        return TilingDecision(
            auto_tiling_enabled=auto_enabled,
            decision="explicit",
            explicit_override=True,
            tile_frames=max(1, explicit_tile_frames or 1),
            tile_spatial=max(1, explicit_tile_spatial or 1),
            tile_overlap=tile_overlap,
            policy_memory_budget_gb=policy_budget_gb,
            vae_decode_budget_gb=None,
            estimate=estimate,
        )

    if not auto_enabled:
        return TilingDecision(
            auto_tiling_enabled=False,
            decision="disabled",
            explicit_override=False,
            tile_frames=1,
            tile_spatial=1,
            tile_overlap=tile_overlap,
            policy_memory_budget_gb=policy_budget_gb,
            vae_decode_budget_gb=None,
            estimate=estimate,
        )

    if policy_budget_gb is None:
        return TilingDecision(
            auto_tiling_enabled=True,
            decision="auto_missing_device_budget",
            explicit_override=False,
            tile_frames=1,
            tile_spatial=1,
            tile_overlap=tile_overlap,
            policy_memory_budget_gb=None,
            vae_decode_budget_gb=None,
            estimate=estimate,
        )

    vae_budget_gb = round(max(2.0, min(12.0, policy_budget_gb * 0.18)), 3)
    if estimate.predicted_peak_gb <= policy_budget_gb:
        return TilingDecision(
            auto_tiling_enabled=True,
            decision="auto_no_tiling",
            explicit_override=False,
            tile_frames=1,
            tile_spatial=1,
            tile_overlap=tile_overlap,
            policy_memory_budget_gb=policy_budget_gb,
            vae_decode_budget_gb=vae_budget_gb,
            estimate=estimate,
        )

    pressure = estimate.predicted_peak_gb / policy_budget_gb
    tile_frames = 1
    if workload.frames >= 65:
        tile_frames = 4 if pressure >= 1.5 else 2
    residual_pressure = pressure / tile_frames**0.6
    tile_spatial = 2 if residual_pressure > 1.0 and min(workload.height, workload.width) >= 512 else 1
    return TilingDecision(
        auto_tiling_enabled=True,
        decision="auto_tiled",
        explicit_override=False,
        tile_frames=tile_frames,
        tile_spatial=tile_spatial,
        tile_overlap=tile_overlap,
        policy_memory_budget_gb=policy_budget_gb,
        vae_decode_budget_gb=vae_budget_gb,
        estimate=estimate,
    )
