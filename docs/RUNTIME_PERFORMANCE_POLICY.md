# Runtime performance policy

The fork release is versioned `0.14.20.dev1`. Product integrations should pin an
exact commit and validate the selected Python environment before launching:

```bash
python -m ltx_pipelines_mlx.utils.runtime_info
```

The JSON response includes the pipelines/core/MLX versions, source commit,
dirty state, and MLX device identity. `LTX2_RUNTIME_COMMIT` may supply the
commit when the installed wheel is not inside a Git worktree.

## Profiling and estimator

`generate --profile-json PATH` appends crash-resilient schema-2 JSONL. The
`run_start.metadata` object includes:

- runtime commit/version and device family;
- model precision/family and execution mode;
- dimensions, frame count, explicit tiling, VAE budget, and cache choice;
- `performance_policy`, including estimator version, predicted/static peak,
  compatible sample count, confidence/fallback reason, calibration multiplier,
  policy memory budget, and final tiling decision.

The per-run terminal event records MLX peak, current-process RSS peak, and the
Darwin lifetime physical-footprint high-water mark. MLX peak accounting is
reset when each profiled run begins.

The `ltx.memory.v1` adaptive estimator reads `--performance-ledger PATH`
(repeatable), `LTX2_PERFORMANCE_LEDGER` (path-separator delimited), and the
current `--profile-json` file. Calibration accepts only:

- terminal `status=success` runs;
- declared/inferred BF16 weights;
- the same runtime major/minor, model family, device family, and execution mode;
- samples no more than 45 days old.

At least three compatible samples are required. The newest 64 are bounded to a
reasonable observed/static ratio, median/MAD-filtered, and reduced to the
larger of a p90 and recent-five envelope with a 5% safety margin. Any missing or
incompatible evidence uses the static workload model and reports the fallback
reason.

## Automatic tiling

Automatic tiling is disabled by default pending fixed-seed quality evidence:

```bash
ltx-2-mlx generate ... --auto-tiling --performance-ledger profile.jsonl
```

Explicit `--tile-frames`, `--tile-spatial`, `--tile-overlap`, and
`LTX2_VAE_DECODE_BUDGET_GB` values win. When enabled without explicit values,
the policy compares predicted peak to 78% of MLX's recommended working set,
selects bounded modality tiling under pressure, and derives a 2–12 GB VAE
decode budget. The decision and evidence are written to profiling metadata.

`LTX2_MEDIA_WRITE_OVERLAP=1` (default) permits one already-evaluated frame to
be written to ffmpeg while the producer prepares the next. Set it to `0` for a
synchronous control. The single-worker queue is ordered and bounded to one
in-flight frame.

TeaCache remains opt-in and applies only to dev two-stage/HQ stage 1. The
distilled pipeline does not expose it as an acceleration lane.

## Qualification matrix

Planning is safe and does not render:

```bash
uv run --extra dev python scripts/qualify_tiling.py \
  --model /path/to/ltx-2.3-bf16 \
  --output-dir /tmp/ltx-tiling-qualification
```

Add `--image reference.png` for matched I2V cases. QA may add `--execute` to
run the fixed-seed BF16 controls. The matrix independently compares:

- no modality tiling with a 12 GB VAE budget;
- two temporal modality tiles with the same VAE budget;
- no modality tiling with a 0.25 GB VAE budget to force temporal VAE tiling.

Media-write overlap is disabled for every matrix run. The result manifest
contains decoded-frame/audio fingerprints, video PSNR, audio sample L1, and
audio STFT L1. `qualified` intentionally remains unset until product quality
thresholds are approved.

Exact per-dispatch GPU duration is not exposed by the MLX Python API. Use the
opt-in `scripts/capture_metal_system_trace.py` wrapper to collect an xctrace
Metal System Trace for a scoped command; ordinary profiling should remain
capture-free.
