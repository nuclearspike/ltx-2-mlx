"""The tiling qualification hook plans isolated fixed-seed comparisons."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script = Path(__file__).parents[1] / "scripts" / "qualify_tiling.py"
    spec = importlib.util.spec_from_file_location("qualify_tiling", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_matrix_isolates_modality_and_vae_tiling(tmp_path: Path) -> None:
    module = _load_module()
    image = tmp_path / "reference.png"
    matrix = module.build_matrix(
        model="local/ltx-2.3-bf16",
        output_dir=tmp_path,
        image=image,
    )

    assert len(matrix) == 6
    by_key = {(run.group, run.candidate): run for run in matrix}
    control = by_key[("t2v", "control")]
    modality = by_key[("t2v", "modality_temporal_2")]
    vae = by_key[("t2v", "vae_temporal")]
    assert "--tile-frames" not in control.command
    assert modality.command[modality.command.index("--tile-frames") + 1] == "2"
    assert vae.environment["LTX2_VAE_DECODE_BUDGET_GB"] == "0.25"
    assert control.environment["LTX2_VAE_DECODE_BUDGET_GB"] == "12.0"
    assert all(run.environment["LTX2_MEDIA_WRITE_OVERLAP"] == "0" for run in matrix)
    assert "--image" in by_key[("i2v", "control")].command
