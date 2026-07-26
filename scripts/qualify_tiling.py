#!/usr/bin/env python3
"""Plan or execute fixed-seed modality/VAE tiling qualification pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

SEED = 9803402
PROMPT = (
    "A woman with auburn hair walks along a sunlit cobblestone street while "
    "the camera tracks beside her, cinematic natural light, consistent motion."
)


@dataclass(frozen=True)
class QualificationRun:
    group: str
    candidate: str
    command: list[str]
    environment: dict[str, str]
    output: str
    profile: str


def _command(
    *,
    model: str,
    output: Path,
    profile: Path,
    image: Path | None,
    extra: list[str],
) -> list[str]:
    command = [
        "uv",
        "run",
        "ltx-2-mlx",
        "generate",
        "--prompt",
        PROMPT,
        "--distilled",
        "--model",
        model,
        "--model-precision",
        "bf16",
        "--seed",
        str(SEED),
        "--height",
        "512",
        "--width",
        "768",
        "--frames",
        "129",
        "--frame-rate",
        "24",
        "--profile-json",
        str(profile),
    ]
    if image is not None:
        command.extend(["--image", str(image)])
    command.extend([*extra, "--output", str(output)])
    return command


def build_matrix(
    *,
    model: str,
    output_dir: Path,
    image: Path | None,
) -> list[QualificationRun]:
    """Build isolated controls for modality and VAE temporal tiling."""
    groups = [("t2v", None)]
    if image is not None:
        groups.append(("i2v", image))

    runs: list[QualificationRun] = []
    for group, conditioning_image in groups:
        for candidate, extra, vae_budget in (
            ("control", [], "12.0"),
            ("modality_temporal_2", ["--tile-frames", "2"], "12.0"),
            ("vae_temporal", [], "0.25"),
        ):
            output = output_dir / f"{group}__{candidate}.mp4"
            profile = output_dir / f"{group}__{candidate}.jsonl"
            runs.append(
                QualificationRun(
                    group=group,
                    candidate=candidate,
                    command=_command(
                        model=model,
                        output=output,
                        profile=profile,
                        image=conditioning_image,
                        extra=extra,
                    ),
                    environment={
                        "LTX2_MEDIA_WRITE_OVERLAP": "0",
                        "LTX2_VAE_DECODE_BUDGET_GB": vae_budget,
                    },
                    output=str(output),
                    profile=str(profile),
                )
            )
    return runs


def _array_sha256(value: object) -> str:
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def _compare(group: str, output_dir: Path) -> dict[str, object]:
    from tests.regression.metrics import (
        audio_l1,
        audio_stft_l1,
        decode_audio,
        decode_video,
        video_psnr,
    )

    control_path = output_dir / f"{group}__control.mp4"
    control_video = decode_video(control_path)
    control_audio = decode_audio(control_path)
    comparisons: dict[str, object] = {}
    for candidate in ("modality_temporal_2", "vae_temporal"):
        candidate_path = output_dir / f"{group}__{candidate}.mp4"
        candidate_video = decode_video(candidate_path)
        candidate_audio = decode_audio(candidate_path)
        comparisons[candidate] = {
            "video_psnr_db": video_psnr(control_video, candidate_video),
            "audio_sample_l1": audio_l1(control_audio, candidate_audio),
            "audio_stft_l1": audio_stft_l1(control_audio, candidate_audio),
            "control_video_sha256": _array_sha256(control_video),
            "candidate_video_sha256": _array_sha256(candidate_video),
            "control_audio_sha256": _array_sha256(control_audio),
            "candidate_audio_sha256": _array_sha256(candidate_audio),
            # Promotion remains a QA decision until product-specific thresholds
            # are established from representative fixed-seed evidence.
            "qualified": None,
        }
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="BF16 model repo or directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=None, help="Optional I2V reference")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the heavy matrix; otherwise print the plan only",
    )
    args = parser.parse_args()

    matrix = build_matrix(
        model=args.model,
        output_dir=args.output_dir,
        image=args.image,
    )
    plan = {
        "schema": "ltx.tiling-qualification.v1",
        "seed": SEED,
        "runs": [asdict(run) for run in matrix],
        "executed": args.execute,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for run in matrix:
        environment = {**os.environ, **run.environment}
        subprocess.run(run.command, env=environment, check=True)
    plan["comparisons"] = {
        group: _compare(group, args.output_dir)
        for group in sorted({run.group for run in matrix})
    }
    manifest = args.output_dir / "qualification.json"
    manifest.write_text(json.dumps(plan, indent=2, allow_nan=True) + "\n")
    print(manifest)


if __name__ == "__main__":
    main()
