#!/usr/bin/env python3
"""Measure render PSNR only on ground-truth foreground pixels.

The render summary is produced by ``render_ff_reconstructed_scene.py``.  Its
``renders`` records contain a released RGB path, a matched-camera prediction,
and a frame index.  Foreground is defined solely by the released instance mask:
all labels except 255 are foreground.  In particular, the predicted mask is
never intersected with the ground-truth mask, so missing predicted geometry is
penalized instead of silently excluded.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-summary", type=Path, required=True)
    parser.add_argument("--instance-mask-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--background-label",
        type=int,
        default=255,
        help="Released instance-mask value reserved for background.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path, summary_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    relative_to_summary = summary_dir / candidate
    if relative_to_summary.exists():
        return relative_to_summary
    raise FileNotFoundError(path)


def mask_for_render(
    render: dict[str, Any],
    mask_dir: Path,
    mask_paths: list[Path],
) -> tuple[Path, str]:
    """Resolve a mask by RGB stem first, then by the summary frame index."""
    rgb_stem = Path(render["input_rgb"]).stem
    exact = mask_dir / f"{rgb_stem}.png"
    if exact.exists():
        return exact, "rgb_stem"

    frame_index = int(render["frame_index"])
    if not 0 <= frame_index < len(mask_paths):
        raise IndexError(
            f"frame index {frame_index} outside mask list of length "
            f"{len(mask_paths)}"
        )
    return mask_paths[frame_index], "sorted_frame_index"


def psnr_from_mse(mse: float) -> float:
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def main() -> None:
    args = parse_args()
    summary_path = args.render_summary.resolve()
    summary = json.loads(summary_path.read_text())
    summary_dir = summary_path.parent
    mask_dir = args.instance_mask_dir.resolve()
    mask_paths = sorted(mask_dir.glob("*.png"))
    if not mask_paths:
        raise FileNotFoundError(f"no PNG masks under {mask_dir}")

    per_view: list[dict[str, Any]] = []
    total_squared_error = 0.0
    total_channel_samples = 0
    total_foreground_pixels = 0

    for render in summary["renders"]:
        gt_path = resolve_path(render["input_rgb"], summary_dir)
        pred_path = resolve_path(render["prediction_rgba"], summary_dir)
        mask_path, mask_resolution = mask_for_render(render, mask_dir, mask_paths)

        prediction_rgba = Image.open(pred_path).convert("RGBA")
        prediction = Image.alpha_composite(
            Image.new("RGBA", prediction_rgba.size, (255, 255, 255, 255)),
            prediction_rgba,
        ).convert("RGB")
        target = Image.open(gt_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if target.size != prediction.size:
            target = target.resize(prediction.size, Image.Resampling.BILINEAR)
        if mask.size != prediction.size:
            mask = mask.resize(prediction.size, Image.Resampling.NEAREST)

        pred_array = np.asarray(prediction, dtype=np.float64)
        target_array = np.asarray(target, dtype=np.float64)
        foreground = np.asarray(mask, dtype=np.uint8) != args.background_label
        foreground_pixels = int(foreground.sum())
        if foreground_pixels == 0:
            per_view.append(
                {
                    "frame_index": int(render["frame_index"]),
                    "input_rgb": str(gt_path),
                    "prediction_rgb": str(pred_path),
                    "instance_mask": str(mask_path),
                    "mask_resolution": mask_resolution,
                    "foreground_pixels": 0,
                    "foreground_fraction": 0.0,
                    "foreground_mse": None,
                    "foreground_psnr_db": None,
                    "status": "skipped_empty_foreground",
                }
            )
            continue

        difference = pred_array[foreground] - target_array[foreground]
        squared_error = float(np.square(difference).sum())
        channel_samples = foreground_pixels * 3
        mse = squared_error / channel_samples
        total_squared_error += squared_error
        total_channel_samples += channel_samples
        total_foreground_pixels += foreground_pixels
        per_view.append(
            {
                "frame_index": int(render["frame_index"]),
                "input_rgb": str(gt_path),
                "prediction_rgb": str(pred_path),
                "instance_mask": str(mask_path),
                "mask_resolution": mask_resolution,
                "foreground_pixels": foreground_pixels,
                "foreground_fraction": foreground_pixels
                / float(foreground.size),
                "foreground_mse": mse,
                "foreground_psnr_db": psnr_from_mse(mse),
                "status": "ok",
            }
        )

    valid_psnr = [
        float(record["foreground_psnr_db"])
        for record in per_view
        if record["status"] == "ok"
    ]
    if not valid_psnr:
        raise RuntimeError("none of the selected frames contains foreground")
    pooled_mse = total_squared_error / total_channel_samples
    output = {
        "schema": "ff_holoscene_foreground_render_psnr_v1",
        "scene_id": summary.get("scene_id"),
        "render_summary": str(summary_path),
        "instance_mask_dir": str(mask_dir),
        "foreground_definition": f"released instance label != {args.background_label}",
        "mask_policy": (
            "ground-truth foreground only; no predicted-mask intersection; "
            "all RGB channels pooled"
        ),
        "selected_view_count": len(per_view),
        "valid_view_count": len(valid_psnr),
        "empty_foreground_view_count": len(per_view) - len(valid_psnr),
        "total_foreground_pixels": total_foreground_pixels,
        "pooled_foreground_mse": pooled_mse,
        "pooled_foreground_psnr_db": psnr_from_mse(pooled_mse),
        "macro_foreground_psnr_db": float(np.mean(valid_psnr)),
        "median_foreground_psnr_db": float(np.median(valid_psnr)),
        "min_foreground_psnr_db": float(np.min(valid_psnr)),
        "max_foreground_psnr_db": float(np.max(valid_psnr)),
        "per_view": per_view,
    }
    output_path = (
        args.output.resolve()
        if args.output is not None
        else summary_dir / "foreground_psnr.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "per_view"}, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
