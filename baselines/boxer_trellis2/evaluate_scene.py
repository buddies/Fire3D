#!/usr/bin/env python3
"""Evaluate one BoxeR+SAM2+TRELLIS.2 Imaginarium scene."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.scene_reconstruction.evaluators.fire3d import (  # noqa: E402
    imaginarium_utils,
)
from benchmarks.scene_reconstruction.evaluators.fire3d.det_seg_utils import (  # noqa: E402
    calculate_mAP,
    calculate_mIoU,
)
from benchmarks.scene_reconstruction.evaluators.fire3d.geometry_utils import (  # noqa: E402
    calculate_geometry_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "data/Imaginarium",
    )
    parser.add_argument("--boxer-sam2-dir", type=Path, required=True)
    parser.add_argument("--trellis2-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--point-downsample", type=int, default=8)
    parser.add_argument("--iou-threshold", type=float, default=0.25)
    parser.add_argument("--geometry-samples", type=int, default=100_000)
    parser.add_argument("--f1-threshold", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--export-vis", action="store_true")
    return parser.parse_args()


def finite_or_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(item) for item in value]
    if isinstance(value, np.ndarray):
        return finite_or_none(value.tolist())
    if isinstance(value, np.generic):
        return finite_or_none(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def mean_geometry(rows: list[dict[str, float]]) -> dict[str, float | None]:
    if not rows:
        return {"CD": None, "F1": None, "NC": None}
    return {
        key: float(np.mean([row[key] for row in rows if np.isfinite(row[key])]))
        for key in ("CD", "F1", "NC")
    }


def colors(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    result = np.zeros((len(labels), 4), dtype=np.uint8)
    result[:, 3] = 255
    result[:, 0] = (labels * 37 + 83) % 255
    result[:, 1] = (labels * 67 + 29) % 255
    result[:, 2] = (labels * 97 + 151) % 255
    result[labels == 0, :3] = 110
    return result


def load_trellis_meshes(root: Path | None) -> dict[int, trimesh.Trimesh]:
    if root is None:
        return {}
    summary_path = root / "trellis2_summary.json"
    if not summary_path.is_file():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = {}
    for row in summary.get("objects", []):
        path = Path(row["world_geometry_path"])
        if path.is_file():
            result[int(row["index"])] = trimesh.load(
                path, force="mesh", process=False
            )
    return result


def main() -> None:
    args = parse_args()
    imaginarium_utils.gt_root_dir = os.fspath(args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred_payload = json.loads(
        (args.boxer_sam2_dir / "object_obbs.json").read_text(encoding="utf-8")
    )
    pred_bboxes = pred_payload["objects"]
    raw_pred_labels = np.load(
        args.boxer_sam2_dir
        / f"point_instance_labels_downsample{args.point_downsample}.npy"
    ).astype(np.int64)

    points, gt_labels = imaginarium_utils.get_point_cloud(
        args.scene_id, downsample=args.point_downsample
    )
    gt_labels = gt_labels.astype(np.int64)
    if len(raw_pred_labels) != len(gt_labels):
        raise ValueError(
            f"point-label length mismatch: pred={len(raw_pred_labels)}, "
            f"gt={len(gt_labels)}"
        )
    gt_obbs_all, gt_meshes = imaginarium_utils.get_obbs_and_meshes(args.scene_id)
    visible_gt_ids = set(int(value) for value in np.unique(gt_labels) if value > 0)
    gt_obbs = [
        obb for obb in gt_obbs_all if int(obb["index"]) in visible_gt_ids
    ]

    map_score, matched_pairs = calculate_mAP(
        pred_bboxes, gt_obbs, iou_threshold=args.iou_threshold
    )
    remapped_pred_labels = np.zeros_like(raw_pred_labels)
    for pred_bbox, gt_bbox, _ in matched_pairs:
        remapped_pred_labels[
            raw_pred_labels == int(pred_bbox["index"])
        ] = int(gt_bbox["index"])
    num_classes = max(visible_gt_ids | {0}) + 1
    miou, iou_per_class = calculate_mIoU(
        remapped_pred_labels, gt_labels, num_classes
    )

    pred_meshes = load_trellis_meshes(args.trellis2_dir)
    pair_results = []
    geometry_rows = []
    for pred_bbox, gt_bbox, obb_iou in matched_pairs:
        pred_index = int(pred_bbox["index"])
        gt_index = int(gt_bbox["index"])
        pair = {
            "pred_index": pred_index,
            "gt_index": gt_index,
            "pred_name": pred_bbox.get("name"),
            "gt_name": gt_bbox.get("name"),
            "obb_iou": float(obb_iou),
            "geometry": None,
        }
        if pred_index in pred_meshes and gt_index in gt_meshes:
            np.random.seed(
                args.seed + pred_index * 1_000_003 + gt_index
            )
            metrics = calculate_geometry_metrics(
                pred_meshes[pred_index],
                gt_meshes[gt_index],
                n_points=args.geometry_samples,
                f1_threshold=args.f1_threshold,
            )
            pair["geometry"] = metrics
            geometry_rows.append(metrics)
        pair_results.append(pair)

    result = {
        "schema": "ff_boxer_sam2_trellis2_imaginarium_eval_v1",
        "scene_name": args.scene_id,
        "num_predictions": len(pred_bboxes),
        "num_gt_visible": len(gt_obbs),
        "num_matched": len(matched_pairs),
        "matched_recall": len(matched_pairs) / max(len(gt_obbs), 1),
        "iou_threshold": args.iou_threshold,
        "mAP": map_score,
        "mIoU": miou,
        "iou_per_class": iou_per_class,
        "geometry": mean_geometry(geometry_rows),
        "matched_pairs": pair_results,
    }
    (args.output_dir / "eval_results.json").write_text(
        json.dumps(finite_or_none(result), indent=2) + "\n",
        encoding="utf-8",
    )

    if args.export_vis:
        vis = args.output_dir / "eval_vis"
        vis.mkdir(parents=True, exist_ok=True)
        trimesh.points.PointCloud(
            points, colors=colors(gt_labels)
        ).export(vis / "gt_point_labels.ply")
        trimesh.points.PointCloud(
            points, colors=colors(remapped_pred_labels)
        ).export(vis / "pred_sam2_point_labels_matched.ply")
        for index, mesh in pred_meshes.items():
            mesh.export(vis / f"pred_world_mesh_{index:04d}.ply")

    print(
        json.dumps(
            {
                "scene": args.scene_id,
                "mAP": result["mAP"],
                "mIoU": result["mIoU"],
                "matched": f"{len(matched_pairs)}/{len(gt_obbs)}",
                "geometry": result["geometry"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
