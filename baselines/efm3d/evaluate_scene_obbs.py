#!/usr/bin/env python3
"""Evaluate predicted world OBB coverage against labeled scene points."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import trimesh


def quaternion_wxyz_to_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(values)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"Invalid quaternion: {values.tolist()}")
    w, x, y, z = values / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_points(path: Path) -> np.ndarray:
    cloud = trimesh.load(path, process=False)
    if hasattr(cloud, "vertices"):
        points = np.asarray(cloud.vertices)
    elif hasattr(cloud, "geometry"):
        points = np.concatenate(
            [np.asarray(geometry.vertices) for geometry in cloud.geometry.values()],
            axis=0,
        )
    else:
        raise ValueError(f"Could not load points from {path}")
    return np.asarray(points, dtype=np.float64)


def load_indices(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "indices" in archive:
            indices = archive["indices"]
        elif len(archive.files) == 1:
            indices = archive[archive.files[0]]
        else:
            raise KeyError(f"No indices array in {path}: {archive.files}")
    return np.asarray(indices, dtype=np.int64).reshape(-1)


def load_obbs(path: Path, probability_threshold: float) -> list[dict]:
    obbs = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            probability = float(row.get("prob", 1.0))
            if probability < probability_threshold:
                continue
            rotation = quaternion_wxyz_to_matrix(
                [
                    float(row["qw_world_object"]),
                    float(row["qx_world_object"]),
                    float(row["qy_world_object"]),
                    float(row["qz_world_object"]),
                ]
            )
            obbs.append(
                {
                    "center": np.asarray(
                        [
                            float(row["tx_world_object"]),
                            float(row["ty_world_object"]),
                            float(row["tz_world_object"]),
                        ]
                    ),
                    "rotation_world_object": rotation,
                    "scale": np.asarray(
                        [float(row["scale_x"]), float(row["scale_y"]), float(row["scale_z"])]
                    ),
                    "probability": probability,
                    "name": row.get("name", ""),
                }
            )
    return obbs


def points_inside_obb(points_world: np.ndarray, obb: dict) -> np.ndarray:
    points_object = (points_world - obb["center"]) @ obb["rotation_world_object"]
    return (np.abs(points_object) <= 0.5 * obb["scale"] + 1e-6).all(axis=1)


def evaluate_obb_coverage(
    points_world: np.ndarray,
    instance_ids: np.ndarray,
    obbs: list[dict],
    min_instance_points: int = 20,
) -> dict:
    if points_world.shape != (instance_ids.shape[0], 3):
        raise ValueError(
            f"Point/index mismatch: {points_world.shape} versus {instance_ids.shape}"
        )
    finite = np.isfinite(points_world).all(axis=1)
    foreground = finite & (instance_ids > 0)
    kept_ids = [
        int(instance_id)
        for instance_id in np.unique(instance_ids[foreground])
        if np.count_nonzero(foreground & (instance_ids == instance_id))
        >= min_instance_points
    ]
    if not kept_ids:
        raise ValueError("No foreground instances satisfy min_instance_points")

    inside_by_box = (
        np.stack([points_inside_obb(points_world, obb) for obb in obbs], axis=0)
        if obbs
        else np.zeros((0, points_world.shape[0]), dtype=bool)
    )
    inside_any = inside_by_box.any(axis=0) if obbs else np.zeros(points_world.shape[0], dtype=bool)

    best_point_fractions = []
    center_hits = []
    for instance_id in kept_ids:
        instance_mask = foreground & (instance_ids == instance_id)
        fractions = inside_by_box[:, instance_mask].mean(axis=1) if obbs else np.zeros(0)
        best_point_fractions.append(float(fractions.max()) if len(fractions) else 0.0)
        center = points_world[instance_mask].mean(axis=0, keepdims=True)
        center_hits.append(
            any(bool(points_inside_obb(center, obb)[0]) for obb in obbs)
        )

    box_purities = []
    for box_mask in inside_by_box:
        box_foreground_ids = instance_ids[box_mask & foreground]
        if len(box_foreground_ids) == 0:
            box_purities.append(0.0)
            continue
        counts = np.unique(box_foreground_ids, return_counts=True)[1]
        box_purities.append(float(counts.max() / counts.sum()))

    fractions = np.asarray(best_point_fractions)
    foreground_kept = foreground & np.isin(instance_ids, kept_ids)
    return {
        "num_gt_instances": len(kept_ids),
        "num_pred_boxes": len(obbs),
        "gt_center_recall": float(np.mean(center_hits)),
        "gt_instance_recall_point10": float(np.mean(fractions >= 0.10)),
        "gt_instance_recall_point50": float(np.mean(fractions >= 0.50)),
        "mean_best_instance_point_fraction": float(fractions.mean()),
        "foreground_point_coverage": float(inside_any[foreground_kept].mean()),
        "mean_pred_box_dominant_instance_purity": float(np.mean(box_purities))
        if box_purities
        else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--prediction-csv", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--probability-threshold", type=float, default=0.0)
    parser.add_argument("--min-instance-points", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.prediction_csv) != len(args.label):
        raise ValueError("Each --prediction-csv requires one --label")
    scene_dir = args.scene_dir.resolve()
    points = load_points(scene_dir / "semi_points.ply")
    indices = load_indices(scene_dir / "semi_points_indices.npz")
    results = {}
    for label, prediction_csv in zip(args.label, args.prediction_csv):
        obbs = load_obbs(prediction_csv.resolve(), args.probability_threshold)
        results[label] = evaluate_obb_coverage(
            points,
            indices,
            obbs,
            min_instance_points=args.min_instance_points,
        )
    payload = {
        "schema": "ff_efm3d_point_obb_coverage_v1",
        "scene_dir": str(scene_dir),
        "probability_threshold": args.probability_threshold,
        "min_instance_points": args.min_instance_points,
        "results": results,
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
