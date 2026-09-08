#!/usr/bin/env python3
"""Evaluate SimRecon 3D instances with the FF scene-perception protocol."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("ithor", "imaginarium"),
        default="imaginarium",
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--prepared-scene-dir", type=Path, required=True)
    parser.add_argument("--semantic-label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-downsample", type=int, default=16)
    parser.add_argument("--transfer-radius-multiplier", type=float, default=3.0)
    parser.add_argument("--transfer-radius-min", type=float, default=0.02)
    parser.add_argument("--transfer-radius-max", type=float, default=0.08)
    parser.add_argument("--obb-quantile", type=float, default=0.01)
    parser.add_argument("--obb-margin", type=float, default=1.05)
    parser.add_argument("--min-pred-points", type=int, default=20)
    return parser.parse_args()


def read_ply_xyz(path: Path) -> np.ndarray:
    vertex = PlyData.read(path)["vertex"].data
    return np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)


def load_scene_data(args: argparse.Namespace) -> dict:
    sys.path.insert(0, str(args.repo_root))
    if args.dataset == "imaginarium":
        from utils.data_imaginarium import (
            get_inference_data,
            load_imaginarium_data,
        )

        data_list = load_imaginarium_data(data_root=str(args.data_root))
    elif args.dataset == "ithor":
        from utils.data_ithor import get_inference_data, load_ithor_data

        data_list = load_ithor_data(data_root=str(args.data_root))
    else:
        raise ValueError(args.dataset)
    matches = [
        (index, row)
        for index, row in enumerate(data_list)
        if row["scene_id"] == args.scene_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one scene {args.scene_id}, found {len(matches)}")
    index, _ = matches[0]
    return get_inference_data(
        data_list,
        index,
        image_downsample=args.image_downsample,
    )


def load_instance_scores(
    prepared_scene_dir: Path, instance_info_path: Path
) -> tuple[dict[int, float], dict[int, dict]]:
    score_by_frame_mask: dict[tuple[int, int], float] = {}
    for score_path in sorted((prepared_scene_dir / "sam" / "scores").glob("*.json")):
        row = json.loads(score_path.read_text(encoding="utf-8"))
        frame_id = int(row["frame_index"])
        for mask in row["masks"]:
            score_by_frame_mask[(frame_id, int(mask["mask_id"]))] = float(mask["score"])

    info = json.loads(instance_info_path.read_text(encoding="utf-8"))
    scores, diagnostics = {}, {}
    for instance in info["instances"]:
        instance_id = int(instance["instance_id"])
        values = []
        for source in instance.get("source_masks", []):
            key = (int(source["frame_id"]), int(source["mask_id"]))
            if key in score_by_frame_mask:
                values.append(score_by_frame_mask[key])
        values.sort(reverse=True)
        top_values = values[:3]
        score = float(np.mean(top_values)) if top_values else 0.0
        scores[instance_id] = score
        diagnostics[instance_id] = {
            "num_source_mask_scores": len(values),
            "top3_source_mask_scores": top_values,
            "aggregated_confidence": score,
        }
    return scores, diagnostics


def fit_z_up_obb(
    points: np.ndarray, quantile: float, margin: float
) -> tuple[np.ndarray, float, float, np.ndarray]:
    xy_center = np.median(points[:, :2], axis=0)
    centered_xy = points[:, :2] - xy_center
    covariance = centered_xy.T @ centered_xy / max(centered_xy.shape[0], 1)
    values, vectors = np.linalg.eigh(covariance)
    principal = vectors[:, int(np.argmax(values))]
    yaw = float(np.arctan2(principal[1], principal[0]))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation_xy = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    local_xy = centered_xy @ rotation_xy
    lower_xy = np.quantile(local_xy, quantile, axis=0)
    upper_xy = np.quantile(local_xy, 1.0 - quantile, axis=0)
    lower_z = float(np.quantile(points[:, 2], quantile))
    upper_z = float(np.quantile(points[:, 2], 1.0 - quantile))
    local_center_xy = 0.5 * (lower_xy + upper_xy)
    center_xy = xy_center + local_center_xy @ rotation_xy.T
    center = np.array([center_xy[0], center_xy[1], 0.5 * (lower_z + upper_z)])
    extents = np.array(
        [upper_xy[0] - lower_xy[0], upper_xy[1] - lower_xy[1], upper_z - lower_z]
    )
    scale = float(max(np.max(extents) * margin, 1e-4))
    quaternion = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    return center, scale, yaw, quaternion


def adaptive_transfer_radius(
    xyz: np.ndarray, multiplier: float, minimum: float, maximum: float
) -> tuple[float, float]:
    if xyz.shape[0] > 50_000:
        indices = np.linspace(0, xyz.shape[0] - 1, 50_000, dtype=np.int64)
        sample = xyz[indices]
    else:
        sample = xyz
    distances, _ = cKDTree(sample).query(sample, k=2, workers=-1)
    spacing = float(np.median(distances[:, 1]))
    radius = float(np.clip(multiplier * spacing, minimum, maximum))
    return radius, spacing


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene = load_scene_data(args)
    eval_points = np.asarray(scene["points"], dtype=np.float64)
    gt_labels = np.asarray(scene["points_instance_masks"], dtype=np.int32).reshape(-1)
    gt_obbs = list(scene["gt_obbs"])

    gaussian_path = args.semantic_label_dir / "point_cloud.ply"
    if not gaussian_path.exists():
        gaussian_path = args.prepared_scene_dir / "point_cloud.ply"
    gaussian_xyz_raw = read_ply_xyz(gaussian_path)
    labels = np.load(args.semantic_label_dir / "point_cloud_labels.npy").astype(np.int32)
    if gaussian_xyz_raw.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Gaussian/label mismatch: {gaussian_xyz_raw.shape[0]} versus {labels.shape[0]}"
        )

    transform = np.asarray(scene["preprocess_transform"], dtype=np.float64)
    gaussian_xyz = gaussian_xyz_raw @ transform[:3, :3].T + transform[:3, 3]
    valid_gaussians = labels >= 0
    labeled_xyz = gaussian_xyz[valid_gaussians]
    labeled_ids = labels[valid_gaussians]
    if labeled_xyz.shape[0] == 0:
        raise ValueError("SimRecon exported no labeled Gaussians")

    score_path = args.semantic_label_dir / "instance_info.json"
    scores, score_diagnostics = load_instance_scores(
        args.prepared_scene_dir, score_path
    )
    pred_obbs = []
    obb_diagnostics = {}
    for instance_id in sorted(np.unique(labeled_ids).tolist()):
        instance_xyz = gaussian_xyz[labels == instance_id]
        if instance_xyz.shape[0] < args.min_pred_points:
            continue
        center, scale, yaw, quaternion = fit_z_up_obb(
            instance_xyz, args.obb_quantile, args.obb_margin
        )
        pred_obbs.append(
            {
                "index": int(instance_id),
                "translate": center.tolist(),
                "rotation": quaternion.tolist(),
                "scale": scale,
                "confidence": float(scores.get(instance_id, 0.0)),
            }
        )
        obb_diagnostics[int(instance_id)] = {
            "num_labeled_gaussians": int(instance_xyz.shape[0]),
            "yaw_radians": yaw,
            "scale": scale,
            **score_diagnostics.get(instance_id, {}),
        }

    radius, median_spacing = adaptive_transfer_radius(
        labeled_xyz,
        args.transfer_radius_multiplier,
        args.transfer_radius_min,
        args.transfer_radius_max,
    )
    distances, indices = cKDTree(labeled_xyz).query(eval_points, k=1, workers=-1)
    pred_labels = np.full(eval_points.shape[0], -1, dtype=np.int32)
    supported = distances <= radius
    pred_labels[supported] = labeled_ids[indices[supported]]

    from eval.perception.det_seg_eval_utils import calculate_mAP_and_mIoU

    map25, miou, iou_per_class = calculate_mAP_and_mIoU(
        pred_bboxes=pred_obbs,
        gt_bboxes=gt_obbs,
        pred_labels=pred_labels,
        gt_labels=gt_labels,
    )
    metrics = {
        "schema": "ff_simrecon_scene_perception_eval_v2",
        "dataset": args.dataset,
        "scene_id": args.scene_id,
        "protocol": {
            "detection": "class_agnostic_3d_obb_ap_iou_0.25",
            "segmentation": "point_instance_miou_after_obb_iou_0.25_matching",
            "box_source": "robust_z_up_pca_box_fitted_from_labeled_gaussian_centers",
            "mask_transfer": "bounded_nearest_labeled_gaussian",
            "confidence": "mean_top3_source_cropformer_mask_scores",
            "image_downsample": int(args.image_downsample),
        },
        "OBB_AP_at_0.25": float(map25),
        "point_instance_mIoU": float(miou),
        "iou_per_class": [
            None if not np.isfinite(value) else float(value) for value in iou_per_class
        ],
        "counts": {
            "gt_obbs": len(gt_obbs),
            "pred_obbs": len(pred_obbs),
            "evaluation_points": int(eval_points.shape[0]),
            "gaussians": int(gaussian_xyz.shape[0]),
            "labeled_gaussians": int(labeled_xyz.shape[0]),
            "supported_evaluation_points": int(supported.sum()),
        },
        "transfer": {
            "median_labeled_gaussian_spacing": median_spacing,
            "radius": radius,
            "supported_fraction": float(supported.mean()),
            "nearest_distance_mean": float(np.mean(distances)),
            "nearest_distance_median": float(np.median(distances)),
        },
        "input_mode": "shared_dataset_camera_and_depth_geometry_adapter",
        "uses_gt_masks_during_simrecon_inference": False,
        "uses_gt_masks_and_transforms_for_final_scoring_only": True,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "predicted_obbs.json").write_text(
        json.dumps(
            {
                "scene_id": args.scene_id,
                "objects": pred_obbs,
                "diagnostics": obb_diagnostics,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "point_instance_masks.pkl").open("wb") as handle:
        pickle.dump({"pred": pred_labels, "gt": gt_labels}, handle)
    np.savez_compressed(
        args.output_dir / "point_transfer_debug.npz",
        points=eval_points.astype(np.float32),
        pred=pred_labels,
        gt=gt_labels,
        nearest_distance=distances.astype(np.float32),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
