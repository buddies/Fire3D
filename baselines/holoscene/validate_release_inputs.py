#!/usr/bin/env python3
"""Validate and visualize the four matched public HoloScene release scenes.

The HoloScene release archives contain calibrated RGB frames, per-frame instance
masks, a scan mesh, and camera transforms, but no metric depth images. This
script ray-casts the released scan mesh from the released cameras to construct
visible point clouds. Those points are suitable for checking data alignment and
can later serve as an explicit scan-mesh/oracle-depth input to FF-HoloScene.

HoloScene label convention:
    raw mask 255 -> background result ``surface_0.obj``
    raw mask k   -> object result ``surface_{k + 1}.obj``
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image


SCHEMA = "ff_holoscene_release_input_validation_v1"


@dataclass(frozen=True)
class SceneSpec:
    key: str
    input_relpath: str
    result_relpath: str


SCENES = (
    SceneSpec("replica_room_0", "data_dir/replica/room_0", "release_hf/replica_room_0"),
    SceneSpec(
        "scannetpp_67d",
        "data_dir/scannetpp/67d702f2e8",
        "release_hf/scannetpp_67d",
    ),
    SceneSpec(
        "gibson_beechwood_0",
        "data_dir/gibson/Beechwood_0_int",
        "release_hf/gibson_beechwood_0",
    ),
    SceneSpec(
        "uiuc_siebel_game_room",
        "data_dir/custom/siebelgame",
        "release_hf/uiuc_siebel_game_room",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holoscene-root",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2] / "baselines/_upstream/holoscene"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--scene",
        action="append",
        choices=[scene.key for scene in SCENES],
        help="Scene(s) to validate. Repeat as needed; default is all four.",
    )
    parser.add_argument("--overview-views", type=int, default=4)
    parser.add_argument("--cloud-views", type=int, default=24)
    parser.add_argument("--cloud-stride", type=int, default=8)
    parser.add_argument("--overview-stride", type=int, default=2)
    parser.add_argument("--max-preview-points", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def stable_palette(instance_ids: np.ndarray) -> np.ndarray:
    """Return deterministic uint8 colors for HoloScene result-surface ids."""
    ids = np.asarray(instance_ids, dtype=np.int64)
    colors = np.empty(ids.shape + (3,), dtype=np.uint8)
    colors[ids == 0] = (105, 105, 105)
    foreground = ids != 0
    values = ids[foreground].astype(np.uint64)
    values ^= values >> np.uint64(16)
    values *= np.uint64(0x7FEB352D)
    values ^= values >> np.uint64(15)
    values *= np.uint64(0x846CA68B)
    values ^= values >> np.uint64(16)
    colors[foreground, 0] = (55 + (values & 0xBF)).astype(np.uint8)
    colors[foreground, 1] = (55 + ((values >> 8) & 0xBF)).astype(np.uint8)
    colors[foreground, 2] = (55 + ((values >> 16) & 0xBF)).astype(np.uint8)
    return colors


def mask_to_surface_ids(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    result = mask.astype(np.int32) + 1
    result[mask == 255] = 0
    return result


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def uniform_indices(count: int, requested: int) -> list[int]:
    if count <= 0:
        return []
    requested = max(1, min(int(requested), count))
    return sorted(set(np.linspace(0, count - 1, requested).round().astype(int).tolist()))


def load_scene_mesh(path: Path) -> tuple[o3d.t.geometry.RaycastingScene, dict[str, Any]]:
    legacy = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    if len(legacy.vertices) == 0 or len(legacy.triangles) == 0:
        raise ValueError(f"Empty triangle mesh: {path}")
    vertices = np.asarray(legacy.vertices)
    triangles = np.asarray(legacy.triangles)
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
    ray_scene = o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(tensor_mesh)
    return ray_scene, {
        "vertices": int(vertices.shape[0]),
        "triangles": int(triangles.shape[0]),
        "bounds_min": vertices.min(axis=0).tolist(),
        "bounds_max": vertices.max(axis=0).tolist(),
    }


def camera_to_world_cv(transform_matrix: Any, convert_gl_to_cv: bool = True) -> np.ndarray:
    pose = np.asarray(transform_matrix, dtype=np.float64).reshape(4, 4).copy()
    if convert_gl_to_cv:
        pose[:3, 1:3] *= -1.0
    return pose


def cast_view(
    ray_scene: o3d.t.geometry.RaycastingScene,
    camera: dict[str, float],
    pose: np.ndarray,
    *,
    height: int,
    width: int,
    stride: int,
) -> dict[str, np.ndarray]:
    stride = max(1, int(stride))
    ys = np.arange(0, height, stride, dtype=np.int32)
    xs = np.arange(0, width, stride, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    u = grid_x.astype(np.float32) + 0.5
    v = grid_y.astype(np.float32) + 0.5
    camera_dirs = np.stack(
        [
            (u - float(camera["cx"])) / float(camera["fl_x"]),
            (v - float(camera["cy"])) / float(camera["fl_y"]),
            np.ones_like(u),
        ],
        axis=-1,
    )
    world_dirs = camera_dirs @ pose[:3, :3].T
    origins = np.broadcast_to(pose[:3, 3], world_dirs.shape).astype(np.float32)
    rays = np.concatenate([origins, world_dirs.astype(np.float32)], axis=-1)
    cast = ray_scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
    t_hit = cast["t_hit"].numpy()
    primitive_ids = cast["primitive_ids"].numpy()
    hit = np.isfinite(t_hit) & (t_hit > 0.0)
    points = origins + world_dirs * t_hit[..., None]
    points[~hit] = np.nan
    return {
        "xs": grid_x,
        "ys": grid_y,
        "t_hit": t_hit,
        "primitive_ids": primitive_ids,
        "hit": hit,
        "points": points.astype(np.float32),
    }


def write_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    colors = np.asarray(colors).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    colors = colors[valid]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if colors.dtype == np.uint8 or float(colors.max(initial=0)) > 1.0:
        colors = colors.astype(np.float64) / 255.0
    cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False):
        raise OSError(f"Failed to write {path}")


def depth_color(depth: np.ndarray, hit: np.ndarray) -> np.ndarray:
    result = np.full(depth.shape + (3,), 245, dtype=np.uint8)
    valid_values = depth[hit]
    if valid_values.size == 0:
        return result
    low, high = np.percentile(valid_values, [2.0, 98.0])
    if not np.isfinite(high) or high <= low:
        high = low + 1.0
    normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
    mapped = cv2.applyColorMap(
        np.round((1.0 - normalized) * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO
    )[..., ::-1]
    result[hit] = mapped[hit]
    return result


def resize_nearest(array: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(array, (width, height), interpolation=cv2.INTER_NEAREST)


def pca_projection(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    center = np.median(points, axis=0)
    centered = points - center
    _, _, axes = np.linalg.svd(centered[:: max(1, len(centered) // 200000)], full_matrices=False)
    projected = centered @ axes.T
    return projected[:, :2], axes


def compute_centroid_consistency(
    instance_points: dict[int, list[np.ndarray]], scene_diagonal: float
) -> dict[str, Any]:
    normalized_spreads: list[float] = []
    per_instance: dict[str, Any] = {}
    for instance_id, chunks in sorted(instance_points.items()):
        centroids = [np.median(chunk, axis=0) for chunk in chunks if len(chunk) >= 4]
        if len(centroids) < 2:
            continue
        centroids_array = np.stack(centroids)
        center = np.median(centroids_array, axis=0)
        spread = float(np.median(np.linalg.norm(centroids_array - center, axis=1)))
        normalized = spread / max(scene_diagonal, 1e-8)
        normalized_spreads.append(normalized)
        per_instance[str(instance_id)] = {
            "views": len(centroids),
            "centroid_median_spread_world": spread,
            "centroid_median_spread_over_scene_diagonal": normalized,
        }
    return {
        "eligible_instances": len(normalized_spreads),
        "median_centroid_spread_over_scene_diagonal": (
            float(np.median(normalized_spreads)) if normalized_spreads else None
        ),
        "p90_centroid_spread_over_scene_diagonal": (
            float(np.percentile(normalized_spreads, 90.0)) if normalized_spreads else None
        ),
        "per_instance": per_instance,
    }


def audit_masks(
    image_paths: list[Path],
    mask_paths: list[Path],
    frames: list[dict[str, Any]],
) -> tuple[dict[str, Any], np.ndarray]:
    all_raw_ids: set[int] = set()
    shapes: set[tuple[int, int]] = set()
    paired_stems = True
    paired_frame_keys = True
    image_frame_matches = 0
    foreground_pixels = np.zeros((len(mask_paths),), dtype=np.int64)
    for index, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        mask = np.asarray(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        shapes.add(tuple(int(v) for v in mask.shape[:2]))
        all_raw_ids.update(int(value) for value in np.unique(mask))
        foreground_pixels[index] = int(np.count_nonzero(mask != 255))
        paired_stems &= image_path.stem == mask_path.stem
        image_key_match = re.search(r"(\d+)$", image_path.stem)
        mask_key_match = re.search(r"(\d+)$", mask_path.stem)
        paired_frame_keys &= bool(
            image_key_match
            and mask_key_match
            and int(image_key_match.group(1)) == int(mask_key_match.group(1))
        )
        frame_name = Path(str(frames[index].get("file_path", ""))).name
        image_frame_matches += int(frame_name == image_path.name)
    surface_ids = sorted(0 if raw_id == 255 else raw_id + 1 for raw_id in all_raw_ids)
    return (
        {
            "raw_mask_ids": sorted(all_raw_ids),
            "surface_ids_from_masks": surface_ids,
            "mask_shapes": [list(shape) for shape in sorted(shapes)],
            "image_mask_stems_all_match": bool(paired_stems),
            "image_mask_numeric_frame_keys_all_match": bool(paired_frame_keys),
            "image_mask_pairing_valid": bool(paired_stems or paired_frame_keys),
            "transform_frame_filenames_matching": int(image_frame_matches),
            "frames_with_foreground": int(np.count_nonzero(foreground_pixels)),
            "median_foreground_pixels_in_nonempty_frames": (
                float(np.median(foreground_pixels[foreground_pixels > 0]))
                if np.any(foreground_pixels > 0)
                else 0.0
            ),
        },
        foreground_pixels,
    )


def informative_indices(foreground_pixels: np.ndarray, requested: int) -> list[int]:
    """Choose one foreground-rich view per temporal segment."""
    count = len(foreground_pixels)
    if count <= 0:
        return []
    requested = max(1, min(int(requested), count))
    boundaries = np.linspace(0, count, requested + 1).round().astype(int)
    selected: list[int] = []
    for segment in range(requested):
        begin = int(boundaries[segment])
        end = max(begin + 1, int(boundaries[segment + 1]))
        values = foreground_pixels[begin:end]
        selected.append(begin + int(np.argmax(values)))
    return sorted(set(selected))


def scene_center_scale(frames: list[dict[str, Any]]) -> tuple[np.ndarray, float]:
    translations = np.stack(
        [
            np.asarray(frame["transform_matrix"], dtype=np.float64).reshape(4, 4)[:3, 3]
            for frame in frames
        ]
    )
    minimum = translations.min(axis=0)
    maximum = translations.max(axis=0)
    return (minimum + maximum) / 2.0, float(np.max(maximum - minimum))


def validate_scene(
    spec: SceneSpec,
    *,
    root: Path,
    output_root: Path,
    overview_views: int,
    cloud_views: int,
    cloud_stride: int,
    overview_stride: int,
    max_preview_points: int,
    seed: int,
) -> dict[str, Any]:
    scene_dir = root / spec.input_relpath
    result_dir = root / spec.result_relpath
    output_dir = output_root / spec.key
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(path for path in (scene_dir / "images").iterdir() if path.is_file())
    mask_paths = sorted(
        path for path in (scene_dir / "instance_mask").iterdir() if path.is_file()
    )
    camera_data = json.loads((scene_dir / "transforms.json").read_text(encoding="utf-8"))
    frames = camera_data["frames"]
    if not image_paths or not mask_paths:
        raise ValueError(f"Missing images/masks under {scene_dir}")
    if len(image_paths) != len(mask_paths) or len(image_paths) != len(frames):
        raise ValueError(
            f"{spec.key}: images={len(image_paths)}, masks={len(mask_paths)}, "
            f"frames={len(frames)}"
        )

    first_image = np.asarray(Image.open(image_paths[0]).convert("RGB"))
    height, width = first_image.shape[:2]
    declared_size = (int(camera_data["h"]), int(camera_data["w"]))
    if declared_size != (height, width):
        raise ValueError(
            f"{spec.key}: declared HxW={declared_size}, image HxW={(height, width)}"
        )

    ray_scene, mesh_stats = load_scene_mesh(scene_dir / "mesh.ply")
    mask_audit, foreground_pixels = audit_masks(image_paths, mask_paths, frames)
    graph = json.loads((scene_dir / "graph.json").read_text(encoding="utf-8"))
    surface_paths = sorted(
        result_dir.glob("surface_*.obj"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    released_surface_ids = [int(path.stem.rsplit("_", 1)[1]) for path in surface_paths]
    expected_surface_ids = mask_audit["surface_ids_from_masks"]
    center, scale = scene_center_scale(frames)

    cloud_indices = uniform_indices(len(frames), cloud_views)
    overview_indices = informative_indices(foreground_pixels, overview_views)
    camera = {
        "fl_x": float(camera_data["fl_x"]),
        "fl_y": float(camera_data["fl_y"]),
        "cx": float(camera_data["cx"]),
        "cy": float(camera_data["cy"]),
    }

    all_points: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    object_points: dict[int, list[np.ndarray]] = {}
    per_view: list[dict[str, Any]] = []
    raw_pose_foreground_hits: list[float] = []

    for frame_index in cloud_indices:
        image = np.asarray(Image.open(image_paths[frame_index]).convert("RGB"))
        raw_mask = np.asarray(Image.open(mask_paths[frame_index]))
        if raw_mask.ndim == 3:
            raw_mask = raw_mask[..., 0]
        surface_ids = mask_to_surface_ids(raw_mask)

        pose_cv = camera_to_world_cv(frames[frame_index]["transform_matrix"], True)
        cast = cast_view(
            ray_scene,
            camera,
            pose_cv,
            height=height,
            width=width,
            stride=cloud_stride,
        )
        sampled_rgb = image[cast["ys"], cast["xs"]]
        sampled_ids = surface_ids[cast["ys"], cast["xs"]]
        hit = cast["hit"]
        points = cast["points"][hit]
        colors = sampled_rgb[hit]
        ids = sampled_ids[hit]
        all_points.append(points)
        all_rgb.append(colors)
        all_ids.append(ids)
        for instance_id in np.unique(ids):
            instance_id = int(instance_id)
            if instance_id <= 0:
                continue
            keep = ids == instance_id
            if int(keep.sum()) >= 4:
                object_points.setdefault(instance_id, []).append(points[keep])

        foreground = sampled_ids > 0
        foreground_hit = float(hit[foreground].mean()) if foreground.any() else math.nan
        per_view.append(
            {
                "frame_index": frame_index,
                "image": image_paths[frame_index].name,
                "mask": mask_paths[frame_index].name,
                "ray_count": int(hit.size),
                "hit_fraction_all": float(hit.mean()),
                "hit_fraction_foreground": foreground_hit,
                "foreground_surface_ids": sorted(
                    int(value) for value in np.unique(sampled_ids[foreground])
                ),
            }
        )

        # Wrong-convention sanity check: use the raw OpenGL pose with +Z CV rays.
        raw_pose = camera_to_world_cv(frames[frame_index]["transform_matrix"], False)
        raw_cast = cast_view(
            ray_scene,
            camera,
            raw_pose,
            height=height,
            width=width,
            stride=max(cloud_stride * 2, 16),
        )
        raw_sampled_ids = surface_ids[raw_cast["ys"], raw_cast["xs"]]
        raw_foreground = raw_sampled_ids > 0
        if raw_foreground.any():
            raw_pose_foreground_hits.append(float(raw_cast["hit"][raw_foreground].mean()))

    points = np.concatenate(all_points, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    ids = np.concatenate(all_ids, axis=0)
    object_keep = ids > 0
    write_point_cloud(output_dir / "visible_scan_points_rgb.ply", points, rgb)
    write_point_cloud(
        output_dir / "visible_scan_points_instances.ply", points, stable_palette(ids)
    )
    write_point_cloud(
        output_dir / "visible_object_points_rgb.ply",
        points[object_keep],
        rgb[object_keep],
    )
    write_point_cloud(
        output_dir / "visible_object_points_instances.ply",
        points[object_keep],
        stable_palette(ids[object_keep]),
    )

    rows = len(overview_indices)
    figure, axes = plt.subplots(rows, 4, figsize=(16, 3.7 * rows), squeeze=False)
    figure.suptitle(
        f"{spec.key}: released RGB / instance masks / mesh-raycast depth / visible points",
        fontsize=15,
        fontweight="bold",
    )
    for row, frame_index in enumerate(overview_indices):
        image = np.asarray(Image.open(image_paths[frame_index]).convert("RGB"))
        raw_mask = np.asarray(Image.open(mask_paths[frame_index]))
        if raw_mask.ndim == 3:
            raw_mask = raw_mask[..., 0]
        surface_ids = mask_to_surface_ids(raw_mask)
        label_colors = stable_palette(surface_ids)
        overlay = np.round(0.58 * image + 0.42 * label_colors).astype(np.uint8)

        pose_cv = camera_to_world_cv(frames[frame_index]["transform_matrix"], True)
        cast = cast_view(
            ray_scene,
            camera,
            pose_cv,
            height=height,
            width=width,
            stride=overview_stride,
        )
        depth_small = depth_color(cast["t_hit"], cast["hit"])
        depth_full = resize_nearest(depth_small, width, height)

        view_points = cast["points"][cast["hit"]]
        view_ids = surface_ids[cast["ys"], cast["xs"]][cast["hit"]]
        if len(view_points) > max_preview_points:
            rng = np.random.default_rng(seed + frame_index)
            keep = rng.choice(len(view_points), max_preview_points, replace=False)
            view_points = view_points[keep]
            view_ids = view_ids[keep]
        projected, _ = pca_projection(view_points)
        plot_colors = stable_palette(view_ids).astype(np.float32) / 255.0

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"RGB: {image_paths[frame_index].name}")
        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(
            f"Mask overlay ({len(np.unique(surface_ids[surface_ids > 0]))} visible objects)"
        )
        axes[row, 2].imshow(depth_full)
        axes[row, 2].set_title("Scan-mesh raycast depth")
        axes[row, 3].scatter(
            projected[:, 0],
            projected[:, 1],
            s=0.2,
            c=plot_colors,
            linewidths=0,
            rasterized=True,
        )
        axes[row, 3].set_aspect("equal", adjustable="datalim")
        axes[row, 3].set_title("Visible 3D points (PCA view)")
        for column in range(4):
            axes[row, column].axis("off")

    figure.tight_layout(rect=(0, 0, 1, 0.975))
    figure.savefig(output_dir / "input_alignment_overview.png", dpi=160)
    plt.close(figure)

    mesh_bounds = np.asarray(mesh_stats["bounds_max"]) - np.asarray(mesh_stats["bounds_min"])
    scene_diagonal = float(np.linalg.norm(mesh_bounds))
    cv_hit_values = [
        row["hit_fraction_foreground"]
        for row in per_view
        if np.isfinite(row["hit_fraction_foreground"])
    ]
    audit = {
        "schema": SCHEMA,
        "scene": spec.key,
        "input_scene_dir": str(scene_dir.resolve()),
        "released_result_dir": str(result_dir.resolve()),
        "input_counts": {
            "images": len(image_paths),
            "instance_masks": len(mask_paths),
            "camera_frames": len(frames),
            "graph_nodes": len(graph),
        },
        "image_size_hw": [height, width],
        "intrinsics": camera,
        "mask_audit": mask_audit,
        "released_surface_ids": released_surface_ids,
        "released_surface_count": len(released_surface_ids),
        "surface_mapping_complete": released_surface_ids == expected_surface_ids,
        "mesh": mesh_stats,
        "holoscene_normalization": {
            "camera_center_world": center.tolist(),
            "camera_range_scale": scale,
            "definition": (
                "Matches datasets/ns_dataset.py: center is midpoint of camera-translation "
                "bounds; scale is max camera-translation range."
            ),
        },
        "raycast": {
            "pose_convention": (
                "transforms.json camera-to-world OpenGL; columns 1 and 2 sign-flipped "
                "to OpenCV before casting +Z camera rays, matching NSDataset."
            ),
            "sampled_frame_indices": cloud_indices,
            "overview_frame_indices": overview_indices,
            "stride": cloud_stride,
            "visible_points_all": int(len(points)),
            "visible_points_objects": int(object_keep.sum()),
            "median_foreground_hit_fraction_correct_cv_pose": (
                float(np.median(cv_hit_values)) if cv_hit_values else None
            ),
            "median_foreground_hit_fraction_wrong_raw_gl_pose": (
                float(np.median(raw_pose_foreground_hits))
                if raw_pose_foreground_hits
                else None
            ),
            "per_view": per_view,
        },
        "cross_view_instance_consistency": compute_centroid_consistency(
            object_points, scene_diagonal
        ),
        "outputs": {
            "overview": str((output_dir / "input_alignment_overview.png").resolve()),
            "scene_rgb_points": str((output_dir / "visible_scan_points_rgb.ply").resolve()),
            "scene_instance_points": str(
                (output_dir / "visible_scan_points_instances.ply").resolve()
            ),
            "object_rgb_points": str((output_dir / "visible_object_points_rgb.ply").resolve()),
            "object_instance_points": str(
                (output_dir / "visible_object_points_instances.ply").resolve()
            ),
        },
        "input_scope_warning": (
            "The release has no metric depth images. These points are generated by "
            "ray-casting mesh.ply and are therefore scan-mesh/oracle-depth inputs, not "
            "native HoloScene RGB-only inputs."
        ),
    }
    atomic_json(output_dir / "audit.json", audit)
    return audit


def write_combined_overview(output_root: Path, audits: list[dict[str, Any]]) -> Path:
    rows: list[np.ndarray] = []
    for audit in audits:
        path = Path(audit["outputs"]["overview"])
        image = np.asarray(Image.open(path).convert("RGB"))
        target_width = 1600
        target_height = max(1, round(image.shape[0] * target_width / image.shape[1]))
        image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        # One representative row from each per-scene overview. Matplotlib leaves
        # a small title/header band, so crop below it before selecting row zero.
        row_count = max(1, len(audit["raycast"]["overview_frame_indices"]))
        header = int(round(target_height * 0.045))
        row_height = max(1, (target_height - header) // row_count)
        representative = image[header : header + row_height]
        label = np.full((58, target_width, 3), 244, dtype=np.uint8)
        cv2.putText(
            label,
            audit["scene"],
            (18, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.15,
            (34, 48, 68),
            2,
            cv2.LINE_AA,
        )
        rows.append(np.concatenate([label, representative], axis=0))
    combined = np.concatenate(rows, axis=0)
    path = output_root / "holoscene_release_data_overview.png"
    Image.fromarray(combined).save(path)
    return path


def main() -> None:
    args = parse_args()
    selected = set(args.scene or [scene.key for scene in SCENES])
    specs = [scene for scene in SCENES if scene.key in selected]
    args.output_root.mkdir(parents=True, exist_ok=True)
    audits = []
    for spec in specs:
        print(f"[validate] {spec.key}", flush=True)
        audit = validate_scene(
            spec,
            root=args.holoscene_root,
            output_root=args.output_root,
            overview_views=args.overview_views,
            cloud_views=args.cloud_views,
            cloud_stride=args.cloud_stride,
            overview_stride=args.overview_stride,
            max_preview_points=args.max_preview_points,
            seed=args.seed,
        )
        audits.append(audit)
        print(
            f"[done] {spec.key}: {audit['raycast']['visible_points_objects']} object points; "
            f"surface mapping={audit['surface_mapping_complete']}",
            flush=True,
        )
    combined_overview = write_combined_overview(args.output_root, audits)
    summary = {
        "schema": SCHEMA,
        "scenes": audits,
        "combined_overview": str(combined_overview.resolve()),
    }
    atomic_json(args.output_root / "summary.json", summary)
    print(combined_overview)


if __name__ == "__main__":
    main()
