#!/usr/bin/env python3
"""Prepare one FF iTHOR/Imaginarium scene for the SimRecon data contract.

The adapter uses only RGB, cameras, and depth-derived geometry for SimRecon
inference. Ground-truth instance masks and object transforms are intentionally
not read here; they remain isolated to the final evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ithor", "imaginarium"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--scene-id")
    choice.add_argument("--benchmark-index", type=int)
    parser.add_argument("--random-seed", type=int, default=20260724)
    parser.add_argument("--depth-stride", type=int, default=8)
    parser.add_argument("--max-init-points", type=int, default=200_000)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def select_scene(args: argparse.Namespace) -> dict:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    scenes = list(manifest["scenes"])
    if args.scene_id is not None:
        matches = [row for row in scenes if row["scene_id"] == args.scene_id]
        if len(matches) != 1:
            raise ValueError(f"Expected one manifest scene {args.scene_id!r}, found {len(matches)}")
        return matches[0]
    if args.benchmark_index is not None:
        matches = [row for row in scenes if int(row["benchmark_index"]) == args.benchmark_index]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one benchmark index {args.benchmark_index}, found {len(matches)}"
            )
        return matches[0]
    return random.Random(args.random_seed).choice(scenes)


def camera_to_world(frame: dict) -> np.ndarray:
    eye = np.asarray(frame["eye"], dtype=np.float64)
    lookat = np.asarray(frame["lookat"], dtype=np.float64)
    up_vector = np.asarray(frame["up"], dtype=np.float64)
    forward = lookat - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up_vector)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.column_stack([right, -up, forward])
    c2w[:3, 3] = eye
    return c2w


def rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0)) * 2
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = np.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0)) * 2
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0)) * 2
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quat /= np.linalg.norm(quat)
    return quat


def read_depth(path: Path) -> np.ndarray:
    archive = np.load(path)
    if "depth" in archive:
        return archive["depth"].astype(np.float32)
    if "arr_0" in archive:
        return archive["arr_0"].astype(np.float32)
    raise KeyError(f"Unsupported depth keys in {path}: {list(archive.keys())}")


def write_point_cloud(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    vertices = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = xyz.T.astype(np.float32)
    vertices["nx"] = vertices["ny"] = vertices["nz"] = 0.0
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T.astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def main() -> None:
    args = parse_args()
    if args.depth_stride <= 0:
        raise ValueError("--depth-stride must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_dataset = str(manifest.get("dataset", "imaginarium")).lower()
    dataset = args.dataset or manifest_dataset
    if dataset != manifest_dataset:
        raise ValueError(
            f"--dataset={dataset!r} does not match manifest dataset "
            f"{manifest_dataset!r}"
        )
    selected = select_scene(args)
    scene_id = selected["scene_id"]
    scene_dir = args.output_root / scene_id
    metadata_path = scene_dir / "simrecon_input.json"
    if metadata_path.exists() and not args.force:
        print(metadata_path)
        return

    relative_frames = Path(selected["frames_dir"])
    relative_depths = Path(selected["depths_dir"])
    camera_path = args.data_root / selected["camera_path"]
    frames_dir = args.data_root / relative_frames
    depths_dir = args.data_root / relative_depths
    camera_data = json.loads(camera_path.read_text(encoding="utf-8"))
    intrinsics = np.asarray(camera_data["K"], dtype=np.float64)
    width, height = int(camera_data["width"]), int(camera_data["height"])
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    depth_paths = sorted(depths_dir.glob("*.npz"))
    frames = camera_data["frames"]
    if not (len(frame_paths) == len(depth_paths) == len(frames)):
        raise ValueError(
            f"Frame/depth/camera mismatch: {len(frame_paths)}, {len(depth_paths)}, {len(frames)}"
        )

    if scene_dir.exists() and args.force:
        shutil.rmtree(scene_dir)
    images_dir = scene_dir / "images"
    sparse_dir = scene_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    for source in frame_paths:
        destination = images_dir / source.name
        if args.copy_images:
            shutil.copy2(source, destination)
        else:
            destination.symlink_to(source.resolve())

    cameras_txt = (
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {width} {height} "
        f"{intrinsics[0, 0]:.17g} {intrinsics[1, 1]:.17g} "
        f"{intrinsics[0, 2]:.17g} {intrinsics[1, 2]:.17g}\n"
    )
    (sparse_dir / "cameras.txt").write_text(cameras_txt, encoding="utf-8")

    image_lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    all_xyz, all_rgb = [], []
    stride = args.depth_stride
    yy, xx = np.meshgrid(
        np.arange(0, height, stride, dtype=np.int32),
        np.arange(0, width, stride, dtype=np.int32),
        indexing="ij",
    )
    for frame_index, (frame_path, depth_path, frame) in enumerate(
        zip(frame_paths, depth_paths, frames)
    ):
        c2w = camera_to_world(frame)
        w2c = np.linalg.inv(c2w)
        quaternion = rotation_matrix_to_quaternion_wxyz(w2c[:3, :3])
        translation = w2c[:3, 3]
        values = [
            str(frame_index + 1),
            *(f"{value:.17g}" for value in quaternion),
            *(f"{value:.17g}" for value in translation),
            "1",
            frame_path.name,
        ]
        image_lines.append(" ".join(values))
        image_lines.append("")

        depth = read_depth(depth_path)
        rgb = np.asarray(Image.open(frame_path).convert("RGB"))
        sampled_depth = depth[yy, xx]
        valid = (
            np.isfinite(sampled_depth)
            & (sampled_depth > 0)
            & (sampled_depth <= 50.0)
        )
        z = sampled_depth[valid].astype(np.float64)
        x = (xx[valid] - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y = (yy[valid] - intrinsics[1, 2]) * z / intrinsics[1, 1]
        camera_xyz = np.stack([x, y, z], axis=1)
        world_xyz = camera_xyz @ c2w[:3, :3].T + c2w[:3, 3]
        all_xyz.append(world_xyz)
        all_rgb.append(rgb[yy[valid], xx[valid]])

    (sparse_dir / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    if xyz.shape[0] > args.max_init_points:
        rng = np.random.default_rng(args.random_seed)
        indices = np.sort(
            rng.choice(xyz.shape[0], size=args.max_init_points, replace=False)
        )
        xyz, rgb = xyz[indices], rgb[indices]
    write_point_cloud(sparse_dir / "points3D.ply", xyz, rgb)

    metadata = {
        "schema": "ff_simrecon_scene_input_v2",
        "dataset": dataset,
        "scene_id": scene_id,
        "benchmark_index": int(selected["benchmark_index"]),
        "raw_index": int(selected["raw_index"]),
        "legacy_eval_index": int(selected["legacy_eval_index"]),
        "selection_seed": int(args.random_seed),
        "num_frames": len(frame_paths),
        "num_gt_objects_manifest_only": int(selected["num_gt_objects"]),
        "num_visible_objects_manifest_only": int(selected["num_visible_objects"]),
        "rgb_source": str(frames_dir),
        "camera_source": str(camera_path),
        "geometry_source": str(depths_dir),
        "uses_gt_instance_masks_for_inference": False,
        "uses_gt_object_transforms_for_inference": False,
        "depth_stride": int(args.depth_stride),
        "num_initial_points": int(xyz.shape[0]),
        "input_mode": "shared_dataset_camera_and_depth_geometry_adapter",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(metadata_path)


if __name__ == "__main__":
    main()
