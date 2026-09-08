#!/usr/bin/env python3
"""Convert per-object ShapeR pickles into per-scene frame assets.

The ShapeR eval pickles are object-centric: every pickle repeats the camera
stream and stores that object's points in its own model frame. This script
uses the repeated camera/images from the first sorted object and z-buffers the
projected object points from all sorted pickles into scene-level depth,
valid-mask, and instance-mask frames.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.shaper.preprocessing.camera import (  # noqa: E402
    CameraTW,
    param_to_matrix,
    rectify_video,
)
from baselines.shaper.preprocessing.projection_utils import (  # noqa: E402
    fisheye624_project,
    pinhole_project,
)

DEFAULT_MAX_RESOLUTION = 512
DEFAULT_MAX_NUM_FRAMES = 60


STREAM_KEYS = {
    "rgb": {
        "images": "rgb_image_data",
        "camera_params": "rgb_camera_params",
        "Ts_camera_model": "Ts_rgbCamera_model",
        "visible_points_model": "rgb_visible_points_model",
    },
    "slam": {
        "images": "image_data",
        "camera_params": "camera_params",
        "Ts_camera_model": "Ts_camera_model",
        "visible_points_model": "visible_points_model",
    },
}


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def decode_image(encoded: bytes, force_rgb: bool = True) -> np.ndarray:
    image = Image.open(io.BytesIO(encoded))
    image = image.convert("RGB" if force_rgb else image.mode)
    return np.asarray(image)


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 4:
        return image[..., :3]
    return image


def make_dirs(save_dir: Path) -> None:
    for name in [
        "frames",
        "frames_fisheye",
        "depths",
        "masks",
        "depths_fisheye",
        "masks_fisheye",
        "instance_masks",
        "instance_masks_fisheye",
        "meshes_canonical",
        "meshes_world",
    ]:
        (save_dir / name).mkdir(parents=True, exist_ok=True)


def save_png(path: Path, image: np.ndarray) -> None:
    Image.fromarray(ensure_rgb(image).astype(np.uint8)).save(path)


def save_npz(path: Path, array: np.ndarray, key: str) -> None:
    np.savez_compressed(path, **{key: array})


def resize_image_and_camera(
    image: np.ndarray, camera_params: np.ndarray, max_resolution: int | None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Resize image to max side length and scale Fisheye624 f/c parameters."""
    image = ensure_rgb(image)
    height, width = image.shape[:2]
    if max_resolution is None or max_resolution <= 0:
        return image, camera_params.astype(np.float32), 1.0

    max_side = max(height, width)
    scale = min(1.0, float(max_resolution) / float(max_side))
    if scale == 1.0:
        return image, camera_params.astype(np.float32), 1.0

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = Image.fromarray(image).resize((new_width, new_height), Image.BILINEAR)
    scaled_params = camera_params.astype(np.float32).copy()
    if scaled_params.shape[-1] == 15:
        scaled_params[0:3] *= scale
    elif scaled_params.shape[-1] >= 4:
        scaled_params[0:4] *= scale
    else:
        raise ValueError(f"Unsupported camera parameter shape: {scaled_params.shape}")
    return np.asarray(resized), scaled_params, scale


def rectify_rgb_image(
    image: np.ndarray, camera_params: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rectify one Fisheye624 frame to pinhole and return image plus 4x4 K."""
    image = ensure_rgb(image)
    height, width = image.shape[:2]

    video = (
        torch.from_numpy(image.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div(255.0)
    )
    cam = CameraTW.from_surreal(
        width=torch.tensor([float(width)]),
        height=torch.tensor([float(height)]),
        params=torch.from_numpy(camera_params.astype(np.float32)),
        type_str="Fisheye624",
    ).unsqueeze(0)

    rectified, rectified_cam = rectify_video(video, cam, pinhole_fxy_factor=1.0)
    rectified_image = (
        rectified[0].permute(1, 2, 0).clamp(0.0, 1.0).mul(255.0).byte().numpy()
    )
    rectified_K = param_to_matrix(rectified_cam.params[0]).cpu().numpy()
    return rectified_image, rectified_K.astype(np.float32)


def write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for vertex in vertices:
            f.write(f"{vertex[0]:.8g} {vertex[1]:.8g} {vertex[2]:.8g}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def transform_points(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    vertices_h = np.concatenate(
        [vertices, np.ones((vertices.shape[0], 1), dtype=vertices.dtype)],
        axis=1,
    )
    transformed = (transform @ vertices_h.T).T
    denom = transformed[:, 3:4]
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    return (transformed[:, :3] / denom).astype(np.float32)


def get_model_to_world_transform(sample: dict[str, Any]) -> np.ndarray:
    if "T_model_world" in sample:
        return np.linalg.inv(as_numpy(sample["T_model_world"])).astype(np.float32)
    if "T_zup_obj" in sample:
        return np.linalg.inv(as_numpy(sample["T_zup_obj"])).astype(np.float32)
    return np.eye(4, dtype=np.float32)


def rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    """Convert a 3x3 rotation matrix to a quaternion in x, y, z, w order."""
    m = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    quat /= max(np.linalg.norm(quat), 1e-12)
    return quat.astype(float).tolist()


def decompose_obb_transform(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    linear = transform[:3, :3].astype(np.float64)
    axis_scales = np.linalg.norm(linear, axis=0)
    safe_scales = np.where(axis_scales < 1e-12, 1.0, axis_scales)
    rotation = linear / safe_scales
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation.astype(np.float32), axis_scales.astype(np.float32)


def compute_object_obb(
    vertices: np.ndarray, model_to_world: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any]]:
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center_model = (vmin + vmax) * 0.5
    scale_model = vmax - vmin

    rotation_world, axis_scales = decompose_obb_transform(model_to_world)
    center_world = transform_points(center_model[None], model_to_world)[0]
    scale_world = scale_model * axis_scales
    quat_world = rotation_matrix_to_quaternion_xyzw(rotation_world)

    canonical_obb = {
        "translation": center_model.astype(float).tolist(),
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "scale": scale_model.astype(float).tolist(),
    }
    world_obb = {
        "translation": center_world.astype(float).tolist(),
        "rotation_xyzw": quat_world,
        "scale": scale_world.astype(float).tolist(),
    }
    return canonical_obb, world_obb


def export_object_obbs(
    samples: list[dict[str, Any]], pkl_paths: list[Path], save_dir: Path
) -> list[dict[str, Any]]:
    obb_entries = []
    for obj_idx, (sample, path) in enumerate(zip(samples, pkl_paths), start=1):
        if "mesh_vertices" in sample:
            vertices = as_numpy(sample["mesh_vertices"]).astype(np.float32)
        elif "points_model" in sample:
            vertices = as_numpy(sample["points_model"]).astype(np.float32)
        else:
            continue

        model_to_world = get_model_to_world_transform(sample)
        canonical_obb, world_obb = compute_object_obb(vertices, model_to_world)
        obb_entries.append(
            {
                "index": obj_idx,
                "name": path.stem,
                "path": str(path),
                "category": sample.get("category"),
                "caption": sample.get("caption"),
                "canonical": canonical_obb,
                "world": world_obb,
            }
        )

    obbs_json = {
        "input_dir": str(pkl_paths[0].parent) if pkl_paths else "",
        "rotation_convention": "xyzw",
        "scale_convention": "full side lengths along OBB local axes",
        "objects": obb_entries,
    }
    with (save_dir / "obbs.json").open("w") as f:
        json.dump(obbs_json, f, indent=2)
    return obb_entries


def export_object_meshes(
    samples: list[dict[str, Any]], pkl_paths: list[Path], save_dir: Path
) -> list[dict[str, str]]:
    mesh_entries = []
    for obj_idx, (sample, path) in enumerate(zip(samples, pkl_paths), start=1):
        if "mesh_vertices" not in sample or "mesh_faces" not in sample:
            continue

        vertices = as_numpy(sample["mesh_vertices"]).astype(np.float32)
        faces = as_numpy(sample["mesh_faces"]).astype(np.int32)
        mesh_name = f"{obj_idx:03d}_{path.stem}.ply"
        canonical_path = save_dir / "meshes_canonical" / mesh_name
        world_path = save_dir / "meshes_world" / mesh_name

        write_ply(canonical_path, vertices, faces)
        world_transform = get_model_to_world_transform(sample)
        world_vertices = transform_points(vertices, world_transform)
        write_ply(world_path, world_vertices, faces)

        mesh_entries.append(
            {
                "index": obj_idx,
                "canonical_mesh": str(canonical_path),
                "world_mesh": str(world_path),
            }
        )
    return mesh_entries


def project_fisheye(
    points_model: np.ndarray,
    T_camera_model: np.ndarray,
    camera_params: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_h = np.concatenate(
        [points_model, np.ones((points_model.shape[0], 1), dtype=points_model.dtype)],
        axis=1,
    )
    points_camera_h = (T_camera_model @ points_h.T).T
    denom = points_camera_h[:, 3:4]
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    points_camera = points_camera_h[:, :3] / denom

    xyz = torch.from_numpy(points_camera.astype(np.float32)).unsqueeze(0)
    params = torch.from_numpy(camera_params.astype(np.float32)).unsqueeze(0)
    uv = fisheye624_project(xyz, params)[0].cpu().numpy()
    depth = points_camera[:, 2]
    valid = (
        np.isfinite(uv).all(axis=1)
        & (depth > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    return uv[valid], depth[valid], valid


def project_perspective(
    points_model: np.ndarray,
    T_camera_model: np.ndarray,
    intrinsics: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_h = np.concatenate(
        [points_model, np.ones((points_model.shape[0], 1), dtype=points_model.dtype)],
        axis=1,
    )
    points_camera_h = (T_camera_model @ points_h.T).T
    denom = points_camera_h[:, 3:4]
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    points_camera = points_camera_h[:, :3] / denom

    params = np.array(
        [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]],
        dtype=np.float32,
    )
    xyz = torch.from_numpy(points_camera.astype(np.float32)).unsqueeze(0)
    uv = pinhole_project(xyz, torch.from_numpy(params).unsqueeze(0))[0].cpu().numpy()
    depth = points_camera[:, 2]
    valid = (
        np.isfinite(uv).all(axis=1)
        & (depth > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    return uv[valid], depth[valid], valid


def splat_depth_and_instance(
    uv: np.ndarray,
    depth: np.ndarray,
    object_index: int,
    depth_map: np.ndarray,
    instance_map: np.ndarray,
) -> None:
    if uv.size == 0:
        return

    xs = np.rint(uv[:, 0]).astype(np.int64)
    ys = np.rint(uv[:, 1]).astype(np.int64)
    keep = (
        (xs >= 0)
        & (xs < depth_map.shape[1])
        & (ys >= 0)
        & (ys < depth_map.shape[0])
        & np.isfinite(depth)
        & (depth > 0)
    )
    xs = xs[keep]
    ys = ys[keep]
    depth = depth[keep].astype(np.float32)
    if depth.size == 0:
        return

    flat_idx = ys * depth_map.shape[1] + xs
    order = np.argsort(depth)
    flat_idx = flat_idx[order]
    depth = depth[order]
    _, first = np.unique(flat_idx, return_index=True)
    flat_idx = flat_idx[first]
    depth = depth[first]

    flat_depth = depth_map.reshape(-1)
    flat_inst = instance_map.reshape(-1)
    closer = depth < flat_depth[flat_idx]
    flat_depth[flat_idx[closer]] = depth[closer]
    flat_inst[flat_idx[closer]] = object_index


def get_c2w_opencv(sample: dict[str, Any], T_camera_model: np.ndarray) -> np.ndarray:
    """Return camera-to-world assuming T_model_world maps world to model."""
    if "T_model_world" in sample:
        T_model_world = as_numpy(sample["T_model_world"])
        world_to_camera = T_camera_model @ T_model_world
        return np.linalg.inv(world_to_camera).astype(np.float32)
    return np.linalg.inv(T_camera_model).astype(np.float32)


def get_stream_frame_count(sample: dict[str, Any], keys: dict[str, str]) -> int:
    return min(
        len(sample[keys["images"]]),
        len(sample[keys["camera_params"]]),
        len(sample[keys["Ts_camera_model"]]),
        len(sample[keys["visible_points_model"]]),
    )


def build_observation_groups(
    samples: list[dict[str, Any]],
    pkl_paths: list[Path],
    keys: dict[str, str],
    pose_decimals: int = 4,
) -> list[dict[str, Any]]:
    """Group object observations that were captured from the same camera pose."""
    groups_by_pose = {}
    groups = []
    for obj_idx, (sample, path) in enumerate(zip(samples, pkl_paths), start=1):
        frame_count = get_stream_frame_count(sample, keys)
        for frame_idx in range(frame_count):
            T_camera_model = as_numpy(sample[keys["Ts_camera_model"]][frame_idx]).astype(
                np.float32
            )
            c2w = get_c2w_opencv(sample, T_camera_model)
            pose_key = tuple(np.round(c2w.reshape(-1), pose_decimals))
            if pose_key not in groups_by_pose:
                groups_by_pose[pose_key] = {
                    "c2w": c2w,
                    "observations": [],
                    "first_object_index": obj_idx,
                    "first_object_name": path.stem,
                    "first_frame_index": frame_idx,
                }
                groups.append(groups_by_pose[pose_key])
            groups_by_pose[pose_key]["observations"].append(
                {
                    "object_index": obj_idx,
                    "object_name": path.stem,
                    "sample": sample,
                    "frame_index": frame_idx,
                }
            )
    return groups


def process_scene(
    input_dir: Path,
    save_dir: Path,
    stream: str,
    max_num_frames: int | None,
    frame_stride: int,
    max_resolution: int | None,
) -> None:
    pkl_paths = sorted(input_dir.glob("*.pkl"), key=lambda p: p.name)
    if not pkl_paths:
        raise FileNotFoundError(f"No .pkl files found in {input_dir}")

    keys = STREAM_KEYS[stream]
    samples = []
    for path in pkl_paths:
        with path.open("rb") as f:
            sample = pickle.load(f)
        missing = [key for key in keys.values() if key not in sample]
        if missing:
            raise KeyError(f"{path} is missing keys for stream '{stream}': {missing}")
        samples.append(sample)

    observation_groups = build_observation_groups(samples, pkl_paths, keys)
    frame_group_indices = list(range(0, len(observation_groups), frame_stride))
    if max_num_frames is not None and len(frame_group_indices) > max_num_frames:
        sample_positions = np.linspace(0, len(frame_group_indices) - 1, max_num_frames)
        frame_group_indices = [
            frame_group_indices[int(round(pos))] for pos in sample_positions
        ]

    make_dirs(save_dir)
    object_entries = [
        {"index": idx, "name": path.stem, "path": str(path)}
        for idx, path in enumerate(pkl_paths, start=1)
    ]
    mesh_entries = export_object_meshes(samples, pkl_paths, save_dir)
    obb_entries = export_object_obbs(samples, pkl_paths, save_dir)
    camera_frames = []

    for out_idx, group_idx in enumerate(frame_group_indices):
        group = observation_groups[group_idx]
        reference_observation = group["observations"][0]
        reference = reference_observation["sample"]
        reference_frame_idx = reference_observation["frame_index"]
        fisheye_image = decode_image(reference[keys["images"]][reference_frame_idx])
        camera_params = as_numpy(
            reference[keys["camera_params"]][reference_frame_idx]
        ).astype(
            np.float32
        )
        fisheye_image, camera_params, resize_scale = resize_image_and_camera(
            fisheye_image, camera_params, max_resolution
        )
        height_f, width_f = fisheye_image.shape[:2]

        perspective_image, perspective_K = rectify_rgb_image(
            fisheye_image, camera_params
        )
        height_p, width_p = perspective_image.shape[:2]

        depth_f = np.full((height_f, width_f), np.inf, dtype=np.float32)
        inst_f = np.zeros((height_f, width_f), dtype=np.int32)
        depth_p = np.full((height_p, width_p), np.inf, dtype=np.float32)
        inst_p = np.zeros((height_p, width_p), dtype=np.int32)

        for observation in group["observations"]:
            obj_idx = observation["object_index"]
            sample = observation["sample"]
            frame_idx = observation["frame_index"]
            visible_points = as_numpy(sample[keys["visible_points_model"]][frame_idx])
            if visible_points.size == 0:
                continue
            T_camera_model = as_numpy(sample[keys["Ts_camera_model"]][frame_idx]).astype(
                np.float32
            )
            obj_camera_params = as_numpy(sample[keys["camera_params"]][frame_idx]).astype(
                np.float32
            )
            obj_camera_params = obj_camera_params.copy()
            if obj_camera_params.shape[-1] == 15:
                obj_camera_params[0:3] *= resize_scale
            elif obj_camera_params.shape[-1] >= 4:
                obj_camera_params[0:4] *= resize_scale

            uv_f, z_f, _ = project_fisheye(
                visible_points.astype(np.float32),
                T_camera_model,
                obj_camera_params,
                height_f,
                width_f,
            )
            splat_depth_and_instance(uv_f, z_f, obj_idx, depth_f, inst_f)

            uv_p, z_p, _ = project_perspective(
                visible_points.astype(np.float32),
                T_camera_model,
                perspective_K,
                height_p,
                width_p,
            )
            splat_depth_and_instance(uv_p, z_p, obj_idx, depth_p, inst_p)

        mask_f = np.isfinite(depth_f)
        mask_p = np.isfinite(depth_p)
        depth_f[~mask_f] = 0.0
        depth_p[~mask_p] = 0.0

        stem = f"{out_idx:06d}"
        save_png(save_dir / "frames_fisheye" / f"{stem}.png", fisheye_image)
        save_png(save_dir / "frames" / f"{stem}.png", perspective_image)
        save_npz(save_dir / "depths_fisheye" / f"{stem}.npz", depth_f, "depth")
        save_npz(save_dir / "masks_fisheye" / f"{stem}.npz", mask_f, "mask")
        save_npz(
            save_dir / "instance_masks_fisheye" / f"{stem}.npz",
            inst_f,
            "instance_mask",
        )
        save_npz(save_dir / "depths" / f"{stem}.npz", depth_p, "depth")
        save_npz(save_dir / "masks" / f"{stem}.npz", mask_p, "mask")
        save_npz(save_dir / "instance_masks" / f"{stem}.npz", inst_p, "instance_mask")

        camera_frames.append(
            {
                "frame_index": int(reference_frame_idx),
                "pose_group_index": int(group_idx),
                "file_index": int(out_idx),
                "source_object_index": int(reference_observation["object_index"]),
                "source_object_name": reference_observation["object_name"],
                "source_frame_index": int(reference_frame_idx),
                "num_observations": len(group["observations"]),
                "c2w_opencv": group["c2w"].tolist(),
                "resize_scale": float(resize_scale),
                "perspective": {
                    "intrinsics": perspective_K.tolist(),
                    "width": int(width_p),
                    "height": int(height_p),
                },
                "fisheye": {
                    "intrinsics": camera_params.tolist(),
                    "model": "Fisheye624",
                    "width": int(width_f),
                    "height": int(height_f),
                },
            }
        )

        print(
            f"[{out_idx + 1}/{len(frame_group_indices)}] saved pose group {group_idx} "
            f"({mask_p.sum()} perspective points, {mask_f.sum()} fisheye points)"
        )

    camera_json = {
        "input_dir": str(input_dir),
        "stream": stream,
        "frame_stride": frame_stride,
        "max_num_frames": max_num_frames,
        "max_resolution": max_resolution,
        "num_pose_groups": len(observation_groups),
        "objects": object_entries,
        "meshes": mesh_entries,
        "obbs": obb_entries,
        "frames": camera_frames,
    }
    with (save_dir / "camera.json").open("w") as f:
        json.dump(camera_json, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build scene-level frames/depths/instance masks from ShapeR per-object pickles."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing object .pkl files.")
    parser.add_argument("save_dir", type=Path, help="Directory to write scene outputs.")
    parser.add_argument(
        "--stream",
        choices=sorted(STREAM_KEYS),
        default="rgb",
        help="Camera stream to export. Default: rgb.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Deprecated alias for --max-num-frames.",
    )
    parser.add_argument(
        "--max-num-frames",
        type=int,
        default=DEFAULT_MAX_NUM_FRAMES,
        help=(
            "Maximum number of frames to export. If the stream has more frames, "
            "sample with np.linspace(0, num_frames - 1, max_num_frames). Default: 60."
        ),
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Export every Kth frame. Default: 1.",
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=DEFAULT_MAX_RESOLUTION,
        help="Maximum exported image side length. Use 0 to disable resizing. Default: 512.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    max_num_frames = (
        args.max_frames if args.max_frames is not None else args.max_num_frames
    )
    if max_num_frames is not None and max_num_frames < 1:
        raise ValueError("--max-num-frames must be >= 1")
    max_resolution = args.max_resolution if args.max_resolution > 0 else None
    process_scene(
        input_dir=args.input_dir,
        save_dir=args.save_dir,
        stream=args.stream,
        max_num_frames=max_num_frames,
        frame_stride=args.frame_stride,
        max_resolution=max_resolution,
    )


if __name__ == "__main__":
    main()
