import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm


OUTPUT_ROOT = Path("results/data_processing/object_point_clouds")


def _load_npz_array(path, preferred_key):
    data = np.load(path)
    if preferred_key in data:
        return data[preferred_key]
    return data[data.files[0]]


def _camera_axes(frame):
    extrinsics = frame["extrinsics"]
    eye = np.asarray(extrinsics["eye"], dtype=np.float64)
    lookat = np.asarray(extrinsics["lookat"], dtype=np.float64)
    up = np.asarray(extrinsics["up"], dtype=np.float64)

    forward = lookat - eye
    forward /= np.linalg.norm(forward) + 1e-12
    up /= np.linalg.norm(up) + 1e-12
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, forward)
    up /= np.linalg.norm(up) + 1e-12
    return eye, right, up, forward


def depth_to_world_points(depth, mask, image, frame):
    intrinsics = frame["intrinsics"]
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    valid = mask.astype(bool) & np.isfinite(depth) & (depth > 0)
    v, u = np.nonzero(valid)
    if len(u) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    z = depth[v, u].astype(np.float64)
    x = (u.astype(np.float64) - cx) * z / fx
    y = -(v.astype(np.float64) - cy) * z / fy

    eye, right, up, forward = _camera_axes(frame)
    points = (
        eye[None, :]
        + x[:, None] * right[None, :]
        + y[:, None] * up[None, :]
        + z[:, None] * forward[None, :]
    )
    colors = image[v, u].astype(np.float64) / 255.0
    return points, colors


def write_point_cloud(points, colors, save_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(save_path), pcd, write_ascii=False)


def infer_dataset_and_object(object_save_dir):
    object_save_dir = Path(object_save_dir).resolve()
    dataset_name = object_save_dir.parent.name
    object_name = object_save_dir.name
    return dataset_name, object_name


def generate_object_pcds(object_save_dir, output_root=OUTPUT_ROOT, overwrite=False):
    object_save_dir = Path(object_save_dir)
    cameras_path = object_save_dir / "cameras.json"
    frames_dir = object_save_dir / "frames"
    depth_dir = object_save_dir / "depth"
    mask_dir = object_save_dir / "mask"

    with open(cameras_path, "r") as f:
        cameras = json.load(f)

    dataset_name, object_name = infer_dataset_and_object(object_save_dir)
    object_output_dir = Path(output_root) / dataset_name / object_name
    object_output_dir.mkdir(parents=True, exist_ok=True)

    for frame in tqdm(cameras["frames"], desc=f"{dataset_name}/{object_name}", unit="frame"):
        frame_file = Path(frame["file"])
        frame_stem = frame_file.stem
        save_path = object_output_dir / f"{frame_stem}.ply"
        if save_path.exists() and not overwrite:
            continue

        image = np.asarray(Image.open(frames_dir / frame_file).convert("RGB"))
        depth = _load_npz_array(depth_dir / f"{frame_stem}.npz", "depth").astype(np.float32)
        mask = _load_npz_array(mask_dir / f"{frame_stem}.npz", "mask")

        points, colors = depth_to_world_points(depth, mask, image, frame)
        write_point_cloud(points, colors, save_path)

    return object_output_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "object_save_dir",
        type=str,
        help="Object render directory containing frames, depths, masks, and transforms.",
    )
    parser.add_argument("--output_root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = generate_object_pcds(
        object_save_dir=args.object_save_dir,
        output_root=Path(args.output_root),
        overwrite=args.overwrite,
    )
    print(f"Saved point clouds to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
