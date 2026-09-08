"""Prepare one FF iTHOR or Imaginarium scene for EFM3D EVL inference."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET_SUBDIRS = {"ithor": "ithor", "imaginarium": "Imaginarium"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one FF validation scene to the custom EFM3D scene contract."
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_SUBDIRS), required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument(
        "--fire3d-test-root",
        type=Path,
        default=REPO_ROOT / "data",
    )
    parser.add_argument(
        "--rgb-frames-dir",
        type=Path,
        help="Optional exact-rerender RGB directory; camera, depth, and masks remain from validation data.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-id", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sorted_files(path: Path, suffixes: tuple[str, ...]) -> list[Path]:
    suffixes = tuple(suffix.lower() for suffix in suffixes)
    return sorted(
        child for child in path.iterdir() if child.is_file() and child.suffix.lower() in suffixes
    )


def load_parallel(paths: list[Path], loader: Callable[[str], np.ndarray], workers: int) -> np.ndarray:
    if workers <= 1:
        arrays = [loader(str(path)) for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            arrays = list(executor.map(lambda path: loader(str(path)), paths))
    return np.stack(arrays)


def point_colors(instance_ids: np.ndarray) -> np.ndarray:
    instance_ids = np.asarray(instance_ids, dtype=np.int64)
    unique_ids = np.unique(instance_ids)
    rng = np.random.default_rng(0)
    palette = {int(value): rng.integers(45, 240, size=3, dtype=np.uint8) for value in unique_ids}
    palette[0] = np.asarray([45, 45, 45], dtype=np.uint8)
    return np.stack([palette[int(value)] for value in instance_ids])


def export_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    colors = np.asarray(colors)
    if np.issubdtype(colors.dtype, np.floating):
        colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    cloud = trimesh.points.PointCloud(np.asarray(points, dtype=np.float32), colors=colors)
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud.export(path)


def write_camera_json(
    path: Path,
    c2ws: np.ndarray,
    fisheye_params: np.ndarray,
    width: int,
    height: int,
) -> None:
    frames = []
    for frame_index, (c2w, params) in enumerate(zip(c2ws, fisheye_params)):
        frames.append(
            {
                "file_path": f"frames/{frame_index:06d}.jpg",
                "depth_path": f"depth/{frame_index:06d}.npz",
                "mask_path": f"masks/{frame_index:06d}.npz",
                "camera_model": "FISHEYE624",
                "fisheye624_params": params.tolist(),
                "c2w": c2w.tolist(),
            }
        )
    payload = {
        "camera_model": "FISHEYE624",
        "width": int(width),
        "height": int(height),
        "frames": frames,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def prepare_scene(args: argparse.Namespace) -> dict:
    if args.video_id < 0 or args.workers <= 0:
        raise ValueError("--video-id must be nonnegative and --workers must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")

    from utils import data_imaginarium_fisheye as converter

    dataset_subdir = DATASET_SUBDIRS[args.dataset]
    dataset_root = args.fire3d_test_root.resolve() / dataset_subdir
    render_root = dataset_root / "renders" / args.scene_id
    camera_path = render_root / f"{args.video_id}.json"
    depth_dir = render_root / f"{args.video_id}_depth"
    masks_dir = render_root / f"{args.video_id}_masks"
    default_rgb_dir = render_root / f"{args.video_id}_frames"
    rgb_dir = args.rgb_frames_dir.resolve() if args.rgb_frames_dir else default_rgb_dir

    for required in (camera_path, depth_dir, masks_dir, rgb_dir):
        if not required.exists():
            raise FileNotFoundError(required)

    rgb_paths = sorted_files(rgb_dir, (".jpg", ".jpeg", ".png"))
    depth_paths = sorted_files(depth_dir, (".npz",))
    mask_paths = sorted_files(masks_dir, (".npz",))
    frame_count = min(len(rgb_paths), len(depth_paths), len(mask_paths))
    if frame_count == 0 or len({len(rgb_paths), len(depth_paths), len(mask_paths)}) != 1:
        raise ValueError(
            f"RGB/depth/mask count mismatch: {len(rgb_paths)}/{len(depth_paths)}/{len(mask_paths)}"
        )
    if args.max_frames is not None:
        frame_count = min(frame_count, args.max_frames)
    rgb_paths = rgb_paths[:frame_count]
    depth_paths = depth_paths[:frame_count]
    mask_paths = mask_paths[:frame_count]

    intrinsics, c2ws, (height, width) = converter.read_cameras(str(camera_path))
    intrinsics = intrinsics[:frame_count]
    c2ws = c2ws[:frame_count]
    if len(intrinsics) != frame_count:
        raise ValueError(f"Camera count {len(intrinsics)} does not cover {frame_count} frames")

    rgbs = load_parallel(rgb_paths, converter._read_single_rgb, args.workers)
    depths = load_parallel(depth_paths, converter._read_single_depth, args.workers)
    masks = load_parallel(mask_paths, converter._read_single_mask_v2, args.workers)
    if rgbs.shape != (frame_count, height, width, 3):
        raise ValueError(f"Unexpected RGB shape: {rgbs.shape}")
    if depths.shape != (frame_count, height, width) or masks.shape != depths.shape:
        raise ValueError(f"Unexpected depth/mask shapes: {depths.shape}/{masks.shape}")

    points, point_rgbs = converter.project_depth_to_points(
        rgbs, depths.copy(), intrinsics, c2ws, downsample=1
    )
    semidense_mask = converter._semi_dense_point_mask(rgbs, depths).reshape(-1)
    if not semidense_mask.any():
        semidense_mask = (
            np.isfinite(depths) & (depths > 0.0) & (depths < 50.0)
        ).reshape(-1)
    semidense_points = points[semidense_mask]
    semidense_rgbs = point_rgbs[semidense_mask]
    semidense_instances = masks.reshape(-1)[semidense_mask]
    if len(semidense_points) == 0 or not np.isfinite(semidense_points).all():
        raise ValueError("Prepared semidense points are empty or non-finite")

    fisheye_rgbs, fisheye_depths, fisheye_masks, fisheye_params = (
        converter._perspective_to_fisheye(rgbs, depths, masks, intrinsics, height, width)
    )

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    frames_out = output_dir / "frames"
    depth_out = output_dir / "depth"
    masks_out = output_dir / "masks"
    frames_out.mkdir(parents=True)
    depth_out.mkdir()
    masks_out.mkdir()
    converter._save_rgb_frames(fisheye_rgbs, str(frames_out))
    converter._save_depth_frames(fisheye_depths, str(depth_out))
    converter._save_mask_frames(fisheye_masks, str(masks_out))
    export_point_cloud(output_dir / "semi_points.ply", semidense_points, semidense_rgbs)
    export_point_cloud(
        output_dir / "semi_points_instances.ply",
        semidense_points,
        point_colors(semidense_instances),
    )
    np.savez_compressed(
        output_dir / "semi_points_indices.npz",
        indices=semidense_instances.astype(np.int32, copy=False),
    )
    write_camera_json(
        output_dir / "camera.json", c2ws, fisheye_params, width=width, height=height
    )

    finite_depth = np.isfinite(depths) & (depths > 0.0) & (depths < 50.0)
    summary = {
        "schema": "ff_efm3d_fire3d_scene_v1",
        "dataset": args.dataset,
        "scene_id": args.scene_id,
        "video_id": args.video_id,
        "source": {
            "camera": str(camera_path.resolve()),
            "rgb_frames": str(rgb_dir.resolve()),
            "depth": str(depth_dir.resolve()),
            "masks": str(masks_dir.resolve()),
            "rgb_override": args.rgb_frames_dir is not None,
        },
        "output": str(output_dir),
        "camera_model": "FISHEYE624",
        "frame_count": frame_count,
        "image_size": [width, height],
        "valid_depth_fraction": float(finite_depth.mean()),
        "semidense_point_count": int(len(semidense_points)),
        "semidense_bounds": [
            semidense_points.min(axis=0).tolist(),
            semidense_points.max(axis=0).tolist(),
        ],
        "camera_center_bounds": [
            c2ws[:, :3, 3].min(axis=0).tolist(),
            c2ws[:, :3, 3].max(axis=0).tolist(),
        ],
    }
    (output_dir / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    summary = prepare_scene(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
