#!/usr/bin/env python3
"""Extract fair per-object ShapeR inputs into a reusable, deduplicated cache.

Complete GT meshes are intentionally never copied.  Each object receives ordered
multiview image references plus sparse visible model-frame points, their image
projections, and camera-space depth.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import sys
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.shaper_object_cache import (
    FORBIDDEN_INFERENCE_KEYS,
    MANIFEST_NAME,
    SCHEMA,
    SUMMARY_NAME,
    atomic_savez_compressed,
    atomic_write_bytes,
    atomic_write_json,
    image_identity,
    validate_metadata,
)


STREAMS = {
    "rgb": {
        "images": "rgb_image_data",
        "camera_params": "rgb_camera_params",
        "Ts_camera_model": "Ts_rgbCamera_model",
        "visible_points_model": "rgb_visible_points_model",
        "projections": "rgb_object_point_projections",
    },
    "slam": {
        "images": "image_data",
        "camera_params": "camera_params",
        "Ts_camera_model": "Ts_camera_model",
        "visible_points_model": "visible_points_model",
        "projections": "object_point_projections",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory containing ShapeR object .pkl files")
    parser.add_argument("output_dir", type=Path, help="Destination cache root")
    parser.add_argument("--stream", choices=tuple(STREAMS), default="rgb")
    parser.add_argument("--scene", action="append", default=[], help="Only extract these scene prefixes")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=512,
        help="Resize each RGB view so its longest side is at most this value; <=0 disables",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--prune-unreferenced-images",
        action="store_true",
        help="After a failure-free extraction, remove content-addressed image blobs not referenced by the new manifests",
    )
    parser.add_argument("--allow-failures", action="store_true")
    return parser.parse_args()


def as_numpy(value: Any, *, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def scalar_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def scene_id_from_path(path: Path) -> str:
    stem = path.stem
    return stem.split("__", 1)[0] if "__" in stem else path.parent.name


def resize_encoded_rgb(encoded: bytes, max_resolution: int | None) -> tuple[bytes, int, int, float]:
    with Image.open(io.BytesIO(encoded)) as source:
        image = source.convert("RGB")
        width, height = image.size
        if max_resolution is None or int(max_resolution) <= 0:
            return encoded, int(height), int(width), 1.0
        scale = min(1.0, float(max_resolution) / float(max(height, width)))
        if scale == 1.0:
            return encoded, int(height), int(width), 1.0
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resampling = getattr(Image, "Resampling", Image)
        image = image.resize((new_width, new_height), resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95, subsampling=0)
        return buffer.getvalue(), int(new_height), int(new_width), float(scale)


def cached_entry(
    output_root: Path, metadata_path: Path, max_resolution: int | None
) -> dict[str, Any] | None:
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_metadata(metadata)
    expected_resolution = (
        None if max_resolution is None or int(max_resolution) <= 0 else int(max_resolution)
    )
    if metadata.get("max_resolution") != expected_resolution:
        return None
    observations = output_root / metadata["observations_relpath"]
    if not observations.is_file():
        return None
    missing_images = [
        relpath for relpath in metadata["image_relpaths"] if not (output_root / relpath).is_file()
    ]
    if missing_images:
        return None
    return manifest_entry(metadata)


def manifest_entry(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "sample_id": metadata["sample_id"],
        "scene_id": metadata["scene_id"],
        "metadata_relpath": metadata["metadata_relpath"],
        "observations_relpath": metadata["observations_relpath"],
        "num_views": metadata["num_views"],
        "num_visible_observations": metadata["num_visible_observations"],
    }


def _extract_one(
    source_path_text: str,
    output_root_text: str,
    stream: str,
    overwrite: bool,
    max_resolution: int | None = 512,
) -> dict[str, Any]:
    source_path = Path(source_path_text)
    output_root = Path(output_root_text)
    sample_id = source_path.stem
    scene_id = scene_id_from_path(source_path)
    object_dir = output_root / "objects" / scene_id / sample_id
    metadata_path = object_dir / "metadata.json"
    if not overwrite:
        cached = cached_entry(output_root, metadata_path, max_resolution)
        if cached is not None:
            return {"status": "cached", "entry": cached}

    with source_path.open("rb") as handle:
        sample = pickle.load(handle)
    keys = STREAMS[stream]
    missing = [key for key in keys.values() if key not in sample]
    if missing:
        raise KeyError(f"{source_path} is missing {stream} fields: {missing}")
    images = sample[keys["images"]]
    cameras = sample[keys["camera_params"]]
    transforms = sample[keys["Ts_camera_model"]]
    visible_points = sample[keys["visible_points_model"]]
    projections = sample[keys["projections"]]
    num_raw_views = min(
        len(images), len(cameras), len(transforms), len(visible_points), len(projections)
    )
    if num_raw_views <= 0:
        raise ValueError(f"{source_path} has no complete {stream} views")

    image_info = []
    for view_index in range(num_raw_views):
        encoded, height, width, resize_scale = resize_encoded_rgb(
            bytes(images[view_index]), max_resolution
        )
        digest, extension, height, width = image_identity(encoded)
        image_info.append((encoded, digest, extension, height, width, resize_scale))
    modal_hw = Counter((item[3], item[4]) for item in image_info).most_common(1)[0][0]
    selected_raw_views = [
        index for index, item in enumerate(image_info) if (item[3], item[4]) == modal_hw
    ]
    if not selected_raw_views:
        raise RuntimeError(f"Could not find a common image resolution for {source_path}")

    image_relpaths = []
    image_hw = []
    image_resize_scales = []
    Ts_camera_model = []
    camera_params = []
    points_chunks = []
    uv_chunks = []
    depth_chunks = []
    view_offsets = [0]
    observation_counts = []

    for raw_view_index in selected_raw_views:
        encoded, digest, extension, height, width, resize_scale = image_info[raw_view_index]
        image_relpath = Path("images") / digest[:2] / f"{digest}{extension}"
        atomic_write_bytes(output_root / image_relpath, encoded)
        image_relpaths.append(str(image_relpath))
        image_hw.append([height, width])
        image_resize_scales.append(float(resize_scale))

        transform = as_numpy(transforms[raw_view_index], dtype=np.float32).reshape(4, 4)
        camera = as_numpy(cameras[raw_view_index], dtype=np.float32).reshape(-1).copy()
        if camera.shape[0] == 15:
            camera[0:3] *= resize_scale
        elif camera.shape[0] >= 4:
            camera[0:4] *= resize_scale
        points = as_numpy(visible_points[raw_view_index], dtype=np.float32).reshape(-1, 3)
        uv = (
            as_numpy(projections[raw_view_index], dtype=np.float32).reshape(-1, 2)
            * float(resize_scale)
        )
        count = min(points.shape[0], uv.shape[0])
        points, uv = points[:count], uv[:count]
        if count:
            points_h = np.concatenate(
                [points, np.ones((count, 1), dtype=np.float32)], axis=1
            )
            points_camera = points_h @ transform.T
            denominator = points_camera[:, 3:4]
            safe_denominator = np.where(np.abs(denominator) < 1e-9, 1.0, denominator)
            depth = (points_camera[:, 2:3] / safe_denominator).reshape(-1)
            valid = (
                np.isfinite(points).all(axis=1)
                & np.isfinite(uv).all(axis=1)
                & np.isfinite(depth)
                & (depth > 0)
            )
            points, uv, depth = points[valid], uv[valid], depth[valid]
        else:
            depth = np.zeros((0,), dtype=np.float32)
        points_chunks.append(points.astype(np.float32, copy=False))
        uv_chunks.append(uv.astype(np.float32, copy=False))
        depth_chunks.append(depth.astype(np.float32, copy=False))
        observation_counts.append(int(points.shape[0]))
        view_offsets.append(view_offsets[-1] + int(points.shape[0]))
        Ts_camera_model.append(transform)
        camera_params.append(camera)

    points_model = as_numpy(sample["points_model"], dtype=np.float32).reshape(-1, 3)
    bounds = as_numpy(sample["bounds"], dtype=np.float32).reshape(3)
    pose_source_key = "T_model_world" if "T_model_world" in sample else "T_zup_obj"
    if pose_source_key not in sample:
        raise KeyError(f"{source_path} has neither T_model_world nor T_zup_obj")
    T_model_world = as_numpy(sample[pose_source_key], dtype=np.float32).reshape(4, 4)
    T_world_model = np.linalg.inv(T_model_world).astype(np.float32)

    observations_relpath = Path("objects") / scene_id / sample_id / "observations.npz"
    arrays = {
        "points_model": points_model,
        "view_points_model": np.concatenate(points_chunks, axis=0),
        "view_uv": np.concatenate(uv_chunks, axis=0),
        "view_depth": np.concatenate(depth_chunks, axis=0),
        "view_offsets": np.asarray(view_offsets, dtype=np.int64),
        "Ts_camera_model": np.stack(Ts_camera_model).astype(np.float32),
        "camera_params": np.stack(camera_params).astype(np.float32),
        "source_view_indices": np.asarray(selected_raw_views, dtype=np.int32),
    }
    if "inv_dist_std" in sample:
        arrays["inv_dist_std"] = as_numpy(sample["inv_dist_std"], dtype=np.float32).reshape(-1)
    if "dist_std" in sample:
        arrays["dist_std"] = as_numpy(sample["dist_std"], dtype=np.float32).reshape(-1)
    atomic_savez_compressed(output_root / observations_relpath, **arrays)

    metadata_relpath = Path("objects") / scene_id / sample_id / "metadata.json"
    metadata = {
        "schema": SCHEMA,
        "sample_id": sample_id,
        "scene_id": scene_id,
        "stream": stream,
        "source_pkl": str(source_path.resolve()),
        "source_pkl_size": int(source_path.stat().st_size),
        "metadata_relpath": str(metadata_relpath),
        "observations_relpath": str(observations_relpath),
        "image_relpaths": image_relpaths,
        "image_hw": image_hw,
        "image_resize_scales": image_resize_scales,
        "max_resolution": None if max_resolution is None else int(max_resolution),
        "source_view_indices": selected_raw_views,
        "num_raw_views": int(num_raw_views),
        "num_views": len(selected_raw_views),
        "num_visible_observations": int(view_offsets[-1]),
        "visible_observations_per_view": observation_counts,
        "bounds": bounds.tolist(),
        "T_model_world": T_model_world.tolist(),
        "T_world_model": T_world_model.tolist(),
        "pose_source_key": pose_source_key,
        "category": scalar_json(sample.get("category")),
        "caption": scalar_json(sample.get("caption")),
        "is_ariagen2": scalar_json(sample.get("is_ariagen2")),
        "source_seq_name": scalar_json(sample.get("source_seq_name")),
        "source_object_name": scalar_json(sample.get("source_object_name")),
        "inference_content": [
            "multiview_rgb",
            "points_model",
            "per_view_visible_points_model",
            "per_view_image_projections",
            "per_view_camera_depth",
            "camera_parameters",
            "camera_to_model_transforms",
            "annotated_pose_and_bounds",
        ],
        "excluded_gt_geometry_keys": sorted(FORBIDDEN_INFERENCE_KEYS),
    }
    validate_metadata(metadata)
    atomic_write_json(metadata_path, metadata)
    return {"status": "written", "entry": manifest_entry(metadata)}


def discover_paths(args: argparse.Namespace) -> list[Path]:
    paths = sorted(args.input_dir.rglob("*.pkl"))
    if args.scene:
        requested = set(args.scene)
        paths = [path for path in paths if scene_id_from_path(path) in requested]
    if args.limit is not None:
        paths = paths[: max(int(args.limit), 0)]
    if not paths:
        raise FileNotFoundError(f"No matching ShapeR object pickles under {args.input_dir}")
    return paths


def prune_unreferenced_images(output_root: Path, entries: list[dict[str, Any]]) -> dict[str, int]:
    referenced = set()
    for entry in entries:
        metadata = json.loads(
            (output_root / entry["metadata_relpath"]).read_text(encoding="utf-8")
        )
        referenced.update(metadata["image_relpaths"])
    removed_files = 0
    removed_bytes = 0
    images_root = output_root / "images"
    if images_root.is_dir():
        for path in images_root.rglob("*.*"):
            relative = str(path.relative_to(output_root))
            if relative in referenced:
                continue
            removed_bytes += path.stat().st_size
            path.unlink()
            removed_files += 1
        for directory in sorted(
            (path for path in images_root.rglob("*") if path.is_dir()), reverse=True
        ):
            try:
                directory.rmdir()
            except OSError:
                pass
    return {"removed_files": removed_files, "removed_bytes": removed_bytes}


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = discover_paths(args)
    workers = max(int(args.workers), 1)
    results = []
    failures = []
    if workers == 1:
        for path in paths:
            try:
                results.append(
                    _extract_one(
                        str(path),
                        str(args.output_dir),
                        args.stream,
                        args.overwrite,
                        args.max_resolution,
                    )
                )
            except Exception as error:  # noqa: BLE001 - record every bad source object.
                failures.append(
                    {"source_pkl": str(path), "error": repr(error), "traceback": traceback.format_exc()}
                )
                if not args.allow_failures:
                    break
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _extract_one,
                    str(path),
                    str(args.output_dir),
                    args.stream,
                    args.overwrite,
                    args.max_resolution,
                ): path
                for path in paths
            }
            for future in as_completed(futures):
                path = futures[future]
                try:
                    results.append(future.result())
                except Exception as error:  # noqa: BLE001
                    failures.append(
                        {"source_pkl": str(path), "error": repr(error), "traceback": traceback.format_exc()}
                    )

    entries = sorted((result["entry"] for result in results), key=lambda item: item["sample_id"])
    temporary_manifest = args.output_dir / f".{MANIFEST_NAME}.tmp.{os.getpid()}"
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    os.replace(temporary_manifest, args.output_dir / MANIFEST_NAME)
    prune_summary = {"removed_files": 0, "removed_bytes": 0}
    if args.prune_unreferenced_images and not failures:
        prune_summary = prune_unreferenced_images(args.output_dir, entries)
    summary = {
        "schema": SCHEMA,
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "stream": args.stream,
        "max_resolution": args.max_resolution,
        "num_requested": len(paths),
        "num_objects": len(entries),
        "num_written": sum(result["status"] == "written" for result in results),
        "num_cached": sum(result["status"] == "cached" for result in results),
        "num_failures": len(failures),
        "num_views": sum(int(entry["num_views"]) for entry in entries),
        "num_visible_observations": sum(
            int(entry["num_visible_observations"]) for entry in entries
        ),
        "failures": failures,
        "gt_geometry_copied": False,
        "pruned_unreferenced_images": prune_summary,
    }
    atomic_write_json(args.output_dir / SUMMARY_NAME, summary)
    print(json.dumps(summary, indent=2))
    if failures and not args.allow_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
