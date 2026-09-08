#!/usr/bin/env python3
"""Adapt one frozen FF scene to LiteReality's object-stage RGB input contract."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import re
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fire3d-test-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--scene-id")
    parser.add_argument("--random-seed", type=int, default=20260724)
    parser.add_argument("--views-per-object", type=int, default=4)
    parser.add_argument("--bbox-padding-ratio", type=float, default=0.04)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        if len(archive.files) != 1:
            raise ValueError(f"Expected one array in {path}, got {archive.files}")
        return archive[archive.files[0]]


def clean_semantic(mesh_name: str) -> str:
    value = re.sub(r"^[0-9a-zA-Z]+_", "", str(mesh_name))
    value = re.sub(r"\.\d+$", "", value)
    value = re.sub(r"[-_.]+", " ", value)
    value = re.sub(r"(?i)\bSM\b", " ", value)
    value = re.sub(r"(?i)\b\d+k\s+packed\b$", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or "object"


def bbox_from_mask(mask: np.ndarray, padding_ratio: float) -> list[int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Cannot construct bbox from an empty mask")
    height, width = mask.shape
    pad_x = max(2, int(round((xs.max() - xs.min() + 1) * padding_ratio)))
    pad_y = max(2, int(round((ys.max() - ys.min() + 1) * padding_ratio)))
    x1 = max(0, int(xs.min()) - pad_x)
    y1 = max(0, int(ys.min()) - pad_y)
    x2 = min(width, int(xs.max()) + 1 + pad_x)
    y2 = min(height, int(ys.max()) + 1 + pad_y)
    return [x1, y1, x2, y2]


def camera_to_world(frame: dict) -> np.ndarray:
    eye = np.asarray(frame["eye"], dtype=np.float64)
    lookat = np.asarray(frame["lookat"], dtype=np.float64)
    up = np.asarray(frame["up"], dtype=np.float64)
    forward = lookat - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = right
    matrix[:3, 1] = true_up
    matrix[:3, 2] = -forward
    matrix[:3, 3] = eye
    return matrix


def choose_scene(manifest: dict, scene_id: str | None, seed: int) -> dict:
    scenes = manifest["scenes"]
    if scene_id is not None:
        matches = [scene for scene in scenes if scene["scene_id"] == scene_id]
        if len(matches) != 1:
            raise ValueError(f"Expected one scene {scene_id!r}, found {len(matches)}")
        return matches[0]
    return random.Random(seed).choice(scenes)


def main() -> None:
    args = parse_args()
    started = time.time()
    manifest = json.loads(args.manifest.read_text())
    scene = choose_scene(manifest, args.scene_id, args.random_seed)
    dataset_root = args.fire3d_test_root / manifest["dataset_subdir"]
    scene_id = scene["scene_id"]
    input_root = args.baseline_root / "input" / "object_stage" / scene_id
    if input_root.exists() and args.overwrite:
        shutil.rmtree(input_root)
    input_root.mkdir(parents=True, exist_ok=True)

    frames_dir = dataset_root / scene["frames_dir"]
    masks_dir = dataset_root / scene["masks_dir"]
    transforms_path = dataset_root / scene["transforms_path"]
    mesh_dir = dataset_root / scene["mesh_dir"]
    camera = json.loads((dataset_root / scene["camera_path"]).read_text())
    transforms = pickle.loads(transforms_path.read_bytes())
    intrinsic = np.asarray(camera["K"], dtype=np.float64)
    frame_records = camera["frames"]
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    mask_paths = sorted(masks_dir.glob("mask_*.npz"))
    if len(frame_paths) != len(mask_paths) or len(frame_paths) != len(frame_records):
        raise ValueError(
            f"Frame contract mismatch: rgb={len(frame_paths)} mask={len(mask_paths)} "
            f"camera={len(frame_records)}"
        )
    masks = [load_npz(path) for path in mask_paths]

    records = []
    for object_id in scene["visible_object_ids"]:
        transform_key = f"object_{int(object_id):04d}"
        transform = transforms[transform_key]
        semantic = clean_semantic(transform.get("mesh_name", transform_key))
        visibility = [
            (int(np.count_nonzero(mask == object_id)), frame_index)
            for frame_index, mask in enumerate(masks)
        ]
        selected = [
            frame_index
            for pixels, frame_index in sorted(
                visibility, key=lambda item: (-item[0], item[1])
            )
            if pixels > 0
        ][: args.views_per_object]
        if not selected:
            raise RuntimeError(f"No visible frames for object {object_id}")

        object_name = f"Object_{int(object_id):04d}"
        object_root = input_root / object_name
        images_root = object_root / "images"
        images_root.mkdir(parents=True, exist_ok=True)
        bbox_info: dict[str, object] = {"semantic": semantic}
        camera_info: dict[str, object] = {}
        selected_records = []
        for rank, frame_index in enumerate(selected):
            src = frame_paths[frame_index]
            dst = images_root / src.name
            shutil.copy2(src, dst)
            object_mask = masks[frame_index] == object_id
            bbox = bbox_from_mask(object_mask, args.bbox_padding_ratio)
            bbox_info[src.stem] = bbox
            c2w = camera_to_world(frame_records[frame_index])
            camera_info[src.stem] = {
                "intrinsic": intrinsic.tolist(),
                "pose": c2w.tolist(),
                "bbox": bbox,
                "dimensions": [int(camera["width"]), int(camera["height"])],
            }
            selected_records.append(
                {
                    "rank": rank,
                    "frame_index": frame_index,
                    "rgb_path": str(src),
                    "mask_path": str(mask_paths[frame_index]),
                    "visible_pixels": int(object_mask.sum()),
                    "bbox_xyxy_exclusive": bbox,
                }
            )
        (object_root / "bbox_info_updated.json").write_text(
            json.dumps(bbox_info, indent=2)
        )
        (object_root / "camera_pose_info.json").write_text(
            json.dumps(camera_info, indent=2)
        )
        records.append(
            {
                "object_id": int(object_id),
                "object_name": object_name,
                "semantic": semantic,
                "raw_mesh_name": transform.get("mesh_name"),
                "gt_mesh_path": str(mesh_dir / f"object_{int(object_id):04d}.glb"),
                "gt_transform": {
                    "scale": float(transform["scale"]),
                    "angles": [float(value) for value in transform["angles"]],
                    "translation": [float(value) for value in transform["trans"]],
                },
                "selected_views": selected_records,
            }
        )

    adapter_manifest = {
        "schema": "ff_litereality_scene_adapter_v1",
        "dataset": manifest["dataset"],
        "source_manifest": str(args.manifest.resolve()),
        "selection": {
            "mode": "explicit" if args.scene_id else "python_random_choice",
            "random_seed": args.random_seed,
        },
        "scene_id": scene_id,
        "benchmark_index": scene["benchmark_index"],
        "raw_index": scene["raw_index"],
        "views_per_object": args.views_per_object,
        "bbox_padding_ratio": args.bbox_padding_ratio,
        "oracle_fields": [
            "object_identity",
            "instance_mask_for_crop_and_view_selection",
            "semantic_description_from_gt_mesh_name",
            "gt_pose_for_later_evaluation",
        ],
        "num_objects": len(records),
        "objects": records,
        "elapsed_seconds": time.time() - started,
    }
    (input_root / "adapter_manifest.json").write_text(
        json.dumps(adapter_manifest, indent=2)
    )
    print(json.dumps({
        "scene_id": scene_id,
        "num_objects": len(records),
        "input_root": str(input_root),
        "elapsed_seconds": adapter_manifest["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
