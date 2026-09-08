#!/usr/bin/env python3
"""Associate BoxeR detections to fused objects and obtain SAM2 instance masks.

This stage deliberately keeps responsibilities separate:

* BoxeR owns 2D proposals, 3D lifting, identity fusion, and OBBs.
* SAM2 receives only BoxeR's retained image boxes and predicts visible masks.
* RGB-D points receive an instance label only when both the SAM2 mask and an
  expanded fused BoxeR OBB agree.
* The best SAM2 observation of each fused object is exported as an RGBA crop
  for TRELLIS.2.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Sam2Model, Sam2Processor


ROOT = Path(__file__).resolve().parents[2]
BOXER_ROOT = ROOT / "baselines/_upstream/boxer"
if str(BOXER_ROOT) not in sys.path:
    sys.path.insert(0, str(BOXER_ROOT))

from utils.file_io import read_obb_csv  # noqa: E402
from utils.fuse_3d_boxes import BoundingBox3DFuser  # noqa: E402


PALETTE = np.asarray(
    [
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 212],
        [0, 128, 128],
        [220, 190, 255],
        [170, 110, 40],
        [255, 250, 200],
        [128, 0, 0],
        [170, 255, 195],
        [128, 128, 0],
        [255, 215, 180],
        [0, 0, 128],
        [128, 128, 128],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "data/Imaginarium",
    )
    parser.add_argument("--boxer-scene-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sam2-model", default="facebook/sam2.1-hiera-large"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--point-downsample", type=int, default=8)
    parser.add_argument("--obb-expand-ratio", type=float, default=0.15)
    parser.add_argument("--obb-expand-min-m", type=float, default=0.05)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--max-hole-area", type=float, default=64.0)
    parser.add_argument("--max-sprinkle-area", type=float, default=32.0)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--crop-margin-ratio", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_associations(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def reproduce_fusion(
    raw_csv: Path,
) -> tuple[list[Any], dict[tuple[int, int], int], dict[int, Any]]:
    """Re-run official fusion and retain its otherwise-unserialized memberships."""
    timed = read_obb_csv(os.fspath(raw_csv))
    timestamps = list(timed)
    flattened = torch.cat([timed[timestamp] for timestamp in timestamps], dim=0)
    confidence_mask = flattened.prob.squeeze() >= 0.55
    kept_raw_indices = torch.nonzero(confidence_mask, as_tuple=False).flatten()

    fuser = BoundingBox3DFuser(
        iou_threshold=0.3,
        min_detections=4,
        conf_threshold=0.55,
    )
    instances = fuser.fuse(flattened)

    raw_index_to_frame_row: dict[int, tuple[int, int]] = {}
    offset = 0
    for timestamp in timestamps:
        for row in range(len(timed[timestamp])):
            raw_index_to_frame_row[offset + row] = (int(timestamp), row)
        offset += len(timed[timestamp])

    membership: dict[tuple[int, int], int] = {}
    fused_by_id: dict[int, Any] = {}
    for fused_id, instance in enumerate(instances, start=1):
        fused_by_id[fused_id] = instance.obb
        for filtered_index in instance.detection_indices:
            raw_index = int(kept_raw_indices[int(filtered_index)])
            membership[raw_index_to_frame_row[raw_index]] = fused_id
    return instances, membership, fused_by_id


def serialize_fused_objects(instances: list[Any]) -> list[dict[str, Any]]:
    objects = []
    for fused_id, instance in enumerate(instances, start=1):
        obb = instance.obb
        extents = (obb.bb3_max_object - obb.bb3_min_object).detach().cpu().numpy()
        pose = obb.T_world_object
        text_value = obb.text_string()
        if isinstance(text_value, str):
            label = text_value
        else:
            label = text_value[0] if len(text_value) else "object"
        objects.append(
            {
                "index": fused_id,
                "name": f"object_{fused_id:04d}",
                "category": label,
                "confidence": float(obb.prob.squeeze()),
                "support_count": int(instance.support_count),
                "translation": pose.t.detach().cpu().numpy().tolist(),
                "rotation": pose.q.detach().cpu().numpy().tolist(),
                "scale": extents.tolist(),
                "obb_world": {
                    "translation": pose.t.detach().cpu().numpy().tolist(),
                    "rotation": pose.q.detach().cpu().numpy().tolist(),
                    "scale": extents.tolist(),
                },
            }
        )
    return objects


def read_depth(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        if "depth" in archive:
            return archive["depth"].astype(np.float32)
        if "arr_0" in archive:
            return archive["arr_0"].astype(np.float32)
        raise KeyError(f"Unsupported depth keys in {path}: {archive.files}")


def normalize_masks(masks: torch.Tensor | np.ndarray, expected: int) -> np.ndarray:
    result = np.asarray(masks)
    while result.ndim > 3 and result.shape[1] == 1:
        result = result[:, 0]
    if result.ndim == 2:
        result = result[None]
    if len(result) != expected:
        raise RuntimeError(
            f"SAM2 produced {len(result)} masks for {expected} boxes; "
            f"shape={result.shape}"
        )
    return result.astype(bool)


def square_rgba_crop(
    image: np.ndarray,
    mask: np.ndarray,
    size: int,
    margin_ratio: float,
) -> Image.Image:
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        raise ValueError("Cannot crop an empty mask")
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    side = max(x1 - x0, y1 - y0)
    side = max(int(math.ceil(side * (1.0 + 2.0 * margin_ratio))), 2)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    sx0 = int(math.floor(cx - side / 2))
    sy0 = int(math.floor(cy - side / 2))
    sx1 = sx0 + side
    sy1 = sy0 + side

    rgba = np.zeros((side, side, 4), dtype=np.uint8)
    source_x0 = max(sx0, 0)
    source_y0 = max(sy0, 0)
    source_x1 = min(sx1, image.shape[1])
    source_y1 = min(sy1, image.shape[0])
    target_x0 = source_x0 - sx0
    target_y0 = source_y0 - sy0
    target_x1 = target_x0 + source_x1 - source_x0
    target_y1 = target_y0 + source_y1 - source_y0
    rgba[target_y0:target_y1, target_x0:target_x1, :3] = image[
        source_y0:source_y1, source_x0:source_x1
    ]
    rgba[target_y0:target_y1, target_x0:target_x1, 3] = (
        mask[source_y0:source_y1, source_x0:source_x1].astype(np.uint8) * 255
    )
    return Image.fromarray(rgba, mode="RGBA").resize(
        (size, size), Image.Resampling.LANCZOS
    )


def mask_quality(
    image: np.ndarray,
    mask: np.ndarray,
    detection_score: float,
    sam_score: float,
) -> tuple[float, dict[str, float]]:
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return -float("inf"), {"area": 0.0}
    height, width = mask.shape
    area = float(mask.sum())
    area_ratio = area / float(height * width)
    border_clearance = min(
        float(xx.min()) / width,
        float(width - 1 - xx.max()) / width,
        float(yy.min()) / height,
        float(height - 1 - yy.max()) / height,
    )
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    quality = (
        math.log1p(area)
        + 1.5 * max(border_clearance, -0.25)
        + 0.15 * math.log1p(max(sharpness, 0.0))
        + 0.75 * float(detection_score)
        + 0.5 * float(sam_score)
    )
    return quality, {
        "area": area,
        "area_ratio": area_ratio,
        "border_clearance": border_clearance,
        "sharpness": sharpness,
        "detection_score": float(detection_score),
        "sam_score": float(sam_score),
        "quality": quality,
    }


def world_points_for_eval_grid(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
    downsample: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = depth.shape
    grid_height = height // downsample
    grid_width = width // downsample
    sampled = depth[::downsample, ::downsample][:grid_height, :grid_width]
    half = downsample // 2
    yy = np.arange(grid_height, dtype=np.float32) * downsample + half
    xx = np.arange(grid_width, dtype=np.float32) * downsample + half
    z = sampled
    x = (xx[None, :] - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (yy[:, None] - intrinsic[1, 2]) * z / intrinsic[1, 1]
    camera = np.stack(
        [
            np.broadcast_to(x, z.shape),
            np.broadcast_to(y, z.shape),
            z,
        ],
        axis=-1,
    )
    world = (
        camera.reshape(-1, 3) @ camera_to_world[:3, :3].T
        + camera_to_world[:3, 3]
    )
    return world, (grid_height, grid_width)


def points_inside_expanded_obb(
    points_world: np.ndarray,
    obb: Any,
    ratio: float,
    minimum: float,
) -> np.ndarray:
    transform = obb.T_world_object.matrix.detach().cpu().numpy()
    inverse = np.linalg.inv(transform)
    local = points_world @ inverse[:3, :3].T + inverse[:3, 3]
    lower = obb.bb3_min_object.detach().cpu().numpy()
    upper = obb.bb3_max_object.detach().cpu().numpy()
    expansion = np.maximum((upper - lower) * ratio, minimum)
    valid = np.isfinite(local).all(axis=1)
    return valid & (local >= lower - expansion).all(axis=1) & (
        local <= upper + expansion
    ).all(axis=1)


def font(size: int = 18) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def overlay_instances(
    image: np.ndarray,
    labels: np.ndarray,
    detections: list[dict[str, Any]],
) -> Image.Image:
    output = image.astype(np.float32).copy()
    for fused_id in np.unique(labels):
        if fused_id <= 0:
            continue
        active = labels == fused_id
        color = PALETTE[(int(fused_id) - 1) % len(PALETTE)].astype(np.float32)
        output[active] = output[active] * 0.48 + color * 0.52
    panel = Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(panel)
    text_font = font(max(14, image.shape[1] // 70))
    for detection in detections:
        fused_id = int(detection["fused_id"])
        box = [int(round(value)) for value in detection["box_xyxy_source"]]
        color = tuple(int(v) for v in PALETTE[(fused_id - 1) % len(PALETTE)])
        draw.rectangle(box, outline=color, width=max(2, image.shape[1] // 400))
        draw.text(
            (box[0], box[1]),
            f"{fused_id}:{detection['label']}",
            fill=color,
            font=text_font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    return panel


def make_contact_sheet(paths: list[Path], output_path: Path) -> None:
    if not paths:
        return
    selected_indices = np.linspace(0, len(paths) - 1, min(8, len(paths))).round().astype(int)
    images = [Image.open(paths[index]).convert("RGB") for index in selected_indices]
    target_width = 640
    resized = []
    for image in images:
        height = int(round(image.height * target_width / image.width))
        resized.append(image.resize((target_width, height), Image.Resampling.LANCZOS))
    sheet = Image.new(
        "RGB",
        (target_width * 2, sum(image.height for image in resized[::2])),
        (245, 247, 250),
    )
    y_offsets = [0, 0]
    for index, image in enumerate(resized):
        column = index % 2
        sheet.paste(image, (column * target_width, y_offsets[column]))
        y_offsets[column] += image.height
    sheet = sheet.crop((0, 0, sheet.width, max(y_offsets)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_output_dir = args.output_dir / "frames"
    crop_dir = args.output_dir / "crops"
    frame_output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    association_path = args.boxer_scene_dir / "boxer_frame_associations.jsonl"
    raw_csv = args.boxer_scene_dir / "boxer_3dbbs.csv"
    associations = load_associations(association_path)
    instances, membership, fused_by_id = reproduce_fusion(raw_csv)
    objects = serialize_fused_objects(instances)

    timed_raw = read_obb_csv(os.fspath(raw_csv))
    by_timestamp = {int(timestamp): obbs for timestamp, obbs in timed_raw.items()}
    for frame in associations:
        timestamp = int(frame["timestamp_ns"])
        raw_count = len(by_timestamp.get(timestamp, []))
        if raw_count != len(frame["detections"]):
            raise ValueError(
                f"timestamp {timestamp}: CSV has {raw_count} rows but "
                f"association JSON has {len(frame['detections'])}"
            )
        kept = []
        for detection in frame["detections"]:
            key = (timestamp, int(detection["raw_3d_row"]))
            if key not in membership:
                continue
            detection = dict(detection)
            detection["fused_id"] = int(membership[key])
            kept.append(detection)
        frame["detections"] = kept

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    processor = Sam2Processor.from_pretrained(args.sam2_model)
    model = Sam2Model.from_pretrained(args.sam2_model, torch_dtype=dtype).to(device)
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    network_seconds = 0.0
    wall_start = time.perf_counter()
    best_views: dict[int, dict[str, Any]] = {}
    overlay_paths = []
    all_point_labels: dict[int, np.ndarray] = {}
    per_frame_records = []

    camera_json = json.loads(
        (
            args.dataset_root
            / "renders"
            / args.scene_id
            / "0.json"
        ).read_text(encoding="utf-8")
    )
    intrinsic = np.asarray(camera_json["K"], dtype=np.float32).reshape(3, 3)
    expected_frames = len(camera_json["frames"])

    for frame in associations:
        frame_index = int(frame["frame_index"])
        image = Image.open(frame["frame_path"]).convert("RGB")
        image_np = np.asarray(image)
        detections = frame["detections"]
        if not detections:
            all_point_labels[frame_index] = np.zeros(
                (
                    image.height // args.point_downsample,
                    image.width // args.point_downsample,
                ),
                dtype=np.uint16,
            )
            continue

        boxes = [item["box_xyxy_source"] for item in detections]
        inputs = processor(
            images=image,
            input_boxes=[boxes],
            return_tensors="pt",
        )
        inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            network_start = time.perf_counter()
        outputs = model(**inputs, multimask_output=False)
        if device.type == "cuda":
            end_event.record()
            torch.cuda.synchronize(device)
            network_seconds += start_event.elapsed_time(end_event) / 1000.0
        else:
            network_seconds += time.perf_counter() - network_start

        masks = processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            mask_threshold=args.mask_threshold,
            binarize=True,
            max_hole_area=args.max_hole_area,
            max_sprinkle_area=args.max_sprinkle_area,
        )[0]
        masks = normalize_masks(masks, len(detections))
        sam_scores = outputs.iou_scores.detach().float().cpu().numpy().reshape(
            len(detections), -1
        )[:, 0]

        resolved = np.zeros((image.height, image.width), dtype=np.uint16)
        confidence = np.full((image.height, image.width), -np.inf, dtype=np.float32)
        for detection, mask, sam_score in zip(detections, masks, sam_scores):
            fused_id = int(detection["fused_id"])
            combined = (
                float(detection["score_combined"]) + float(sam_score)
            ) * 0.5
            take = mask & (combined > confidence)
            resolved[take] = fused_id
            confidence[take] = combined

            quality, quality_terms = mask_quality(
                image_np,
                mask,
                float(detection["score_combined"]),
                float(sam_score),
            )
            current = best_views.get(fused_id)
            if current is None or quality > current["quality"]:
                crop_path = crop_dir / f"object_{fused_id:04d}.png"
                square_rgba_crop(
                    image_np,
                    mask,
                    args.crop_size,
                    args.crop_margin_ratio,
                ).save(crop_path)
                best_views[fused_id] = {
                    "fused_id": fused_id,
                    "quality": quality,
                    "quality_terms": quality_terms,
                    "frame_index": frame_index,
                    "frame_path": frame["frame_path"],
                    "box_xyxy_source": detection["box_xyxy_source"],
                    "label": detection["label"],
                    "crop_path": os.fspath(crop_path),
                    "camera_to_world": frame["camera_to_world"],
                }

        depth = read_depth(Path(frame["depth_path"]))
        if depth.shape != resolved.shape:
            depth = cv2.resize(
                depth,
                (resolved.shape[1], resolved.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        points_world, grid_shape = world_points_for_eval_grid(
            depth,
            intrinsic,
            np.asarray(frame["camera_to_world"], dtype=np.float32),
            args.point_downsample,
        )
        point_labels = resolved[
            :: args.point_downsample, :: args.point_downsample
        ][: grid_shape[0], : grid_shape[1]].reshape(-1)
        for fused_id in np.unique(point_labels):
            if fused_id <= 0:
                continue
            selected = point_labels == fused_id
            inside = points_inside_expanded_obb(
                points_world[selected],
                fused_by_id[int(fused_id)],
                args.obb_expand_ratio,
                args.obb_expand_min_m,
            )
            selected_indices = np.flatnonzero(selected)
            point_labels[selected_indices[~inside]] = 0
        all_point_labels[frame_index] = point_labels.reshape(grid_shape)

        np.savez_compressed(
            frame_output_dir / f"frame_{frame_index:04d}_sam2.npz",
            fused_ids=np.asarray(
                [item["fused_id"] for item in detections], dtype=np.int32
            ),
            boxes_xyxy=np.asarray(boxes, dtype=np.float32),
            sam_scores=sam_scores.astype(np.float32),
            masks=masks,
            resolved_labels=resolved,
        )
        overlay_path = frame_output_dir / f"frame_{frame_index:04d}_overlay.jpg"
        overlay_instances(image_np, resolved, detections).save(
            overlay_path, quality=93
        )
        overlay_paths.append(overlay_path)
        per_frame_records.append(
            {
                "frame_index": frame_index,
                "num_boxer_associated": len(detections),
                "num_visible_fused_ids": len(
                    set(int(item["fused_id"]) for item in detections)
                ),
                "overlay_path": os.fspath(overlay_path),
            }
        )

    grid_height = int(camera_json["height"]) // args.point_downsample
    grid_width = int(camera_json["width"]) // args.point_downsample
    zero_grid = np.zeros((grid_height, grid_width), dtype=np.uint16)
    ordered_point_labels = np.stack(
        [all_point_labels.get(index, zero_grid) for index in range(expected_frames)]
    )
    np.save(
        args.output_dir
        / f"point_instance_labels_downsample{args.point_downsample}.npy",
        ordered_point_labels.reshape(-1),
    )

    for obj in objects:
        obj["best_view"] = best_views.get(int(obj["index"]))
        if obj["best_view"] is not None:
            obj["crop_path"] = obj["best_view"]["crop_path"]
    atomic_json(
        args.output_dir / "object_obbs.json",
        {
            "schema": "ff_boxer_sam2_objects_v1",
            "scene_id": args.scene_id,
            "objects": objects,
        },
    )
    atomic_json(
        args.output_dir / "sam2_summary.json",
        {
            "schema": "ff_boxer_sam2_scene_summary_v1",
            "scene_id": args.scene_id,
            "boxer_scene_dir": os.fspath(args.boxer_scene_dir),
            "sam2_model": args.sam2_model,
            "num_fused_objects": len(objects),
            "num_objects_with_crop": len(best_views),
            "num_processed_frames": len(per_frame_records),
            "network_seconds": network_seconds,
            "wall_seconds": time.perf_counter() - wall_start,
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
            "peak_cuda_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda"
                else None
            ),
            "point_downsample": args.point_downsample,
            "obb_expand_ratio": args.obb_expand_ratio,
            "obb_expand_min_m": args.obb_expand_min_m,
            "frames": per_frame_records,
        },
    )
    make_contact_sheet(
        overlay_paths,
        args.output_dir / "boxer_sam2_contact_sheet.jpg",
    )
    print(
        json.dumps(
            {
                "scene_id": args.scene_id,
                "num_fused_objects": len(objects),
                "num_objects_with_crop": len(best_views),
                "network_seconds": network_seconds,
                "wall_seconds": time.perf_counter() - wall_start,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
