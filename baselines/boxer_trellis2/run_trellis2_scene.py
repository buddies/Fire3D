#!/usr/bin/env python3
"""Reconstruct BoxeR+SAM2 object crops with TRELLIS.2 and place them in world."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from torchvision import transforms
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
TRELLIS_ROOT = ROOT / "baselines/_upstream/trellis2"
BOXER_ROOT = ROOT / "baselines/_upstream/boxer"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))

import o_voxel  # noqa: E402
from trellis2.pipelines import Trellis2ImageTo3DPipeline  # noqa: E402
from trellis2.pipelines import rembg  # noqa: E402
from trellis2.modules import image_feature_extractor  # noqa: E402

from baselines.boxer_trellis2.mesh_scene_utils import (  # noqa: E402
    compose_world_scene,
    load_mesh_or_scene,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--boxer-sam2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    parser.add_argument(
        "--pipeline-type",
        choices=("512", "1024", "1024_cascade", "1536_cascade"),
        default="1024_cascade",
    )
    parser.add_argument("--texture-size", type=int, default=512)
    parser.add_argument("--decimation-target", type=int, default=100_000)
    parser.add_argument("--max-num-tokens", type=int, default=49_152)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--max-generation-attempts",
        type=int,
        default=3,
        help=(
            "Retry with deterministic alternate seeds when TRELLIS.2 samples "
            "an empty sparse structure."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--extent-quantile",
        type=float,
        default=0.0,
        help=(
            "Symmetric quantile used to normalize TRELLIS geometry. The default "
            "uses full bounds so the placed mesh cannot exceed the BoxeR OBB."
        ),
    )
    parser.add_argument(
        "--local-dinov3",
        type=Path,
        default=ROOT / "checkpoints/Fire3D/external/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        help="Exact official DINOv3-L checkpoint used when HF access is gated.",
    )
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def quat_wxyz_to_matrix(quaternion: list[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q /= np.linalg.norm(q).clip(1e-12)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def canonical_to_world(
    vertices: np.ndarray,
    obj: dict[str, Any],
    quantile: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit robust TRELLIS bounds to the BoxeR OBB without GT information."""
    lower = np.quantile(vertices, quantile, axis=0)
    upper = np.quantile(vertices, 1.0 - quantile, axis=0)
    center = 0.5 * (lower + upper)
    extent = np.maximum(upper - lower, 1e-5)
    target_extent = np.maximum(
        np.asarray(obj["obb_world"]["scale"], dtype=np.float64), 1e-5
    )
    scale = target_extent / extent
    rotation = quat_wxyz_to_matrix(obj["obb_world"]["rotation"])
    translation = np.asarray(
        obj["obb_world"]["translation"], dtype=np.float64
    )

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation @ np.diag(scale)
    transform[:3, 3] = translation - transform[:3, :3] @ center
    world_vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
    metadata = {
        "source_robust_lower": lower.tolist(),
        "source_robust_upper": upper.tolist(),
        "source_robust_center": center.tolist(),
        "source_robust_extent": extent.tolist(),
        "target_obb_extent": target_extent.tolist(),
        "anisotropic_scale": scale.tolist(),
        "canonical_to_world": transform.tolist(),
        "up_axis": "z",
    }
    return world_vertices, transform, metadata


def export_textured_glb(
    mesh: Any,
    output_path: Path,
    texture_size: int,
    decimation_target: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )
    glb.export(os.fspath(output_path), extension_webp=True)


class LocalDinoV3FeatureExtractor:
    """TRELLIS-compatible exact DINOv3-L loader using the cached Meta weights."""

    def __init__(self, model_name: str, image_size: int = 512):
        del model_name
        boxer_path = os.fspath(BOXER_ROOT)
        sys.path.insert(0, boxer_path)
        try:
            from boxernet.dinov3_wrapper import dinov3_vitl16
        finally:
            sys.path.remove(boxer_path)
        checkpoint = os.environ["FF_TRELLIS2_LOCAL_DINOV3"]
        self.model = dinov3_vitl16(
            pretrained=True,
            weights=checkpoint,
            check_hash=False,
        )
        self.model.eval()
        self.image_size = int(image_size)
        self.transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ]
        )

    def to(self, device):
        self.model.to(device)
        return self

    def cuda(self):
        self.model.cuda()
        return self

    def cpu(self):
        self.model.cpu()
        return self

    @torch.no_grad()
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            batch = image
        elif isinstance(image, list):
            batch = []
            for item in image:
                resized = item.resize(
                    (self.image_size, self.image_size), Image.Resampling.LANCZOS
                )
                array = np.asarray(resized.convert("RGB")).astype(np.float32) / 255.0
                batch.append(torch.from_numpy(array).permute(2, 0, 1))
            batch = torch.stack(batch).cuda()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        batch = self.transform(batch).cuda()
        hidden = self.model.forward_features(batch)["x_prenorm"]
        return F.layer_norm(hidden, hidden.shape[-1:])


class AlphaOnlyBackgroundRemover:
    """No-op rembg component because every input already has a SAM2 alpha mask."""

    def __init__(self, *args, **kwargs):
        del args, kwargs

    def to(self, device):
        del device
        return self

    def cuda(self):
        return self

    def cpu(self):
        return self

    def __call__(self, image):
        return image.convert("RGBA")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    object_dir = args.output_dir / "objects"
    object_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads(
        (args.boxer_sam2_dir / "object_obbs.json").read_text(encoding="utf-8")
    )
    objects = [obj for obj in source["objects"] if obj.get("crop_path")]
    if args.limit is not None:
        objects = objects[: args.limit]

    state_path = args.output_dir / "trellis2_summary.json"
    previous: dict[int, dict[str, Any]] = {}
    if args.resume and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        previous = {
            int(record["index"]): record for record in state.get("objects", [])
        }

    if not args.local_dinov3.is_file():
        raise FileNotFoundError(
            f"Exact local DINOv3-L checkpoint is missing: {args.local_dinov3}"
        )
    os.environ["FF_TRELLIS2_LOCAL_DINOV3"] = os.fspath(args.local_dinov3)
    image_feature_extractor.DinoV3FeatureExtractor = LocalDinoV3FeatureExtractor
    rembg.BiRefNet = AlphaOnlyBackgroundRemover

    load_start = time.perf_counter()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()
    load_seconds = time.perf_counter() - load_start

    records = []
    failures = []
    world_glbs = []
    for position, obj in enumerate(objects, start=1):
        index = int(obj["index"])
        name = obj["name"]
        obj_root = object_dir / name
        canonical_ply = obj_root / "canonical_geometry.ply"
        canonical_glb = obj_root / "canonical_textured.glb"
        world_ply = obj_root / "world_geometry.ply"
        world_glb = obj_root / "world_textured.glb"
        transform_path = obj_root / "placement.json"
        if (
            args.resume
            and index in previous
            and canonical_ply.is_file()
            and canonical_glb.is_file()
            and world_ply.is_file()
            and world_glb.is_file()
        ):
            records.append(previous[index])
            world_glbs.append(world_glb)
            print(f"[{position}/{len(objects)}] resume {name}", flush=True)
            continue

        obj_root.mkdir(parents=True, exist_ok=True)
        crop = Image.open(obj["crop_path"]).convert("RGBA")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        infer_start = time.perf_counter()
        mesh = None
        attempt_errors = []
        generation_seed = None
        for attempt in range(args.max_generation_attempts):
            generation_seed = args.seed + index + attempt * 1_000_003
            try:
                mesh = pipeline.run(
                    crop,
                    seed=generation_seed,
                    pipeline_type=args.pipeline_type,
                    max_num_tokens=args.max_num_tokens,
                )[0]
                break
            except RuntimeError as exc:
                message = str(exc)
                if "input.numel() == 0" not in message:
                    raise
                attempt_errors.append(message)
                print(
                    f"[{position}/{len(objects)}] {name}: empty sparse sample "
                    f"on attempt {attempt + 1}/{args.max_generation_attempts}",
                    flush=True,
                )
                torch.cuda.empty_cache()
        if mesh is None:
            failure = {
                "index": index,
                "name": name,
                "category": obj.get("category"),
                "crop_path": obj["crop_path"],
                "attempts": args.max_generation_attempts,
                "seeds": [
                    args.seed + index + attempt * 1_000_003
                    for attempt in range(args.max_generation_attempts)
                ],
                "error": attempt_errors[-1] if attempt_errors else "unknown",
            }
            failures.append(failure)
            atomic_json(
                state_path,
                {
                    "schema": "ff_boxer_trellis2_scene_summary_v1",
                    "status": "running",
                    "scene_id": args.scene_id,
                    "model": args.model,
                    "pipeline_type": args.pipeline_type,
                    "texture_size": args.texture_size,
                    "decimation_target": args.decimation_target,
                    "model_load_seconds": load_seconds,
                    "objects": records,
                    "failed_objects": failures,
                },
            )
            print(
                f"[{position}/{len(objects)}] {name}: failed after "
                f"{args.max_generation_attempts} empty samples; skipping",
                flush=True,
            )
            continue
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - infer_start
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())

        vertices = mesh.vertices.detach().float().cpu().numpy()
        faces = mesh.faces.detach().long().cpu().numpy()
        canonical_mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces, process=False
        )
        canonical_mesh.export(canonical_ply)

        export_start = time.perf_counter()
        export_textured_glb(
            mesh,
            canonical_glb,
            texture_size=args.texture_size,
            decimation_target=args.decimation_target,
        )
        export_seconds = time.perf_counter() - export_start

        world_vertices, transform, placement = canonical_to_world(
            vertices, obj, args.extent_quantile
        )
        world_mesh = trimesh.Trimesh(
            vertices=world_vertices, faces=faces, process=False
        )
        world_mesh.export(world_ply)
        textured_world = load_mesh_or_scene(canonical_glb)
        textured_world.apply_transform(transform)
        textured_world.export(world_glb)
        atomic_json(
            transform_path,
            {
                "schema": "ff_boxer_trellis2_placement_v1",
                "scene_id": args.scene_id,
                "index": index,
                "name": name,
                "category": obj.get("category"),
                "boxer_obb": obj["obb_world"],
                **placement,
            },
        )

        record = {
            "index": index,
            "name": name,
            "category": obj.get("category"),
            "crop_path": obj["crop_path"],
            "canonical_geometry_path": os.fspath(canonical_ply),
            "canonical_textured_path": os.fspath(canonical_glb),
            "world_geometry_path": os.fspath(world_ply),
            "world_textured_path": os.fspath(world_glb),
            "placement_path": os.fspath(transform_path),
            "num_vertices": int(len(vertices)),
            "num_faces": int(len(faces)),
            "generation_seed": generation_seed,
            "inference_seconds": inference_seconds,
            "texture_export_seconds": export_seconds,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "process_peak_rss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
        }
        records.append(record)
        world_glbs.append(world_glb)
        atomic_json(
            state_path,
            {
                "schema": "ff_boxer_trellis2_scene_summary_v1",
                "status": "running",
                "scene_id": args.scene_id,
                "model": args.model,
                "pipeline_type": args.pipeline_type,
                "texture_size": args.texture_size,
                "decimation_target": args.decimation_target,
                "model_load_seconds": load_seconds,
                "objects": records,
                "failed_objects": failures,
            },
        )
        print(
            f"[{position}/{len(objects)}] {name}: "
            f"infer={inference_seconds:.2f}s export={export_seconds:.2f}s",
            flush=True,
        )

    composed_path = args.output_dir / "predicted_textured_world_scene.glb"
    compose_world_scene(world_glbs, composed_path)
    inference_records = []
    placement_records = []
    for record in records:
        placement = json.loads(
            Path(record["placement_path"]).read_text(encoding="utf-8")
        )
        inference_records.append(
            {
                "object_id": int(record["index"]),
                "status": "decoded",
                "textured_glb": record["canonical_textured_path"],
                "geometry_ply": record["canonical_geometry_path"],
            }
        )
        placement_records.append(
            {
                "instance_id": int(record["index"]),
                "object_to_world": placement["canonical_to_world"],
                "textured_glb": record["canonical_textured_path"],
            }
        )
    atomic_json(
        args.output_dir / "inference_summary.json",
        {
            "schema": "ff_boxer_trellis2_appearance_compat_inference_v1",
            "scene_id": args.scene_id,
            "objects": inference_records,
        },
    )
    atomic_json(
        args.output_dir / "appearance/appearance_summary.json",
        {
            "schema": "ff_boxer_trellis2_appearance_compat_world_v1",
            "scene_id": args.scene_id,
            "composed_world_scene": {
                "path": os.fspath(composed_path),
                "objects": placement_records,
            },
        },
    )
    atomic_json(
        args.output_dir / "object_obbs.json",
        json.loads(
            (args.boxer_sam2_dir / "object_obbs.json").read_text(encoding="utf-8")
        ),
    )
    atomic_json(
        state_path,
        {
            "schema": "ff_boxer_trellis2_scene_summary_v1",
            "status": "complete",
            "scene_id": args.scene_id,
            "model": args.model,
            "pipeline_type": args.pipeline_type,
            "texture_size": args.texture_size,
            "decimation_target": args.decimation_target,
            "model_load_seconds": load_seconds,
            "num_requested_objects": len(objects),
            "num_completed_objects": len(records),
            "num_failed_objects": len(failures),
            "network_inference_seconds": float(
                sum(record["inference_seconds"] for record in records)
            ),
            "texture_export_seconds": float(
                sum(record["texture_export_seconds"] for record in records)
            ),
            "composed_world_scene_path": os.fspath(composed_path),
            "objects": records,
            "failed_objects": failures,
        },
    )
    print(state_path)


if __name__ == "__main__":
    main()
