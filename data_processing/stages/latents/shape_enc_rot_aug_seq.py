#!/usr/bin/env python3
"""Encode right-angle-rotated shape ovoxels into pre-X2 TRELLIS shape latents.

This creates the training data expected by trellis2/configs/shape_vae_x2.yaml:

    <output-root>/<dataset>/metadata.csv
    <output-root>/<dataset>/latents/trellis2_shape_encoding/<asset>__rot090.npz

The source is the already-normalized shape O-Voxel tree configured through
``FIRE3D_OBJECT_POSTPROCESS_ROOT``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

REPO_ROOT = Path(__file__).resolve().parents[3]
TRELLIS_ROOT = REPO_ROOT / "trellis2_x2"
for import_path in (REPO_ROOT, TRELLIS_ROOT):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.append(import_path_str)

import numpy as np
import o_voxel
import pandas as pd
import torch
from tqdm import tqdm

import trellis2.models as models
import trellis2.modules.sparse as sp
from trellis2.modules.sparse.basic import sparse_cat, sparse_unbind

torch.set_grad_enabled(False)

OBJECT_ROOT = Path(
    os.environ.get("FIRE3D_OBJECT_ROOT", REPO_ROOT / "data/training_objects")
)
DEFAULT_POSTPROCESS_ROOT = Path(
    os.environ.get("FIRE3D_OBJECT_POSTPROCESS_ROOT", OBJECT_ROOT / "postprocess")
)
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("FIRE3D_SHAPE_LATENT_ROOT", OBJECT_ROOT / "shape_latents_rotated")
)
LATENT_KEY = "trellis2_shape_encoding"

DATASET_NAMES = {
    "3d-future": "3D-FUTURE",
    "3D-FUTURE": "3D-FUTURE",
    "abo": "ABO",
    "ABO": "ABO",
    "hssd": "HSSD",
    "HSSD": "HSSD",
    "objaversexl_github": "ObjaverseXL_github",
    "ObjaverseXL_github": "ObjaverseXL_github",
    "objaversexl_sketchfab": "ObjaverseXL_sketchfab",
    "ObjaverseXL_sketchfab": "ObjaverseXL_sketchfab",
    "objathor": "objathor",
    "Objathor": "objathor",
    "OBJATHOR": "objathor",
    "procthor": "procthor",
    "ProcTHOR": "procthor",
    "PROCTHOR": "procthor",
    "scenesmith": "scenesmith",
    "SceneSmith": "scenesmith",
    "SCENESMITH": "scenesmith",
    "internscenes": "internscenes",
    "InternScenes": "internscenes",
    "INTERNSCENES": "internscenes",
    "sage10k": "sage10k",
    "Sage10K": "sage10k",
    "SAGE10K": "sage10k",
    "SAGE-10k": "sage10k",
}


def rotation_label(degrees: float) -> str:
    return f"{int(round(float(degrees))) % 360:03d}"


def normalize_rotations(rotations: Iterable[float]) -> list[float]:
    normalized = []
    seen = set()
    for degrees in rotations:
        label = rotation_label(degrees)
        if label in seen:
            continue
        value = float(int(label))
        if int(value) not in {0, 90, 180, 270}:
            raise ValueError(f"Only 0/90/180/270 rotations are supported, got {degrees}")
        seen.add(label)
        normalized.append(value)
    return normalized


def clone_attr(attr: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in attr.items()}


def pack_intersected(bits: torch.Tensor) -> torch.Tensor:
    bits = bits.to(torch.uint8)
    return (bits[:, 0:1] + 2 * bits[:, 1:2] + 4 * bits[:, 2:3]).to(torch.uint8)


def unpack_intersected(intersected: torch.Tensor) -> torch.Tensor:
    if intersected.ndim == 2 and intersected.shape[1] == 3:
        return intersected.bool()
    packed = intersected.reshape(-1, 1).to(torch.uint8)
    return torch.cat(
        [
            packed % 2,
            packed // 2 % 2,
            packed // 4 % 2,
        ],
        dim=-1,
    ).bool()


def rotate_coords(coords: torch.Tensor, degrees: float, grid_size: int) -> torch.Tensor:
    deg = int(round(float(degrees))) % 360
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    if deg == 0:
        return coords.clone()
    if deg == 90:
        return torch.stack([grid_size - 1 - y, x, z], dim=1)
    if deg == 180:
        return torch.stack([grid_size - 1 - x, grid_size - 1 - y, z], dim=1)
    if deg == 270:
        return torch.stack([y, grid_size - 1 - x, z], dim=1)
    raise ValueError(f"Only 0/90/180/270 rotations are supported, got {degrees}")


def rotate_dual_vertices(dual_vertices: torch.Tensor, degrees: float) -> torch.Tensor:
    deg = int(round(float(degrees))) % 360
    x, y, z = dual_vertices[:, 0], dual_vertices[:, 1], dual_vertices[:, 2]
    if deg == 0:
        return dual_vertices.clone()
    if deg == 90:
        return torch.stack([255 - y, x, z], dim=1).to(torch.uint8)
    if deg == 180:
        return torch.stack([255 - x, 255 - y, z], dim=1).to(torch.uint8)
    if deg == 270:
        return torch.stack([y, 255 - x, z], dim=1).to(torch.uint8)
    raise ValueError(f"Only 0/90/180/270 rotations are supported, got {degrees}")


def rotate_intersected(intersected: torch.Tensor, degrees: float) -> torch.Tensor:
    del intersected, degrees
    raise RuntimeError(
        "FDG intersected flags cannot be rotated without remapping coordinate anchors; "
        "use rotate_intersected_with_coords instead."
    )


def linearize_coords(coords: torch.Tensor, grid_size: int) -> torch.Tensor:
    coords = coords.to(torch.long)
    return (coords[:, 0] * grid_size + coords[:, 1]) * grid_size + coords[:, 2]


def rotate_intersected_with_coords(
    coords: torch.Tensor,
    rotated_coords: torch.Tensor,
    intersected: torch.Tensor,
    degrees: float,
    grid_size: int,
) -> torch.Tensor:
    """Rotate FDG edge flags while preserving their canonical anchor rows."""
    deg = int(round(float(degrees))) % 360
    bits = unpack_intersected(intersected)
    if deg == 0:
        return pack_intersected(bits)

    edge_mappings = {
        90: ((0, 1, (-1, 0, 0)), (1, 0, (0, 0, 0)), (2, 2, (-1, 0, 0))),
        180: ((0, 0, (0, -1, 0)), (1, 1, (-1, 0, 0)), (2, 2, (-1, -1, 0))),
        270: ((0, 1, (0, 0, 0)), (1, 0, (0, -1, 0)), (2, 2, (0, -1, 0))),
    }
    if deg not in edge_mappings:
        raise ValueError(f"Only 0/90/180/270 local-Z rotations are supported, got {degrees}")

    del coords
    rotated_coords = rotated_coords.to(torch.int32)
    out_bits = torch.zeros_like(bits)
    sorted_keys, sorted_order = torch.sort(linearize_coords(rotated_coords, grid_size))
    size = len(sorted_keys)

    for in_axis, out_axis, shift_values in edge_mappings[deg]:
        mask = bits[:, in_axis]
        if not bool(mask.any()):
            continue
        shift = torch.tensor(shift_values, dtype=rotated_coords.dtype, device=rotated_coords.device)
        target_coords = rotated_coords[mask] + shift
        in_bounds = ((target_coords >= 0) & (target_coords < grid_size)).all(dim=1)
        if not bool(in_bounds.any()):
            continue
        target_keys = linearize_coords(target_coords[in_bounds], grid_size)
        positions = torch.searchsorted(sorted_keys, target_keys)
        valid = positions < size
        if not bool(valid.any()):
            continue
        positions = positions[valid]
        target_keys = target_keys[valid]
        matched = sorted_keys[positions] == target_keys
        if not bool(matched.any()):
            continue
        out_bits[sorted_order[positions[matched]], out_axis] = True
    return pack_intersected(out_bits)


def rotate_shape_ovoxel(
    coords: torch.Tensor,
    attr: dict[str, torch.Tensor],
    degrees: float,
    grid_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rotated_attr = clone_attr(attr)
    vertex_key = "vertices" if "vertices" in rotated_attr else "dual_vertices"
    rotated_coords = rotate_coords(coords, degrees, grid_size).to(torch.int32)
    rotated_attr[vertex_key] = rotate_dual_vertices(rotated_attr[vertex_key], degrees)
    rotated_attr["intersected"] = rotate_intersected_with_coords(
        coords,
        rotated_coords,
        rotated_attr["intersected"],
        degrees,
        grid_size,
    )
    return rotated_coords, rotated_attr


def is_valid_sparse_tensor(tensor: sp.SparseTensor) -> bool:
    return torch.isfinite(tensor.feats).all() and torch.isfinite(tensor.coords).all()


def clear_cuda_error() -> None:
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    torch.cuda.empty_cache()


def configure_hf_token(token_file: Path | None) -> None:
    if token_file is None or not token_file.exists():
        return
    token = token_file.read_text().strip()
    if not token:
        return
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)


def make_shape_encoder_inputs(
    coords: torch.Tensor,
    attr: dict[str, torch.Tensor],
) -> tuple[sp.SparseTensor, sp.SparseTensor]:
    vertex_key = "vertices" if "vertices" in attr else "dual_vertices"
    vertices = sp.SparseTensor(
        (attr[vertex_key] / 255.0).float(),
        torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1),
    )
    bits = unpack_intersected(attr["intersected"])
    intersected = vertices.replace(bits.bool())
    return vertices, intersected


def load_voxel_item(voxel_path: Path, asset_id: str) -> tuple[str, torch.Tensor, dict[str, torch.Tensor]] | None:
    try:
        coords, attr = o_voxel.io.read_vxz(str(voxel_path), num_threads=2)
        if "intersected" not in attr or not ("vertices" in attr or "dual_vertices" in attr):
            raise ValueError(f"unexpected VXZ attrs: {sorted(attr.keys())}")
        return asset_id, coords, attr
    except Exception as exc:
        print(f"[Skip] {asset_id}: load error: {exc}", flush=True)
        return None


def save_latent(save_path: Path, feats_np: np.ndarray, coords_np: np.ndarray) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, feats=feats_np, coords=coords_np)


def save_skip_marker(save_path: Path, payload: dict[str, object]) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp_path.replace(save_path)


def read_asset_ids(args: argparse.Namespace, voxel_dir: Path) -> list[str]:
    if args.asset_id:
        return sorted(set(args.asset_id))
    if args.asset_list:
        return [line.strip() for line in args.asset_list.read_text().splitlines() if line.strip()]
    return sorted(path.stem for path in voxel_dir.glob("*.vxz"))


def augmented_asset_id(asset_id: str, degrees: float) -> str:
    return f"{asset_id}__rot{rotation_label(degrees)}"


def variant_terminal(latent_dir: Path, skip_dir: Path, latent_name: str) -> bool:
    return (latent_dir / f"{latent_name}.npz").exists() or (skip_dir / f"{latent_name}.skip.json").exists()


def write_metadata(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sha256",
        f"latent_{LATENT_KEY}",
        f"{LATENT_KEY}_tokens",
        "source_dataset",
        "source_asset_id",
        "rotation_degrees",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def merge_metadata(output_dir: Path) -> Path:
    latent_dir = output_dir / "latents" / LATENT_KEY
    part_dir = output_dir / "metadata_parts"
    records: list[dict[str, object]] = []
    if part_dir.exists():
        for part_path in sorted(part_dir.glob("metadata_rank*.csv")):
            part = pd.read_csv(part_path)
            records.extend(part.to_dict("records"))

    by_id: dict[str, dict[str, object]] = {}
    for record in records:
        by_id[str(record["sha256"])] = record

    for latent_path in sorted(latent_dir.glob("*.npz")):
        if latent_path.stem in by_id:
            continue
        source_asset_id, _, rot = latent_path.stem.rpartition("__rot")
        try:
            rotation_degrees = int(rot)
        except ValueError:
            source_asset_id = latent_path.stem
            rotation_degrees = -1
        try:
            with np.load(latent_path) as data:
                token_count = int(data["coords"].shape[0])
        except Exception:
            token_count = -1
        by_id[latent_path.stem] = {
            "sha256": latent_path.stem,
            f"latent_{LATENT_KEY}": True,
            f"{LATENT_KEY}_tokens": token_count,
            "source_dataset": output_dir.name,
            "source_asset_id": source_asset_id,
            "rotation_degrees": rotation_degrees,
        }

    out_path = output_dir / "metadata.csv"
    write_metadata([by_id[key] for key in sorted(by_id)], out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", "--dataset_name", required=True, choices=sorted(DATASET_NAMES.keys()))
    parser.add_argument("--postprocess-root", type=Path, default=DEFAULT_POSTPROCESS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--enc-pretrained",
        type=str,
        default="microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16",
    )
    parser.add_argument("--hf-token-file", type=Path, default=None)
    parser.add_argument("--rotations", type=float, nargs="+", default=[0.0, 90.0, 180.0, 270.0])
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4, help="Number of rotated variants per encoder batch")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--max-assets", type=int, default=None)
    parser.add_argument("--asset-id", action="append", default=[])
    parser.add_argument("--asset-list", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-loader-workers", type=int, default=4)
    parser.add_argument("--num-saver-workers", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=16)
    parser.add_argument("--merge-metadata-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_name = DATASET_NAMES[args.dataset_name]
    args.rotations = normalize_rotations(args.rotations)
    output_dir = args.output_root / dataset_name
    latent_dir = output_dir / "latents" / LATENT_KEY
    skip_dir = output_dir / "skips" / LATENT_KEY
    part_dir = output_dir / "metadata_parts"
    latent_dir.mkdir(parents=True, exist_ok=True)
    skip_dir.mkdir(parents=True, exist_ok=True)
    part_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_metadata_only:
        out_path = merge_metadata(output_dir)
        print(json.dumps({"metadata": str(out_path)}, indent=2), flush=True)
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TRELLIS shape encoding")

    voxel_dir = args.postprocess_root / "shape_ovoxels" / dataset_name
    if not voxel_dir.exists():
        raise FileNotFoundError(voxel_dir)

    asset_ids = read_asset_ids(args, voxel_dir)
    if args.max_assets is not None:
        asset_ids = asset_ids[: args.max_assets]
    total_before_shard = len(asset_ids)
    start = total_before_shard * args.rank // args.world_size
    end = total_before_shard * (args.rank + 1) // args.world_size
    asset_ids = asset_ids[start:end]

    if not args.overwrite:
        kept = []
        for asset_id in asset_ids:
            if all(variant_terminal(latent_dir, skip_dir, augmented_asset_id(asset_id, degrees)) for degrees in args.rotations):
                continue
            kept.append(asset_id)
        asset_ids = kept

    print(
        json.dumps(
            {
                "dataset": dataset_name,
                "voxel_dir": str(voxel_dir),
                "output_dir": str(output_dir),
                "latent_key": LATENT_KEY,
                "rotations": args.rotations,
                "rank": args.rank,
                "world_size": args.world_size,
                "shard_start": start,
                "shard_end": end,
                "total_assets_before_shard": total_before_shard,
                "assets_to_process": len(asset_ids),
            },
            indent=2,
        ),
        flush=True,
    )

    configure_hf_token(args.hf_token_file)
    encoder = models.from_pretrained(args.enc_pretrained).eval().cuda()

    loader_pool = ThreadPoolExecutor(max_workers=args.num_loader_workers, thread_name_prefix="rot_aug_loader")
    saver_pool = ThreadPoolExecutor(max_workers=args.num_saver_workers, thread_name_prefix="rot_aug_saver")

    pending = deque()
    save_futures = deque()
    next_idx = 0
    records: list[dict[str, object]] = []
    variant_vertices: list[sp.SparseTensor] = []
    variant_intersected: list[sp.SparseTensor] = []
    variant_names: list[str] = []
    variant_source_ids: list[str] = []
    variant_rotations: list[int] = []

    def submit_next() -> None:
        nonlocal next_idx
        while next_idx < len(asset_ids) and len(pending) < max(args.prefetch, args.batch_size):
            asset_id = asset_ids[next_idx]
            pending.append((asset_id, loader_pool.submit(load_voxel_item, voxel_dir / f"{asset_id}.vxz", asset_id)))
            next_idx += 1

    def drain_saves() -> None:
        while save_futures and save_futures[0].done():
            save_futures.popleft().result()

    def mark_skipped(latent_name: str, source_id: str, rotation: int, reason: str, message: str) -> None:
        payload = {
            "sha256": latent_name,
            "source_dataset": dataset_name,
            "source_asset_id": source_id,
            "rotation_degrees": int(rotation),
            "reason": reason,
            "message": message,
        }
        save_futures.append(saver_pool.submit(save_skip_marker, skip_dir / f"{latent_name}.skip.json", payload))
        drain_saves()

    def mark_asset_load_skipped(asset_id: str, message: str) -> None:
        for degrees in args.rotations:
            rotation = int(round(float(degrees))) % 360
            latent_name = augmented_asset_id(asset_id, degrees)
            if variant_terminal(latent_dir, skip_dir, latent_name) and not args.overwrite:
                continue
            mark_skipped(latent_name, asset_id, rotation, "load_error", message)

    def encode_variant_batch(
        vertices_items: list[sp.SparseTensor],
        intersected_items: list[sp.SparseTensor],
        names: list[str],
        source_ids: list[str],
        rotations: list[int],
    ) -> None:
        if not names:
            return
        try:
            vertices_cat = sparse_cat(vertices_items, dim=0)
            intersected_cat = sparse_cat(intersected_items, dim=0)
            z = encoder(vertices_cat.cuda(), intersected_cat.cuda())
            if not torch.isfinite(z.feats).all():
                if len(names) > 1:
                    mid = len(names) // 2
                    print(f"[Split] Non-finite latent batch of {len(names)} variants", flush=True)
                    encode_variant_batch(vertices_items[:mid], intersected_items[:mid], names[:mid], source_ids[:mid], rotations[:mid])
                    encode_variant_batch(vertices_items[mid:], intersected_items[mid:], names[mid:], source_ids[mid:], rotations[mid:])
                else:
                    print(f"[Skip] Non-finite latent: {names[0]}", flush=True)
                    mark_skipped(names[0], source_ids[0], rotations[0], "non_finite_latent", "Non-finite latent")
                clear_cuda_error()
                return

            z_list = sparse_unbind(z, dim=0)
            for z_item, latent_name, source_id, rotation in zip(z_list, names, source_ids, rotations):
                feats_np = z_item.feats.detach().cpu().numpy().astype(np.float32)
                coords_np = z_item.coords[:, 1:].detach().cpu().numpy().astype(np.uint8)
                save_path = latent_dir / f"{latent_name}.npz"
                save_futures.append(saver_pool.submit(save_latent, save_path, feats_np, coords_np))
                records.append(
                    {
                        "sha256": latent_name,
                        f"latent_{LATENT_KEY}": True,
                        f"{LATENT_KEY}_tokens": int(coords_np.shape[0]),
                        "source_dataset": dataset_name,
                        "source_asset_id": source_id,
                        "rotation_degrees": int(rotation),
                    }
                )
            drain_saves()
        except torch.cuda.OutOfMemoryError as exc:
            clear_cuda_error()
            if len(names) <= 1:
                print(f"[Skip] {names[0]}: CUDA OOM for single variant: {exc}", flush=True)
                mark_skipped(names[0], source_ids[0], rotations[0], "cuda_oom", str(exc))
                return
            mid = len(names) // 2
            print(f"[Split] CUDA OOM for {len(names)} variants; retrying as {mid}+{len(names) - mid}", flush=True)
            encode_variant_batch(vertices_items[:mid], intersected_items[:mid], names[:mid], source_ids[:mid], rotations[:mid])
            encode_variant_batch(vertices_items[mid:], intersected_items[mid:], names[mid:], source_ids[mid:], rotations[mid:])
        except RuntimeError as exc:
            message = str(exc)
            retryable = (
                "negative dimension" in message
                or "CUDA error" in message
                or "illegal memory access" in message
                or "out of memory" in message.lower()
            )
            if not retryable:
                raise
            clear_cuda_error()
            if len(names) <= 1:
                print(f"[Skip] {names[0]}: retryable RuntimeError for single variant: {exc}", flush=True)
                mark_skipped(names[0], source_ids[0], rotations[0], "retryable_runtime_error", str(exc))
                return
            mid = len(names) // 2
            print(f"[Split] retryable RuntimeError for {len(names)} variants; retrying as {mid}+{len(names) - mid}: {exc}", flush=True)
            encode_variant_batch(vertices_items[:mid], intersected_items[:mid], names[:mid], source_ids[:mid], rotations[:mid])
            encode_variant_batch(vertices_items[mid:], intersected_items[mid:], names[mid:], source_ids[mid:], rotations[mid:])
        finally:
            torch.cuda.empty_cache()

    def flush_batch() -> None:
        nonlocal variant_vertices, variant_intersected, variant_names, variant_source_ids, variant_rotations
        if not variant_names:
            return
        vertices_items = variant_vertices
        intersected_items = variant_intersected
        names = variant_names
        source_ids = variant_source_ids
        rotations = variant_rotations
        variant_vertices = []
        variant_intersected = []
        variant_names = []
        variant_source_ids = []
        variant_rotations = []
        try:
            encode_variant_batch(vertices_items, intersected_items, names, source_ids, rotations)
        finally:
            torch.cuda.empty_cache()

    try:
        submit_next()
        pbar = tqdm(total=len(asset_ids), desc=f"Encoding rotated {dataset_name} shape latents")
        while pending:
            pending_asset_id, future = pending.popleft()
            result = future.result()
            pbar.update(1)
            submit_next()
            if result is None:
                mark_asset_load_skipped(pending_asset_id, "load_voxel_item returned None")
                continue
            asset_id, coords, attr = result
            for degrees in args.rotations:
                latent_name = augmented_asset_id(asset_id, degrees)
                save_path = latent_dir / f"{latent_name}.npz"
                skip_path = skip_dir / f"{latent_name}.skip.json"
                if variant_terminal(latent_dir, skip_dir, latent_name) and not args.overwrite:
                    continue
                if args.overwrite and skip_path.exists():
                    skip_path.unlink()
                rotated_coords, rotated_attr = rotate_shape_ovoxel(coords, attr, degrees, args.grid_size)
                vertices, intersected = make_shape_encoder_inputs(rotated_coords, rotated_attr)
                if not (is_valid_sparse_tensor(vertices) and is_valid_sparse_tensor(intersected)):
                    print(f"[Skip] {latent_name}: NaN/Inf in rotated encoder input", flush=True)
                    mark_skipped(latent_name, asset_id, int(round(float(degrees))) % 360, "non_finite_input", "NaN/Inf in rotated encoder input")
                    continue
                variant_vertices.append(vertices)
                variant_intersected.append(intersected)
                variant_names.append(latent_name)
                variant_source_ids.append(asset_id)
                variant_rotations.append(int(round(float(degrees))) % 360)
                if len(variant_names) >= args.batch_size:
                    flush_batch()

        flush_batch()
        pbar.close()
        for future in save_futures:
            future.result()
    finally:
        loader_pool.shutdown(wait=True)
        saver_pool.shutdown(wait=True)

    part_path = part_dir / f"metadata_rank{args.rank:04d}.csv"
    write_metadata(records, part_path)
    metadata_path = None
    if args.world_size == 1:
        metadata_path = merge_metadata(output_dir)

    print(
        json.dumps(
            {
                "dataset": dataset_name,
                "encoded_variants": len(records),
                "metadata_part": str(part_path),
                "metadata": str(metadata_path) if metadata_path is not None else None,
                "latent_dir": str(latent_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
