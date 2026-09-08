#!/usr/bin/env python3
"""Generic scene Shape rot-aug preparation.

This utility prepares scene-local Shape VAE X2 rot-aug data for one resolver-backed
scene dataset without modifying the existing rot000 roots.

Stages:

1. audit: inspect split coverage.
2. encode-bg: rotate scene-background shape VXZs and encode original TRELLIS2 latents.
3. materialize-raw: hardlink/copy foreground source-cache latents into scene-local names,
   and link background rot000 from the existing scene-local TRELLIS2 root.
4. write-source-x2-manifest: write source-object X2 manifest for nonzero rotations.
5. materialize-x2-rot000: link existing Shape X2 rot000 latents into the new rot-aug root.
6. materialize-x2-objects: hardlink/copy source-object X2 rot latents scene-locally.
7. write-x2-manifest: write a JSONL manifest for remaining nonzero scene-local raw latents
   so encode_shape_vae_x2_latents.py can encode them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
OBJECT_ENCODE_DIR = REPO_ROOT / "trellis2_x2" / "data_toolkits" / "objects" / "encode"
for path in (SCRIPT_DIR, REPO_ROOT, OBJECT_ENCODE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from resolvers import (  # noqa: E402
    DEFAULT_POSTPROCESS_ROOT,
    DEFAULT_SHAPE_OBJECT_LATENT_ROOT,
    DEFAULT_TRAINING_ROOT,
    DATASET_FOLDERS,
    SHAPE_OBJECT_LATENT_KEY,
    LatentItem,
    PathConfig,
    build_manifest,
    normalize_dataset_name,
    read_split_records,
    shard_by_index,
    unique_items_by_target,
)


DEFAULT_SPLIT_JSON = Path(
    os.environ.get(
        "FIRE3D_SHAPE_SCENE_SPLIT",
        REPO_ROOT / "data/training_manifests/shape_scene_train.json",
    )
)
DEFAULT_SCENE_RAW_ROT_NAME = "trellis2_shape_latents_rot_aug"
DEFAULT_SCENE_X2_ROT_NAME = "shape_hcvae_latents_rotated"
DEFAULT_SCENE_X2_NAME = "shape_hcvae_latents"
DEFAULT_SOURCE_X2_ROOT = Path(
    os.environ.get(
        "FIRE3D_SHAPE_HCVAE_LATENT_ROOT",
        REPO_ROOT / "data/training_objects/shape_hcvae_latents",
    )
)
DEFAULT_SOURCE_X2_LATENT_KEY = "trellis2_shape_x2_encoding"
DEFAULT_ERROR_LIMIT = 20


@dataclass(frozen=True)
class ScenePaths:
    training_root: Path
    postprocess_root: Path
    source_rot_root: Path
    source_x2_root: Path
    scene_raw_rot_root: Path
    scene_x2_root: Path
    scene_x2_rot_root: Path


def normalize_rotation_label(value: str | int | float) -> str:
    text = str(value)
    if text.startswith("rot"):
        text = text[3:]
    try:
        degrees = int(round(float(text))) % 360
    except ValueError as exc:
        raise ValueError(f"invalid rotation label: {value!r}") from exc
    if degrees not in {0, 90, 180, 270}:
        raise ValueError(f"only 0/90/180/270 rotations are supported, got {value!r}")
    return f"{degrees:03d}"


def parse_rotations(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or value == "":
        return list(default)
    labels = []
    seen = set()
    for part in value.replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        label = normalize_rotation_label(part)
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
    if not labels:
        raise ValueError("rotation list is empty")
    return labels


def source_rot_latent_path(paths: ScenePaths, source_dataset: str, source_key: str, rotation: str) -> Path:
    return (
        paths.source_rot_root
        / source_dataset
        / "latents"
        / SHAPE_OBJECT_LATENT_KEY
        / f"{source_key}__rot{rotation}.npz"
    )


def source_x2_rot_latent_path(paths: ScenePaths, source_dataset: str, source_key: str, rotation: str) -> Path:
    return (
        paths.source_x2_root
        / source_dataset
        / "latents"
        / DEFAULT_SOURCE_X2_LATENT_KEY
        / f"{source_key}__rot{rotation}.npz"
    )


def scene_raw_rot_path(paths: ScenePaths, item: LatentItem, rotation: str) -> Path:
    return paths.scene_raw_rot_root / item.scene_id / f"{item.local_name}__rot{rotation}.npz"


def scene_x2_path(paths: ScenePaths, item: LatentItem) -> Path:
    return paths.scene_x2_root / item.scene_id / f"{item.local_name}.npz"


def scene_x2_rot_path(paths: ScenePaths, item: LatentItem, rotation: str) -> Path:
    return paths.scene_x2_rot_root / item.scene_id / f"{item.local_name}__rot{rotation}.npz"


def is_valid_npz(path: Path, *, coord_max_exclusive: int | None = None) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as data:
            valid = (
                "feats" in data
                and "coords" in data
                and data["feats"].ndim == 2
                and data["coords"].ndim == 2
                and data["feats"].shape[0] == data["coords"].shape[0]
                and data["feats"].shape[0] > 0
            )
            if not valid:
                return False
            if coord_max_exclusive is not None:
                coords = np.asarray(data["coords"])
                spatial_coords = coords[:, -3:]
                if spatial_coords.size and (
                    spatial_coords.min() < 0
                    or spatial_coords.max() >= coord_max_exclusive
                ):
                    return False
            return True
    except Exception:
        return False


def target_exists(path: Path, validate: bool) -> bool:
    if path.with_suffix(path.suffix + ".skip.json").exists():
        return True
    return is_valid_npz(path) if validate else path.exists()


def parse_datasets(value: str) -> list[str]:
    datasets = []
    seen = set()
    for part in str(value).replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        name = normalize_dataset_name(part)
        if name in seen:
            continue
        seen.add(name)
        datasets.append(name)
    if not datasets:
        raise ValueError("--dataset must contain at least one dataset")
    return datasets


def scene_filter(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def build_scene_items(args: argparse.Namespace) -> tuple[list[LatentItem], list[str], list[str]]:
    datasets = parse_datasets(args.dataset)
    selected_scene_ids = set(getattr(args, "scene_id", []) or [])
    selected_scene_ids.update(scene_filter(getattr(args, "scene_list", None)))
    records = read_split_records(
        args.split_json,
        datasets=datasets,
        scene_ids=selected_scene_ids or None,
        limit_scenes=args.limit_scenes,
    )
    config = PathConfig(
        training_root=args.training_root,
        postprocess_root=args.postprocess_root,
        split_json=args.split_json,
        shape_object_latent_root=args.source_rot_root,
        object_cache_rotation="000",
        scene_bg_only=bool(getattr(args, "scene_bg_only", False)),
        strict=False,
    )
    manifest = build_manifest(records, config)
    items = sorted(manifest.items, key=lambda item: (item.scene_id, item.local_name))
    return items, manifest.errors, manifest.warnings


def unique_source_object_items(items: Iterable[LatentItem]) -> list[LatentItem]:
    return unique_items_by_target([item for item in items if item.is_object], "shape", use_cache=True)


def make_paths(args: argparse.Namespace) -> ScenePaths:
    datasets = parse_datasets(args.dataset)
    if len(datasets) != 1:
        raise ValueError("This utility expects exactly one scene dataset per run")
    scene_root = args.training_root / DATASET_FOLDERS[datasets[0]]
    scene_raw_rot_root = args.scene_raw_rot_root or scene_root / DEFAULT_SCENE_RAW_ROT_NAME
    scene_x2_root = args.scene_x2_root or scene_root / DEFAULT_SCENE_X2_NAME
    scene_x2_rot_root = args.scene_x2_rot_root or scene_root / DEFAULT_SCENE_X2_ROT_NAME
    return ScenePaths(
        training_root=args.training_root,
        postprocess_root=args.postprocess_root,
        source_rot_root=args.source_rot_root,
        source_x2_root=args.source_x2_root,
        scene_raw_rot_root=scene_raw_rot_root,
        scene_x2_root=scene_x2_root,
        scene_x2_rot_root=scene_x2_rot_root,
    )


def hardlink_or_copy(src: Path, dst: Path, *, overwrite: bool = False) -> bool:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return False
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return True


def count_existing(paths: Iterable[Path], validate: bool) -> int:
    return sum(1 for path in paths if target_exists(path, validate))


def audit(args: argparse.Namespace) -> int:
    rotations = parse_rotations(args.rotations, ("000", "090", "180", "270"))
    paths = make_paths(args)
    items, manifest_errors, manifest_warnings = build_scene_items(args)
    object_items = [item for item in items if item.is_object]
    bg_items = [item for item in items if item.is_scene_bg]
    source_items = unique_source_object_items(items)

    source_missing_vxz = [item for item in source_items if not item.shape_vxz.exists()]
    bg_missing_vxz = [item for item in bg_items if not item.shape_vxz.exists()]
    current_raw_missing = [item for item in items if not item.shape_output.exists()]
    current_x2_missing = [item for item in items if not scene_x2_path(paths, item).exists()]

    per_rotation: dict[str, dict[str, int]] = {}
    missing_source_by_rotation: dict[str, list[str]] = {}
    for rotation in rotations:
        source_paths = [
            source_rot_latent_path(paths, str(item.source_dataset), str(item.source_key), rotation)
            for item in source_items
            if item.source_key is not None
        ]
        object_scene_paths = [scene_raw_rot_path(paths, item, rotation) for item in object_items]
        bg_scene_paths = [scene_raw_rot_path(paths, item, rotation) for item in bg_items]
        x2_paths = [scene_x2_rot_path(paths, item, rotation) for item in items]
        source_x2_paths = [
            source_x2_rot_latent_path(paths, str(item.source_dataset), str(item.source_key), rotation)
            for item in source_items
            if item.source_key is not None
        ]

        missing_sources = [
            str(item.source_key)
            for item in source_items
            if item.source_key is not None and not source_rot_latent_path(paths, str(item.source_dataset), str(item.source_key), rotation).exists()
        ]
        missing_source_by_rotation[rotation] = missing_sources
        per_rotation[rotation] = {
            "source_raw_unique_required": len(source_paths),
            "source_raw_unique_existing": count_existing(source_paths, args.validate_npz),
            "source_raw_unique_missing": len(missing_sources),
            "scene_raw_object_required": len(object_scene_paths),
            "scene_raw_object_existing": count_existing(object_scene_paths, args.validate_npz),
            "scene_raw_bg_required": len(bg_scene_paths),
            "scene_raw_bg_existing": count_existing(bg_scene_paths, args.validate_npz),
            "scene_raw_total_required": len(object_scene_paths) + len(bg_scene_paths),
            "scene_raw_total_existing": count_existing(object_scene_paths + bg_scene_paths, args.validate_npz),
            "scene_x2_total_required": len(x2_paths),
            "scene_x2_total_existing": count_existing(x2_paths, args.validate_npz),
            "source_x2_unique_required": len(source_x2_paths),
            "source_x2_unique_existing": count_existing(source_x2_paths, args.validate_npz),
        }

    source_counts = Counter(item.source_dataset for item in source_items)
    summary = {
        "dataset": parse_datasets(args.dataset)[0],
        "split_json": str(args.split_json),
        "training_root": str(args.training_root),
        "scene_raw_rot_root": str(paths.scene_raw_rot_root),
        "scene_x2_rot_root": str(paths.scene_x2_rot_root),
        "rotations": rotations,
        "items": len(items),
        "scenes": len({item.scene_id for item in items}),
        "object_items": len(object_items),
        "background_items": len(bg_items),
        "unique_source_objects": len(source_items),
        "unique_source_by_dataset": dict(sorted(source_counts.items())),
        "manifest_errors": len(manifest_errors),
        "manifest_warnings": len(manifest_warnings),
        "missing_unique_source_vxz": len(source_missing_vxz),
        "missing_background_vxz": len(bg_missing_vxz),
        "missing_current_scene_raw": len(current_raw_missing),
        "missing_current_scene_x2": len(current_x2_missing),
        "per_rotation": per_rotation,
        "examples": {
            "manifest_errors": manifest_errors[: args.max_errors],
            "manifest_warnings": manifest_warnings[: args.max_errors],
            "missing_unique_source_vxz": [f"{item.scene_id}/{item.local_name}->{item.shape_vxz}" for item in source_missing_vxz[: args.max_errors]],
            "missing_background_vxz": [f"{item.scene_id}/{item.local_name}->{item.shape_vxz}" for item in bg_missing_vxz[: args.max_errors]],
            "missing_current_scene_raw": [f"{item.scene_id}/{item.local_name}->{item.shape_output}" for item in current_raw_missing[: args.max_errors]],
            "missing_current_scene_x2": [f"{item.scene_id}/{item.local_name}->{scene_x2_path(paths, item)}" for item in current_x2_missing[: args.max_errors]],
        },
    }

    if args.asset_list_dir is not None:
        args.asset_list_dir.mkdir(parents=True, exist_ok=True)
        required_path = args.asset_list_dir / "required_source_keys.txt"
        required_path.write_text("".join(f"{item.source_key}\n" for item in source_items if item.source_key is not None))
        summary["required_source_list"] = str(required_path)
        missing_paths = {}
        for rotation, missing_sources in missing_source_by_rotation.items():
            path = args.asset_list_dir / f"missing_source_rot{rotation}.txt"
            path.write_text("".join(f"{source}\n" for source in missing_sources))
            missing_paths[rotation] = str(path)
        summary["missing_source_lists"] = missing_paths

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not manifest_errors or not args.strict else 2


def lazy_import_rot_encoder():
    import torch
    import o_voxel
    from shape_enc_rot_aug_seq import (  # type: ignore
        configure_hf_token,
        is_valid_sparse_tensor,
        make_shape_encoder_inputs,
        models,
        rotate_shape_ovoxel,
        sparse_cat,
        sparse_unbind,
    )

    return {
        "torch": torch,
        "o_voxel": o_voxel,
        "configure_hf_token": configure_hf_token,
        "is_valid_sparse_tensor": is_valid_sparse_tensor,
        "make_shape_encoder_inputs": make_shape_encoder_inputs,
        "models": models,
        "rotate_shape_ovoxel": rotate_shape_ovoxel,
        "sparse_cat": sparse_cat,
        "sparse_unbind": sparse_unbind,
    }


def save_npz_atomic(path: Path, feats: np.ndarray, coords: np.ndarray, compressed: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("wb") as f:
        if compressed:
            np.savez_compressed(f, feats=feats, coords=coords)
        else:
            np.savez(f, feats=feats, coords=coords)
    os.replace(tmp_path, path)


def write_skip(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_path, path)


def encode_bg(args: argparse.Namespace) -> int:
    rotations = parse_rotations(args.rotations, ("090", "180", "270"))
    paths = make_paths(args)
    items, manifest_errors, _manifest_warnings = build_scene_items(args)
    if manifest_errors and args.strict:
        print(json.dumps({"manifest_errors": manifest_errors[: args.max_errors]}, indent=2), file=sys.stderr)
        return 2

    bg_items = [item for item in items if item.is_scene_bg]
    bg_items = shard_by_index(bg_items, args.rank, args.world_size)
    if args.max_items is not None:
        bg_items = bg_items[: args.max_items]

    rot = lazy_import_rot_encoder()
    torch = rot["torch"]
    if args.torch_threads is not None and int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    if args.torch_interop_threads is not None and int(args.torch_interop_threads) > 0:
        try:
            torch.set_num_interop_threads(int(args.torch_interop_threads))
        except RuntimeError:
            pass
    o_voxel = rot["o_voxel"]
    rot["configure_hf_token"](args.hf_token_file)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for background TRELLIS2 shape encoding")
    encoder = rot["models"].from_pretrained(args.enc_pretrained).eval().to(device)

    processed = 0
    encoded = 0
    skipped_existing = 0
    failed = 0
    pending_vertices = []
    pending_intersected = []
    pending_meta: list[tuple[LatentItem, str, Path]] = []

    def write_variant_skip(item: LatentItem, rotation: str, target: Path, error: str, reason: str) -> None:
        nonlocal failed
        failed += 1
        write_skip(
            target.with_suffix(target.suffix + ".skip.json"),
            {
                "dataset_name": item.dataset_name,
                "scene_id": item.scene_id,
                "local_name": item.local_name,
                "kind": item.kind,
                "rotation": rotation,
                "source_vxz": str(item.shape_vxz),
                "target": str(target),
                "reason": reason,
                "error": error,
            },
        )
        if getattr(args, "fail_on_encode_errors", False):
            raise RuntimeError(f"{item.scene_id}/{item.local_name} rot{rotation}: {error}")

    def encode_batch(vertices_list, intersected_list, meta, *, allow_split: bool) -> None:
        nonlocal encoded
        try:
            vertices_cat = rot["sparse_cat"](vertices_list, dim=0).to(device)
            intersected_cat = rot["sparse_cat"](intersected_list, dim=0).to(device)
            z = encoder(vertices_cat, intersected_cat)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if not torch.isfinite(z.feats).all().item():
                raise ValueError("non-finite encoded background batch")
            # Split on the producing device. Moving the whole sparse batch to CPU
            # first can trip sparse layout code on large or edge-case batches.
            z_items = [z] if len(meta) == 1 else rot["sparse_unbind"](z, dim=0)
            for z_item, (item, rotation, target) in zip(z_items, meta, strict=True):
                z_cpu = z_item.to("cpu", non_blocking=True)
                feats = z_cpu.feats.detach().numpy().astype(np.float32)
                coords_int = z_cpu.coords[:, 1:].detach().numpy()
                if feats.shape[0] == 0:
                    raise ValueError(f"empty encoded raw latent: {item.scene_id}/{item.local_name} rot{rotation}")
                if coords_int.ndim != 2 or coords_int.shape[1] != 3:
                    raise ValueError(
                        f"invalid encoded coordinate shape {coords_int.shape}: "
                        f"{item.scene_id}/{item.local_name} rot{rotation}"
                    )
                if coords_int.size and (coords_int.min() < 0 or coords_int.max() >= 32):
                    raise ValueError(
                        f"encoded raw coordinates outside [0,31]: "
                        f"[{coords_int.min()},{coords_int.max()}] for "
                        f"{item.scene_id}/{item.local_name} rot{rotation}"
                    )
                coords = coords_int.astype(np.uint8, copy=False)
                save_npz_atomic(target, feats, coords, compressed=not args.uncompressed)
                encoded += 1
        except Exception as exc:  # noqa: BLE001
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if allow_split and len(meta) > 1:
                print(f"[warn] batch encode failed for {len(meta)} variants; retrying singly: {exc!r}", flush=True)
                for one_vertices, one_intersected, one_meta in zip(vertices_list, intersected_list, meta, strict=True):
                    encode_batch([one_vertices], [one_intersected], [one_meta], allow_split=False)
                return
            for item, rotation, target in meta:
                write_variant_skip(item, rotation, target, repr(exc), reason="encode_error")

    def flush_batch() -> None:
        nonlocal pending_vertices, pending_intersected, pending_meta
        if not pending_meta:
            return
        try:
            encode_batch(pending_vertices, pending_intersected, pending_meta, allow_split=True)
        finally:
            pending_vertices = []
            pending_intersected = []
            pending_meta = []
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for item in bg_items:
        targets: list[tuple[str, Path]] = []
        for rotation in rotations:
            target = scene_raw_rot_path(paths, item, rotation)
            skip_path = target.with_suffix(target.suffix + ".skip.json")
            processed += 1
            if not args.overwrite and (
                is_valid_npz(target, coord_max_exclusive=32) or skip_path.exists()
            ):
                skipped_existing += 1
                continue
            if args.overwrite and skip_path.exists():
                skip_path.unlink()
            targets.append((rotation, target))
        if not targets:
            continue
        if not item.shape_vxz.exists():
            print(f"[missing bg vxz] {item.scene_id}/{item.local_name}: {item.shape_vxz}", flush=True)
            for rotation, target in targets:
                write_variant_skip(item, rotation, target, f"missing shape vxz: {item.shape_vxz}", reason="missing_shape_vxz")
            continue
        try:
            coords, attr = o_voxel.io.read_vxz(str(item.shape_vxz), num_threads=args.vxz_threads)
        except Exception as exc:  # noqa: BLE001
            print(f"[load error] {item.scene_id}/{item.local_name}: {exc!r}", flush=True)
            for rotation, target in targets:
                write_variant_skip(item, rotation, target, repr(exc), reason="load_error")
            continue
        for rotation, target in targets:
            try:
                rotated_coords, rotated_attr = rot["rotate_shape_ovoxel"](
                    coords,
                    attr,
                    int(rotation),
                    args.grid_size,
                )
                vertices, intersected = rot["make_shape_encoder_inputs"](rotated_coords, rotated_attr)
                if not (rot["is_valid_sparse_tensor"](vertices) and rot["is_valid_sparse_tensor"](intersected)):
                    raise ValueError("non-finite rotated shape encoder input")
                pending_vertices.append(vertices)
                pending_intersected.append(intersected)
                pending_meta.append((item, rotation, target))
                if len(pending_meta) >= args.batch_size:
                    flush_batch()
            except Exception as exc:  # noqa: BLE001
                write_variant_skip(item, rotation, target, repr(exc), reason="prepare_error")

    flush_batch()
    summary = {
        "stage": "encode-bg",
        "rank": args.rank,
        "world_size": args.world_size,
        "rotations": rotations,
        "background_items": len(bg_items),
        "processed_variants": processed,
        "encoded": encoded,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "torch_threads": None if args.torch_threads is None else int(args.torch_threads),
        "torch_interop_threads": None if args.torch_interop_threads is None else int(args.torch_interop_threads),
        "fail_on_encode_errors": bool(getattr(args, "fail_on_encode_errors", False)),
        "scene_raw_rot_root": str(paths.scene_raw_rot_root),
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if failed == 0 or not getattr(args, "fail_on_encode_errors", False) else 2


def materialize_raw(args: argparse.Namespace) -> int:
    rotations = parse_rotations(args.rotations, ("000", "090", "180", "270"))
    paths = make_paths(args)
    items, manifest_errors, _manifest_warnings = build_scene_items(args)
    if manifest_errors and args.strict:
        print(json.dumps({"manifest_errors": manifest_errors[: args.max_errors]}, indent=2), file=sys.stderr)
        return 2

    object_items = [item for item in items if item.is_object]
    bg_items = [item for item in items if item.is_scene_bg]
    tasks: list[tuple[Path, Path, str, str, str]] = []

    for item in object_items:
        if item.source_key is None:
            continue
        for rotation in rotations:
            src = source_rot_latent_path(paths, str(item.source_dataset), str(item.source_key), rotation)
            dst = scene_raw_rot_path(paths, item, rotation)
            tasks.append((src, dst, item.scene_id, item.local_name, rotation))

    if "000" in rotations:
        for item in bg_items:
            tasks.append((item.shape_output, scene_raw_rot_path(paths, item, "000"), item.scene_id, item.local_name, "000"))

    tasks = shard_by_index(tasks, args.rank, args.world_size)
    if args.max_items is not None:
        tasks = tasks[: args.max_items]

    written = 0
    skipped = 0
    failed = 0
    errors = []

    def run_task(task: tuple[Path, Path, str, str, str]) -> str:
        src, dst, scene_id, local_name, rotation = task
        try:
            did_write = hardlink_or_copy(src, dst, overwrite=args.overwrite)
            return "written" if did_write else "skipped"
        except Exception as exc:  # noqa: BLE001
            return json.dumps(
                {
                    "src": str(src),
                    "dst": str(dst),
                    "scene_id": scene_id,
                    "local_name": local_name,
                    "rotation": rotation,
                    "error": repr(exc),
                },
                sort_keys=True,
            )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(run_task, tasks):
            if result == "written":
                written += 1
            elif result == "skipped":
                skipped += 1
            else:
                failed += 1
                if len(errors) < args.max_errors:
                    errors.append(json.loads(result))

    summary = {
        "stage": "materialize-raw",
        "rank": args.rank,
        "world_size": args.world_size,
        "rotations": rotations,
        "tasks": len(tasks),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "errors": errors,
        "scene_raw_rot_root": str(paths.scene_raw_rot_root),
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if failed == 0 or not args.strict else 2


def materialize_x2_rot000(args: argparse.Namespace) -> int:
    paths = make_paths(args)
    items, manifest_errors, _manifest_warnings = build_scene_items(args)
    if manifest_errors and args.strict:
        print(json.dumps({"manifest_errors": manifest_errors[: args.max_errors]}, indent=2), file=sys.stderr)
        return 2

    tasks = [(scene_x2_path(paths, item), scene_x2_rot_path(paths, item, "000"), item.scene_id, item.local_name) for item in items]
    tasks = shard_by_index(tasks, args.rank, args.world_size)
    if args.max_items is not None:
        tasks = tasks[: args.max_items]

    written = 0
    skipped = 0
    failed = 0
    errors = []

    def run_task(task: tuple[Path, Path, str, str]) -> str:
        src, dst, scene_id, local_name = task
        try:
            did_write = hardlink_or_copy(src, dst, overwrite=args.overwrite)
            return "written" if did_write else "skipped"
        except Exception as exc:  # noqa: BLE001
            return json.dumps(
                {
                    "src": str(src),
                    "dst": str(dst),
                    "scene_id": scene_id,
                    "local_name": local_name,
                    "rotation": "000",
                    "error": repr(exc),
                },
                sort_keys=True,
            )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(run_task, tasks):
            if result == "written":
                written += 1
            elif result == "skipped":
                skipped += 1
            else:
                failed += 1
                if len(errors) < args.max_errors:
                    errors.append(json.loads(result))

    summary = {
        "stage": "materialize-x2-rot000",
        "rank": args.rank,
        "world_size": args.world_size,
        "tasks": len(tasks),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "errors": errors,
        "scene_x2_rot_root": str(paths.scene_x2_rot_root),
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if failed == 0 or not args.strict else 2


def write_source_x2_manifest(args: argparse.Namespace) -> int:
    rotations = parse_rotations(args.rotations, ("090", "180", "270"))
    paths = make_paths(args)
    items, manifest_errors, _manifest_warnings = build_scene_items(args)
    if manifest_errors and args.strict:
        print(json.dumps({"manifest_errors": manifest_errors[: args.max_errors]}, indent=2), file=sys.stderr)
        return 2

    source_items = unique_source_object_items(items)
    entries = []
    missing_inputs = []
    skipped_existing = 0
    skipped_raw_skip_markers = 0
    for item in source_items:
        if item.source_key is None:
            continue
        for rotation in rotations:
            input_path = source_rot_latent_path(paths, str(item.source_dataset), str(item.source_key), rotation)
            output_path = source_x2_rot_latent_path(paths, str(item.source_dataset), str(item.source_key), rotation)
            if not input_path.exists() and not args.include_missing_inputs:
                if input_path.with_suffix(input_path.suffix + ".skip.json").exists():
                    skipped_raw_skip_markers += 1
                    continue
                if len(missing_inputs) < args.max_errors:
                    missing_inputs.append(str(input_path))
                continue
            if output_path.exists() and args.skip_existing_outputs:
                skipped_existing += 1
                continue
            entries.append(
                {
                    "kind": "object",
                    "dataset": str(item.source_dataset),
                    "name": f"{item.source_dataset}/{item.source_key}__rot{rotation}.npz",
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                }
            )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

    summary = {
        "stage": "write-source-x2-manifest",
        "manifest": str(args.manifest),
        "rotations": rotations,
        "unique_source_items": len(source_items),
        "entries": len(entries),
        "missing_inputs_examples": missing_inputs,
        "missing_inputs_truncated": len(missing_inputs) >= args.max_errors,
        "skipped_existing_outputs": skipped_existing,
        "skipped_raw_skip_markers": skipped_raw_skip_markers,
        "source_rot_root": str(paths.source_rot_root),
        "source_x2_root": str(paths.source_x2_root),
    }
    summary_path = args.manifest.with_suffix(args.manifest.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if missing_inputs and args.strict:
        return 2
    return 0


def materialize_x2_objects(args: argparse.Namespace) -> int:
    rotations = parse_rotations(args.rotations, ("090", "180", "270"))
    paths = make_paths(args)
    items, manifest_errors, _manifest_warnings = build_scene_items(args)
    if manifest_errors and args.strict:
        print(json.dumps({"manifest_errors": manifest_errors[: args.max_errors]}, indent=2), file=sys.stderr)
        return 2

    object_items = [item for item in items if item.is_object]
    tasks: list[tuple[Path, Path, str, str, str, str]] = []
    for item in object_items:
        if item.source_key is None:
            continue
        for rotation in rotations:
            src = source_x2_rot_latent_path(paths, str(item.source_dataset), str(item.source_key), rotation)
            dst = scene_x2_rot_path(paths, item, rotation)
            tasks.append((src, dst, item.scene_id, item.local_name, str(item.source_key), rotation))

    tasks = shard_by_index(tasks, args.rank, args.world_size)
    if args.max_items is not None:
        tasks = tasks[: args.max_items]

    written = 0
    skipped = 0
    failed = 0
    errors = []

    def run_task(task: tuple[Path, Path, str, str, str, str]) -> str:
        src, dst, scene_id, local_name, source_key, rotation = task
        try:
            did_write = hardlink_or_copy(src, dst, overwrite=args.overwrite)
            return "written" if did_write else "skipped"
        except Exception as exc:  # noqa: BLE001
            return json.dumps(
                {
                    "src": str(src),
                    "dst": str(dst),
                    "scene_id": scene_id,
                    "local_name": local_name,
                    "source_key": source_key,
                    "rotation": rotation,
                    "error": repr(exc),
                },
                sort_keys=True,
            )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(run_task, tasks):
            if result == "written":
                written += 1
            elif result == "skipped":
                skipped += 1
            else:
                failed += 1
                if len(errors) < args.max_errors:
                    errors.append(json.loads(result))

    summary = {
        "stage": "materialize-x2-objects",
        "rank": args.rank,
        "world_size": args.world_size,
        "rotations": rotations,
        "tasks": len(tasks),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "errors": errors,
        "source_x2_root": str(paths.source_x2_root),
        "scene_x2_rot_root": str(paths.scene_x2_rot_root),
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if failed == 0 or not args.strict else 2


def write_x2_manifest(args: argparse.Namespace) -> int:
    rotations = parse_rotations(args.rotations, ("090", "180", "270"))
    paths = make_paths(args)
    items, manifest_errors, _manifest_warnings = build_scene_items(args)
    if manifest_errors and args.strict:
        print(json.dumps({"manifest_errors": manifest_errors[: args.max_errors]}, indent=2), file=sys.stderr)
        return 2

    entries = []
    missing_inputs = []
    skipped_existing = 0
    skipped_raw_skip_markers = 0
    for item in items:
        for rotation in rotations:
            input_path = scene_raw_rot_path(paths, item, rotation)
            output_path = scene_x2_rot_path(paths, item, rotation)
            if not input_path.exists() and not args.include_missing_inputs:
                if input_path.with_suffix(input_path.suffix + ".skip.json").exists():
                    skipped_raw_skip_markers += 1
                    continue
                if len(missing_inputs) < args.max_errors:
                    missing_inputs.append(str(input_path))
                continue
            if output_path.exists() and args.skip_existing_outputs:
                skipped_existing += 1
                continue
            entries.append(
                {
                    "kind": "scene",
                    "dataset": item.dataset_name,
                    "name": f"{item.dataset_name}/{item.scene_id}/{item.local_name}__rot{rotation}.npz",
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                }
            )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

    summary = {
        "stage": "write-x2-manifest",
        "manifest": str(args.manifest),
        "rotations": rotations,
        "entries": len(entries),
        "missing_inputs_examples": missing_inputs,
        "missing_inputs_truncated": len(missing_inputs) >= args.max_errors,
        "skipped_existing_outputs": skipped_existing,
        "skipped_raw_skip_markers": skipped_raw_skip_markers,
        "scene_raw_rot_root": str(paths.scene_raw_rot_root),
        "scene_x2_rot_root": str(paths.scene_x2_rot_root),
    }
    summary_path = args.manifest.with_suffix(args.manifest.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if missing_inputs and args.strict:
        return 2
    return 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="SceneSmith")
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--postprocess-root", type=Path, default=DEFAULT_POSTPROCESS_ROOT)
    parser.add_argument("--source-rot-root", type=Path, default=DEFAULT_SHAPE_OBJECT_LATENT_ROOT)
    parser.add_argument("--source-x2-root", type=Path, default=DEFAULT_SOURCE_X2_ROOT)
    parser.add_argument("--scene-raw-rot-root", type=Path, default=None)
    parser.add_argument("--scene-x2-root", type=Path, default=None)
    parser.add_argument("--scene-x2-rot-root", type=Path, default=None)
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--scene-id", action="append", default=[])
    parser.add_argument("--scene-list", type=Path, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-errors", type=int, default=DEFAULT_ERROR_LIMIT)
    parser.add_argument("--scene-bg-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    audit_parser = subparsers.add_parser("audit")
    add_common_args(audit_parser)
    audit_parser.add_argument("--rotations", default="000,090,180,270")
    audit_parser.add_argument("--validate-npz", action="store_true")
    audit_parser.add_argument("--asset-list-dir", type=Path, default=None)
    audit_parser.set_defaults(func=audit)

    bg_parser = subparsers.add_parser("encode-bg")
    add_common_args(bg_parser)
    bg_parser.add_argument("--rotations", default="090,180,270")
    bg_parser.add_argument("--enc-pretrained", default="microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16")
    bg_parser.add_argument("--hf-token-file", type=Path, default=None)
    bg_parser.add_argument("--grid-size", type=int, default=512)
    bg_parser.add_argument("--batch-size", type=int, default=8)
    bg_parser.add_argument("--device", default="cuda:0")
    bg_parser.add_argument("--rank", type=int, default=0)
    bg_parser.add_argument("--world-size", type=int, default=1)
    bg_parser.add_argument("--vxz-threads", type=int, default=2)
    bg_parser.add_argument("--torch-threads", type=int, default=None)
    bg_parser.add_argument("--torch-interop-threads", type=int, default=1)
    bg_parser.add_argument("--overwrite", action="store_true")
    bg_parser.add_argument("--uncompressed", action="store_true")
    bg_parser.add_argument("--fail-on-encode-errors", action="store_true")
    bg_parser.set_defaults(func=encode_bg)

    mat_parser = subparsers.add_parser("materialize-raw")
    add_common_args(mat_parser)
    mat_parser.add_argument("--rotations", default="000,090,180,270")
    mat_parser.add_argument("--rank", type=int, default=0)
    mat_parser.add_argument("--world-size", type=int, default=1)
    mat_parser.add_argument("--workers", type=int, default=16)
    mat_parser.add_argument("--overwrite", action="store_true")
    mat_parser.set_defaults(func=materialize_raw)

    x2_000_parser = subparsers.add_parser("materialize-x2-rot000")
    add_common_args(x2_000_parser)
    x2_000_parser.add_argument("--rank", type=int, default=0)
    x2_000_parser.add_argument("--world-size", type=int, default=1)
    x2_000_parser.add_argument("--workers", type=int, default=16)
    x2_000_parser.add_argument("--overwrite", action="store_true")
    x2_000_parser.set_defaults(func=materialize_x2_rot000)

    source_x2_manifest_parser = subparsers.add_parser("write-source-x2-manifest")
    add_common_args(source_x2_manifest_parser)
    source_x2_manifest_parser.add_argument("--rotations", default="090,180,270")
    source_x2_manifest_parser.add_argument("--manifest", type=Path, required=True)
    source_x2_manifest_parser.add_argument("--include-missing-inputs", action="store_true")
    source_x2_manifest_parser.add_argument("--skip-existing-outputs", action=argparse.BooleanOptionalAction, default=True)
    source_x2_manifest_parser.set_defaults(func=write_source_x2_manifest)

    x2_objects_parser = subparsers.add_parser("materialize-x2-objects")
    add_common_args(x2_objects_parser)
    x2_objects_parser.add_argument("--rotations", default="090,180,270")
    x2_objects_parser.add_argument("--rank", type=int, default=0)
    x2_objects_parser.add_argument("--world-size", type=int, default=1)
    x2_objects_parser.add_argument("--workers", type=int, default=16)
    x2_objects_parser.add_argument("--overwrite", action="store_true")
    x2_objects_parser.set_defaults(func=materialize_x2_objects)

    x2_manifest_parser = subparsers.add_parser("write-x2-manifest")
    add_common_args(x2_manifest_parser)
    x2_manifest_parser.add_argument("--rotations", default="090,180,270")
    x2_manifest_parser.add_argument("--manifest", type=Path, required=True)
    x2_manifest_parser.add_argument("--include-missing-inputs", action="store_true")
    x2_manifest_parser.add_argument("--skip-existing-outputs", action=argparse.BooleanOptionalAction, default=True)
    x2_manifest_parser.set_defaults(func=write_x2_manifest)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "rank") and (args.rank < 0 or args.rank >= args.world_size):
        parser.error("--rank must satisfy 0 <= rank < --world-size")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
