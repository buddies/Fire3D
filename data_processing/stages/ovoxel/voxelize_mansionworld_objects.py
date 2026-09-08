#!/usr/bin/env python3
"""Voxelize normalized MansionWorld object assets into <MansionWorld>/ovoxels.

Shared Blender dumping and VXZ writing helpers live in utils.py. MansionWorld
canonical asset loading lives in mansionworld_utils.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

from mansionworld_utils import (
    DEFAULT_AI2THOR_ASSET_DIR,
    DEFAULT_MANSIONWORLD_ROOT,
    configure_mansionworld_import,
    load_canonical_asset_mesh,
    wait_for_dependencies,
)
from utils import (
    DEFAULT_BLENDER_PATH,
    DEFAULT_DEPENDENCY_ROOT,
    apply_local_z_rotation,
    dual_grid_from_dump,
    dump_with_blender,
    pbr_from_dump,
    rotation_degrees_label,
    save_glb,
    vxz_is_valid,
    write_records,
)


def collect_asset_ids(scene_data_dir: Path) -> list[str]:
    asset_ids: set[str] = set()

    def visit(value):
        if isinstance(value, dict):
            asset_id = value.get("assetId")
            if isinstance(asset_id, str) and asset_id:
                asset_ids.add(asset_id)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for scene_path in sorted(scene_data_dir.glob("*/floor_*.json")):
        with scene_path.open() as f:
            visit(json.load(f))
    return sorted(asset_ids)


def safe_name(asset_id: str) -> str:
    return quote(asset_id, safe="")


def output_dir_for_rotation(base_output_dir: Path, degrees: float) -> Path:
    normalized = float(degrees) % 360.0
    if abs(normalized) < 1e-6:
        return base_output_dir
    return base_output_dir.parent / f"{base_output_dir.name}_{rotation_degrees_label(normalized)}"


def debug_dir_for_rotation(base_debug_dir: Path | None, degrees: float) -> Path | None:
    if base_debug_dir is None:
        return None
    normalized = float(degrees) % 360.0
    if abs(normalized) < 1e-6:
        return base_debug_dir
    return base_debug_dir.parent / f"{base_debug_dir.name}_{rotation_degrees_label(normalized)}"


def normalized_rotation_list(values: list[float]) -> list[float]:
    rotations: list[float] = []
    seen: set[str] = set()
    for value in values:
        normalized = float(value) % 360.0
        if abs(normalized - 360.0) < 1e-6:
            normalized = 0.0
        label = rotation_degrees_label(normalized)
        if label not in seen:
            seen.add(label)
            rotations.append(normalized)
    return rotations


def process_asset(asset_id: str, args: argparse.Namespace) -> dict[str, object]:
    safe = safe_name(asset_id)
    rotation = float(args.local_z_rotation_degrees)
    rotation_label = rotation_degrees_label(rotation)
    output = {
        "asset_id": asset_id,
        "safe_name": safe,
        "status": "started",
        "local_z_rotation_degrees": rotation,
        "rotation_label": rotation_label,
    }
    shape_path = args.output_dir / "shape" / f"{safe}.vxz"
    pbr_path = args.output_dir / "pbr" / f"{safe}.vxz"
    if args.skip_existing:
        shape_done = args.mode == "pbr" or vxz_is_valid(shape_path)
        pbr_done = args.mode == "shape" or vxz_is_valid(pbr_path)
        if shape_done and pbr_done:
            output["status"] = "skipped_existing"
            return output

    mansion_utils = configure_mansionworld_import(args)
    mesh = load_canonical_asset_mesh(asset_id, mansion_utils)
    if mesh is None:
        output["status"] = "missing_mesh"
        return output
    if abs(rotation % 360.0) > 1e-6:
        mesh = apply_local_z_rotation(mesh, rotation)

    if args.debug_dump_normalized_geometry:
        debug_dir = args.debug_geometry_dir or args.output_dir / "debug_normalized_geometry"
        debug_path = debug_dir / f"{safe}.glb"
        save_glb(mesh, debug_path)
        output["debug_geometry_path"] = str(debug_path)

    with tempfile.TemporaryDirectory(dir=args.temp_dir) as tmp:
        tmp_dir = Path(tmp)
        glb_path = tmp_dir / f"{safe}.glb"
        mesh_dump_path = tmp_dir / f"{safe}_mesh.pkl"
        pbr_dump_path = tmp_dir / f"{safe}_pbr.pkl"
        save_glb(mesh, glb_path)
        if args.mode in {"shape", "both"}:
            if not dump_with_blender(args.blender_path, "dump_mesh.py", glb_path, mesh_dump_path):
                output["status"] = "dump_mesh_failed"
                return output
            if not dual_grid_from_dump(mesh_dump_path, shape_path, args.resolution):
                output["status"] = "shape_voxel_failed"
                return output
        if args.mode in {"pbr", "both"}:
            if not dump_with_blender(args.blender_path, "dump_pbr.py", glb_path, pbr_dump_path):
                output["status"] = "dump_pbr_failed"
                return output
            if not pbr_from_dump(pbr_dump_path, pbr_path, args.resolution):
                output["status"] = "pbr_voxel_failed"
                return output

    output["status"] = "finished"
    if args.mode in {"shape", "both"}:
        output["shape_path"] = str(shape_path)
    if args.mode in {"pbr", "both"}:
        output["pbr_path"] = str(pbr_path)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mansionworld-root", type=Path, default=DEFAULT_MANSIONWORLD_ROOT)
    parser.add_argument("--objathor-root", type=Path, default=DEFAULT_MANSIONWORLD_ROOT / "objathor")
    parser.add_argument("--ai2thor-asset-dir", type=Path, default=DEFAULT_AI2THOR_ASSET_DIR)
    parser.add_argument(
        "--objathor-status",
        type=Path,
        default=DEFAULT_DEPENDENCY_ROOT / "objathor_download.status",
    )
    parser.add_argument(
        "--ai2thor-status",
        type=Path,
        default=DEFAULT_DEPENDENCY_ROOT / "hf_download_procthor.status",
    )
    parser.add_argument("--blender-path", type=Path, default=DEFAULT_BLENDER_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--asset-list", type=Path, default=None)
    parser.add_argument("--asset-id", action="append", default=[])
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--mode", choices=["shape", "pbr", "both"], default="both")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug-dump-normalized-geometry", action="store_true")
    parser.add_argument("--debug-geometry-dir", type=Path, default=None)
    parser.add_argument("--local-z-rotation", type=float, default=None)
    parser.add_argument("--local-z-rotations", type=float, nargs="+", default=None)
    parser.add_argument("--wait-for-deps", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.mansionworld_root / "ovoxels"
    if args.local_z_rotation is not None and args.local_z_rotations is not None:
        parser.error("Use either --local-z-rotation or --local-z-rotations, not both")
    if args.local_z_rotations is not None:
        args.rotation_degrees = normalized_rotation_list(args.local_z_rotations)
    elif args.local_z_rotation is not None:
        args.rotation_degrees = normalized_rotation_list([args.local_z_rotation])
    else:
        args.rotation_degrees = [0.0]
    if args.overwrite:
        args.skip_existing = False
    return args


def build_variant_args(args: argparse.Namespace, degrees: float) -> argparse.Namespace:
    variant = copy.copy(args)
    variant.local_z_rotation_degrees = float(degrees)
    variant.output_dir = output_dir_for_rotation(args.output_dir, degrees)
    variant.debug_geometry_dir = debug_dir_for_rotation(args.debug_geometry_dir, degrees)
    variant.temp_dir = variant.output_dir / "tmp"
    variant.temp_dir.mkdir(parents=True, exist_ok=True)
    return variant


def run_rotation_variant(args: argparse.Namespace, asset_ids: list[str], selected: list[str]) -> None:
    rotation = float(args.local_z_rotation_degrees)
    label = rotation_degrees_label(rotation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "shape").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pbr").mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "asset_ids.json"
    manifest = {
        "local_z_rotation_degrees": rotation,
        "rotation_label": label,
        "asset_ids": asset_ids,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Rotation {label} deg: total assets {len(asset_ids)}; selected for rank {args.rank}/{args.world_size}: {len(selected)}",
        flush=True,
    )
    print(f"Manifest: {manifest_path}", flush=True)
    if args.dry_run:
        for asset_id in selected[:20]:
            print(asset_id)
        return

    records_path = args.output_dir / "records" / f"rank_{args.rank:04d}.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and records_path.exists():
        records_path.unlink()
    if args.max_workers <= 1:
        for asset_id in selected:
            write_records(records_path, [process_asset(asset_id, args)])
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(process_asset, asset_id, args) for asset_id in selected]
            batch = []
            for future in as_completed(futures):
                batch.append(future.result())
                if len(batch) >= 16:
                    write_records(records_path, batch)
                    batch.clear()
            if batch:
                write_records(records_path, batch)
    print(f"Records: {records_path}", flush=True)


def main() -> int:
    args = parse_args()
    wait_for_dependencies(args)
    scene_data_dir = args.mansionworld_root / "mansionworld"
    if args.asset_id:
        asset_ids = sorted(set(args.asset_id))
    elif args.asset_list:
        asset_ids = [line.strip() for line in args.asset_list.read_text().splitlines() if line.strip()]
    else:
        asset_ids = collect_asset_ids(scene_data_dir)
    if args.limit is not None:
        asset_ids = asset_ids[: args.limit]
    selected = asset_ids[len(asset_ids) * args.rank // args.world_size : len(asset_ids) * (args.rank + 1) // args.world_size]

    for degrees in args.rotation_degrees:
        run_rotation_variant(build_variant_args(args, degrees), asset_ids, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
