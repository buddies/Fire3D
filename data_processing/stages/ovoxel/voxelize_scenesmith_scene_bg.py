#!/usr/bin/env python3
"""Voxelize SceneSmith exported room background GLBs.

SceneSmith export writes one normalized layout GLB per room plus a transform PKL.
The GLB has the same export-only ``-90deg X`` save rotation used by other raw
kit exporters, so this script passes it directly to Blender dumping and lets the
validators undo that frame with ``--input-glb-frame blender-x-minus-90``.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scenesmith_utils import (  # noqa: E402
    DEFAULT_SCENESMITH_ROOT,
    collect_room_names,
    copy_normalized_glb,
    layout_latent,
    load_transform_dict,
    safe_name,
    scene_dir_for,
    split_room_names,
    transform_path_for,
    write_validator_transforms,
)
from utils import (  # noqa: E402
    DEFAULT_BLENDER_PATH,
    dual_grid_from_dump,
    dump_with_blender,
    pbr_from_dump,
    vxz_is_valid,
    write_records,
)


def process_room(room_name: str, args: argparse.Namespace) -> list[dict[str, object]]:
    record: dict[str, object] = {"room_name": room_name, "status": "started"}
    try:
        scene_dir = scene_dir_for(args.scenesmith_root, room_name)
        transform_path = transform_path_for(args.scenesmith_root, room_name)
        transforms = load_transform_dict(transform_path)
        latent = layout_latent(room_name, transforms)
        source_glb = scene_dir / f"{latent}.glb"
        if not source_glb.exists():
            raise FileNotFoundError(f"Missing SceneSmith layout GLB: {source_glb}")

        stem = safe_name(latent)
        shape_path = args.output_dir / "shape" / f"{stem}.vxz"
        pbr_path = args.output_dir / "pbr" / f"{stem}.vxz"
        record.update({"latent": latent, "shape_path": str(shape_path), "pbr_path": str(pbr_path)})

        if args.transforms_dir is not None:
            transform_json = write_validator_transforms(room_name, transforms, args.transforms_dir)
            record["validator_transforms"] = str(transform_json)
        if args.normalized_glb_dir is not None:
            copied = copy_normalized_glb(source_glb, args.normalized_glb_dir / f"{stem}.glb", args.normalized_dump_mode)
            record["normalized_glb_path"] = copied
        if args.debug_dump_normalized_geometry:
            debug_path = args.debug_geometry_dir / f"{stem}.glb"
            copied = copy_normalized_glb(source_glb, debug_path, args.normalized_dump_mode)
            record["debug_geometry_path"] = copied
        if args.dump_normalized_only:
            record["status"] = "debug_normalized_written"
            return [record]

        if args.skip_existing:
            shape_done = args.mode == "pbr" or vxz_is_valid(shape_path)
            pbr_done = args.mode == "shape" or vxz_is_valid(pbr_path)
            if shape_done and pbr_done:
                record["status"] = "skipped_existing"
                return [record]

        with tempfile.TemporaryDirectory(dir=args.temp_dir) as tmp:
            tmp_dir = Path(tmp)
            mesh_dump_path = tmp_dir / f"{stem}_mesh.pkl"
            pbr_dump_path = tmp_dir / f"{stem}_pbr.pkl"
            if args.mode in {"shape", "both"}:
                if not dump_with_blender(args.blender_path, "dump_mesh.py", source_glb, mesh_dump_path):
                    record["status"] = "dump_mesh_failed"
                    return [record]
                if not dual_grid_from_dump(mesh_dump_path, shape_path, args.resolution):
                    record["status"] = "shape_voxel_failed"
                    return [record]
            if args.mode in {"pbr", "both"}:
                if not dump_with_blender(args.blender_path, "dump_pbr.py", source_glb, pbr_dump_path):
                    record["status"] = "dump_pbr_failed"
                    return [record]
                if not pbr_from_dump(pbr_dump_path, pbr_path, args.resolution):
                    record["status"] = "pbr_voxel_failed"
                    return [record]
    except Exception as exc:  # noqa: BLE001 - keep batch jobs moving.
        record["status"] = "error"
        record["error"] = repr(exc)
        return [record]

    record["status"] = "finished"
    return [record]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenesmith-root", type=Path, default=DEFAULT_SCENESMITH_ROOT)
    parser.add_argument("--blender-path", type=Path, default=DEFAULT_BLENDER_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--normalized-glb-dir", type=Path, default=None)
    parser.add_argument("--normalized-dump-mode", choices=["copy", "hardlink", "symlink"], default="copy")
    parser.add_argument("--debug-dump-normalized-geometry", action="store_true")
    parser.add_argument("--debug-geometry-dir", type=Path, default=None)
    parser.add_argument("--dump-normalized-only", action="store_true")
    parser.add_argument("--transforms-dir", type=Path, default=None)
    parser.add_argument("--room-name", action="append", default=[])
    parser.add_argument("--room-list", type=Path, default=None)
    parser.add_argument("--trajectory-id", default="0")
    parser.add_argument("--require-renders", action="store_true")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--mode", choices=["shape", "pbr", "both"], default="both")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--limit-rooms", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.scenesmith_root / "ovoxels" / "scene_bg"
    if args.normalized_glb_dir is None and not args.dump_normalized_only:
        args.normalized_glb_dir = args.output_dir / "normalized_glbs"
    if args.debug_geometry_dir is None:
        args.debug_geometry_dir = args.output_dir / "debug_normalized_geometry"
    if args.dump_normalized_only:
        args.debug_dump_normalized_geometry = True
    if args.transforms_dir is None:
        args.transforms_dir = args.scenesmith_root / "ovoxels" / "transforms"
    if args.overwrite:
        args.skip_existing = False
    args.temp_dir = args.output_dir / "tmp"
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    return args


def main() -> int:
    args = parse_args()
    room_names = collect_room_names(args)
    selected = split_room_names(room_names, args.rank, args.world_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "shape").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pbr").mkdir(parents=True, exist_ok=True)
    if args.normalized_glb_dir is not None:
        args.normalized_glb_dir.mkdir(parents=True, exist_ok=True)
    if args.debug_dump_normalized_geometry:
        args.debug_geometry_dir.mkdir(parents=True, exist_ok=True)
    args.transforms_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "room_names.json"
    manifest_path.write_text(__import__("json").dumps(room_names, indent=2) + "\n")
    print(f"Total rooms: {len(room_names)}; selected for rank {args.rank}/{args.world_size}: {len(selected)}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)

    if args.dry_run:
        for room_name in selected[:20]:
            print(room_name)
        return 0

    records_path = args.output_dir / "records" / f"rank_{args.rank:04d}.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and records_path.exists():
        records_path.unlink()

    if args.max_workers <= 1:
        for room_name in selected:
            write_records(records_path, process_room(room_name, args))
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(process_room, room_name, args) for room_name in selected]
            for future in as_completed(futures):
                write_records(records_path, future.result())
    print(f"Records: {records_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
