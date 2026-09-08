#!/usr/bin/env python3
"""Voxelize normalized InternScenes scene objects."""

from __future__ import annotations

import argparse
import json
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

from internscenes_utils import (  # noqa: E402
    DEFAULT_INTERNSCENES_ROOT,
    DEFAULT_RAW_KIT_DIR,
    DEFAULT_SPLIT_JSON,
    collect_scene_names,
    composer_for,
    ensure_layout_extracted,
    instance_object_asset_id,
    load_layout,
    normalized_background,
    normalized_scene_object,
    scene_name_to_safe,
    split_items,
    transform_entries_for_scene,
    write_transform_entries,
)
from utils import (  # noqa: E402
    DEFAULT_BLENDER_PATH,
    dual_grid_from_dump,
    dump_with_blender,
    pbr_from_dump,
    save_glb,
    vxz_is_valid,
    write_records,
)


def process_object(
    scene_name: str,
    instance: dict[str, object],
    normalized_mesh,
    args: argparse.Namespace,
) -> dict[str, object]:
    instance_id = int(instance.get("id", 0))
    asset_id = instance_object_asset_id(scene_name, instance_id)
    shape_path = args.output_dir / "shape" / f"{asset_id}.vxz"
    pbr_path = args.output_dir / "pbr" / f"{asset_id}.vxz"
    record: dict[str, object] = {
        "scene_name": scene_name,
        "instance_id": instance_id,
        "asset_id": asset_id,
        "model_uid": instance.get("model_uid"),
        "category": instance.get("category"),
        "status": "started",
        "shape_path": str(shape_path),
        "pbr_path": str(pbr_path),
    }
    try:
        if args.normalized_glb_dir is not None:
            normalized_path = args.normalized_glb_dir / f"{asset_id}.glb"
            save_glb(normalized_mesh, normalized_path)
            record["normalized_glb_path"] = str(normalized_path)
        if args.debug_dump_normalized_geometry:
            debug_path = args.debug_geometry_dir / f"{asset_id}.glb"
            save_glb(normalized_mesh, debug_path)
            record["debug_geometry_path"] = str(debug_path)
        if args.dump_normalized_only:
            record["status"] = "debug_normalized_written"
            return record

        if args.skip_existing:
            shape_done = args.mode == "pbr" or vxz_is_valid(shape_path)
            pbr_done = args.mode == "shape" or vxz_is_valid(pbr_path)
            if shape_done and pbr_done:
                record["status"] = "skipped_existing"
                return record

        with tempfile.TemporaryDirectory(dir=args.temp_dir) as tmp:
            tmp_dir = Path(tmp)
            glb_path = tmp_dir / f"{asset_id}.glb"
            mesh_dump_path = tmp_dir / f"{asset_id}_mesh.pkl"
            pbr_dump_path = tmp_dir / f"{asset_id}_pbr.pkl"
            save_glb(normalized_mesh, glb_path)

            if args.mode in {"shape", "both"}:
                if not dump_with_blender(args.blender_path, "dump_mesh.py", glb_path, mesh_dump_path):
                    record["status"] = "dump_mesh_failed"
                    return record
                if not dual_grid_from_dump(mesh_dump_path, shape_path, args.resolution):
                    record["status"] = "shape_voxel_failed"
                    return record
            if args.mode in {"pbr", "both"}:
                if not dump_with_blender(args.blender_path, "dump_pbr.py", glb_path, pbr_dump_path):
                    record["status"] = "dump_pbr_failed"
                    return record
                if not pbr_from_dump(pbr_dump_path, pbr_path, args.resolution):
                    record["status"] = "pbr_voxel_failed"
                    return record
    except Exception as exc:  # noqa: BLE001 - batch jobs should keep moving.
        record["status"] = "error"
        record["error"] = repr(exc)
        return record

    record["status"] = "finished"
    return record


def process_scene(scene_name: str, args: argparse.Namespace) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    try:
        ensure_layout_extracted(scene_name, args.layout_root, args.layout_tar, args.extract_layout)
        layout = load_layout(scene_name, args.layout_root)
        composer, _module = composer_for(args.raw_kit_dir, args.internscenes_root, args.layout_root)
        bg_transform = None
        if args.transforms_dir is not None:
            _bg_mesh, bg_transform = normalized_background(scene_name, args.layout_root)

        object_transforms = {}
        processed = 0
        for instance in sorted(layout, key=lambda x: int(x.get("id", 0))):
            if not instance.get("model_uid"):
                continue
            if args.limit_objects is not None and processed >= args.limit_objects:
                break
            normalized_mesh, transform = normalized_scene_object(composer, instance)
            instance_id = int(instance.get("id", 0))
            object_transforms[instance_id] = transform
            records.append(process_object(scene_name, instance, normalized_mesh, args))
            processed += 1

        if args.transforms_dir is not None:
            entries = transform_entries_for_scene(scene_name, layout, bg_transform, object_transforms)
            transform_path = write_transform_entries(scene_name, entries, args.transforms_dir)
            records.insert(
                0,
                {
                    "scene_name": scene_name,
                    "status": "transforms_written",
                    "transforms_path": str(transform_path),
                    "transform_count": len(entries),
                },
            )
    except Exception as exc:  # noqa: BLE001 - batch jobs should keep moving.
        records.append({"scene_name": scene_name, "status": "error", "error": repr(exc)})
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internscenes-root", type=Path, default=DEFAULT_INTERNSCENES_ROOT)
    parser.add_argument("--raw-kit-dir", type=Path, default=DEFAULT_RAW_KIT_DIR)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--from-split", action="store_true")
    parser.add_argument("--layout-root", type=Path, default=None)
    parser.add_argument("--layout-tar", type=Path, default=None)
    parser.add_argument("--extract-layout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--blender-path", type=Path, default=DEFAULT_BLENDER_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--normalized-glb-dir", type=Path, default=None)
    parser.add_argument("--no-normalized-glbs", action="store_true", help="Do not save normalized GLBs during large production voxelization runs.")
    parser.add_argument("--transforms-dir", type=Path, default=None)
    parser.add_argument("--scene-name", action="append", default=[])
    parser.add_argument("--scene-list", type=Path, default=None)
    parser.add_argument("--trajectory-id", default="0")
    parser.add_argument("--require-renders", action="store_true")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--mode", choices=["shape", "pbr", "both"], default="both")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--limit-objects", type=int, default=None)
    parser.add_argument("--debug-dump-normalized-geometry", action="store_true")
    parser.add_argument("--debug-geometry-dir", type=Path, default=None)
    parser.add_argument("--dump-normalized-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    downloaded = args.internscenes_root / "downloaded"
    if args.layout_root is None:
        args.layout_root = downloaded / "Layout_info"
    if args.layout_tar is None:
        args.layout_tar = downloaded / "Layout_info.tar.gz"
    if args.output_dir is None:
        args.output_dir = args.internscenes_root / "ovoxels" / "objects"
    if args.no_normalized_glbs and not args.dump_normalized_only:
        args.normalized_glb_dir = None
    elif args.normalized_glb_dir is None and not args.dump_normalized_only:
        args.normalized_glb_dir = args.output_dir / "normalized_glbs"
    if args.transforms_dir is None:
        args.transforms_dir = args.internscenes_root / "ovoxels" / "transforms"
    if args.debug_geometry_dir is None:
        args.debug_geometry_dir = args.output_dir / "debug_normalized_geometry"
    if args.dump_normalized_only:
        args.debug_dump_normalized_geometry = True
    if args.overwrite:
        args.skip_existing = False
    args.temp_dir = args.output_dir / "tmp"
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    return args


def main() -> int:
    args = parse_args()
    scene_names = collect_scene_names(args)
    selected = split_items(scene_names, args.rank, args.world_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "shape").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pbr").mkdir(parents=True, exist_ok=True)
    if args.normalized_glb_dir is not None:
        args.normalized_glb_dir.mkdir(parents=True, exist_ok=True)
    if args.debug_dump_normalized_geometry:
        args.debug_geometry_dir.mkdir(parents=True, exist_ok=True)
    if args.transforms_dir is not None:
        args.transforms_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "scene_names.json"
    manifest_path.write_text(json.dumps(scene_names, indent=2) + "\n")
    print(f"Total scenes: {len(scene_names)}; selected for rank {args.rank}/{args.world_size}: {len(selected)}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)

    if args.dry_run:
        for scene_name in selected[:20]:
            print(scene_name)
        return 0

    records_path = args.output_dir / "records" / f"rank_{args.rank:04d}.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and records_path.exists():
        records_path.unlink()

    if args.max_workers <= 1:
        for scene_name in selected:
            write_records(records_path, process_scene(scene_name, args))
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(process_scene, scene_name, args) for scene_name in selected]
            for future in as_completed(futures):
                write_records(records_path, future.result())
    print(f"Records: {records_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
