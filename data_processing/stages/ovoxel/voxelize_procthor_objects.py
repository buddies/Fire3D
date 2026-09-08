#!/usr/bin/env python3
"""Voxelize normalized ProcTHOR scene object instances."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from procthor_utils import (  # noqa: E402
    DEFAULT_AI2THOR_HAB_ROOT,
    DEFAULT_PROCTHOR_ROOT,
    configure_procthor_import,
    wait_for_dependencies,
)
from utils import (  # noqa: E402
    DEFAULT_BLENDER_PATH,
    DEFAULT_DEPENDENCY_ROOT,
    dual_grid_from_dump,
    dump_with_blender,
    pbr_from_dump,
    save_glb,
    vxz_is_valid,
    write_records,
)


def read_scene_room_list(scene_room_list: Path) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for raw_line in scene_room_list.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t") if "\t" in line else line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Expected '<scene_name>\t<room_id>' in {scene_room_list}: {raw_line!r}")
        mapping.setdefault(parts[0], set()).add(parts[1])
    return mapping


def scene_room_slug(scene_name: str, room_id: str) -> str:
    return f"{scene_name.replace('/', '_')}_room_{room_id}"


def safe_object_name(scene_name: str, room_id: str, object_name: str) -> str:
    raw = f"{scene_room_slug(scene_name, room_id)}__{object_name}"
    return quote(raw.replace(" ", "_").replace("|", "_"), safe="")


def collect_scene_names(args: argparse.Namespace) -> list[str]:
    if args.scene_name:
        scene_names = list(args.scene_name)
    elif args.scene_list:
        scene_names = [line.strip() for line in args.scene_list.read_text().splitlines() if line.strip()]
    elif args.scene_room_list:
        scene_names = sorted(read_scene_room_list(args.scene_room_list))
    else:
        procthor_utils = configure_procthor_import(args)
        scene_names = procthor_utils.get_all_procthor_scenes()
    scene_names = sorted(scene_names)
    if args.limit_scenes is not None:
        scene_names = scene_names[: args.limit_scenes]
    return scene_names


def matrix_payload(matrix) -> list[list[float]]:
    return np.asarray(matrix, dtype=np.float64).reshape(4, 4).tolist()


def process_object_mesh(
    scene_name: str,
    room_id: str,
    object_name: str,
    mesh,
    args: argparse.Namespace,
) -> dict[str, object]:
    safe = safe_object_name(scene_name, room_id, object_name)
    record: dict[str, object] = {
        "scene_name": scene_name,
        "room_id": room_id,
        "object_name": object_name,
        "safe_name": safe,
        "status": "started",
    }
    shape_path = args.output_dir / "shape" / f"{safe}.vxz"
    pbr_path = args.output_dir / "pbr" / f"{safe}.vxz"
    try:
        if args.skip_existing:
            shape_done = args.mode == "pbr" or vxz_is_valid(shape_path)
            pbr_done = args.mode == "shape" or vxz_is_valid(pbr_path)
            if shape_done and pbr_done:
                record["status"] = "skipped_existing"
                return record

        if args.debug_dump_normalized_geometry:
            debug_dir = args.debug_geometry_dir or args.output_dir / "debug_normalized_geometry"
            debug_path = debug_dir / f"{safe}.glb"
            save_glb(mesh, debug_path)
            record["debug_geometry_path"] = str(debug_path)

        with tempfile.TemporaryDirectory(dir=args.temp_dir) as tmp:
            tmp_dir = Path(tmp)
            glb_path = tmp_dir / f"{safe}.glb"
            mesh_dump_path = tmp_dir / f"{safe}_mesh.pkl"
            pbr_dump_path = tmp_dir / f"{safe}_pbr.pkl"
            save_glb(mesh, glb_path)
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
    except Exception as exc:
        record["status"] = "error"
        record["error"] = repr(exc)
        return record

    record["status"] = "finished"
    if args.mode in {"shape", "both"}:
        record["shape_path"] = str(shape_path)
    if args.mode in {"pbr", "both"}:
        record["pbr_path"] = str(pbr_path)
    return record


def room_lookup(room_geoms_dict: dict, room_id: str):
    return room_geoms_dict[int(room_id)] if room_id.isdigit() and int(room_id) in room_geoms_dict else room_geoms_dict[room_id]


def process_scene(scene_name: str, args: argparse.Namespace) -> list[dict[str, object]]:
    procthor_utils = configure_procthor_import(args)
    _room_regions, room_geoms_dict = procthor_utils.export_rooms_with_canonical_meshes(scene_name)
    room_ids = sorted(str(room_id) for room_id in room_geoms_dict)
    room_filter = getattr(args, "scene_room_filter", None)
    if room_filter is not None:
        allowed = room_filter.get(scene_name, set())
        room_ids = [room_id for room_id in room_ids if str(room_id) in allowed]
    if args.room_id:
        wanted = {str(room_id) for room_id in args.room_id}
        room_ids = [room_id for room_id in room_ids if room_id in wanted]
    if args.limit_rooms_per_scene is not None:
        room_ids = room_ids[: args.limit_rooms_per_scene]

    records: list[dict[str, object]] = []
    for room_id in room_ids:
        room_geoms = room_lookup(room_geoms_dict, room_id)
        transform_entries = []

        if args.scene_bg_debug_dir is not None:
            bg_safe = scene_room_slug(scene_name, room_id)
            bg_path = args.scene_bg_debug_dir / f"{bg_safe}.glb"
            save_glb(room_geoms["bg"]["canonical_mesh"], bg_path)
            records.append(
                {
                    "scene_name": scene_name,
                    "room_id": room_id,
                    "safe_name": bg_safe,
                    "status": "debug_bg_written",
                    "debug_geometry_path": str(bg_path),
                }
            )
        transform_entries.append(
            {
                "name": f"layout_{scene_room_slug(scene_name, room_id)}",
                "asset_id": None,
                "matrix": matrix_payload(room_geoms["bg"]["total_transform"]),
            }
        )

        for object_name in sorted(name for name in room_geoms if name != "bg"):
            info = room_geoms[object_name]
            safe = safe_object_name(scene_name, room_id, object_name)
            transform_entries.append(
                {
                    "name": safe,
                    "asset_id": safe,
                    "source_name": object_name,
                    "matrix": matrix_payload(info["total_transform"]),
                }
            )
            records.append(process_object_mesh(scene_name, room_id, object_name, info["canonical_mesh"], args))

        if args.transforms_dir is not None:
            args.transforms_dir.mkdir(parents=True, exist_ok=True)
            transform_path = args.transforms_dir / f"{scene_room_slug(scene_name, room_id)}.json"
            transform_path.write_text(json.dumps(transform_entries, indent=2) + "\n")
            records.append(
                {
                    "scene_name": scene_name,
                    "room_id": room_id,
                    "status": "transforms_written",
                    "transforms_path": str(transform_path),
                    "transform_count": len(transform_entries),
                }
            )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--procthor-root", type=Path, default=DEFAULT_PROCTHOR_ROOT)
    parser.add_argument("--ai2thor-hab-root", type=Path, default=DEFAULT_AI2THOR_HAB_ROOT)
    parser.add_argument(
        "--ai2thor-status",
        type=Path,
        default=DEFAULT_DEPENDENCY_ROOT / "hf_download_procthor.status",
    )
    parser.add_argument("--blender-path", type=Path, default=DEFAULT_BLENDER_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scene-name", action="append", default=[])
    parser.add_argument("--scene-list", type=Path, default=None)
    parser.add_argument("--scene-room-list", type=Path, default=None)
    parser.add_argument("--room-id", action="append", default=[])
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--mode", choices=["shape", "pbr", "both"], default="both")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--limit-rooms-per-scene", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug-dump-normalized-geometry", action="store_true")
    parser.add_argument("--debug-geometry-dir", type=Path, default=None)
    parser.add_argument("--scene-bg-debug-dir", type=Path, default=None)
    parser.add_argument("--transforms-dir", type=Path, default=None)
    parser.add_argument("--wait-for-deps", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.procthor_root / "ovoxels" / "objects"
    if args.overwrite:
        args.skip_existing = False
    args.temp_dir = args.output_dir / "tmp"
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    return args


def main() -> int:
    args = parse_args()
    wait_for_dependencies(args)
    args.scene_room_filter = read_scene_room_list(args.scene_room_list) if args.scene_room_list else None
    scene_names = collect_scene_names(args)
    selected = scene_names[
        len(scene_names) * args.rank // args.world_size : len(scene_names) * (args.rank + 1) // args.world_size
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "shape").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pbr").mkdir(parents=True, exist_ok=True)
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
