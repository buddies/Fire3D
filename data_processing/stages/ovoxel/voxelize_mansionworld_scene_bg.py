#!/usr/bin/env python3
"""Voxelize normalized MansionWorld room background geometry.

Scene background meshes come from the released MansionWorld adapter via
get_room_geoms(...)[room_id]["bg"]["canonical_mesh"]. Outputs are written under
<MansionWorld>/ovoxels/scene_bg by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mansionworld_utils import (  # noqa: E402
    DEFAULT_AI2THOR_ASSET_DIR,
    DEFAULT_MANSIONWORLD_ROOT,
    configure_mansionworld_import,
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


def resolve_scene_path(scene_path: str, scene_data_dir: Path) -> Path:
    path = Path(scene_path)
    if path.is_absolute():
        return path
    return scene_data_dir / path


def scene_rel_path(scene_path: Path, scene_data_dir: Path) -> str:
    try:
        return str(scene_path.relative_to(scene_data_dir))
    except ValueError:
        return str(scene_path)


def safe_room_name(scene_path: Path, scene_data_dir: Path, room_id: str) -> str:
    rel = scene_rel_path(scene_path, scene_data_dir)
    scene_stem = "__".join(Path(rel).with_suffix("").parts)
    raw = f"{scene_stem}__room_{room_id}"
    return quote(raw.replace(" ", "_").replace("|", "_"), safe="")


def read_scene_room_list(scene_room_list: Path, scene_data_dir: Path) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for raw_line in scene_room_list.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t") if "\t" in line else line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Expected '<scene_path>\t<room_id>' in {scene_room_list}: {raw_line!r}")
        scene_path = resolve_scene_path(parts[0], scene_data_dir)
        rel = scene_rel_path(scene_path, scene_data_dir)
        mapping.setdefault(rel, set()).add(parts[1])
    return mapping


def collect_scene_paths(args: argparse.Namespace) -> list[Path]:
    scene_data_dir = args.mansionworld_root / "mansionworld"
    if args.scene_path:
        scene_paths = [resolve_scene_path(path, scene_data_dir) for path in args.scene_path]
    elif args.scene_list:
        scene_paths = [
            resolve_scene_path(line.strip(), scene_data_dir)
            for line in args.scene_list.read_text().splitlines()
            if line.strip()
        ]
    elif args.scene_room_list:
        scene_paths = [resolve_scene_path(path, scene_data_dir) for path in read_scene_room_list(args.scene_room_list, scene_data_dir)]
    else:
        mansion_utils = configure_mansionworld_import(args)
        scene_paths = [Path(path) for path in mansion_utils.get_all_scene_paths()]
    scene_paths = sorted(scene_paths)
    if args.limit_scenes is not None:
        scene_paths = scene_paths[: args.limit_scenes]
    return scene_paths


def process_bg_mesh(scene_path: Path, room_id: str, mesh, args: argparse.Namespace) -> dict[str, object]:
    scene_data_dir = args.mansionworld_root / "mansionworld"
    safe = safe_room_name(scene_path, scene_data_dir, room_id)
    rel = scene_rel_path(scene_path, scene_data_dir)
    record: dict[str, object] = {
        "scene_path": rel,
        "room_id": room_id,
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


def process_scene(scene_path: Path, args: argparse.Namespace) -> list[dict[str, object]]:
    mansion_utils = configure_mansionworld_import(args)
    scene_geoms, _room_polygons = mansion_utils.get_room_geoms(str(scene_path))
    room_ids = sorted(scene_geoms)
    room_filter = getattr(args, "scene_room_filter", None)
    if room_filter is not None:
        scene_data_dir = args.mansionworld_root / "mansionworld"
        allowed = room_filter.get(scene_rel_path(scene_path, scene_data_dir), set())
        room_ids = [room_id for room_id in room_ids if str(room_id) in allowed]
    if args.limit_rooms_per_scene is not None:
        room_ids = room_ids[: args.limit_rooms_per_scene]
    records = []
    for room_id in room_ids:
        bg_mesh = scene_geoms[room_id]["bg"]["canonical_mesh"]
        records.append(process_bg_mesh(scene_path, room_id, bg_mesh, args))
    return records


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
    parser.add_argument("--scene-path", action="append", default=[])
    parser.add_argument("--scene-list", type=Path, default=None)
    parser.add_argument("--scene-room-list", type=Path, default=None)
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
    parser.add_argument("--wait-for-deps", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.mansionworld_root / "ovoxels" / "scene_bg"
    if args.overwrite:
        args.skip_existing = False
    args.temp_dir = args.output_dir / "tmp"
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    return args


def main() -> int:
    args = parse_args()
    wait_for_dependencies(args)
    scene_data_dir = args.mansionworld_root / "mansionworld"
    args.scene_room_filter = read_scene_room_list(args.scene_room_list, scene_data_dir) if args.scene_room_list else None
    scene_paths = collect_scene_paths(args)
    selected = scene_paths[
        len(scene_paths) * args.rank // args.world_size : len(scene_paths) * (args.rank + 1) // args.world_size
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "shape").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pbr").mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "scene_paths.json"
    scene_data_dir = args.mansionworld_root / "mansionworld"
    manifest_path.write_text(json.dumps([scene_rel_path(path, scene_data_dir) for path in scene_paths], indent=2) + "\n")
    print(f"Total scenes: {len(scene_paths)}; selected for rank {args.rank}/{args.world_size}: {len(selected)}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)

    if args.dry_run:
        for scene_path in selected[:20]:
            print(scene_rel_path(scene_path, scene_data_dir))
        return 0

    records_path = args.output_dir / "records" / f"rank_{args.rank:04d}.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and records_path.exists():
        records_path.unlink()

    if args.max_workers <= 1:
        for scene_path in selected:
            write_records(records_path, process_scene(scene_path, args))
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(process_scene, scene_path, args) for scene_path in selected]
            for future in as_completed(futures):
                write_records(records_path, future.result())
    print(f"Records: {records_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
