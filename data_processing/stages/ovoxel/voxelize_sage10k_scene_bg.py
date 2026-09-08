#!/usr/bin/env python3
"""Voxelize SAGE-10k layout background geometry.

Scene archives are extracted on demand and converted through the released
SAGE-10k helpers. Outputs are written under
<SAGE-10k>/ovoxels/scene_bg by default.
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

from sage10k_utils import (  # noqa: E402
    DEFAULT_SAGE10K_ROOT,
    background_mesh_ids,
    collect_scene_ids,
    layout_id_from_scene_id,
    load_scene_mesh_dict,
    resolve_scene_dir,
    save_sage_textured_glb,
    wait_for_dependencies,
)
from utils import (  # noqa: E402
    DEFAULT_BLENDER_PATH,
    DEFAULT_DEPENDENCY_ROOT,
    dual_grid_from_dump,
    dump_with_blender,
    pbr_from_dump,
    vxz_is_valid,
    write_records,
)


def safe_scene_name(scene_id: str) -> str:
    return quote(scene_id.replace(" ", "_").replace("|", "_"), safe="")


def process_scene(scene_id: str, args: argparse.Namespace) -> list[dict[str, object]]:
    safe = safe_scene_name(scene_id)
    layout_id = layout_id_from_scene_id(scene_id)
    record: dict[str, object] = {
        "scene_id": scene_id,
        "layout_id": layout_id,
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
                return [record]

        scene_dir = resolve_scene_dir(scene_id, args)
        record["scene_dir"] = str(scene_dir)
        mesh_dict = load_scene_mesh_dict(scene_dir, scene_id)
        mesh_ids = background_mesh_ids(mesh_dict)
        record["mesh_ids"] = mesh_ids
        if not mesh_ids:
            record["status"] = "no_background_meshes"
            return [record]

        if args.debug_dump_normalized_geometry:
            debug_dir = args.debug_geometry_dir or args.output_dir / "debug_normalized_geometry"
            debug_path = debug_dir / f"{safe}.glb"
            save_sage_textured_glb(mesh_dict, mesh_ids, debug_path)
            record["debug_geometry_path"] = str(debug_path)

        with tempfile.TemporaryDirectory(dir=args.temp_dir) as tmp:
            tmp_dir = Path(tmp)
            glb_path = tmp_dir / f"{safe}.glb"
            mesh_dump_path = tmp_dir / f"{safe}_mesh.pkl"
            pbr_dump_path = tmp_dir / f"{safe}_pbr.pkl"
            save_sage_textured_glb(mesh_dict, mesh_ids, glb_path)

            if args.mode in {"shape", "both"}:
                if not dump_with_blender(args.blender_path, "dump_mesh.py", glb_path, mesh_dump_path):
                    record["status"] = "dump_mesh_failed"
                    return [record]
                if not dual_grid_from_dump(mesh_dump_path, shape_path, args.resolution):
                    record["status"] = "shape_voxel_failed"
                    return [record]
            if args.mode in {"pbr", "both"}:
                if not dump_with_blender(args.blender_path, "dump_pbr.py", glb_path, pbr_dump_path):
                    record["status"] = "dump_pbr_failed"
                    return [record]
                if not pbr_from_dump(pbr_dump_path, pbr_path, args.resolution):
                    record["status"] = "pbr_voxel_failed"
                    return [record]
    except Exception as exc:
        record["status"] = "error"
        record["error"] = repr(exc)
        return [record]

    record["status"] = "finished"
    if args.mode in {"shape", "both"}:
        record["shape_path"] = str(shape_path)
    if args.mode in {"pbr", "both"}:
        record["pbr_path"] = str(pbr_path)
    return [record]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sage10k-root", type=Path, default=DEFAULT_SAGE10K_ROOT)
    parser.add_argument("--scene-zip-dir", type=Path, default=None)
    parser.add_argument("--extracted-dir", type=Path, default=None)
    parser.add_argument(
        "--sage10k-status",
        type=Path,
        default=DEFAULT_DEPENDENCY_ROOT / "hf_download_sage10k.status",
    )
    parser.add_argument("--blender-path", type=Path, default=DEFAULT_BLENDER_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scene-id", action="append", default=[])
    parser.add_argument("--scene-list", type=Path, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--mode", choices=["shape", "pbr", "both"], default="both")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug-dump-normalized-geometry", action="store_true")
    parser.add_argument("--debug-geometry-dir", type=Path, default=None)
    parser.add_argument("--extract-zips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wait-for-deps", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.scene_zip_dir is None:
        args.scene_zip_dir = args.sage10k_root / "scenes"
    if args.extracted_dir is None:
        args.extracted_dir = args.sage10k_root / "scenes_extracted"
    if args.output_dir is None:
        args.output_dir = args.sage10k_root / "ovoxels" / "scene_bg"
    if args.overwrite:
        args.skip_existing = False
    args.temp_dir = args.output_dir / "tmp"
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    return args


def main() -> int:
    args = parse_args()
    wait_for_dependencies(args)
    scene_ids = collect_scene_ids(args)
    selected = scene_ids[
        len(scene_ids) * args.rank // args.world_size : len(scene_ids) * (args.rank + 1) // args.world_size
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "shape").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pbr").mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "scene_ids.json"
    manifest_path.write_text(json.dumps(scene_ids, indent=2) + "\n")
    print(f"Total scenes: {len(scene_ids)}; selected for rank {args.rank}/{args.world_size}: {len(selected)}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)

    if args.dry_run:
        for scene_id in selected[:20]:
            print(scene_id)
        return 0

    records_path = args.output_dir / "records" / f"rank_{args.rank:04d}.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and records_path.exists():
        records_path.unlink()

    if args.max_workers <= 1:
        for scene_id in selected:
            write_records(records_path, process_scene(scene_id, args))
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(process_scene, scene_id, args) for scene_id in selected]
            for future in as_completed(futures):
                write_records(records_path, future.result())
    print(f"Records: {records_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
