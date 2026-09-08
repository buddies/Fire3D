#!/usr/bin/env python3
"""Export one or more iTHOR scenes with canonical and transformed elements."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import (
    EXPORT_DIR,
    export_scene,
    export_scene_with_canonical_meshes,
    get_all_ithor_scenes,
    save_glb,
)


def export_one(scene_name: str, output_root: Path) -> None:
    scene_key = scene_name.replace("/", "_")
    output_root.mkdir(parents=True, exist_ok=True)
    save_glb(export_scene(scene_name), str(output_root / f"{scene_key}.glb"))

    scene_dir = output_root / scene_key
    canonical_dir = scene_dir / "canonical_mesh"
    transformed_dir = scene_dir / "transformed_mesh"
    transformed_glb_dir = scene_dir / "transformed_glb_mesh"
    for directory in (canonical_dir, transformed_dir, transformed_glb_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for mesh_name, mesh_info in export_scene_with_canonical_meshes(scene_name).items():
        mesh_info["canonical_mesh"].export(canonical_dir / f"{mesh_name}.ply")
        transformed = mesh_info["transformed_mesh"]
        transformed.export(transformed_dir / f"{mesh_name}.ply")
        save_glb(transformed, str(transformed_glb_dir / f"{mesh_name}.glb"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-name", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path(EXPORT_DIR))
    args = parser.parse_args()
    if not args.scene_name and not args.all:
        parser.error("provide --scene-name or --all")
    return args


def main() -> None:
    args = parse_args()
    scenes = get_all_ithor_scenes() if args.all else args.scene_name
    for scene_name in sorted(scenes):
        export_one(scene_name, args.output_root)
        print(f"Exported {scene_name}", flush=True)


if __name__ == "__main__":
    main()
