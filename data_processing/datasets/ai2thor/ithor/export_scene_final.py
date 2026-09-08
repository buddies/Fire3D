#!/usr/bin/env python3
"""Export canonical iTHOR scene elements in deterministic instance order."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from utils import export_scene_with_canonical_meshes, get_all_ithor_scenes, save_glb


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_AI2THOR_ROOT = Path(
    os.environ.get(
        "FIRE3D_AI2THOR_ROOT",
        REPO_ROOT / "data/training_scenes/ai2thor-hab",
    )
)


def export_ordered_scene(scene_name: str, output_root: Path) -> Path:
    scene_geometries = export_scene_with_canonical_meshes(scene_name)
    object_names = sorted(name for name in scene_geometries if name != "bg")
    ordered_names = ["bg", *object_names]
    scene_output = output_root / scene_name.replace("/", "_")
    scene_output.mkdir(parents=True, exist_ok=True)
    for index, mesh_name in enumerate(ordered_names):
        transformed_mesh = scene_geometries[mesh_name]["transformed_mesh"]
        save_glb(transformed_mesh, str(scene_output / f"{index:04d}.glb"))
    return scene_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-name", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_AI2THOR_ROOT / "ithor_data/scenes_transformed",
    )
    args = parser.parse_args()
    if not args.scene_name and not args.all:
        parser.error("provide --scene-name or --all")
    return args


def main() -> None:
    args = parse_args()
    scenes = get_all_ithor_scenes() if args.all else args.scene_name
    for scene_name in sorted(scenes):
        output = export_ordered_scene(scene_name, args.output_root)
        print(f"Exported {scene_name} to {output}", flush=True)


if __name__ == "__main__":
    main()
