#!/usr/bin/env python3
"""Export MansionWorld rooms as canonical and scene-space geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import (
    MANSIONWORLD_ROOT,
    get_all_scene_paths,
    get_room_geoms,
    save_glb,
    visualize_room_polygons,
)


def export_scene(scene_path: Path, output_root: Path, *, render_room_map: bool) -> None:
    scene_path = scene_path.expanduser().resolve()
    scene_name = "_".join(scene_path.with_suffix("").parts[-2:])
    scene_geoms, room_polygons = get_room_geoms(str(scene_path))
    scene_output = output_root / scene_name
    scene_output.mkdir(parents=True, exist_ok=True)

    if render_room_map:
        visualize_room_polygons(
            room_polygons,
            str(scene_output / "room_polygons.png"),
        )

    for raw_room_id, room_geoms in scene_geoms.items():
        room_id = str(raw_room_id)
        world_dir = scene_output / room_id
        world_ply_dir = scene_output / f"{room_id}_ply"
        canonical_dir = scene_output / f"{room_id}_canonical"
        for directory in (world_dir, world_ply_dir, canonical_dir):
            directory.mkdir(parents=True, exist_ok=True)

        for mesh_name, geometry in room_geoms.items():
            canonical = geometry["canonical_mesh"]
            canonical.export(canonical_dir / f"{mesh_name}.ply")
            world = canonical.copy()
            world.apply_transform(geometry["transform"])
            world.export(world_ply_dir / f"{mesh_name}.ply")
            save_glb(world, str(world_dir / f"{mesh_name}.glb"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-path", type=Path, action="append", default=[])
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export every scene discovered under the configured MansionWorld root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(MANSIONWORLD_ROOT) / "exports",
    )
    parser.add_argument(
        "--room-map",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if not args.scene_path and not args.all:
        parser.error("provide --scene-path or --all")
    return args


def main() -> None:
    args = parse_args()
    scenes = [Path(path) for path in get_all_scene_paths()] if args.all else args.scene_path
    args.output_root.mkdir(parents=True, exist_ok=True)
    for scene_path in scenes:
        print(f"Exporting {scene_path}", flush=True)
        export_scene(scene_path, args.output_root, render_room_map=args.room_map)


if __name__ == "__main__":
    main()
