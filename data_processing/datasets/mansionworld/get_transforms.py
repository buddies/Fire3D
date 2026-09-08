#!/usr/bin/env python3
"""Export MansionWorld object placement transforms used by Fire3D training."""

from __future__ import annotations

import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import trimesh
from tqdm import tqdm

from utils import MANSIONWORLD_ROOT, get_all_scene_paths, get_room_geoms


def process_scene(scene_path: str, output_dir: str, overwrite: bool) -> list[str]:
    source = Path(scene_path)
    target_root = Path(output_dir)
    scene_name = "_".join(source.with_suffix("").parts[-2:])
    scene_geometries, _room_polygons = get_room_geoms(str(source))
    written: list[str] = []

    for raw_room_id, room_geometries in scene_geometries.items():
        room_id = str(raw_room_id)
        room_name = f"{scene_name}_room_{room_id}"
        output = target_root / f"{room_name}.pkl"
        if output.exists() and not overwrite:
            continue

        object_names = sorted(name for name in room_geometries if name != "bg")
        records = {}
        for index, mesh_name in enumerate(["bg", *object_names]):
            geometry = room_geometries[mesh_name]
            scale, _shear, angles, translation, _perspective = (
                trimesh.transformations.decompose_matrix(geometry["transform"])
            )
            latent = f"layout_{room_name}" if index == 0 else f"object_{index:04d}"
            records[latent] = {
                "scale": float(scale.reshape(3)[0]),
                "angles": [float(value) for value in angles],
                "trans": [float(value) for value in translation],
                "latent": latent,
                "asset_id": None if index == 0 else geometry["asset_id"],
            }

        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as stream:
            pickle.dump(records, stream, protocol=pickle.HIGHEST_PROTOCOL)
        written.append(str(output))
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-path", type=Path, action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(MANSIONWORLD_ROOT) / "transforms",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.scene_path and not args.all:
        parser.error("provide --scene-path or --all")
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def main() -> None:
    args = parse_args()
    scenes = [Path(path) for path in get_all_scene_paths()] if args.all else args.scene_path
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_scene,
                str(path),
                str(args.output_dir),
                args.overwrite,
            )
            for path in scenes
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="MansionWorld"):
            future.result()


if __name__ == "__main__":
    main()
