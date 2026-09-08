#!/usr/bin/env python3
"""Apply saved Imaginarium placements and export scene-space GLB assets."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = Path(
    os.environ.get(
        "FIRE3D_IMAGINARIUM_ROOT",
        REPO_ROOT / "data/training_scenes/Imaginarium",
    )
)


def save_glb(mesh_or_scene, save_path: Path) -> None:
    output = mesh_or_scene.copy()
    output.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    trimesh.exchange.export.export_mesh(output, save_path)


def load_glb(file_path: Path):
    mesh_or_scene = trimesh.load(file_path, process=False)
    mesh_or_scene.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    )
    return mesh_or_scene


def transform_from_record(record: dict) -> np.ndarray:
    scale = record["scale"]
    if np.isscalar(scale):
        scale = [scale, scale, scale]
    return trimesh.transformations.compose_matrix(
        scale=scale,
        angles=record["angles"],
        translate=record["trans"],
    )


def export_transformed_scene(
    scene_dir: Path,
    transforms_path: Path,
    export_scene_dir: Path,
) -> None:
    export_scene_dir.mkdir(parents=True, exist_ok=True)
    with transforms_path.open("rb") as stream:
        transforms = pickle.load(stream)

    glb_paths = sorted(scene_dir.glob("*.glb"))
    if not glb_paths:
        raise FileNotFoundError(f"No GLB files found in {scene_dir}")
    for glb_path in glb_paths:
        if glb_path.stem not in transforms:
            raise KeyError(f"No transform found for {glb_path.stem} in {transforms_path}")
        mesh_or_scene = load_glb(glb_path)
        mesh_or_scene.apply_transform(transform_from_record(transforms[glb_path.stem]))
        save_glb(mesh_or_scene, export_scene_dir / glb_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--scene-name", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--scenes-dir", type=Path)
    parser.add_argument("--transforms-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.scene_name and not args.all:
        parser.error("provide --scene-name or --all")
    args.scenes_dir = args.scenes_dir or args.dataset_root / "scenes"
    args.transforms_dir = args.transforms_dir or args.dataset_root / "transforms"
    args.output_dir = args.output_dir or args.dataset_root / "scenes_transformed"
    return args


def main() -> None:
    args = parse_args()
    scene_names = (
        sorted(path.name for path in args.scenes_dir.iterdir() if path.is_dir())
        if args.all
        else args.scene_name
    )
    for scene_name in scene_names:
        print(f"Exporting {scene_name}", flush=True)
        export_transformed_scene(
            args.scenes_dir / scene_name,
            args.transforms_dir / f"{scene_name}.pkl",
            args.output_dir / scene_name,
        )


if __name__ == "__main__":
    main()
