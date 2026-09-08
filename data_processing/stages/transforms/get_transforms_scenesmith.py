#!/usr/bin/env python3
"""Export canonical SceneSmith room assets and placement transforms."""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import sys
from pathlib import Path

import trimesh


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = Path(
    os.environ.get(
        "FIRE3D_SCENESMITH_ROOT",
        REPO_ROOT / "data/training_scenes/Scenesmith",
    )
)
DEFAULT_KIT_ROOT = Path(
    os.environ.get(
        "FIRE3D_SCENESMITH_KIT_ROOT",
        REPO_ROOT / "data_processing/_upstream/scenesmith",
    )
)


def load_glb_utils(kit_root: Path):
    module_path = kit_root / "scripts/glb_utils.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"Missing SceneSmith helper {module_path}; fetch the upstream source "
            "or set FIRE3D_SCENESMITH_KIT_ROOT"
        )
    spec = importlib.util.spec_from_file_location("fire3d_scenesmith_glb_utils", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    try:
        sys.path[:0] = [str(kit_root), str(module_path.parent)]
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


def export_scene(scene_path: Path, dataset_root: Path, kit_root: Path) -> int:
    glb_utils = load_glb_utils(kit_root)
    scene_output = dataset_root / "scenes"
    transform_output = dataset_root / "transforms"
    scene_output.mkdir(parents=True, exist_ok=True)
    transform_output.mkdir(parents=True, exist_ok=True)

    rooms = glb_utils.load_scene_meshes_by_room_canonical(scene_path)
    scene_name = "_".join(scene_path.parts[-2:])
    count = 0
    for raw_room_id, room in rooms.items():
        room_id = str(raw_room_id)
        room_name = f"{scene_name}_room_{room_id}"
        objects_output = scene_output / room_name
        objects_output.mkdir(parents=True, exist_ok=True)
        names = ["bg", *sorted(name for name in room if name != "bg")]
        records = {}
        for index, mesh_name in enumerate(names):
            local_name = f"layout_{room_name}" if index == 0 else f"object_{index:04d}"
            info = room[mesh_name]
            glb_utils.save_glb(info["mesh_canonical"], str(objects_output / f"{local_name}.glb"))
            scale, _shear, angles, translation, _perspective = (
                trimesh.transformations.decompose_matrix(info["transform_matrix"])
            )
            records[local_name] = {
                "scale": float(scale.reshape(3)[0]),
                "angles": [float(value) for value in angles],
                "trans": [float(value) for value in translation],
                "latent": local_name,
                "mesh_name": mesh_name,
            }
        with (transform_output / f"{room_name}.pkl").open("wb") as stream:
            pickle.dump(records, stream, protocol=pickle.HIGHEST_PROTOCOL)
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_path", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--kit-root", type=Path, default=DEFAULT_KIT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = export_scene(
        args.scene_path.expanduser().resolve(),
        args.dataset_root.expanduser().resolve(),
        args.kit_root.expanduser().resolve(),
    )
    print(f"Exported {count} SceneSmith rooms")


if __name__ == "__main__":
    main()
