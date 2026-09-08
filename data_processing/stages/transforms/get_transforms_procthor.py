#!/usr/bin/env python3
"""Export canonical-to-scene transforms for ProcTHOR rooms."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import trimesh


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_processing.datasets.ai2thor.procthor import utils as procthor  # noqa: E402


DEFAULT_TRAINING_ROOT = Path(
    os.environ.get("FIRE3D_TRAINING_ROOT", REPO_ROOT / "data/training_scenes")
)
DEFAULT_PROCTHOR_ROOT = Path(
    os.environ.get("FIRE3D_PROCTHOR_ROOT", DEFAULT_TRAINING_ROOT / "ProcTHOR")
)


def transform_record(local_name: str, matrix) -> dict:
    scale, _shear, angles, translation, _perspective = (
        trimesh.transformations.decompose_matrix(matrix)
    )
    return {
        "latent": local_name,
        "scale": float(scale.reshape(3)[0]),
        "angles": [float(value) for value in angles],
        "trans": [float(value) for value in translation],
    }


def export_scene(scene_name: str, output_dir: Path) -> list[Path]:
    _room_regions, rooms = procthor.export_rooms_with_canonical_meshes(scene_name)
    scene_key = scene_name.replace("/", "_")
    outputs = []
    for raw_room_id, room in rooms.items():
        room_id = str(raw_room_id)
        room_key = f"{scene_key}_room_{room_id}"
        names = ["bg", *sorted(name for name in room if name != "bg")]
        records = {}
        for index, mesh_name in enumerate(names):
            local_name = f"layout_{room_key}" if index == 0 else f"object_{index:04d}"
            records[local_name] = transform_record(
                local_name,
                room[mesh_name]["total_transform"],
            )
        output = output_dir / f"{room_key}.pkl"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as stream:
            pickle.dump(records, stream, protocol=pickle.HIGHEST_PROTOCOL)
        outputs.append(output)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-name", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--procthor-root", type=Path, default=DEFAULT_PROCTHOR_ROOT)
    parser.add_argument("--ai2thor-hab-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.scene_name and not args.all:
        parser.error("provide --scene-name or --all")
    args.ai2thor_hab_root = args.ai2thor_hab_root or args.procthor_root / "ai2thor-hab"
    args.output_dir = args.output_dir or args.procthor_root / "transforms"
    return args


def main() -> None:
    args = parse_args()
    procthor.configure_paths(args.ai2thor_hab_root)
    scene_names = procthor.get_all_procthor_scenes() if args.all else args.scene_name
    for scene_name in sorted(scene_names):
        outputs = export_scene(scene_name, args.output_dir)
        print(f"Exported {scene_name}: {len(outputs)} rooms", flush=True)


if __name__ == "__main__":
    main()
