#!/usr/bin/env python3
"""Export canonical-to-scene transforms for InternScenes assets."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import trimesh


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_processing.stages.ovoxel.internscenes_utils import (  # noqa: E402
    DEFAULT_INTERNSCENES_ROOT,
    DEFAULT_RAW_KIT_DIR,
    composer_for,
    ensure_layout_extracted,
    load_layout,
    normalized_background,
    normalized_scene_object,
    scene_name_to_safe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--internscenes-root", type=Path, default=DEFAULT_INTERNSCENES_ROOT)
    parser.add_argument("--raw-kit-dir", type=Path, default=DEFAULT_RAW_KIT_DIR)
    parser.add_argument("--layout-root", type=Path)
    parser.add_argument("--layout-tar", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--extract-layout", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    downloaded = args.internscenes_root / "downloaded"
    layout_root = args.layout_root or downloaded / "Layout_info"
    layout_tar = args.layout_tar or downloaded / "Layout_info.tar.gz"
    output_dir = args.output_dir or args.internscenes_root / "transforms"

    ensure_layout_extracted(args.scene_name, layout_root, layout_tar, args.extract_layout)
    layout = load_layout(args.scene_name, layout_root)
    composer, _module = composer_for(args.raw_kit_dir, args.internscenes_root, layout_root)
    _background, background_transform = normalized_background(args.scene_name, layout_root)
    object_transforms = {}
    for instance in sorted(layout, key=lambda item: int(item.get("id", 0))):
        if not instance.get("model_uid"):
            continue
        _mesh, transform = normalized_scene_object(composer, instance)
        object_transforms[int(instance.get("id", 0))] = transform
    scene_key = scene_name_to_safe(args.scene_name)
    matrices = {f"layout_{scene_key}_bg": background_transform}
    matrices.update(
        {f"{instance_id:04d}": matrix for instance_id, matrix in object_transforms.items()}
    )
    records = {}
    for local_name, matrix in matrices.items():
        scale, _shear, angles, translation, _perspective = (
            trimesh.transformations.decompose_matrix(matrix)
        )
        records[local_name] = {
            "scale": float(scale.reshape(3)[0]),
            "angles": [float(value) for value in angles],
            "trans": [float(value) for value in translation],
            "latent": local_name,
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{scene_key}.pkl"
    with output.open("wb") as stream:
        pickle.dump(records, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {len(records)} transforms to {output}")


if __name__ == "__main__":
    main()
