#!/usr/bin/env python3
"""Compose a SceneSmith exported room folder into one GLB.

Input format matches ``export_scene_and_transforms.py``:

- ``scene_dir`` contains normalized/canonical ``<latent>.glb`` files.
- ``transform_pkl`` maps each latent id to ``scale``, ``angles``, and ``trans``.

The script first rotates each loaded GLB back from the export-only rotation
added by ``glb_utils.save_glb`` (``-pi/2`` around X), then reconstructs each
latent's global transform and writes a single GLB containing the transformed
layout and objects.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

# ``glb_utils.save_glb`` applies this export-only rotation before writing each GLB.
SAVE_GLB_ROTATION = trimesh.transformations.rotation_matrix(-np.pi / 2.0, [1, 0, 0])
SAVE_GLB_ROTATION_INV = np.linalg.inv(SAVE_GLB_ROTATION)


def load_transform_dict(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"transform pkl must contain a dict, got {type(data)!r}: {path}")
    return data


def entry_to_matrix(entry: dict[str, Any]) -> np.ndarray:
    """Reconstruct the matrix saved by export_scene_and_transforms.py."""
    if "matrix" in entry:
        matrix = np.asarray(entry["matrix"], dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"matrix entry must be 4x4, got {matrix.shape}")
        return matrix

    scale_value = float(entry.get("scale", 1.0))
    angles = np.asarray(entry.get("angles", [0.0, 0.0, 0.0]), dtype=np.float64)
    trans = np.asarray(entry.get("trans", [0.0, 0.0, 0.0]), dtype=np.float64)
    if angles.shape != (3,):
        raise ValueError(f"angles must have shape (3,), got {angles.shape}")
    if trans.shape != (3,):
        raise ValueError(f"trans must have shape (3,), got {trans.shape}")

    return trimesh.transformations.compose_matrix(
        scale=[scale_value, scale_value, scale_value],
        angles=angles,
        translate=trans,
    )


def add_scene_with_prefix(
    composed: trimesh.Scene,
    source: trimesh.Scene,
    prefix: str,
    transform: np.ndarray,
) -> int:
    """Add all geometry nodes from source to composed with a latent-name prefix."""
    count = 0
    nodes = list(source.graph.nodes_geometry)
    if nodes:
        for node_name in nodes:
            local_transform, geom_name = source.graph[node_name]
            geom = source.geometry[geom_name].copy()
            composed.add_geometry(
                geom,
                node_name=f"{prefix}/{node_name}",
                geom_name=f"{prefix}/{geom_name}",
                transform=transform @ local_transform,
            )
            count += 1
        return count

    # Fallback for unusual GLBs without graph nodes.
    for geom_name, geom in source.geometry.items():
        composed.add_geometry(
            geom.copy(),
            node_name=f"{prefix}/{geom_name}",
            geom_name=f"{prefix}/{geom_name}",
            transform=transform,
        )
        count += 1
    return count


def compose_scene(
    scene_dir: Path,
    transform_pkl: Path,
    output_path: Path,
    skip_missing: bool,
    rotate_back: bool,
) -> tuple[int, int]:
    transforms = load_transform_dict(transform_pkl)
    composed = trimesh.Scene()
    missing = 0
    added_nodes = 0

    for latent, entry in transforms.items():
        latent_name = str(entry.get("latent", latent))
        glb_path = scene_dir / f"{latent_name}.glb"
        if not glb_path.exists():
            missing += 1
            message = f"missing GLB for latent {latent_name}: {glb_path}"
            if skip_missing:
                print(f"warning: {message}")
                continue
            raise FileNotFoundError(message)

        source = trimesh.load(glb_path, force="scene", process=False)
        if not isinstance(source, trimesh.Scene):
            source = trimesh.Scene(source)
        if rotate_back:
            source.apply_transform(SAVE_GLB_ROTATION_INV)
        matrix = entry_to_matrix(entry)
        added_nodes += add_scene_with_prefix(composed, source, latent_name, matrix)

    if added_nodes == 0:
        raise RuntimeError("no geometry was added to the composed scene")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    composed.export(output_path)
    return added_nodes, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", type=Path, help="Exported room folder containing <latent>.glb files.")
    parser.add_argument("transform_pkl", type=Path, help="Transform PKL for the same exported room.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output composed GLB. Default: <scene_dir>/composed.glb",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip transform entries whose <latent>.glb is missing instead of failing.",
    )
    parser.add_argument(
        "--no-rotate-back",
        action="store_true",
        help="Do not undo the -pi/2 X export rotation added by glb_utils.save_glb.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_dir = args.scene_dir.expanduser().resolve()
    transform_pkl = args.transform_pkl.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else scene_dir / "composed.glb"
    )

    if not scene_dir.is_dir():
        raise SystemExit(f"scene_dir does not exist: {scene_dir}")
    if not transform_pkl.is_file():
        raise SystemExit(f"transform_pkl does not exist: {transform_pkl}")

    added_nodes, missing = compose_scene(
        scene_dir,
        transform_pkl,
        output_path,
        args.skip_missing,
        rotate_back=not args.no_rotate_back,
    )
    print(f"wrote {output_path}")
    print(f"added_geometry_nodes={added_nodes} missing_latents={missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
