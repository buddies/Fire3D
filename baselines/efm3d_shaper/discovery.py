"""Discover ShapeR meshes reconstructed from EFM3D detections."""

from __future__ import annotations

from pathlib import Path


def discover(run: Path, dataset: str, scene_id: str) -> list[tuple[int, Path, str]]:
    # Predicted detections have names rather than GT ids. Stable one-based ids
    # preserve instance 0 for the repository-wide background convention.
    mesh_dir = run / "reconstruction" / dataset / scene_id
    paths = sorted(mesh_dir.glob(f"{scene_id}__*.ply"))
    return [
        (index, path, path.stem.split("__", 1)[1])
        for index, path in enumerate(paths, start=1)
    ]
