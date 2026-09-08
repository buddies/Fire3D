"""Mesh-scene composition helpers for the composed reconstruction baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import trimesh


def load_mesh_or_scene(path: Path) -> trimesh.Scene:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        return loaded
    scene = trimesh.Scene()
    scene.add_geometry(loaded)
    return scene


def compose_world_scene(world_glbs: list[Path], output_path: Path) -> None:
    """Compose GLBs while baking every source scene-graph node transform."""
    composed = trimesh.Scene()
    for path in world_glbs:
        scene = load_mesh_or_scene(path)
        for index, node_name in enumerate(sorted(scene.graph.nodes_geometry)):
            transform, geometry_name = scene.graph[node_name]
            geometry: Any = scene.geometry[geometry_name].copy()
            geometry.apply_transform(transform)
            unique_name = (
                f"{path.parent.name}_{path.stem}_{index:04d}_{geometry_name}"
            )
            composed.add_geometry(
                geometry,
                geom_name=unique_name,
                node_name=unique_name,
            )
    if composed.geometry:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        composed.export(output_path)
