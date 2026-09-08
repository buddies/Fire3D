"""Discover SAM 3D Objects meshes in the arranged result layout."""

from __future__ import annotations

import re
from pathlib import Path


def discover(run: Path, dataset: str, scene_id: str) -> list[tuple[int, Path, str]]:
    rows = []
    object_dir = run / "sam3d" / dataset / scene_id / "objects"
    for path in sorted(object_dir.glob("object_*.glb")):
        match = re.search(r"object_(\d+)\.glb$", path.name)
        if match:
            rows.append((int(match.group(1)), path, path.stem))
    return rows
