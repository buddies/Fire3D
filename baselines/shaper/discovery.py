"""Discover ShapeR object meshes in supported result layouts."""

from __future__ import annotations

import re
from pathlib import Path


def discover(run: Path, dataset: str, scene_id: str) -> list[tuple[int, Path, str]]:
    rows = []
    arranged = run / "shaper" / dataset / scene_id / "objects"
    if arranged.is_dir():
        for path in sorted(arranged.glob("object_*.ply")):
            match = re.search(r"object_(\d+)\.ply$", path.name)
            if match:
                rows.append((int(match.group(1)), path, path.stem))
        if rows:
            return rows

    pattern = f"worker_*/flat_output/{dataset}__{scene_id}__object_*.ply"
    for path in sorted((run / "shaper/workers").glob(pattern)):
        match = re.search(r"object_(\d+)\.ply$", path.name)
        if match:
            rows.append((int(match.group(1)), path, path.stem))
    return rows
