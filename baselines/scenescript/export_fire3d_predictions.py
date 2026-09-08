#!/usr/bin/env python3
"""Convert normalized SceneScript boxes into the Fire3D OBB interchange schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text())
    boxes = payload.get("boxes", payload if isinstance(payload, list) else None)
    if not isinstance(boxes, list):
        raise ValueError("Expected a list or an object containing 'boxes'")
    normalized = []
    for index, box in enumerate(boxes):
        center = box.get("center")
        extent = box.get("extent", box.get("size"))
        rotation = box.get("rotation_matrix")
        if not (
            isinstance(center, list)
            and len(center) == 3
            and isinstance(extent, list)
            and len(extent) == 3
            and isinstance(rotation, list)
            and len(rotation) == 3
            and all(isinstance(row, list) and len(row) == 3 for row in rotation)
        ):
            raise ValueError(f"Invalid normalized SceneScript box at index {index}")
        normalized.append(
            {
                "instance_id": int(box.get("instance_id", index + 1)),
                "score": float(box.get("score", 1.0)),
                "center": [float(value) for value in center],
                "extent": [float(value) for value in extent],
                "rotation_matrix": [[float(value) for value in row] for row in rotation],
            }
        )
    output = {"schema": "fire3d.obb_predictions.v1", "boxes": normalized}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
