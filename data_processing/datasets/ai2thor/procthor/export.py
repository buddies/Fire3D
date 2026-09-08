#!/usr/bin/env python3
"""Export a ProcTHOR scene GLB from the ai2thor-hab assets."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import EXPORT_DIR, export_scene, get_all_procthor_scenes, save_glb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-name", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path(EXPORT_DIR))
    args = parser.parse_args()
    if not args.scene_name and not args.all:
        parser.error("provide --scene-name or --all")
    return args


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    scene_names = get_all_procthor_scenes() if args.all else args.scene_name
    for scene_name in sorted(scene_names):
        output = args.output_root / f"{scene_name.replace('/', '_')}.glb"
        save_glb(export_scene(scene_name), str(output))
        print(f"Exported {scene_name} to {output}", flush=True)


if __name__ == "__main__":
    main()
